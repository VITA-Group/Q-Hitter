'''
KV Cache Outlier
'''

import os
import sys
import pdb
import math
import copy
import types
import torch
from typing import Optional, Tuple

from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaForCausalLM,
)

__all__ = ['QH2OLlamaForCausalLM', 'QH2OLlamaAttention']

def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

## Simulation Quantizationl; Real Implementation are based on https://github.com/FMInference/H2O/blob/main/h2o_flexgen/flexgen/compression.py
def Quantization_simulated(w, n_bit=8, inplace=False):
    # BS, HEADS, TOKEN, DIMS
    org_w_shape = w.shape

    max_val = w.amax(dim=-1, keepdim=True)
    min_val = w.amin(dim=-1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w, scales


def Qerror(w, n_bit=8, inplace=False, normalize=False, norm=2):
    w_copy = copy.deepcopy(w)
    qw, _ = Quantization_simulated(w, n_bit, inplace)
    error = (qw - w_copy).norm(p=norm, dim=-1)

    if normalize:
        min_error, max_error = error.min(), error.max()
        n_error = (error - min_error) / (max_error - min_error)
        return n_error

    return error


class QH2OKVCache:
    def __init__(
        self,
        hh_size=0.2,
        recent_size=0,
        k_seq_dim=2,
        v_seq_dim=2,
        kbits=4,
        vbits=4,
        lambda_hh=1,
        default_ratio=0.07,
    ):  
        self.default_ratio = default_ratio
        self.hh_size_ratio = hh_size - self.default_ratio
        self.recent_size_ratio = recent_size
        self.cache_size_ratio = hh_size + recent_size + self.default_ratio
        self.cache_size = None
        self.hh_size = None
        self.default_size = None
        self.recent_size = None
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_bits = kbits
        self.v_bits = vbits
        self.hh_score = None
        self.q_score = None
        self.combination_score = None
        self.lambda_hh = lambda_hh

        assert self.cache_size_ratio > 0

    def __call__(self, past_key_values, attn_score_cache):

        self._update_hh_score(attn_score_cache)
        self._update_q_score(past_key_values)
        self._update_combination_score()

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        if self.default_size > 0:
            select_default_scores = self.hh_score[:, :seq_len - self.recent_size]
            _, keep_topk_default = torch.topk(select_default_scores, self.default_size, dim=-1)
            keep_topk_default = keep_topk_default.sort().values
        else:
            keep_topk_default = None

        if self.hh_size > 0:
            select_hh_scores = self.combination_score[:, :seq_len - self.recent_size]

            if keep_topk_default is not None:
                select_hh_scores = select_hh_scores.scatter(-1, keep_topk_default, -1)

            _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
            keep_topk = keep_topk.sort().values
        else:
            keep_topk = None

        if keep_topk_default is not None:
            keep_topk = torch.cat([keep_topk, keep_topk_default], dim=-1)

        if self.recent_size > 0:
            keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values[0].device).repeat(num_heads, 1)
            if keep_topk is not None:
                keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
            else:
                keep_idx = keep_recent
        else:
            keep_idx = keep_topk

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)
        self.q_score= self.q_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            # set-up cache size
            self.hh_size = int(self.hh_size_ratio * num_new_tokens)
            self.recent_size = int(self.recent_size_ratio * num_new_tokens)
            self.default_size = int(self.default_ratio * num_new_tokens)
            self.cache_size = self.hh_size + self.recent_size + self.default_size

            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    ## Using raw quantization error
    def _update_q_score(self, past_key_values):

        inplace_k = True
        inplace_v = True

        if self.q_score is None:
            kerror = Qerror(past_key_values[0][0], n_bit=self.k_bits, inplace=inplace_k, normalize=True)
            verror = Qerror(past_key_values[1][0], n_bit=self.v_bits, inplace=inplace_v, normalize=True)
            self.q_score = (2 - kerror - verror)/2
        else:
            kerror = Qerror(past_key_values[0][0,:,-1:,:], n_bit=self.k_bits, inplace=inplace_k, normalize=True)
            verror = Qerror(past_key_values[1][0,:,-1:,:], n_bit=self.v_bits, inplace=inplace_v, normalize=True)
            new_q_score = (2 - kerror - verror)/2
            self.q_score = torch.cat([self.q_score, new_q_score], dim=-1)

    def _update_combination_score(self):
        min_hh, max_hh = self.hh_score.min(), self.hh_score.max()
        self.combination_score = self.lambda_hh * (self.hh_score - min_hh) / (max_hh - min_hh) + (1 - self.lambda_hh) * self.q_score

    def _clean_scores(self):
        self.hh_score = None
        self.q_score = None
        self.combination_score = None
        self.default_size = None


class QH2OLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    """H2O version of LlamaAttention with Absolute Position Embedding"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = QH2OKVCache(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            kbits=config.kbits,
            vbits=config.vbits,
            lambda_hh=config.alpha
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # key/value are already rotated
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QH2OLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = QH2OLlamaAttention(config)



