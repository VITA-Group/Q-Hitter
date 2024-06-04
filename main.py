import os
import tqdm
import json
import copy
import math
import torch
import logging
import argparse
import numpy as np
import dataclasses
from xopen import xopen
from rouge import Rouge
import matplotlib.pyplot as plt 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.modify_llama import QH2OLlamaForCausalLM, QH2OLlamaAttention

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": None,
    "llama_qh2o": QH2OLlamaForCausalLM,
}

TAGET_MODULE = {
    "llama": None,
    "llama_qh2o": QH2OLlamaAttention,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")

    # KV Cache Policy
    parser.add_argument("--hh_size", type=float, default=0.15)
    parser.add_argument("--recent_size", type=float, default=0.05)
    # For quantization
    parser.add_argument("--kbits", type=int, default=4)
    parser.add_argument("--vbits", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument('--enable_qh2o_cache', action='store_true')

    parser.add_argument("--seed", type=int, default=2, help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if args.enable_qh2o_cache:
        if args.alpha == 1:
            print('Enabling Quantization H2O KV cache')
        else:
            print('Enabling Q-Hitter Cache')
        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        config.kbits = args.kbits
        config.vbits = args.vbits
        config.alpha = args.alpha
        model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_qh2o'].from_pretrained(model_name, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))
    results = []
    rouge = Rouge()
    rouge_score_list = []

    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['article']
            label = request['summary_gt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
            )

            if args.enable_qh2o_cache:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE['llama_qh2o']):
                        m._clean_cache()

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            scores = rouge.get_scores(generate_text, label)[0]
            rouge_score_list.append(scores['rouge-l']['f'])
            results.append(result)
            print('{:.6f}'.format(np.mean(rouge_score_list)))

    print('Rouge-L: {:.6f}'.format(np.mean(rouge_score_list)))
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
