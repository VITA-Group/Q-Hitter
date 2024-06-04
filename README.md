# Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Codes for this paper **Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache [MLSys'24]**

Zhenyu Zhang\*, Shiwei Liu*, Runjin Chen, Bhavya Kailkhura, Beidi Chen, Zhangyang Wang

## Overview

This paper focuses on addressing the substantial memory footprints and bandwidth costs associated with the deployment of Large Language Models (LLMs). LLMs, characterized by their extensive context length, inherently demands vast memory resource and traffic to store and load the attention key and value embeddings within self-attention modules, referred to as the KV Cache. In an effort to alleviate these resource-intensive aspects of LLM inference, techniques such as sparsification and quantization for KV Cache reduction have been investigated as separate endeavors within the realm of LLMs. However, this paper illuminates the critical importance of considering the compound effects of these techniques when employed together, as a simplistic amalgamation of sparsification and quantization can yield sub-optimal performance. For instance, the Heavy Hitter Oracle (H20) has demonstrated that preserving just 20 percent of the KV Cache attributed to pivotal tokens, denoted as Heavy Hitters, can yield substantial memory savings while upholding the model's original performance. Furthermore, the KV Cache of these Heavy Hitter tokens, which are identified as those with the highest accumulated attention scores, can be further quantized with encouraging throughput saving. Nevertheless, our investigation uncovers two primary deficiencies in such unrefined post-sparsification quantization in low-bit scenarios: (1) the application of low-bit KV Cache quantization, significantly diminishes the accuracy of Heavy Hitter selection during the generation phase, particularly in deeper layers; (2) tokens selected by the Heavy Hitter Oracle are not necessarily well-suited for quantization, and their quantization can lead to sub-optimal performance. To overcome these challenges, we propose a novel rule-of-thumb for token selection during LLM generation, termed Q-Hitter. This approach combines both accumulated attention scores and Quantization Friendliness metrics for different layers, identifying tokens that are not only pivotal for preserving the generalization capabilities of LLMs but are also more amenable to KV Cache quantization. Q-Hitter naturally offers a free lunch of KV Cache quantization and can further escalate the affordability of state-of-the-art LLMs. Additionally, we also demonstrate that Q-Hitter empowers LLMs to effectively handle inputs of infinite sequence length, enhancing the capacity of LLMs to process a more extensive range of informations.

## Environment

```
conda env create -f environment.yaml
```

## Usage

```
# llama-2-7b-chat on xsum with full cache
bash src/full.sh

# llama-2-7b-chat on xsum with h2o+quantization cache policy
bash src/h2o_w_q.sh

# llama-2-7b-chat on xsum with q-hitter cache policy
bash src/qhitter.sh
```

## Citation

```
@article{zhang2024q,
  title={Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache},
  author={Zhang, Zhenyu and Liu, Shiwei and Chen, Runjin and Kailkhura, Bhavya and Chen, Beidi and Wang, Atlas},
  journal={Proceedings of Machine Learning and Systems},
  volume={6},
  pages={381--394},
  year={2024}
}
```
