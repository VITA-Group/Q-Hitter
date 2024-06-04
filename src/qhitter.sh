CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --input_path data/xsum.jsonl \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --enable_qh2o_cache \
    --alpha 0.5 \
    --output_path output/xsum_llama_2_7b_chat_q_hitter.jsonl 
