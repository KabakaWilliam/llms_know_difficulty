#!/bin/bash

# Simple script to run multiple models
# Edit the models list below to add/remove models

# List of models to run (under 14B parameters)
models=(
    "Qwen/Qwen2.5-Coder-1.5B"
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"

#     "Qwen/Qwen2.5-Math-1.5B-Instruct"
#     "Qwen/Qwen2.5-Math-7B"
#     "Qwen/Qwen2.5-Math-7B-Instruct"
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "Qwen/Qwen2.5-Math-1.5B"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-3B"
#     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#     "HuggingFaceTB/FineMath-Llama-3B"
#     "nvidia/OpenMath2-Llama3.1-8B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    # "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)


# CHOSEN_DEVICE=2
# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --model_path HuggingFaceTB/FineMath-Llama-3B\
#     --use_k_fold \

# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --model_path HuggingFaceTB/FineMath-Llama-3B \
#     --use_k_fold \
#     --n_train 800 \
#     --n_test 200 \
#     --batch_size 16 \
#     --generation_batch_size 8 \
#     --max_new_tokens 2000 \
#     --subset_datasets E2H-AMC E2H-GSM8K \
#     --evaluation_datasets E2H-AMC \
#     --model_alias my_custom_alias \
#     --save_dir /path/to/custom/save/dir
# Qwen/Qwen2.5-Math-7B
# allenai/Olmo-3-7B-Think
#  Qwen/Qwen2.5-Math-1.5B-Instruct 3000
CHOSEN_DEVICE=1
CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-1.5B\
    --use_k_fold \
    --batch_size 16 \
    --n_train 1000 \
    --n_test 500 \
    --subset_datasets predicting_learnability  \
    --evaluation_datasets predicting_learnability\
    --generation_batch_size 8 \
    --max_new_tokens 2000 \

# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --config_file runs/Qwen2.5-Math-7B/config.yaml \
#     --generate_responses \
#     --resume_from_step 2
# echo "All models completed!"


