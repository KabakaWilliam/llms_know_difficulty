#!/bin/bash

# Activate conda environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate diff-direction

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
#     --resume_from_step 2 \
#     --save_dir /path/to/custom/save/dir
# Qwen/Qwen2.5-Math-7B
# allenai/Olmo-3-7B-Think
#  Qwen/Qwen2.5-Math-1.5B-Instruct 3000
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# CHOSEN_DEVICE=1 #not run yet
# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \
#     --use_k_fold \
#     --batch_size 16 \
#     --n_train 12000 \
#     --n_test 500 \
#     --subset_datasets predicting_MATH_learnability \
#     --subdirectory MATH \
#     --max_tokens 3000 \
#     --k 1 \
#     --temperature 0.0 \
#     --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025\
#     --generation_batch_size 8 \
#     --max_new_tokens 2000 \

## MATH Probe config
CHOSEN_DEVICE=2
CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-7B-Instruct \
    --use_k_fold \
    --batch_size 16 \
    --n_train 12000 \
    --n_test 500 \
    --subset_datasets predicting_MATH_learnability \
    --subdirectory MATH \
    --max_tokens 3000 \
    --k 1 \
    --temperature 0.0 \
    --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025 \
    --generation_batch_size 8 \
    --max_new_tokens 2000 \
    # --resume_from_step 3

# #GSM8K Probe config
# CHOSEN_DEVICE=2 
# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --model_path Qwen/Qwen2.5-Math-7B-Instruct \
#     --use_k_fold \
#     --batch_size 16 \
#     --n_train 7473 \
#     --n_test 1319 \
#     --subset_datasets predicting_GSM8K_SR \
#     --subdirectory GSM8K \
#     --max_tokens 3000 \
#     --k 1 \
#     --temperature 0.0 \
#     --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025 \
#     --generation_batch_size 8 \
#     --max_new_tokens 2000 \
#     # --resume_from_step 3