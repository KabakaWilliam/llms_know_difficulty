#!/bin/bash

# Activate conda environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate diff-direction


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

# MATH Probe config
# CHOSEN_DEVICE=2
# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \
#     --use_k_fold \
#     --batch_size 16 \
#     --n_train 12000 \
#     --n_test 500 \
#     --subset_datasets E2H-GSM8K \
#     --subdirectory MATH \
#     --max_tokens 3000 \
#     --k 1 \
#     --temperature 0.0 \
#     --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025 \
#     --generation_batch_size 8 \
#     --max_new_tokens 2000 \
#     # --resume_from_step 3

# #GSM8K Probe config #
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
# #     # --resume_from_step 3


# MATH_X_GSM8K
CHOSEN_DEVICE=3
CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --use_k_fold \
    --batch_size 32 \
    --n_train 14946 \
    --n_test 1000 \
    --subset_datasets predicting_MATH_X_GSM8K_SR \
    --subdirectory MATH_X_GSM8K \
    --max_tokens 3000 \
    --k 1 \
    --temperature 0.0 \
    --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025 \
    --generation_batch_size 8 \
    --max_new_tokens 2000 \
    # --resume_from_step 3