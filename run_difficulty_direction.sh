#!/bin/bash

# Activate conda environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate diff-direction

CHOSEN_DEVICE=2
CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-7B-Instruct \
    --use_k_fold \
    --batch_size 32 \
    --n_train 14946 \
    --n_test 2975 \
    --subset_datasets E2H-AMC \
    --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025 \

curl -d "Finished calculating SR" ntfy.sh/wills-kasirye-site-llm-logs


# CHOSEN_DEVICE=2
# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --model_path Qwen/Qwen2.5-7B-Instruct \
#     --use_k_fold \
#     --batch_size 32 \
#     --n_train 14946 \
#     --n_test 2975 \
#     --subset_datasets E2H-AMC \
#     --evaluation_datasets GSM_HARD E2H-GSM8K AIME_1983_2024 AIME_2025 \
#     # --generation_batch_size 8 \
#     # --max_new_tokens 2000 \
#     # --resume_from_step 3
# curl -d "Finished calculating SR" ntfy.sh/wills-kasirye-site-llm-logs