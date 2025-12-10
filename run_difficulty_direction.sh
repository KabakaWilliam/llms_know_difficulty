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


CHOSEN_DEVICE=0
CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --use_k_fold \
    --batch_size 16 \
    --n_train 12000 \
    --n_test 500 \
    --subset_datasets predicting_MATH_learnability \
    --subdirectory MATH \
    --max_tokens 32768 \
    --k 1 \
    --temperature 0 \
    --evaluation_datasets GSM_HARD E2H-GSM8K \
    --generation_batch_size 8 \
    --max_new_tokens 2000 \
    --resume_from_step 3



# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --config_file runs/Qwen2.5-Math-7B/config.yaml \
#     --generate_responses \
#     --resume_from_step 2
# echo "All models completed!"