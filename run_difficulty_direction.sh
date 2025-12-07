#!/bin/bash



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
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --use_k_fold \
    --batch_size 16 \
    --n_train 2400 \
    --n_test 1600 \
    --subset_datasets E2H-Codeforces \
    --evaluation_datasets E2H-Codeforces \
    --generation_batch_size 8 \
    --max_new_tokens 2000

# CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
#     --config_file runs/Qwen2.5-Math-7B/config.yaml \
#     --generate_responses \
#     --resume_from_step 2
# echo "All models completed!"


