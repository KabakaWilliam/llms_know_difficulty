#!/bin/bash

python predict_success_rate.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --hf_dataset DigitalLearningGmbH/MATH-lighteval \
    --train_scores_path /users/staff/dmi-dmi/bankes0000/llms_know_difficulty/data/MATH_train-Qwen-Qwen2-1.5B-Instruct.parquet \
    --val_scores_path /users/staff/dmi-dmi/bankes0000/llms_know_difficulty/data/MATH_test-Qwen-Qwen2-1.5B-Instruct.parquet \
    --layer_mode attn_lite \
    --pool all_sequence_positions \
    --layers 17 \
    --max_length 512 \
    --lr 1e-4 \
    --batch_size 128 \
    --wandb_run_name "attn_lite_layers_17_full_train" \
    --epochs 10


