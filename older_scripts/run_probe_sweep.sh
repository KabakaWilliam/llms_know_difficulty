#!/bin/bash

# Script to run through multiple GPT OSS models and datasets

# GPU configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

# Define models (low, medium, high reasoning levels)
declare -a MODELS=(
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-Math-1.5B"
    # "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # "Qwen/Qwen2.5-Math-7B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "Qwen/Qwen2.5-Math-7B-Instruct"
    # "Qwen/Qwen2.5-Math-72B-Instruct"
    # "openai/gpt-oss-20b_low"
    # "openai/gpt-oss-20b_medium"
    "openai/gpt-oss-20b_high"
    # "Qwen/Qwen2.5-Coder-3B-Instruct"
    # "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "Qwen2.5-Coder-32B-Instruct"
    
)

# Define datasets
declare -a DATASETS=(
    # "openai_gsm8k"
    "DigitalLearningGmbH_MATH-lighteval"
    # "gneubig_aime-1983-2024"
    # "E2H-AMC"
    # "livecodebench_code_generation_lite"
)

# Define probes to use
declare -a PROBES=(
    "linear_eoi_probe"
    # "tfidf_probe"
    # "length_probe"
)

# Generation parameters (GPT Config)
MAX_LEN=131072
K=5
TEMPERATURE=1.0
LABEL_COLUMN="majority_vote_is_correct" #"majority_vote_is_correct" #"success_rate" #"pass_at_k" "rating"

# # Generation parameters (Qwen Config)
# MAX_LEN=3000
# K=5
# TEMPERATURE=0.7
# LABEL_COLUMN="majority_vote_is_correct" #"majority_vote_is_correct" #"success_rate" #"pass_at_k"

# Generation parameters (Qwen LCB Config)
# MAX_LEN=4096
# K=5
# TEMPERATURE=0.2
# LABEL_COLUMN="pass_at_k" #"majority_vote_is_correct" #"success_rate" #"pass_at_k"

# LABEL_COLUMN="majority_vote_is_correct" #"majority_vote_is_correct" #"success_rate" #"pass_at_k"

# # # Generation parameters (DS Config)
# MAX_LEN=32768
# K=5
# TEMPERATURE=0.6
# LABEL_COLUMN="majority_vote_is_correct" #"majority_vote_is_correct" #"success_rate" #"pass_at_k"

# Change to source directory
cd "$(dirname "$0")" || exit

# Counter for progress
total_runs=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#PROBES[@]}))
current_run=0

echo "========================================"
echo "Starting Model Sweep"
echo "Total runs to execute: $total_runs"
echo "========================================"

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for probe in "${PROBES[@]}"; do
            ((current_run++))
            
            echo ""
            echo "========================================"
            echo "Run $current_run / $total_runs"
            echo "Model: $model"
            echo "Dataset: $dataset"
            echo "Probe: $probe"
            echo "========================================"
            
            python3 src/pika/main.py \
                --probe "$probe" \
                --dataset "$dataset" \
                --model "$model" \
                --max_len "$MAX_LEN" \
                --k "$K" \
                --temperature "$TEMPERATURE" \
                --label_column "$LABEL_COLUMN" \
            
            if [ $? -eq 0 ]; then
                echo "✅ Run $current_run completed successfully"
            else
                echo "❌ Run $current_run failed"
            fi
        done
    done
done

echo ""
echo "========================================"
echo "All runs completed!"
echo "========================================"
