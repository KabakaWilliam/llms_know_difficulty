#!/bin/bash

# Script to run through multiple GPT OSS models and datasets

# GPU configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2

# Define models (low, medium, high reasoning levels)
declare -a MODELS=(
    # "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # "Qwen/Qwen2.5-Math-7B-Instruct"
    # "Qwen/Qwen2.5-Math-72B-Instruct"
    # "openai/gpt-oss-20b_low"
    # "openai/gpt-oss-20b_medium"
    "openai/gpt-oss-20b_high"
)

# Define datasets
declare -a DATASETS=(
    # "gneubig_aime-1983-2024"
    # "openai_gsm8k"
    "DigitalLearningGmbH_MATH-lighteval"
    # "livecodebench_code_generation_lite"
)

# Define probes to use
declare -a PROBES=(
    "linear_eoi_probe"
    # "tfidf_probe"
)

# Generation parameters
MAX_LEN=131072
K=5
TEMPERATURE=1.0
LABEL_COLUMN="majority_vote_is_correct" #"majority_vote_is_correct" #"success_rate" #"pass_at_k"

# MAX_LEN=3000
# K=5
# TEMPERATURE=0.7
# LABEL_COLUMN="success_rate" #"majority_vote_is_correct" #"success_rate" #"pass_at_k"

# Change to source directory
cd "$(dirname "$0")" || exit

# Counter for progress
total_runs=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#PROBES[@]}))
current_run=0

echo "========================================"
echo "Starting GPT OSS Sweep"
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
            
            python3 src/llms_know_difficulty/main.py \
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
