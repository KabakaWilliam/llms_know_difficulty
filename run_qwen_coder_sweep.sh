#!/bin/bash

# Script to run through multiple GPT OSS models and datasets

# GPU configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Define models (low, medium, high reasoning levels)
declare -a MODELS=(
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "Qwen/Qwen2.5-Coder-14B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    # "Qwen/Qwen3-Coder-30B-A3B-Instruct"
)

# Define datasets
declare -a DATASETS=(
    # "DigitalLearningGmbH_MATH-lighteval"
    # "openai_gsm8k"
    # "gneubig_aime-1983-2024"
    # "opencompass_AIME2025"
    "livecodebench_code_generation_lite"
)

# Define probes to use
declare -a PROBES=(
    # "tfidf_probe"
    "linear_eoi_probe"
)

# Generation parameters
MAX_LEN=4096
K=1
TEMPERATURE=0.2
LABEL_COLUMN="pass_at_k"

# Change to source directory
cd "$(dirname "$0")" || exit

# Counter for progress
total_runs=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#PROBES[@]}))
current_run=0

echo "========================================"
echo "Starting Qwen Coder Sweep"
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
                --label_column "$LABEL_COLUMN"
            
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
