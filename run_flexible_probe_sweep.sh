#!/bin/bash

# Flexible probe sweep script that supports different gen configs per model
# This allows running models with their proper maxlen/k/temp settings in one sweep

# GPU configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2

# Define model configurations as an array of "model|maxlen|k|temp"
# Format: "model_name|max_len|k|temperature"
declare -a MODEL_CONFIGS=(
    # "Qwen/Qwen2.5-Math-1.5B-Instruct|3000|5|0.7"
    # "Qwen/Qwen2.5-Math-7B-Instruct|3000|5|0.7"
    # "Qwen/Qwen2.5-1.5B-Instruct|3000|5|0.7"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|32768|5|0.6"
    # "openai/gpt-oss-20b_low|131072|5|1.0"
    "openai/gpt-oss-20b_medium|131072|5|1.0"
    # "openai/gpt-oss-20b_high|131072|5|1.0"
    # Add more models here with their specific configs
    # "Qwen/Qwen2.5-Coder-7B-Instruct|4096|5|0.2"
)

# Define datasets to process
declare -a DATASETS=(
    # "gneubig_aime-1983-2024"
    # "openai_gsm8k"
    "DigitalLearningGmbH_MATH-lighteval"
    # "E2H-AMC"
    # "livecodebench_code_generation_lite"
)

# Define probes to use
declare -a PROBES=(
    "linear_eoi_probe"
    # "tfidf_probe"
    # "length_probe"
)

# Label column to use for all models
LABEL_COLUMN="majority_vote_is_correct"  # or "success_rate" or "pass_at_k"

# Change to source directory
cd "$(dirname "$0")" || exit

# Counter for progress
total_runs=$((${#MODEL_CONFIGS[@]} * ${#DATASETS[@]} * ${#PROBES[@]}))
current_run=0

echo "========================================"
echo "Starting Flexible Model Sweep"
echo "Models: ${#MODEL_CONFIGS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "Probes: ${#PROBES[@]}"
echo "Total runs to execute: $total_runs"
echo "========================================"

# Loop through all combinations
for model_config in "${MODEL_CONFIGS[@]}"; do
    # Parse the config string
    IFS='|' read -r model max_len k temperature <<< "$model_config"
    
    for dataset in "${DATASETS[@]}"; do
        for probe in "${PROBES[@]}"; do
            ((current_run++))
            
            echo ""
            echo "========================================"
            echo "Run $current_run / $total_runs"
            echo "Model: $model"
            echo "Dataset: $dataset"
            echo "Probe: $probe"
            echo "Config: maxlen=$max_len, k=$k, temp=$temperature"
            echo "========================================"
            
            python3 src/pika/main.py \
                --probe "$probe" \
                --dataset "$dataset" \
                --model "$model" \
                --max_len "$max_len" \
                --k "$k" \
                --temperature "$temperature" \
                --label_column "$LABEL_COLUMN"
            
            if [ $? -eq 0 ]; then
                echo "✅ Run $current_run completed successfully"
            else
                echo "❌ Run $current_run failed"
                # Optionally uncomment to stop on first failure:
                # exit 1
            fi
        done
    done
done

echo ""
echo "========================================"
echo "All runs completed!"
echo "========================================"
