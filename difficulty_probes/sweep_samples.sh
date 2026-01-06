#!/bin/bash
# Sweep through different models, sample sizes and temperatures using THOM_MATH data

set -e  # Exit on error

# Configuration
MODELS=(
    "Qwen/Qwen2.5-72B-Instruct"
    "Qwen/Qwen2.5-Math-72B-Instruct"
    # "Qwen/Qwen2.5-Math-7B-Instruct"
    # "Qwen/Qwen2.5-Math-1.5B-Instruct"
    # Add more models here as needed
)
# MODELS=(
#     "Qwen/Qwen2.5-7B-Instruct"
#     "Qwen/Qwen2.5-Math-7B-Instruct"
#     "Qwen/Qwen2-1.5B-Instruct"
#     "Qwen/Qwen2.5-Math-1.5B-Instruct"
#     "Qwen/Qwen2-7B-Instruct"
# )

# Sweep parameters
TEMPERATURES=(1.0 0.5)  # List of temperatures to test
SAMPLE_SIZES=(100 1000)  # List of sample sizes to test (100, 1000)

MAIN_DATA_DIR="DATA"

SELECTED_DATASET="THOM_MATH"

DATASET_DIR="${MAIN_DATA_DIR}/SR_DATA/${SELECTED_DATASET}"

QUESTION_COL="formatted_prompt"
LABEL_COL="success_rate"
LAYERS="all"
BATCH_SIZE=32
GPU=0

# Skip activation extraction if they already exist
SKIP_ACTIVATIONS=false  # Set to true to skip extraction and reuse existing activations

# Regularization parameters
ALPHA_GRID="0,0.001,0.01,0.1,1,10,100,1000,10000"  # Grid search (nested CV)
ALPHA=1000  # Used only if ALPHA_GRID is empty

# Wandb logging
USE_WANDB=true
WANDB_PROJECT="llms-know-difficulty-probes"

# Loop through each model, temperature and sample size combination
for MODEL in "${MODELS[@]}"; do
    MODEL_ALIAS="${MODEL//\//-}"
    
    for TEMPERATURE in "${TEMPERATURES[@]}"; do
        for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
            echo ""
            echo "========================================="
            echo "Processing: Model=${MODEL}, Temperature=${TEMPERATURE}, Samples=${SAMPLE_SIZE}"
            echo "========================================="
            
            # Construct dataset filenames based on THOM_MATH naming convention
            # Format: MATH_train_100-Qwen-Qwen2.5-72B-Instruct-temperature=0.5.parquet
            TRAIN_DATASET="${DATASET_DIR}/MATH_train_${SAMPLE_SIZE}-${MODEL_ALIAS}-temperature=${TEMPERATURE}.parquet"
            TEST_DATASET="${DATASET_DIR}/MATH_test_${SAMPLE_SIZE}-${MODEL_ALIAS}-temperature=${TEMPERATURE}.parquet"
            
            # Construct output paths
            CONFIG_STR="${MODEL_ALIAS}_samples_${SAMPLE_SIZE}_temp_${TEMPERATURE}"
            ACTIVATIONS_DIR="${MAIN_DATA_DIR}/activations/${SELECTED_DATASET}"
            RESULTS_DIR="probe_results/${DATASET_DIR}/${CONFIG_STR}"
            TRAIN_ACTIVATIONS="${ACTIVATIONS_DIR}/${CONFIG_STR}_train.pt"
            TEST_ACTIVATIONS="${ACTIVATIONS_DIR}/${CONFIG_STR}_test.pt"
            
            WANDB_NAME="${CONFIG_STR}"
            
            # Create output directories
            mkdir -p "${ACTIVATIONS_DIR}"
            mkdir -p "${RESULTS_DIR}"
            
            # Check if datasets exist
            if [ ! -f "${TRAIN_DATASET}" ]; then
                echo "WARNING: Training dataset not found: ${TRAIN_DATASET}"
                echo "Skipping this configuration..."
                continue
            fi
            
            if [ ! -f "${TEST_DATASET}" ]; then
                echo "WARNING: Test dataset not found: ${TEST_DATASET}"
                echo "Skipping this configuration..."
                continue
            fi
            
            echo "Found datasets:"
            echo "  Train: ${TRAIN_DATASET}"
            echo "  Test: ${TEST_DATASET}"
            
            # Check if activations already exist
            if [ "${SKIP_ACTIVATIONS}" = "true" ]; then
                if [ -f "${TRAIN_ACTIVATIONS}" ] && [ -f "${TEST_ACTIVATIONS}" ]; then
                    echo "Activations already exist, skipping extraction..."
                    echo "  Train: ${TRAIN_ACTIVATIONS}"
                    echo "  Test: ${TEST_ACTIVATIONS}"
                else
                    echo "WARNING: SKIP_ACTIVATIONS=true but activations not found!"
                    echo "  Train: ${TRAIN_ACTIVATIONS} (exists: $([ -f "${TRAIN_ACTIVATIONS}" ] && echo 'yes' || echo 'no'))"
                    echo "  Test: ${TEST_ACTIVATIONS} (exists: $([ -f "${TEST_ACTIVATIONS}" ] && echo 'yes' || echo 'no'))"
                    echo "Will proceed with extraction..."
                    SKIP_ACTIVATIONS=false
                fi
            fi
            
            # Extract activations only if not skipping
            if [ "${SKIP_ACTIVATIONS}" != "true" ]; then
                echo ""
                echo "Step 1: Extracting training activations..."
                CUDA_VISIBLE_DEVICES=${GPU} python3 -m scripts.extract_activations \
                    --model "${MODEL}" \
                    --dataset_path "${TRAIN_DATASET}" \
                    --output_path "${TRAIN_ACTIVATIONS}" \
                    --layers "${LAYERS}" \
                    --batch_size ${BATCH_SIZE} \
                    --question_column "${QUESTION_COL}" \
                    --label_column "${LABEL_COL}"
                
                echo ""
                echo "Step 2: Extracting test activations..."
                CUDA_VISIBLE_DEVICES=${GPU} python3 -m scripts.extract_activations \
                    --model "${MODEL}" \
                    --dataset_path "${TEST_DATASET}" \
                    --output_path "${TEST_ACTIVATIONS}" \
                    --layers "${LAYERS}" \
                    --batch_size ${BATCH_SIZE} \
                    --question_column "${QUESTION_COL}" \
                    --label_column "${LABEL_COL}"
            fi
            
            echo ""
            echo "Step 3: Training probes..."
            
            # Build train_probe command
            TRAIN_CMD="python3 -m scripts.train_probe \
                --train_activations \"${TRAIN_ACTIVATIONS}\" \
                --test_activations \"${TEST_ACTIVATIONS}\" \
                --output_dir \"${RESULTS_DIR}\" \
                --k_fold \
                --n_folds 5"
            
            # Add alpha grid if specified
            if [ -n "${ALPHA_GRID}" ]; then
                echo "Using alpha grid search: ${ALPHA_GRID}"
                TRAIN_CMD="${TRAIN_CMD} --alpha_grid \"${ALPHA_GRID}\""
            else
                echo "Using fixed alpha: ${ALPHA}"
                TRAIN_CMD="${TRAIN_CMD} --alpha ${ALPHA}"
            fi
            
            # Add wandb if enabled
            if [ "${USE_WANDB}" = "true" ]; then
                echo "Wandb logging enabled: ${WANDB_PROJECT}/${WANDB_NAME}"
                TRAIN_CMD="${TRAIN_CMD} --wandb --wandb_project \"${WANDB_PROJECT}\" --wandb_name \"${WANDB_NAME}\""
            fi
            
            # Execute
            eval ${TRAIN_CMD}
            
            echo ""
            echo "Completed configuration: Model=${MODEL}, Temperature=${TEMPERATURE}, Samples=${SAMPLE_SIZE}"
            echo "Results saved to: ${RESULTS_DIR}"
            
            # Send notification
            curl -d "Finished ${MODEL_ALIAS} temp=${TEMPERATURE} samples=${SAMPLE_SIZE}" ntfy.sh/wills-linear-probe
        done
    done
done

echo ""
echo "========================================="
echo "All configurations processed!"
echo "========================================="

# Final notification
curl -d "Finished all model/sample/temperature sweep experiments" ntfy.sh/wills-linear-probe
