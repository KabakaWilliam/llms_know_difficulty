#!/bin/bash
# Loop through multiple models and run the complete pipeline for each

set -e  # Exit on error

# List of models to process
MODELS=(
    # "Qwen/Qwen2.5-Math-1.5B-Instruct" 
    # "Qwen/Qwen2.5-1.5B"
    # # "openai/gpt-oss-20b"
    # "Qwen/Qwen2.5-1.5B-Instruct" 
    # "Qwen/Qwen2-1.5B-Instruct"
    # "Qwen/Qwen2.5-Math-7B-Instruct" 
    "Qwen/Qwen2.5-Math-72B-Instruct" 
    # "Qwen/Qwen2.5-Math-1.5B"
    # "Qwen/Qwen2.5-1.5B" 
    # "Qwen/Qwen2.5-Math-7B" 
    # "Qwen/Qwen2-1.5B"
    # "Qwen/Qwen2-1.5B-Instruct"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
)

# Configuration (same for all models)
MAX_LEN=3000
K=1  
TEMPERATURE=0.0
GEN_OPTIONS=maxlen_${MAX_LEN}_k_${K}_temp_${TEMPERATURE}

MAIN_DATA_DIR="DATA"
CHOSEN_DATASET="DigitalLearningGmbH_MATH-lighteval"
DATASET_DIR="${MAIN_DATA_DIR}/SR_DATA/${CHOSEN_DATASET}"

QUESTION_COL="formatted_prompt"
LABEL_COL="success_rate"
LAYERS="all"
BATCH_SIZE=32
GPU=1,2

# Skip activation extraction if they already exist
SKIP_ACTIVATIONS=true  # Set to true to skip extraction and reuse existing activations

# Regularization parameters
ALPHA_GRID="0, 0.001,0.01,0.1,1,10,100,1000,10000"  # Grid search (nested CV)
# ALPHA_GRID=""  # Uncomment to disable grid search and use fixed alpha
ALPHA=1000  # Used only if ALPHA_GRID is empty

# Wandb logging
USE_WANDB=false
WANDB_PROJECT="llms-know-difficulty-probes"

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing model: ${MODEL}"
    echo "========================================="
    
    MODEL_ALIAS="${MODEL//\//-}"
    GENERATION_STR=${MODEL_ALIAS}_${GEN_OPTIONS}
    
    TRAIN_DATASET="${DATASET_DIR}/train-${GENERATION_STR}.parquet"
    TEST_DATASET="${DATASET_DIR}/test-${GENERATION_STR}.parquet"
    
    ACTIVATIONS_DIR="${MAIN_DATA_DIR}/activations/${CHOSEN_DATASET}"
    RESULTS_DIR="probe_results/${DATASET_DIR}/${MODEL_ALIAS}_${GEN_OPTIONS}"
    TRAIN_ACTIVATIONS="${ACTIVATIONS_DIR}/${GENERATION_STR}_train.pt"
    TEST_ACTIVATIONS="${ACTIVATIONS_DIR}/${GENERATION_STR}_test.pt"
    
    WANDB_NAME="${MODEL_ALIAS}_${GEN_OPTIONS}"
    
    # Create output directories
    mkdir -p "${ACTIVATIONS_DIR}"
    mkdir -p "${RESULTS_DIR}"
    
    # Check if datasets exist
    if [ ! -f "${TRAIN_DATASET}" ]; then
        echo "WARNING: Training dataset not found: ${TRAIN_DATASET}"
        echo "Skipping model: ${MODEL}"
        continue
    fi
    
    if [ ! -f "${TEST_DATASET}" ]; then
        echo "WARNING: Test dataset not found: ${TEST_DATASET}"
        echo "Skipping model: ${MODEL}"
        continue
    fi
    
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
    echo "Completed pipeline for: ${MODEL}"
    echo "Results saved to: ${RESULTS_DIR}"
    
    # Send notification
    curl -d "Finished ${MODEL_ALIAS}_${GEN_OPTIONS}" ntfy.sh/wills-linear-probe
done

echo ""
echo "========================================="
echo "All models processed!"
echo "========================================="

# Final notification
curl -d "Finished all models in replication pipeline" ntfy.sh/wills-linear-probe
