#!/bin/bash
# Complete pipeline: Extract activations and train probes

set -e  # Exit on error

# Configuration
MODEL="Qwen/Qwen2.5-7B" #not run
# MODEL="Qwen/Qwen2.5-Math-7B"
# MODEL="Qwen/Qwen2-1.5B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL_ALIAS="${MODEL//\//-}"
MAX_LEN=3000
K=50  
TEMPERATURE=1.0
GEN_OPTIONS=maxlen_${MAX_LEN}_k_${K}_temp_${TEMPERATURE}
GENERATION_STR=${MODEL_ALIAS}_${GEN_OPTIONS}

MAIN_DATA_DIR="DATA"
DATASET_DIR="${MAIN_DATA_DIR}/SR_DATA/MATH"
TRAIN_DATASET="${DATASET_DIR}/train-${GENERATION_STR}.parquet"
TEST_DATASET="${DATASET_DIR}/test-${GENERATION_STR}.parquet"

ACTIVATIONS_DIR="${MAIN_DATA_DIR}/activations"
RESULTS_DIR="probe_results/${DATASET_DIR}/${MODEL_ALIAS}_${GEN_OPTIONS}"
TRAIN_ACTIVATIONS="${ACTIVATIONS_DIR}/${GENERATION_STR}.pt"
TEST_ACTIVATIONS="${ACTIVATIONS_DIR}/${GENERATION_STR}.pt"

QUESTION_COL="formatted_prompt"
LABEL_COL="success_rate"
LAYERS="all"
BATCH_SIZE=32
GPU=0

# Regularization parameters
# Set ALPHA_GRID to enable grid search (recommended for proper model selection)
# Leave empty to use fixed ALPHA value
# ALPHA_GRID="0,0.001,0.01,0.1,1,10,100,1000"  # Grid search (nested CV)
ALPHA_GRID="100,1000, 10000, 100000, 1000000"  # Grid search (nested CV)
ALPHA_GRID="0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000"  # Grid search (nested CV)

# ALPHA_GRID=""  # Uncomment to disable grid search and use fixed alpha
ALPHA=1000  # Used only if ALPHA_GRID is empty

# Wandb logging
USE_WANDB=true  # Set to true to enable wandb logging
WANDB_PROJECT="pika-probes"
WANDB_NAME="${MODEL_ALIAS}_${GEN_OPTIONS}"

# Create output directories
mkdir -p "${ACTIVATIONS_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "========================================="
echo "Step 1: Extracting training activations..."
echo "========================================="
CUDA_VISIBLE_DEVICES=${GPU} python3 -m scripts.extract_activations\
    --model "${MODEL}" \
    --dataset_path "${TRAIN_DATASET}" \
    --output_path "${TRAIN_ACTIVATIONS}" \
    --layers "${LAYERS}" \
    --batch_size ${BATCH_SIZE} \
    --question_column "${QUESTION_COL}" \
    --label_column "${LABEL_COL}"

echo ""
echo "========================================="
echo "Step 2: Extracting test activations..."
echo "========================================="
CUDA_VISIBLE_DEVICES=${GPU} python3 -m scripts.extract_activations\
    --model "${MODEL}" \
    --dataset_path "${TEST_DATASET}" \
    --output_path "${TEST_ACTIVATIONS}" \
    --layers "${LAYERS}" \
    --batch_size ${BATCH_SIZE} \
    --question_column "${QUESTION_COL}" \
    --label_column "${LABEL_COL}"

echo ""
echo "========================================="
echo "Step 3: Training probes..."
echo "========================================="

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
echo "========================================="
echo "Pipeline complete!"
echo "Results saved to: ${RESULTS_DIR}"
echo "========================================="

curl -d "Finished replication pipeline for ${MODEL_ALIAS}_${GEN_OPTIONS}" ntfy.sh/wills-linear-probe