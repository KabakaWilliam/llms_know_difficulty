#!/bin/bash
# Complete pipeline: Extract activations and train probes

set -e  # Exit on error

# Configuration
MODEL="Qwen/Qwen2-1.5B-Instruct"
MODEL_ALIAS="${MODEL##*/}"
MAX_LEN=3000
K=1  
TEMPERATURE=0.0
GENERATION_STR=${MODEL_ALIAS}_maxlen_${MAX_LEN}_k_${K}_temp_${TEMPERATURE}

MAIN_DATA_DIR="DATA"
DATASET_DIR="${MAIN_DATA_DIR}/SR_DATA/MATH"
TRAIN_DATASET="${DATASET_DIR}/train-${GENERATION_STR}.parquet"
TEST_DATASET="${DATASET_DIR}/test-${GENERATION_STR}.parquet"

ACTIVATIONS_DIR="${MAIN_DATA_DIR}/activations"
RESULTS_DIR="results/${DATASET_DIR}/${MODEL_ALIAS}_${GENERATION_STR}"
TRAIN_ACTIVATIONS="${ACTIVATIONS_DIR}/${GENERATION_STR}_train.pt"
TEST_ACTIVATIONS="${ACTIVATIONS_DIR}/${GENERATION_STR}_test.pt"

QUESTION_COL="formatted_prompt"
LABEL_COL="success_rate"
LAYERS="all"
BATCH_SIZE=32
GPU=1

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
python3 -m scripts.train_probe\
    --train_activations "${TRAIN_ACTIVATIONS}" \
    --test_activations "${TEST_ACTIVATIONS}" \
    --output_dir "${RESULTS_DIR}" \
    --k_fold \
    --n_folds 5

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "Results saved to: ${RESULTS_DIR}"
echo "========================================="

curl -d "Finished replication pipeline for ${MODEL_ALIAS}_${GENERATION_STR}" ntfy.sh/wills-linear-probe