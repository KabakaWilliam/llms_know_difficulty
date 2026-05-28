#!/bin/bash
# Cross-dataset probe generalisation sweep.
#
# For each (source_dataset, eval_dataset) pair, loads the latest probe trained
# on source_dataset and runs it on eval_dataset. Results are saved under:
#   data/results/{model}/{eval_dataset}/{probe_type}/{gen_str}/label_{metric}/probe_from_{source_dataset}/{timestamp}/
#
# Usage:
#   bash scripts/run_cross_dataset_probe_sweep.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_cross_dataset_probe_sweep.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/opt/anaconda/envs/pika/bin/python3

# ── Configuration ──────────────────────────────────────────────────────────────

PROBE_TYPE="linear_eoi_probe"
GEN_STR="maxlen_3000_k_5_temp_0.7"
LABEL="majority_vote_is_correct"
SPLIT="test"
DATA_DIR="data"

# Models and their generation configs: "model|maxlen|k|temp"
declare -a MODEL_CONFIGS=(
    # "Qwen/Qwen2.5-Math-7B-Instruct|3000|5|0.7"
    # "Qwen/Qwen2.5-Math-1.5B-Instruct|3000|5|0.7"
    # "Qwen/Qwen2.5-1.5B-Instruct|3000|5|0.7"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|32768|5|0.6"
    "openai/gpt-oss-20b_low|131072|5|1.0"
    "openai/gpt-oss-20b_high|131072|5|1.0"
    "openai/gpt-oss-20b_medium|131072|5|1.0"
)

# Datasets to use as probe source / eval target
declare -a DATASETS=(
    "DigitalLearningGmbH_MATH-lighteval"
    "openai_gsm8k"
    "gneubig_aime-1983-2024"
    # "E2H-AMC"
    # "opencompass_AIME2025"   # no probes trained yet; add once probes exist
)

# ── Helpers ────────────────────────────────────────────────────────────────────

# Find the latest timestamp directory for a given probe checkpoint path
latest_checkpoint() {
    local base_dir="$1"
    ls -1d "${base_dir}"/[0-9]* 2>/dev/null | sort | tail -1
}

# ── Sweep ──────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    IFS='|' read -r MODEL MAX_LEN K TEMP <<< "$model_cfg"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Model: ${MODEL}"
    echo "════════════════════════════════════════════════════════════"

    for SOURCE_DATASET in "${DATASETS[@]}"; do
        for EVAL_DATASET in "${DATASETS[@]}"; do
            [ "$SOURCE_DATASET" = "$EVAL_DATASET" ] && continue

            # Find latest probe for SOURCE_DATASET (use model-specific gen_str)
            MODEL_GEN_STR="maxlen_${MAX_LEN}_k_${K}_temp_${TEMP}"
            PROBE_BASE="${DATA_DIR}/results/${MODEL}/${SOURCE_DATASET}/${PROBE_TYPE}/${MODEL_GEN_STR}/label_${LABEL}"
            PROBE_PATH=$(latest_checkpoint "$PROBE_BASE")

            if [ -z "$PROBE_PATH" ]; then
                echo "  ⚠  No probe found for ${SOURCE_DATASET} — skipping"
                continue
            fi

            # Output path for cross-dataset predictions
            OUTPUT_DIR="${DATA_DIR}/results/${MODEL}/${EVAL_DATASET}/${PROBE_TYPE}/${MODEL_GEN_STR}/label_${LABEL}/probe_from_${SOURCE_DATASET}/${TIMESTAMP}"

            echo ""
            echo "  Probe source : ${SOURCE_DATASET}"
            echo "  Eval dataset : ${EVAL_DATASET}"
            echo "  Probe path   : ${PROBE_PATH}"
            echo "  Output       : ${OUTPUT_DIR}"

            $PYTHON src/pika/predict_with_probe.py \
                --probe_path    "${PROBE_PATH}" \
                --probe_dataset "${SOURCE_DATASET}" \
                --dataset       "${EVAL_DATASET}" \
                --model         "${MODEL}" \
                --max_len       "${MAX_LEN}" \
                --k             "${K}" \
                --temperature   "${TEMP}" \
                --label_column  "${LABEL}" \
                --split         "${SPLIT}" \
                --output_dir    "${OUTPUT_DIR}" \
                --batch_size    32

            if [ $? -eq 0 ]; then
                echo "  ✅ Done: ${SOURCE_DATASET} → ${EVAL_DATASET}"
            else
                echo "  ❌ Failed: ${SOURCE_DATASET} → ${EVAL_DATASET}"
            fi
        done
    done
done

echo ""
echo "Cross-dataset sweep complete."
