#!/bin/bash

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Configuration
PROBE_PATH="data/results/Qwen/Qwen2.5-Math-7B-Instruct/Qwen_PolyMath_en/linear_eoi_probe/maxlen_3000_k_10_temp_0.7/label_majority_vote_is_correct/20260123_002712"
DATASET="openai_gsm8k" #"DigitalLearningGmbH_MATH-lighteval" #"openai_gsm8k"
MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
SPLIT="test"
MAX_LEN=3000
K=5
TEMPERATURE=0.7
DEVICE="cuda:0"
BATCH_SIZE=128

# Extract information from probe path for output directory structure
# Path format: data/results/<model>/<probe_dataset>/<probe_name>/<gen_str>/label_<label>/timestamp

# Extract model name from MODEL variable (e.g., "Qwen/Qwen2.5-Math-1.5B-Instruct" -> "Qwen2.5-Math-1.5B-Instruct")
MODEL_NAME=$(echo "$MODEL" | sed 's|.*/||')

# Extract probe name (e.g., "linear_eoi_probe")
PROBE_NAME=$(echo "$PROBE_PATH" | grep -oP '[^/]*_probe(?=/)' | head -1)

# Extract probe dataset (the source dataset used to train the probe)
# This is the directory between model and probe_name
PROBE_DATASET=$(echo "$PROBE_PATH" | sed -n 's/^.*\/Qwen2\.5-Math-1\.5B-Instruct\/\([^/]*\)\/.*_probe.*/\1/p')

# Extract gen_str (e.g., "maxlen_3000_k_5_temp_0.7")
GEN_STR=$(echo "$PROBE_PATH" | grep -oP 'maxlen_\d+_k_\d+_temp_[\d.]+')

# Extract label column from probe path
# Path format: .../label_<LABEL_COLUMN>/timestamp
# Extract the part after "label_" and before the timestamp
LABEL_COLUMN=$(echo "$PROBE_PATH" | grep -oP 'label_\K[^/]+')

if [ -z "$LABEL_COLUMN" ]; then
    echo "⚠️  Warning: Could not extract label column from probe path"
    echo "Using default: is_correct"
    LABEL_COLUMN="is_correct"
fi

# Construct output directory with hierarchical structure: model / probe_dataset_probe_name / gen_str / label / dataset
OUTPUT_DIR="data/predictions/${MODEL_NAME}/${PROBE_DATASET}_${PROBE_NAME}/${GEN_STR}/label_${LABEL_COLUMN}/${DATASET}"

echo "=================================="
echo "🔮 Probe Prediction Configuration"
echo "=================================="
echo "Probe model: $MODEL_NAME"
echo "Probe dataset: $PROBE_DATASET"
echo "Probe name: $PROBE_NAME"
echo "Gen string: $GEN_STR"
echo "Label column: $LABEL_COLUMN"
echo "Dataset to label: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "=================================="
echo ""

python src/pika/predict_with_probe.py \
  --probe_path "$PROBE_PATH" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --split "$SPLIT" \
  --max_len "$MAX_LEN" \
  --k "$K" \
  --temperature "$TEMPERATURE" \
  --label_column "$LABEL_COLUMN" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"