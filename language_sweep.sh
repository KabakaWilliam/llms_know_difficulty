#!/bin/bash

# ============================================================================
# Language Sweep: Run Probe on All Qwen PolyMath Language Splits
# ============================================================================
# Use a trained probe to make predictions on all language variants
# Configure these values directly in the script, then run: ./language_sweep.sh
# ============================================================================

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Configuration (modify these values)
# PROBE_PATH="data/results/Qwen/Qwen2.5-Math-7B-Instruct/Qwen_PolyMath_en/linear_eoi_probe/maxlen_3000_k_10_temp_0.7/label_majority_vote_is_correct/20260123_002712"
PROBE_PATH="data/results/Qwen/Qwen2.5-Math-7B-Instruct/Qwen_PolyMath_en/linear_eoi_probe/maxlen_3000_k_10_temp_0.7/label_success_rate/20260123_011352"
MODEL="Qwen/Qwen2.5-7B-Instruct"
SPLIT="test"
MAX_LEN=3000
K_VALUE=10
TEMPERATURE=0.7
GPU_DEVICE="cuda:0"
BATCH_SIZE=128

# Language suffixes to sweep (from train_probes_all_splits.sh)
LANGUAGE_SUFFIXES=("en" "sw" "zh" "es" "ar" "te")

# Print configuration
echo "============================================================================"
echo "Language Sweep: Running Probe Across All Language Splits"
echo "============================================================================"
echo "Probe path: $PROBE_PATH"
echo "Model:      $MODEL"
echo "Split:      $SPLIT"
echo "Max Length: $MAX_LEN"
echo "K Value:    $K_VALUE"
echo "Temperature: $TEMPERATURE"
echo "Device:     $GPU_DEVICE"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Language splits to process: ${#LANGUAGE_SUFFIXES[@]}"
echo "  ${LANGUAGE_SUFFIXES[@]}"
echo "============================================================================"
echo ""

# Extract information from probe path for output directory structure
# Extract model name from MODEL variable (e.g., "Qwen/Qwen2.5-Math-1.5B-Instruct" -> "Qwen2.5-Math-1.5B-Instruct")
MODEL_NAME=$(echo "$MODEL" | sed 's|.*/||')

# Extract probe name (e.g., "linear_eoi_probe")
PROBE_NAME=$(echo "$PROBE_PATH" | grep -oP '[^/]*_probe(?=/)' | head -1)

# Extract probe dataset (the source dataset used to train the probe)
PROBE_DATASET=$(echo "$PROBE_PATH" | sed -n 's/^.*\/\([^/]*\)\/.*_probe.*/\1/p')

# Extract gen_str (e.g., "maxlen_3000_k_5_temp_0.7")
GEN_STR=$(echo "$PROBE_PATH" | grep -oP 'maxlen_\d+_k_\d+_temp_[\d.]+')

# Extract label column from probe path
LABEL_COLUMN=$(echo "$PROBE_PATH" | grep -oP 'label_\K[^/]+')

if [ -z "$LABEL_COLUMN" ]; then
    echo "⚠️  Warning: Could not extract label column from probe path"
    echo "Using default: is_correct"
    LABEL_COLUMN="is_correct"
fi

echo "Extracted probe info:"
echo "  Probe name:    $PROBE_NAME"
echo "  Probe dataset: $PROBE_DATASET"
echo "  Gen string:    $GEN_STR"
echo "  Label column:  $LABEL_COLUMN"
echo ""

# Track progress
SUCCEEDED=0
FAILED=0
TOTAL=${#LANGUAGE_SUFFIXES[@]}

# Process each language split
for i in "${!LANGUAGE_SUFFIXES[@]}"; do
    SUFFIX=${LANGUAGE_SUFFIXES[$i]}
    DATASET="Qwen_PolyMath_${SUFFIX}"
    COUNT=$((i + 1))
    
    # Construct output directory
    OUTPUT_DIR="data/predictions/${MODEL_NAME}/${PROBE_DATASET}/${PROBE_NAME}/${GEN_STR}/label_${LABEL_COLUMN}/${DATASET}"
    
    echo "[$COUNT/$TOTAL] Processing: $DATASET"
    echo "  Output: $OUTPUT_DIR"
    
    # Build and execute command
    CMD="python src/llms_know_difficulty/predict_with_probe.py \
        --probe_path \"$PROBE_PATH\" \
        --dataset \"$DATASET\" \
        --model \"$MODEL\" \
        --split \"$SPLIT\" \
        --max_len $MAX_LEN \
        --k $K_VALUE \
        --temperature $TEMPERATURE \
        --label_column \"$LABEL_COLUMN\" \
        --output_dir \"$OUTPUT_DIR\" \
        --batch_size $BATCH_SIZE \
        --device \"$GPU_DEVICE\""
    
    if eval $CMD > /tmp/language_sweep_${SUFFIX}.log 2>&1; then
        echo "  ✓ SUCCESS"
        SUCCEEDED=$((SUCCEEDED + 1))
        # Print key metrics from output
        tail -3 /tmp/language_sweep_${SUFFIX}.log | grep -E "(AUC|Spearman|Mean prediction)" || true
    else
        echo "  ✗ FAILED"
        FAILED=$((FAILED + 1))
        echo "  Error log: /tmp/language_sweep_${SUFFIX}.log"
    fi
    echo ""
done

# Summary
echo "============================================================================"
echo "SUMMARY: $SUCCEEDED/$TOTAL succeeded, $FAILED failed"
echo "============================================================================"
echo ""
echo "Results saved to:"
echo "  data/predictions/${MODEL_NAME}/${PROBE_DATASET}_${PROBE_NAME}/${GEN_STR}/label_${LABEL_COLUMN}/"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All language splits processed successfully!"
    exit 0
else
    echo "⚠️  Some splits failed. Check error logs for details."
    exit 1
fi
