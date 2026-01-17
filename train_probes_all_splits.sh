#!/bin/bash

# ============================================================================
# Train Probes for All Language Splits
# ============================================================================
# Configure these values directly in the script, then run: ./train_probes_all_splits.sh
# ============================================================================

# Configuration (modify these values)
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"  # Model architecture
PROBE="linear_eoi_probe"                 # Probe type: linear_eoi_probe, tfidf_probe, attn_probe
MAXLEN=3000                              # Maximum sequence length
K_VALUE=10                               # K value
TEMPERATURE=0.7                          # Temperature
GPU_DEVICE=3                             # GPU device to use

# Language suffixes to train on
LANGUAGE_SUFFIXES=("en" "sw" "zh" "es" "ar" "fr" "bn" "pt" "ru" "id" "de" "ja" "vi" "it" "te" "ko" "th" "ms")

# Print configuration
echo "============================================================================"
echo "Training Probes for All Language Splits"
echo "============================================================================"
echo "Model:       $MODEL"
echo "Probe:       $PROBE"
echo "Max Length:  $MAXLEN"
echo "K Value:     $K_VALUE"
echo "Temperature: $TEMPERATURE"
echo "GPU Device:  $GPU_DEVICE"
echo ""
echo "Language splits to train: ${#LANGUAGE_SUFFIXES[@]}"
echo "  ${LANGUAGE_SUFFIXES[@]}"
echo "============================================================================"
echo ""

# Track progress
SUCCEEDED=0
FAILED=0
TOTAL=${#LANGUAGE_SUFFIXES[@]}

# Train for each language split
for i in "${!LANGUAGE_SUFFIXES[@]}"; do
    SUFFIX=${LANGUAGE_SUFFIXES[$i]}
    DATASET="Qwen_PolyMath_${SUFFIX}"
    COUNT=$((i + 1))
    
    echo "[$COUNT/$TOTAL] Training: $DATASET"
    
    # Build and execute command
    CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 src/llms_know_difficulty/main.py \
        --probe $PROBE \
        --dataset $DATASET \
        --model $MODEL \
        --max_len $MAXLEN \
        --k $K_VALUE \
        --temperature $TEMPERATURE"
    
    # Use expandable segments for attn_probe
    if [ "$PROBE" == "attn_probe" ]; then
        CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $CMD"
    fi
    
    if eval $CMD; then
        echo "  ✓ SUCCESS"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  ✗ FAILED"
        FAILED=$((FAILED + 1))
    fi
done

# Summary
echo ""
echo "============================================================================"
echo "SUMMARY: $SUCCEEDED/$TOTAL succeeded, $FAILED failed"
echo "============================================================================"

exit 0
