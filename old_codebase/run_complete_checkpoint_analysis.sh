#!/bin/bash

# Complete Checkpoint Analysis Pipeline
# This script runs probe training and generates visualizations with model names

set -e  # Exit on any error

# Default values
# CHECKPOINT_DIR=converted_hf_models/Qwen2.5-Math-1.5B_ckpt_1e-5_8_3000_filtered_template_train_BASELINE_grpo_standard_20250917_022323
CHECKPOINT_DIR=converted_hf_models/Llama-3.2-3B-Instruct_ckpt_1e-5_8_3000_filtered_template_train_BASELINE_grpo_standard_20250917_022224
OUTPUT_DIR=""
USE_K_FOLD=true
N_FOLDS=5
CV_SEED=42
BASELINE_MODEL=""
NO_ANNOTATIONS=false
CUDA_DEVICE=2

# Function to display usage
usage() {
    echo "Usage: $0 --checkpoint_dir <path> [options]"
    echo ""
    echo "Required:"
    echo "  --checkpoint_dir <path>    Directory containing checkpoint steps"
    echo ""
    echo "Optional:"
    echo "  --output_dir <path>        Output directory (default: runs/checkpoint_comparison/model_name)"
    echo "  --baseline_model <path>    Path to baseline model for comparison"
    echo "  --no_k_fold               Disable K-fold cross-validation (enabled by default)"
    echo "  --n_folds <int>            Number of folds for K-fold (default: 5)"
    echo "  --cv_seed <int>            Random seed for cross-validation (default: 42)"
    echo "  --no_annotations           Disable layer/position annotations on graphs"
    echo "  --cuda_device <int>        CUDA device to use (default: 2)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --checkpoint_dir /path/to/checkpoints"
    echo "  $0 --checkpoint_dir /path/to/checkpoints --no_k_fold"
    echo "  $0 --checkpoint_dir /path/to/checkpoints --n_folds 3 --cuda_device 1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --baseline_model)
            BASELINE_MODEL="$2"
            shift 2
            ;;
        --use_k_fold)
            USE_K_FOLD=true
            shift
            ;;
        --no_k_fold)
            USE_K_FOLD=false
            shift
            ;;
        --n_folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --cv_seed)
            CV_SEED="$2"
            shift 2
            ;;
        --no_annotations)
            NO_ANNOTATIONS=true
            shift
            ;;
        --cuda_device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: --checkpoint_dir is required"
    usage
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_NAME=$(basename "$CHECKPOINT_DIR")
    OUTPUT_DIR="runs/checkpoint_comparison/$MODEL_NAME"
fi

echo "🎯 Complete Checkpoint Analysis Pipeline"
echo "========================================="
echo "📁 Checkpoint directory: $CHECKPOINT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🖥️  CUDA device: $CUDA_DEVICE"
echo "🔬 K-fold cross-validation: $([ "$USE_K_FOLD" = true ] && echo "enabled ($N_FOLDS folds)" || echo "disabled")"
echo "📊 Annotations: $([ "$NO_ANNOTATIONS" = true ] && echo "disabled" || echo "enabled")"
if [ -n "$BASELINE_MODEL" ]; then
    echo "📍 Baseline model: $BASELINE_MODEL"
fi
echo ""

# Step 1: Train probes for all checkpoints
echo "🔬 Step 1: Training probes for all checkpoint steps..."
echo "======================================================"

# First, let's check how many steps we have
echo "🔍 Checking available checkpoint steps..."
STEP_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "step_*" | wc -l)
echo "📊 Found $STEP_COUNT checkpoint steps in $CHECKPOINT_DIR"
echo ""

TRAIN_CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m difficulty_direction.train_checkpoint_probes --checkpoint_dir \"$CHECKPOINT_DIR\" --output_dir \"$OUTPUT_DIR\""

if [ "$USE_K_FOLD" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_k_fold --n_folds $N_FOLDS --cv_seed $CV_SEED"
fi

if [ -n "$BASELINE_MODEL" ]; then
    TRAIN_CMD="$TRAIN_CMD --baseline_model \"$BASELINE_MODEL\""
fi

echo "Running: $TRAIN_CMD"
eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo "❌ Probe training failed!"
    exit 1
fi

echo ""
echo "✅ Probe training completed!"

# Step 2: Generate visualizations
echo ""
echo "📊 Step 2: Generating visualizations with model names..."
echo "======================================================="

SUMMARY_FILE="$OUTPUT_DIR/checkpoint_probe_summary.json"
VIS_OUTPUT_DIR="$OUTPUT_DIR/visualizations"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "❌ Summary file not found: $SUMMARY_FILE"
    exit 1
fi

VIS_CMD="python -m difficulty_direction.visualize_checkpoint_evolution --summary_file \"$SUMMARY_FILE\" --output_dir \"$VIS_OUTPUT_DIR\""

if [ "$NO_ANNOTATIONS" = true ]; then
    VIS_CMD="$VIS_CMD --no_annotations"
fi

echo "Running: $VIS_CMD"
eval $VIS_CMD

if [ $? -ne 0 ]; then
    echo "❌ Visualization generation failed!"
    exit 1
fi

echo ""
echo "🎉 Complete analysis pipeline finished successfully!"
echo "=================================================="
echo "📁 Results saved to: $OUTPUT_DIR"
echo "📊 Visualizations in: $VIS_OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  📄 $SUMMARY_FILE"
echo "  📊 $(ls "$VIS_OUTPUT_DIR"/*.png 2>/dev/null | wc -l) visualization files"
echo ""
echo "To view results:"
echo "  📊 Open visualization files in: $VIS_OUTPUT_DIR"
echo "  📄 Review summary JSON: $SUMMARY_FILE"
