#!/bin/bash
# Run Layer x Checkpoint Heatmap Visualization
# This script generates heatmaps showing probe performance across layers and checkpoints

set -e

# Default parameters
RESULTS_DIR="runs/checkpoint_comparison"
POSITION=-1
OUTPUT_DIR="."
TASK_NAME="GSM8K (E2H-GSM8K)"

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate layer x checkpoint heatmaps from probe training results"
    echo ""
    echo "Options:"
    echo "  -r, --results-dir DIR     Probe results directory (default: runs/checkpoint_comparison)"
    echo "  -p, --position POS        Token position for position-specific heatmap (default: -1)"
    echo "  -o, --output-dir DIR      Output directory for visualizations (default: .)"
    echo "  -t, --task-name NAME      Task name for plot titles (default: 'AMC (E2H-AMC)')"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use defaults"
    echo "  $0 -r runs/checkpoint_comparison -p -1 # Specify results dir and position"
    echo "  $0 -t 'MATH Difficulty' -o plots     # Custom task name and output dir"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -p|--position)
            POSITION="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo "=== Layer x Checkpoint Heatmap Visualization ==="
echo "Results directory: $RESULTS_DIR"
echo "Position: $POSITION"
echo "Output directory: $OUTPUT_DIR"
echo "Task name: $TASK_NAME"
echo ""

# Check if results directory exists
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: Results directory '$RESULTS_DIR' not found"
    echo "Make sure you have run probe training first"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to the difficulty_direction directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/difficulty_direction"

# Convert relative results dir to absolute path
if [[ ! "$RESULTS_DIR" = /* ]]; then
    RESULTS_DIR="$SCRIPT_DIR/$RESULTS_DIR"
fi

# Run the visualization script
echo "Generating heatmaps..."
python visualize_layer_checkpoint_heatmap.py \
    --results_dir "$RESULTS_DIR" \
    --position "$POSITION" \
    --output_dir "../$OUTPUT_DIR" \
    --task_name "$TASK_NAME"

echo ""
echo "=== Visualization Complete ==="
echo "Check $RESULTS_DIR/visualizations for generated heatmap files:"
echo "  - layer_checkpoint_heatmap_best_*.png (best position per layer)"
echo "  - layer_checkpoint_heatmap_pos${POSITION}_*.png (position $POSITION specific)"
