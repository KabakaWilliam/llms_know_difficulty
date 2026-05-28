#!/bin/bash

# Flexible probe sweep script with multi-GPU parallel scheduling.
# Runs multiple probe jobs concurrently across a pool of GPUs (one job per GPU).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Configuration -----------------------------------------------------------

# GPU pool — edit to change which GPUs are used or pool size.
GPU_IDS=(0 1 2 3)

# Model configurations as "model|max_len|k|temperature".
declare -a MODEL_CONFIGS=(
    "Qwen/Qwen2.5-Math-1.5B-Instruct|3000|5|0.7"
    "Qwen/Qwen2.5-Math-7B-Instruct|3000|5|0.7"
)

declare -a DATASETS=(
    "DigitalLearningGmbH_MATH-lighteval"
)

declare -a PROBES=(
    "linear_eoi_probe"
    "mlp_probe"
)

# Every odd layer from 1 to 27.
declare -a LAYER_CHOICES=(1 3 5 7 9 11 13 15 17 19 21 23 25 27)

LABEL_COLUMN="majority_vote_is_correct"

# ---- Setup -------------------------------------------------------------------

cd "$(dirname "$0")" || exit 1

LOG_DIR="logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Build flat list of job specs: "model|max_len|k|temperature|dataset|probe|layer".
# We unpack model_config here so the spec has no nested '|' delimiters.
declare -a JOBS=()
for model_config in "${MODEL_CONFIGS[@]}"; do
    IFS='|' read -r model max_len k temperature <<< "$model_config"
    for dataset in "${DATASETS[@]}"; do
        for probe in "${PROBES[@]}"; do
            for layer in "${LAYER_CHOICES[@]}"; do
                JOBS+=("$model|$max_len|$k|$temperature|$dataset|$probe|$layer")
            done
        done
    done
done

TOTAL=${#JOBS[@]}
echo "========================================"
echo "Probe sweep starting"
echo "Total jobs : $TOTAL"
echo "GPUs       : ${GPU_IDS[*]} (${#GPU_IDS[@]} parallel slots)"
echo "Logs       : $LOG_DIR"
echo "========================================"

# ---- Scheduler ---------------------------------------------------------------

declare -A gpu_busy       # gpu_id -> pid currently running on that GPU
declare -A pid_to_meta    # pid    -> "gpu|idx|model_config|dataset|probe|layer|log_file"
success_count=0
fail_count=0
declare -a failed_logs=()

# Kill all in-flight children if the user interrupts.
cleanup() {
    echo ""
    echo "Interrupt received — terminating ${#pid_to_meta[@]} in-flight job(s)..."
    for pid in "${!pid_to_meta[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    exit 130
}
trap cleanup INT TERM

launch_job() {
    local gpu=$1 spec=$2 idx=$3
    IFS='|' read -r model max_len k temperature dataset probe layer <<< "$spec"

    local model_safe="${model//\//_}"
    local log_file
    log_file="$LOG_DIR/run_$(printf '%03d' "$idx")_${model_safe}_${dataset}_${probe}_L${layer}.log"

    echo "[GPU $gpu] ▶ launched $idx/$TOTAL: $model | $dataset | $probe | L$layer"

    (
        CUDA_VISIBLE_DEVICES=$gpu python3 src/pika/main.py \
            --probe "$probe" \
            --dataset "$dataset" \
            --model "$model" \
            --max_len "$max_len" \
            --k "$k" \
            --temperature "$temperature" \
            --label_column "$LABEL_COLUMN" \
            --layers "$layer" \
            > "$log_file" 2>&1
    ) &

    local pid=$!
    gpu_busy[$gpu]=$pid
    pid_to_meta[$pid]="$gpu|$idx|$model|$dataset|$probe|$layer|$log_file"
}

collect_pid() {
    local pid=$1
    local meta=${pid_to_meta[$pid]}
    IFS='|' read -r gpu idx model dataset probe layer log_file <<< "$meta"

    wait "$pid"
    local rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "[GPU $gpu] ✅ $idx/$TOTAL ok: $model | $dataset | $probe | L$layer"
        success_count=$((success_count + 1))
    else
        echo "[GPU $gpu] ❌ $idx/$TOTAL FAIL rc=$rc: $model | $dataset | $probe | L$layer  ($log_file)"
        failed_logs+=("$log_file")
        fail_count=$((fail_count + 1))
    fi

    unset "gpu_busy[$gpu]"
    unset "pid_to_meta[$pid]"
}

# Dispatch loop: for each job, find a free GPU (harvesting finished ones), then launch.
job_idx=0
for spec in "${JOBS[@]}"; do
    job_idx=$((job_idx + 1))

    free_gpu=""
    while [ -z "$free_gpu" ]; do
        for gpu in "${GPU_IDS[@]}"; do
            pid=${gpu_busy[$gpu]:-}
            if [ -z "$pid" ]; then
                free_gpu=$gpu
                break
            elif ! kill -0 "$pid" 2>/dev/null; then
                collect_pid "$pid"
                free_gpu=$gpu
                break
            fi
        done
        [ -z "$free_gpu" ] && sleep 2
    done

    launch_job "$free_gpu" "$spec" "$job_idx"
done

# ---- Drain remaining ---------------------------------------------------------

echo ""
echo "All ${TOTAL} jobs dispatched. Waiting on ${#pid_to_meta[@]} remaining..."
while [ ${#pid_to_meta[@]} -gt 0 ]; do
    for pid in "${!pid_to_meta[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            collect_pid "$pid"
        fi
    done
    [ ${#pid_to_meta[@]} -gt 0 ] && sleep 2
done

# ---- Summary -----------------------------------------------------------------

echo ""
echo "========================================"
echo "Sweep complete: $success_count succeeded, $fail_count failed (of $TOTAL)"
echo "Log directory: $LOG_DIR"
if [ ${#failed_logs[@]} -gt 0 ]; then
    echo ""
    echo "Failed job logs:"
    for log in "${failed_logs[@]}"; do
        echo "  $log"
    done
fi
echo "========================================"