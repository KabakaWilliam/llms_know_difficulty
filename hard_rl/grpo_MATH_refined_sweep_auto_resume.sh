#!/bin/bash
# Auto-resuming sweep script with crash detection and automatic resumption
# chmod +x grpo_MATH_refined_sweep_auto_resume.sh
# ./grpo_MATH_refined_sweep_auto_resume.sh
# 
# This script automatically resumes after crashes by:
# 1. Saving sweep state to a file
# 2. Checking for existing incomplete runs on startup
# 3. Automatically resuming from the last checkpoint
# 4. Handling crashes gracefully with retry logic

# Activate the dfr environment
echo "🔧 Activating dfr environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dfr

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate dfr environment"
    exit 1
fi

echo "✅ Environment activated: $(conda info --active-environment)"

export RAY_TMPDIR=/scratch/ray_tmp_$(whoami)
mkdir -p $RAY_TMPDIR

set -x
PROJECT_NAME=HardRL_MATH_Sweep
LR=1e-5
MODEL_NAME=Qwen/Qwen2.5-Math-1.5B
MAX_NEW_TOKENS=1024
MAX_PROMPT_LENGTH=512
MODEL_ALIAS=${MODEL_NAME##*/}
GROUP_SIZE=8
ADVANTAGE_ESTIMATOR="dr_grpo"

MY_HOME=/VData/linna4335/difficulty_check/hard_rl

# Define training data file
TRAIN_FILE="$MY_HOME/data/MATH/train_filtered.parquet"
TEST_FILE="$MY_HOME/data/MATH/test.parquet"

# Auto-detect dataset characteristics for run naming
if [[ "$TRAIN_FILE" == *"filtered"* ]]; then
    DATASET_TYPE="filtered"
else
    DATASET_TYPE="full"
fi

if [[ "$TRAIN_FILE" == *"no_template"* ]]; then
    TEMPLATE_TYPE="no_template"
else
    TEMPLATE_TYPE="template"
fi

# Determine probe type from training file name
if [[ "$TRAIN_FILE" == *"E2H-AMC"* ]]; then
    PROBE_TYPE="E2H-AMC"
elif [[ "$TRAIN_FILE" == *"E2H-GSM8K"* ]]; then
    PROBE_TYPE="E2H-GSM8K"
else
    PROBE_TYPE="standard"
fi

# Configuration
MAX_RETRIES=3
RETRY_DELAY=60  # seconds
STATE_DIR="$MY_HOME/sweep_state"
mkdir -p "$STATE_DIR"

# Function to generate or detect sweep timestamp
get_or_create_sweep_timestamp() {
    local state_file="$STATE_DIR/current_sweep.state"
    
    if [[ -f "$state_file" ]]; then
        # Read existing sweep info
        source "$state_file"
        echo "🔄 Detected existing sweep with timestamp: $SWEEP_TIMESTAMP"
        echo "   Status: $SWEEP_STATUS"
        echo "   Completed runs: $COMPLETED_RUNS"
        echo "   Failed runs: $FAILED_RUNS"
        
        if [[ "$SWEEP_STATUS" == "completed" ]]; then
            echo "✅ Previous sweep already completed. Starting new sweep."
            create_new_sweep_state
        else
            echo "🔄 Resuming incomplete sweep..."
        fi
    else
        echo "🆕 No existing sweep detected. Starting new sweep."
        create_new_sweep_state
    fi
}

create_new_sweep_state() {
    local state_file="$STATE_DIR/current_sweep.state"
    SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    cat > "$state_file" << EOF
SWEEP_TIMESTAMP="$SWEEP_TIMESTAMP"
SWEEP_STATUS="running"
COMPLETED_RUNS=""
FAILED_RUNS=""
TOTAL_RUNS=3
MAX_RETRIES=$MAX_RETRIES
EOF
    
    echo "📝 Created new sweep state: $SWEEP_TIMESTAMP"
}

# Function to update sweep state
update_sweep_state() {
    local run_id=$1
    local status=$2  # "completed" or "failed"
    local state_file="$STATE_DIR/current_sweep.state"
    
    source "$state_file"
    
    if [[ "$status" == "completed" ]]; then
        if [[ ! "$COMPLETED_RUNS" =~ $run_id ]]; then
            COMPLETED_RUNS="$COMPLETED_RUNS $run_id"
        fi
        # Remove from failed runs if it was there
        FAILED_RUNS=$(echo "$FAILED_RUNS" | sed "s/\b$run_id\b//g" | xargs)
    elif [[ "$status" == "failed" ]]; then
        if [[ ! "$FAILED_RUNS" =~ $run_id ]]; then
            FAILED_RUNS="$FAILED_RUNS $run_id"
        fi
    fi
    
    # Clean up the variables
    COMPLETED_RUNS=$(echo "$COMPLETED_RUNS" | xargs)
    FAILED_RUNS=$(echo "$FAILED_RUNS" | xargs)
    
    # Check if sweep is complete
    local completed_count=$(echo "$COMPLETED_RUNS" | wc -w)
    if [[ $completed_count -eq 3 ]]; then
        SWEEP_STATUS="completed"
    fi
    
    # Update state file
    cat > "$state_file" << EOF
SWEEP_TIMESTAMP="$SWEEP_TIMESTAMP"
SWEEP_STATUS="$SWEEP_STATUS"
COMPLETED_RUNS="$COMPLETED_RUNS"
FAILED_RUNS="$FAILED_RUNS"
TOTAL_RUNS=3
MAX_RETRIES=$MAX_RETRIES
EOF
    
    echo "📝 Updated sweep state: Run $run_id -> $status"
    echo "   Completed: [$COMPLETED_RUNS]"
    echo "   Failed: [$FAILED_RUNS]"
}

# Function to check if a run has completed successfully
check_run_completion() {
    local run_id=$1
    local run_name=${MODEL_ALIAS}_${LR}_${GROUP_SIZE}_${MAX_NEW_TOKENS}_${DATASET_TYPE}_${TEMPLATE_TYPE}_train_dr_${ADVANTAGE_ESTIMATOR}_${PROBE_TYPE}_${SWEEP_TIMESTAMP}_run${run_id}
    local checkpoint_dir="$MY_HOME/checkpoints/$PROJECT_NAME/$run_name"
    
    if [[ ! -d "$checkpoint_dir" ]]; then
        echo "no_checkpoint"
        return
    fi
    
    local latest_file="$checkpoint_dir/latest_checkpointed_iteration.txt"
    if [[ ! -f "$latest_file" ]]; then
        echo "no_checkpoint"
        return
    fi
    
    local latest_iteration=$(cat "$latest_file")
    # Consider a run "completed" if it has more than 80 steps
    if [[ $latest_iteration -ge 80 ]]; then
        echo "completed"
    else
        echo "incomplete"
    fi
}

# Function to get checkpoint status
get_checkpoint_status() {
    local run_id=$1
    local run_name=${MODEL_ALIAS}_${LR}_${GROUP_SIZE}_${MAX_NEW_TOKENS}_${DATASET_TYPE}_${TEMPLATE_TYPE}_train_dr_${ADVANTAGE_ESTIMATOR}_${PROBE_TYPE}_${SWEEP_TIMESTAMP}_run${run_id}
    local checkpoint_dir="$MY_HOME/checkpoints/$PROJECT_NAME/$run_name"
    local latest_file="$checkpoint_dir/latest_checkpointed_iteration.txt"
    
    if [[ -f "$latest_file" ]]; then
        local latest_iteration=$(cat "$latest_file")
        echo "Found checkpoint at iteration $latest_iteration"
        return 0
    else
        echo "No checkpoint found"
        return 1
    fi
}

# Function to run a single experiment with retry logic
run_experiment_with_retry() {
    local run_id=$1
    local seed=$2
    local max_retries=$3
    
    local attempt=1
    while [[ $attempt -le $max_retries ]]; do
        echo "========================================="
        echo "Starting Run $run_id/3 - Attempt $attempt/$max_retries"
        echo "Seed: $seed"
        echo "========================================="
        
        # Check if already completed
        local completion_status=$(check_run_completion $run_id)
        if [[ "$completion_status" == "completed" ]]; then
            echo "✅ Run $run_id already completed, skipping"
            update_sweep_state $run_id "completed"
            return 0
        fi
        
        # Check checkpoint status
        local status_msg=$(get_checkpoint_status $run_id)
        echo "📋 Checkpoint status: $status_msg"
        
        # Generate unique run name with run ID
        RUN_NAME=${MODEL_ALIAS}_${LR}_${GROUP_SIZE}_${MAX_NEW_TOKENS}_${DATASET_TYPE}_${TEMPLATE_TYPE}_train_dr_${ADVANTAGE_ESTIMATOR}_${PROBE_TYPE}_${SWEEP_TIMESTAMP}_run${run_id}
        
        # Set checkpoint directory for this run
        CHECKPOINT_DIR="$MY_HOME/checkpoints/$PROJECT_NAME/$RUN_NAME"
        
        # Create a lock file to prevent concurrent runs
        local lock_file="$STATE_DIR/run_${run_id}.lock"
        if [[ -f "$lock_file" ]]; then
            local lock_pid=$(cat "$lock_file")
            if kill -0 "$lock_pid" 2>/dev/null; then
                echo "⚠️  Run $run_id is already running (PID: $lock_pid). Skipping."
                return 1
            else
                echo "🧹 Removing stale lock file"
                rm -f "$lock_file"
            fi
        fi
        
        # Create lock file
        echo $$ > "$lock_file"
        
        # Set up trap to clean up lock file on exit
        trap "rm -f '$lock_file'" EXIT INT TERM
        
        # Run the training
        CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=$ADVANTAGE_ESTIMATOR \
            data.train_files=$TRAIN_FILE \
            data.val_files=$TEST_FILE \
            data.train_batch_size=256 \
            data.max_prompt_length=$MAX_PROMPT_LENGTH \
            data.max_response_length=$MAX_NEW_TOKENS \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path=$MODEL_NAME \
            actor_rollout_ref.actor.optim.lr=$LR \
            actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=256 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
            actor_rollout_ref.rollout.n=$GROUP_SIZE \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.top_p=1.0 \
            algorithm.use_kl_in_reward=False \
            algorithm.norm_adv_by_std_in_grpo=False \
            trainer.critic_warmup=0 \
            trainer.logger='["console","wandb"]' \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.save_freq=10 \
            trainer.test_freq=1 \
            trainer.resume_mode=auto \
            trainer.default_local_dir=$CHECKPOINT_DIR \
            trainer.project_name=$PROJECT_NAME \
            trainer.experiment_name=$RUN_NAME \
            trainer.total_epochs=2
        
        local exit_code=$?
        
        # Clean up lock file
        rm -f "$lock_file"
        trap - EXIT INT TERM
        
        if [ $exit_code -eq 0 ]; then
            # Double-check completion
            local final_status=$(check_run_completion $run_id)
            if [[ "$final_status" == "completed" ]]; then
                echo "✅ Run $run_id completed successfully"
                update_sweep_state $run_id "completed"
                return 0
            else
                echo "⚠️  Run $run_id finished but may not be complete (exit code 0 but checkpoint incomplete)"
            fi
        else
            echo "❌ Run $run_id failed with exit code $exit_code (attempt $attempt/$max_retries)"
        fi
        
        if [[ $attempt -lt $max_retries ]]; then
            echo "⏳ Waiting $RETRY_DELAY seconds before retry..."
            sleep $RETRY_DELAY
        fi
        
        ((attempt++))
    done
    
    echo "❌ Run $run_id failed after $max_retries attempts"
    update_sweep_state $run_id "failed"
    return 1
}

# Function to determine which runs need to be executed
get_runs_to_execute() {
    local state_file="$STATE_DIR/current_sweep.state"
    source "$state_file"
    
    local runs_needed=()
    
    for run_id in {1..3}; do
        # Check if run is already completed
        if [[ "$COMPLETED_RUNS" =~ $run_id ]]; then
            echo "✅ Run $run_id: Already completed" >&2
            continue
        fi
        
        # Check actual completion status from filesystem
        local status=$(check_run_completion $run_id)
        case $status in
            "completed")
                echo "✅ Run $run_id: Found completed, updating state" >&2
                update_sweep_state $run_id "completed"
                ;;
            "incomplete")
                echo "🔄 Run $run_id: Has checkpoint but incomplete, will resume" >&2
                runs_needed+=($run_id)
                ;;
            "no_checkpoint")
                echo "🆕 Run $run_id: No checkpoint found, will start fresh" >&2
                runs_needed+=($run_id)
                ;;
        esac
    done
    
    echo "${runs_needed[@]}"
}

# Function to create resume script for manual restart
create_resume_script() {
    local resume_script="$MY_HOME/resume_current_sweep.sh"
    cat > "$resume_script" << EOF
#!/bin/bash
# Auto-generated resume script
# Run this to manually resume the current sweep

cd "$MY_HOME"
./grpo_MATH_refined_sweep_auto_resume.sh
EOF
    chmod +x "$resume_script"
    echo "📄 Created resume script: $resume_script"
}

# Main execution starts here
echo "🚀 Starting Auto-Resuming GRPO MATH Sweep"
echo "========================================="

# Get or create sweep timestamp
get_or_create_sweep_timestamp

# Create resume script
create_resume_script

# Load current state
source "$STATE_DIR/current_sweep.state"

echo ""
echo "📋 Sweep Configuration:"
echo "   Project: $PROJECT_NAME"
echo "   Timestamp: $SWEEP_TIMESTAMP" 
echo "   Advantage Estimator: $ADVANTAGE_ESTIMATOR"
echo "   Training file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   Max retries per run: $MAX_RETRIES"
echo ""

# Check if sweep is already completed
if [[ "$SWEEP_STATUS" == "completed" ]]; then
    echo "🎉 Sweep already completed!"
    echo "   All 3 runs finished successfully."
    exit 0
fi

# Determine which runs to execute
RUNS_TO_EXECUTE=($(get_runs_to_execute))

if [[ ${#RUNS_TO_EXECUTE[@]} -eq 0 ]]; then
    echo "🎉 All runs are completed!"
    update_sweep_state "" ""  # This will trigger completion check
    source "$STATE_DIR/current_sweep.state"
    if [[ "$SWEEP_STATUS" == "completed" ]]; then
        echo "✅ Sweep marked as completed."
    fi
    exit 0
fi

echo "📋 Runs to execute: ${RUNS_TO_EXECUTE[*]}"
echo ""

# Define seeds for reproducibility
SEEDS=(42 123 456)

# Track results
successful_runs=0
failed_runs=0

# Execute runs with auto-retry
for run_id in "${RUNS_TO_EXECUTE[@]}"; do
    seed=${SEEDS[$((run_id-1))]}
    
    echo "🎯 Executing Run $run_id with seed $seed"
    
    if run_experiment_with_retry $run_id $seed $MAX_RETRIES; then
        ((successful_runs++))
        echo "✅ Run $run_id completed successfully"
    else
        ((failed_runs++))
        echo "❌ Run $run_id failed after all retries"
    fi
    
    # Small delay between runs
    remaining_runs=$((${#RUNS_TO_EXECUTE[@]} - successful_runs - failed_runs))
    if [[ $remaining_runs -gt 0 ]]; then
        echo "⏳ Waiting 30 seconds before next run..."
        sleep 30
    fi
done

# Final summary
echo ""
echo "========================================="
echo "SWEEP EXECUTION COMPLETED"
echo "========================================="

# Reload final state
source "$STATE_DIR/current_sweep.state"

echo "📊 Final Results:"
echo "   Timestamp: $SWEEP_TIMESTAMP"
echo "   Successful runs: $successful_runs/${#RUNS_TO_EXECUTE[@]} (attempted in this session)"
echo "   Failed runs: $failed_runs/${#RUNS_TO_EXECUTE[@]} (attempted in this session)"
echo "   Total completed: $(echo $COMPLETED_RUNS | wc -w)/3"
echo "   Total failed: $(echo $FAILED_RUNS | wc -w)/3"
echo ""

if [[ "$SWEEP_STATUS" == "completed" ]]; then
    echo "🎉 Entire sweep completed successfully!"
    echo "   Check WandB project '$PROJECT_NAME' for detailed metrics"
    echo "   Sweep timestamp: $SWEEP_TIMESTAMP"
    
    # Archive the sweep state
    mv "$STATE_DIR/current_sweep.state" "$STATE_DIR/completed_${SWEEP_TIMESTAMP}.state"
    echo "📁 Sweep state archived to: completed_${SWEEP_TIMESTAMP}.state"
    
    exit 0
else
    echo "⚠️  Some runs still need attention."
    echo ""
    echo "🔄 To continue this sweep later, simply run:"
    echo "   $MY_HOME/resume_current_sweep.sh"
    echo "   # or"
    echo "   ./grpo_MATH_refined_sweep_auto_resume.sh"
    echo ""
    echo "📁 Sweep state saved. The script will automatically resume from where it left off."
    
    exit 1
fi
