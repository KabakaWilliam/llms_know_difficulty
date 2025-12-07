#!/bin/bash
# chmod +x grpo_MATH_sweep.sh
# ./grpo_MATH_sweep.sh
# filepath: /VData/linna4335/difficulty_check/hard_rl/grpo_MATH_sweep.sh
#MATH has 12k observations
# num_training_steps = (MATH_len/batch_size) * num_epochs

# setting up num gradient accumulation steps
# micro_train_batch_size = train_batch_size/grad_accumulation_steps
# ...grad_accumulation_steps = train_batch_size/micro_train_batch_size (256/4 = 64 steps per mini-b)
# https://github.com/volcengine/verl/issues/914

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
MAX_NEW_TOKENS=1024 #1024 #3000
MAX_PROMPT_LENGTH=512 #1024 #512
MODEL_ALIAS=${MODEL_NAME##*/}
GROUP_SIZE=8
# ADVANTAGE_ESTIMATOR="grpo"#grpo_lead_hardcoded_levels#dr_grpo_regret#grpo_lead_hardcoded_levels_pro_hard#dr_grpo#hard_rl_mix#hard_rl_mix_normalizes#grpo_lead_basic_hardcoded_levels_pro_hard#hard_rl_mix_basic#dr_grpo
ADVANTAGE_ESTIMATOR="dr_grpo"

MY_HOME=/VData/linna4335/difficulty_check/hard_rl

# Define training data file
TRAIN_FILE="$MY_HOME/data/MATH/train_filtered.parquet"
# Alternative: TRAIN_FILE="$MY_HOME/data/MATH/train.parquet"#train_filtered#train_E2H-AMC_filtered #train_E2H-AMC.parquet
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

# Generate base timestamp for the sweep
BASE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run a single experiment
run_experiment() {
    local run_id=$1
    local seed=$2
    
    echo "========================================="
    echo "Starting Run $run_id/3 with seed $seed"
    echo "========================================="
    
    # Generate unique run name with run ID
    RUN_NAME=${MODEL_ALIAS}_${LR}_${GROUP_SIZE}_${MAX_NEW_TOKENS}_${DATASET_TYPE}_${TEMPLATE_TYPE}_train_dr_${ADVANTAGE_ESTIMATOR}_${PROBE_TYPE}_${BASE_TIMESTAMP}_run${run_id}
    
    python3 -m verl.trainer.main_ppo \
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
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=1 \
        trainer.resume_mode=auto \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$RUN_NAME \
        trainer.total_epochs=2 \
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Run $run_id completed successfully"
    else
        echo "❌ Run $run_id failed with exit code $exit_code"
    fi
    
    echo "========================================="
    echo "Finished Run $run_id/3"
    echo "========================================="
    echo
    
    return $exit_code
}

# Run the sweep - 3 identical runs with different seeds
echo "Starting GRPO MATH Sweep - 3 runs"
echo "Project: $PROJECT_NAME"
echo "Advantage Estimator: $ADVANTAGE_ESTIMATOR"
echo "Base timestamp: $BASE_TIMESTAMP"
echo "Training file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo

# Define seeds for reproducibility
SEEDS=(42 123 456)

# Track successful runs
successful_runs=0
failed_runs=0

# Run the experiments
for i in {1..3}; do
    seed=${SEEDS[$((i-1))]}

    echo "📋 Run $i/3 Configuration:"
    echo "   Seed: $seed"
    echo "   Model: $MODEL_NAME"
    echo "   LR: $LR"
    echo "   Group Size: $GROUP_SIZE"
    echo "   Max Tokens: $MAX_NEW_TOKENS"
    echo

    run_experiment $i $seed

    if [ $? -eq 0 ]; then
        ((successful_runs++))
    else
        ((failed_runs++))
    fi

    # Add a small delay between runs to avoid resource conflicts
    if [ $i -lt 3 ]; then
        echo "⏳ Waiting 30 seconds before next run..."
        sleep 30
    fi
done

# Summary
echo "========================================="
echo "SWEEP COMPLETED"
echo "========================================="
echo "📊 Results Summary:"
echo "   Successful runs: $successful_runs/3"
echo "   Failed runs: $failed_runs/3"
echo "   Model: $MODEL_ALIAS"
echo "   Dataset: $DATASET_TYPE ($TEMPLATE_TYPE)"
echo "   Probe Type: $PROBE_TYPE"
echo "   Advantage Estimator: $ADVANTAGE_ESTIMATOR"
echo "   Learning Rate: $LR"
echo "   Group Size: $GROUP_SIZE"
echo "   Max New Tokens: $MAX_NEW_TOKENS"
echo "   Project: $PROJECT_NAME"
echo "   Base timestamp: $BASE_TIMESTAMP"
echo

if [ $failed_runs -eq 0 ]; then
    echo "🎉 All runs completed successfully!"
    echo "   Check WandB project '$PROJECT_NAME' for detailed metrics"
    exit 0
else
    echo "⚠️  Some runs failed. Check logs above for details."
    echo "   Successful runs: $successful_runs/3"
    exit 1
fi