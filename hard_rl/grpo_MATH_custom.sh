#MATH has 12k observations
# num_training_steps = (MATH_len/batch_size) * num_epochs

# setting up num gradient accumulation steps
# micro_train_batch_size = train_batch_size/grad_accumulation_steps
# ...grad_accumulation_steps = train_batch_size/micro_train_batch_size (256/4 = 64 steps per mini-b)
# https://github.com/volcengine/verl/issues/914
export RAY_TMPDIR=/scratch/ray_tmp_$(whoami)
mkdir -p $RAY_TMPDIR

# Redirect ALL temporary directories to /scratch to avoid "No space left on device"
export TMPDIR=/scratch/tmp_$(whoami)
export TMP=/scratch/tmp_$(whoami)
export TEMP=/scratch/tmp_$(whoami)
export TEMPDIR=/scratch/tmp_$(whoami)
mkdir -p $TMPDIR

# Memory optimization environment variables
export RAY_memory_usage_threshold=0.85  # Lower threshold to prevent OOM
export RAY_memory_monitor_refresh_ms=0  # Monitor memory every second
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# # Additional memory optimizations
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export CUDA_LAUNCH_BLOCKING=0

set -x
PROJECT_NAME=HardRL_MATH_Custom
LR=1e-5
MODEL_NAME=Qwen/Qwen2.5-Math-1.5B
# MODEL_NAME=Qwen/Qwen2.5-Math-7B
MAX_NEW_TOKENS=1024 #1024 #3000
MAX_PROMPT_LENGTH=512 #1024 #512
MODEL_ALIAS=${MODEL_NAME##*/}
GROUP_SIZE=8
# ADVANTAGE_ESTIMATOR="grpo"#grpo_lead_hardcoded_levels#dr_grpo_regret#grpo_lead_hardcoded_levels_pro_hard#dr_grpo#hard_rl_mix#hard_rl_mix_normalizes#grpo_lead_basic_hardcoded_levels_pro_hard#hard_rl_mix_basic#dr_grpo
ADVANTAGE_ESTIMATOR="grpo"

MY_HOME=/VData/linna4335/difficulty_check/hard_rl

# Define training data file
TRAIN_FILE="$MY_HOME/data/MATH/train_E2H-AMC_filtered.parquet"
# Alternative: TRAIN_FILE="$MY_HOME/data/MATH/train.parquet"#train_filtered#train_E2H-AMC_filtered #train_E2H-AMC.parquet
TEST_FILE="$MY_HOME/data/AIME24/validation.parquet"
# TEST_FILE="$MY_HOME/data/MATH/test.parquet"

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

# Generate unique timestamp ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=${MODEL_ALIAS}_${LR}_${GROUP_SIZE}_${MAX_NEW_TOKENS}_${DATASET_TYPE}_${TEMPLATE_TYPE}_train_dr_${ADVANTAGE_ESTIMATOR}_${PROBE_TYPE}_${TIMESTAMP}

CUDA_VISIBLE_DEVICES=3 python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False\
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=2 \
    custom_reward_function.path=verl/verl/utils/reward_score/math_level.py \
    custom_reward_function.name=compute_level_score_sigmoid \