#!/bin/bash
# Train checkpoint probes for difficulty direction analysis
# MODELS:
# Llama-3.2-3B-Instruct, Qwen2.5-Math-1.5B
# Configuration
GPU_ID=0
PYTHON_MODULE="difficulty_direction.train_checkpoint_probes"
MODEL_NAME=Llama-3.2-3B-Instruct
CHECKPOINT_DIR="/VData/linna4335/difficulty_check/converted_hf_models/${MODEL_NAME}_ckpt_1e-5_8_3000_filtered_template_train_BASELINE_grpo_standard_20250917_022224_HardRL_MATH_Baseline_3000_token_eval_maxprompt_1024"

# CHECKPOINT_DIR="/VData/linna4335/difficulty_check/converted_hf_models/${MODEL_NAME}_HardRL_MATH_Baseline_3000_token_eval_maxprompt_1024"
OUTPUT_DIR="/VData/linna4335/difficulty_check/probe_checkpoint_results/${MODEL_NAME}"
N_FOLDS=5

# Run the command
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m ${PYTHON_MODULE} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_k_fold \
    --n_folds ${N_FOLDS}