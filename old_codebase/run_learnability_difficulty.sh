#!/bin/bash

# Simple script to run multiple models under 14B parameters
# Edit the models list below to add/remove models
CHOSEN_DEVICE=1
# nvidia/OpenMath2-Llama3.1-8B
# allenai/Olmo-3-7B-Think 

CUDA_VISIBLE_DEVICES=$CHOSEN_DEVICE python3 -m difficulty_direction.run \
    --model_path allenai/Olmo-3-7B-Think \
    --use_k_fold \



