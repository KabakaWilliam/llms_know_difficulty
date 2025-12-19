python predict_success_rate_with_probe.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --hf_dataset DigitalLearningGmbH/MATH-lighteval \
    --train_scores_path /home/thomfoster/llms_know_difficulty/thom_replication/data/MATH_train-Qwen-Qwen2-1.5B-Instruct.parquet \
    --val_scores_path /home/thomfoster/llms_know_difficulty/thom_replication/data/MATH_test-Qwen-Qwen2-1.5B-Instruct.parquet \
    --layers all \
    --layer_mode concat_all_sequence_positions \
    --pool all_sequence_positions \
    --max_length 512 \
    --lr 1e-4 \
    --batch_size 128 \

