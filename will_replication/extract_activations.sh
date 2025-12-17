
CUDA_VISIBLE_DEVICES=1 python extract_activations.py \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --dataset_path "../predicting_learnability/data/MATH/MATH_Qwen2.5-Math-1.5B-Instruct-SR_train_max_3000_k_1_temp_0.0.parquet" \
    --output_path "activations/qwen2.5-Math-1.5b_train.pt" \
    --layers "all" \
    --batch_size 32 \
    --question_column "prompt" \
    --label_column "correct" \