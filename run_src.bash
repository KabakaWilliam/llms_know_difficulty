CUDA_VISIBLE_DEVICES=0 python3 src/llms_know_difficulty/main.py --probe linear_eoi_probe --dataset DigitalLearningGmbH_MATH-lighteval --model Qwen/Qwen2.5-Math-1.5B-Instruct --max_len 3000 --k 8 --temperature 0.7

# attn_probe
# linear_eoi_probe