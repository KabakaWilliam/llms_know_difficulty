python -m scripts.train_probe \
    --train_activations "activations/qwen2.5-Math-1.5b_train.pt" \
    --test_activations "activations/qwen2.5-Math-1.5b_test.pt" \
    --output_dir "results/qwen2.5-Math-1.5b_all_layers" \
    --k_fold \
    --n_folds 5