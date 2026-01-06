# Will's Probe Training Pipeline

A simplified and modular pipeline for training linear probes on LLM activations.

## Overview

This pipeline consists of two main scripts:

1. **`extract_activations.py`**: Extracts hidden state activations from LLM layers
2. **`train_probe.py`**: Trains linear probes (Ridge/Logistic regression) on the extracted activations

## Workflow

This pipeline replicates the approach from `get_directions_store_preds.py`:

1. **Post-instruction token extraction**: Instead of extracting from arbitrary positions, we extract activations only from the tokens that follow the instruction in the chat template
   - Auto-detected via: `positions = list(range(-len(model.eoi_toks), 0))`
   - For example, if there are 3 post-instruction tokens, we extract from positions `[-3, -2, -1]`

2. **Multi-position probing**: A separate linear probe is trained for each (layer, position) combination
   - Not just one probe per layer, but one probe per (layer × position)
   - This allows finding the optimal layer AND token position

3. **Cross-validation**: Uses StratifiedKFold (classification) or KFold (regression) to select the best probe

### Step 1: Extract Activations

Extract activations from post-instruction tokens for both train and test sets:

```bash
# Extract training activations
python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "../predicting_learnability/data/MATH/MATH_train.parquet" \
    --output_path "activations/train.pt" \
    --label_column "success_rate" \
    --question_column "question" \
    --layers "all" \
    --batch_size 32

# Extract test activations
python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "../predicting_learnability/data/MATH/MATH_test.parquet" \
    --output_path "activations/test.pt" \
    --label_column "success_rate" \
    --question_column "question" \
    --layers "all" \
    --batch_size 32
```

**Key arguments:**
- `--dataset_path`: Path to local parquet file with questions and labels
- `--label_column`: Column name for labels (e.g., `"success_rate"`, `"difficulty_score"`)
- `--question_column`: Column name for questions/prompts (default: `"question"`)
- `--layers`: Which layers to extract from (`"all"`, `"-1"`, or `"0,1,2"`)
- `--eoi_tokens`: Number of post-instruction tokens (auto-detected if not specified)
- `--use_chat_template`: Add this flag if questions need chat template formatting (not needed if already formatted)
- `--max_samples`: Limit number of samples (useful for testing)

**Output format**: Saved as `.pt` file with shape `[N_samples, N_layers, N_positions, Hidden_size]`

### Step 2: Train Probes

Train linear probes on all (layer, position) combinations:

```bash
python train_probe.py \
    --train_activations "activations/train.pt" \
    --test_activations "activations/test.pt" \
    --output_dir "results/run_1" \
    --task_type "auto" \
    --alpha 1.0 \
    --k_fold \
    --n_folds 5 \
    --seed 42
```

**Key arguments:**
- `--task_type`: `"auto"` (infer from data), `"regression"`, or `"classification"`
- `--alpha`: Regularization strength for Ridge regression (default: 1.0, matching original)
- `--k_fold`: Enable cross-validation for model selection
- `--n_folds`: Number of CV folds (default: 5)

## Output Files

After running `train_probe.py`, you'll get:

1. **`performance.json`**: Performance metrics for all (layer, position) combinations
   - Nested structure: `layer_performance["test"][pos_idx][layer_idx]`
   - Train/test/CV scores for each probe
   - Position indices and layer indices
   - Task type and metric used

2. **`best_probe_predictions.json`**: Predictions from the best probe (selected by CV)
   - Best position and layer indices
   - CV score (selection criterion) and test score
   - Actual labels and predictions on train/test/CV sets
   - Useful for downstream analysis

3. **`all_predictions.json`**: Predictions from all (layer, position) probes
   - Keyed by `"pos_{position}_layer_{layer_idx}"`
   - Useful for analyzing which (layer, position) combinations work best
   - Contains train/test/CV predictions for every probe traineds from the best probe (selected by CV)
   - Actual labels
   - Predictions on train/test/CV sets
## Comparison to Other Implementations

### vs. `predict_success_rate.py` (thom_replication)
- ✅ **Cross-validation**: Built-in k-fold CV vs. simple train/val split
- ✅ **Separation of concerns**: Activation extraction separate from training
- ✅ **Reusability**: Extract activations once, train many probes
- ✅ **Faster iteration**: No need to re-run LLM for each experiment
- ✅ **Multi-position probing**: Extracts from all post-instruction tokens, not just last token

### vs. `get_directions_store_preds.py` (difficulty_direction)
- ✅ **Simpler interface**: No custom ModelBase wrapper needed
- ✅ **Standard HuggingFace**: Works with any HF model directly
- ✅ **Local datasets**: Works with local parquet files, not just HF datasets
- ✅ **Cleaner code**: Single responsibility per script
## Using `run.sh`

The recommended way to run the pipeline is using `run.sh`, which has all configuration at the top:

```bash
# Edit these variables in run.sh:
MODEL="Qwen/Qwen2-1.5B-Instruct"
DATASET_DIR="../predicting_learnability/data/SR_DATASETS/THOMS_MATH"
TRAIN_DATASET="${DATASET_DIR}/MATH_train-Qwen-${MODEL_ALIAS}_lorem.parquet"
TEST_DATASET="${DATASET_DIR}/MATH_test-Qwen-${MODEL_ALIAS}_lorem.parquet"
QUESTION_COL="question"
LABEL_COL="success_rate"
LAYERS="all"
BATCH_SIZE=32
GPU=1

# Then run:
bash run.sh
```

This is DRY and makes it easy to:
- Switch between models (just change `MODEL`)
- Use different datasets (change `DATASET_DIR`)
- Adjust hyperparameters (change `BATCH_SIZE`, `LAYERS`, etc.)
- Run on different GPUs (change `GPU`)

## Manual Pipeline Examples

If you want to run steps separately or experiment:

```bash
# 1. Extract from all layers
python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "data/train.parquet" \
    --output_path "activations/train.pt" \
    --layers "all" \
    --label_column "success_rate"

python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "data/test.parquet" \
    --output_path "activations/test.pt" \
    --layers "all" \
    --label_column "success_rate"

# 2. Train probes with CV
python train_probe.py \
    --train_activations "activations/train.pt" \
    --test_activations "activations/test.pt" \
    --output_dir "results/all_layers" \
    --k_fold \
    --n_folds 5

# 3. Try different regularization
python train_probe.py \
    --train_activations "activations/train.pt" \
    --test_activations "activations/test.pt" \
    --output_dir "results/alpha_0.1" \
    --alpha 0.1 \
    --k_fold

# 4. Fast baseline with last layer only
python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "data/train.parquet" \
    --output_path "activations/train_last.pt" \
    --layers "-1"

python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "data/test.parquet" \
    --output_path "activations/test_last.pt" \
    --layers "-1"

python train_probe.py \
    --train_activations "activations/train_last.pt" \
    --test_activations "activations/test_last.pt" \
    --output_dir "results/last_layer"
``` --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "../predicting_learnability/data/MATH/MATH_Qwen2-1.5B-Instruct-SR_train_max_3000_k_1_temp_0.0.parquet" \
    --output_path "activations/qwen2_train_last.pt" \
    --layers "-1"

python extract_activations.py \
    --model "Qwen/Qwen2-1.5B-Instruct" \
    --dataset_path "../predicting_learnability/data/MATH/MATH_Qwen2-1.5B-Instruct-SR_test_max_3000_k_1_temp_0.0.parquet" \
    --output_path "activations/qwen2_test_last.pt" \
    --layers "-1"

python train_probe.py \
    --train_activations "activations/qwen2_train_last.pt" \
    --test_activations "activations/qwen2_test_last.pt" \
    --output_dir "results/qwen2_last_layer"
```

## Working with Different Datasets

The pipeline works with any parquet file that has:
- A **question/prompt column** (specified with `--question_column`)
- A **label/score column** (specified with `--label_column`)

Example datasets in the repo:
- Success rate: `MATH_*-SR_*.parquet` with columns `question`, `success_rate`
- Custom difficulty: Any parquet with `question`, `difficulty_score`, etc.

```bash
# Example: Using a different label column
python extract_activations.py \
    --model "your-model" \
    --dataset_path "your_dataset.parquet" \
    --label_column "difficulty_label" \
    --question_column "prompt" \
    --output_path "activations/custom.pt"
```

## Notes

- Datasets should be parquet files with question and label columns
- Questions can be pre-formatted with prompt templates (like in the MATH-SR datasets) or raw text
- Use `--use_chat_template` flag if questions need chat template formatting
- Activations are saved as PyTorch tensors for efficiency
- Cross-validation uses Spearman correlation for regression, ROC-AUC for classification
- All probes use `fit_intercept=False` to match the original implementations
- Compatible with datasets from `create_success_rate_dataset.py`

## Technical Notes

**Post-instruction token extraction:**
- Automatically detects the number of post-instruction tokens from the chat template
- For Qwen models, this is typically 2-3 tokens (e.g., `<|im_start|>assistant`)
- Manual override: `--eoi_tokens N` to specify exactly N tokens

**Activation storage:**
- Saved as PyTorch tensors with shape `[N_samples, N_layers, N_positions, Hidden_size]`
- Includes metadata: `positions`, `layer_indices`, `n_eoi_tokens`, `model_name`

**Probe training:**
- Uses Ridge regression for regression tasks, Logistic for classification
- All probes have `fit_intercept=False` to match original implementation
- Regularization: `alpha=1.0` for Ridge, `C=1.0` for Logistic (C is inverse of alpha)
- Cross-validation uses Spearman correlation (regression) or ROC-AUC (classification)

**Performance:**
- Extracting activations is the slow step (requires running the LLM)
- Training probes is fast (sklearn on CPU)
- Extract once, experiment with different probe configurations many times

**Compatibility:**
- Works with datasets from `create_success_rate_dataset.py`
- Questions can be pre-formatted with chat template or raw text
- Use `--use_chat_template` only if questions need formatting (not needed for SR datasets)
