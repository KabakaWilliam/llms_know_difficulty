# Using Local Datasets

This guide explains how to configure and use local (non-HuggingFace) datasets with the difficulty_direction project.

## Configuration

Local datasets are configured in `config.py` under the `DATASET_CONFIGS` dictionary. Here's the structure:

### Example: Local Dataset Configuration

```python
"your_dataset_name": {
    "dataset_type": "local",  # Must be "local" for file-based datasets
    "local_path": "/path/to/your/dataset/directory",
    "file_pattern": "dataset_{split}.parquet",  # {split} will be replaced with train/test/eval
    "file_format": "parquet",  # Supported: parquet, json, jsonl, csv, pkl/pickle
    "splits": ["train", "test"],  # Available splits
    "prompt_column": "prompt",  # Column containing the problem/question
    "answer_column": "answer",  # Column containing the answer
    "has_train_split": True,  # Whether dataset has separate train/test files
    "difficulty_column": "difficulty"  # Column containing difficulty rating (e.g., success_rate)
}
```

### Required Fields

- **`dataset_type`**: Set to `"local"` for file-based datasets
- **`local_path`**: Absolute path to the directory containing your dataset files
- **`file_pattern`**: Filename pattern with `{split}` placeholder that gets replaced with split names (train/test/eval)
- **`file_format`**: File type - supports: `parquet`, `json`, `jsonl`, `csv`, `pkl`, `pickle`
- **`splits`**: List of available splits (e.g., `["train", "test"]`)
- **`prompt_column`**: Name of the column containing prompts/questions
- **`answer_column`**: Name of the column containing ground truth answers
- **`difficulty_column`**: Name of the column containing difficulty scores
- **`has_train_split`**: `True` if you have separate train/test files, `False` if you only have one file

### Optional Fields

- **`train_split_ratio`**: If `has_train_split` is `False`, this ratio (default 0.8) determines train/test split

## Dataset File Requirements

Your dataset files should contain at minimum these columns:
- A column for prompts/questions
- A column for answers
- A column for difficulty ratings

### Example: predicting_learnability Dataset

Current working example in the codebase:

```python
"predicting_learnability": {
    "dataset_type": "local",
    "local_path": "/VData/linna4335/llms_know_difficult/predicting_learnability/data",
    "file_pattern": "MATH_DeepSeek-R1-Distill-Qwen-1.5B-SR_{split}.parquet",
    "file_format": "parquet",
    "splits": ["train", "test"],
    "prompt_column": "prompt",
    "answer_column": "ground_truth",
    "has_train_split": True,
    "difficulty_column": "success_rate"
}
```

This expects files:
- `/VData/linna4335/llms_know_difficult/predicting_learnability/data/MATH_DeepSeek-R1-Distill-Qwen-1.5B-SR_train.parquet`
- `/VData/linna4335/llms_know_difficult/predicting_learnability/data/MATH_DeepSeek-R1-Distill-Qwen-1.5B-SR_test.parquet`

## Adding Custom Prompt Templates

If your dataset needs a custom prompt template (instead of using prompts as-is), add it to `PROMPT_TEMPLATES` in `config.py`:

```python
PROMPT_TEMPLATES = {
    "your_dataset_name": {
        "template": "Your template with {problem} placeholder",
        "prompt_column": "problem"
    }
}
```

If no template is provided (or `template` is `None`), the prompt column will be used directly without formatting.

## File Format Examples

### Parquet (Recommended)
```python
"file_format": "parquet"
```

### JSON
```python
"file_format": "json"
```

### JSONL (JSON Lines)
```python
"file_format": "jsonl"
```

### CSV
```python
"file_format": "csv"
```

### Pickle
```python
"file_format": "pkl"  # or "pickle"
```

## Usage

Once configured, use your local dataset just like HuggingFace datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m difficulty_direction.run \
    --model_path your/model/path \
    --use_k_fold \
    --batch_size 16 \
    --n_train 1000 \
    --n_test 500 \
    --subset_datasets your_dataset_name \
    --evaluation_datasets your_dataset_name \
    --generation_batch_size 8 \
    --max_new_tokens 2000
```

## Switching Between Datasets

You can easily switch which local dataset files to use by changing the `file_pattern`:

```python
# Use different model's success rate data
"file_pattern": "MATH_Different-Model_{split}.parquet"
```

Or create multiple dataset configurations:

```python
"predicting_learnability_model1": {
    "dataset_type": "local",
    "local_path": "/path/to/data",
    "file_pattern": "MATH_Model1_{split}.parquet",
    # ... other fields
},
"predicting_learnability_model2": {
    "dataset_type": "local",
    "local_path": "/path/to/data",
    "file_pattern": "MATH_Model2_{split}.parquet",
    # ... other fields
}
```

## Notes

- Local datasets do not get sampled by default (unlike HuggingFace datasets), as they're assumed to be pre-prepared
- The system automatically handles different file formats based on the `file_format` field
- All local datasets are cached in `difficulty_direction/dataset/raw/` and processed versions in `difficulty_direction/dataset/processed/`
