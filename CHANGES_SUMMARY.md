# Local Dataset Support - Summary of Changes

## Overview
The `difficulty_direction` project has been modified to support local (non-HuggingFace) datasets alongside the existing HuggingFace dataset support. This allows you to use your own dataset files stored locally without needing to upload them to HuggingFace.

## What Changed

### 1. **config.py**
- Added `dataset_type` field to all dataset configurations
  - `"local"` for file-based datasets
  - `"huggingface"` for HuggingFace Hub datasets
- Added new fields for local datasets:
  - `local_path`: Directory containing dataset files
  - `file_pattern`: Filename pattern with `{split}` placeholder
  - `file_format`: File type (parquet, json, jsonl, csv, pkl)

### 2. **dataset/load_dataset.py**
- Added `load_local_dataset()` function to load from local files
- Updated `load_raw_dataset()` to handle both local and HuggingFace datasets
- Updated `get_dataframe_from_dataset()` to handle pandas DataFrames directly
- Updated `process_dataset()` to use unified loading approach
- Updated `prepare_all_datasets()` to auto-detect local datasets and skip sampling
- Updated `apply_prompt_template()` to handle datasets without templates

### Supported File Formats
- Parquet (`.parquet`)
- JSON (`.json`)
- JSON Lines (`.jsonl`)
- CSV (`.csv`)
- Pickle (`.pkl`, `.pickle`)

## New Files Created

1. **`LOCAL_DATASETS_README.md`** - Comprehensive guide on using local datasets
2. **`EXAMPLE_ADD_LOCAL_DATASET.py`** - Code examples for adding new datasets
3. **`generate_dataset_configs.py`** - Helper script to auto-generate configurations

## Example: predicting_learnability Dataset

The `predicting_learnability` dataset has been converted to use the new local dataset system:

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

## How to Add a New Local Dataset

### Quick Steps:

1. **Prepare your data files** with columns for prompts, answers, and difficulty
2. **Add configuration to `config.py`**:
   ```python
   "my_dataset": {
       "dataset_type": "local",
       "local_path": "/path/to/data",
       "file_pattern": "data_{split}.parquet",
       "file_format": "parquet",
       "splits": ["train", "test"],
       "prompt_column": "question",
       "answer_column": "answer",
       "has_train_split": True,
       "difficulty_column": "difficulty"
   }
   ```
3. **(Optional) Add prompt template to `config.py`**:
   ```python
   "my_dataset": {
       "template": None,  # Use prompt as-is, or provide template string
       "prompt_column": "question"
   }
   ```
4. **Run your pipeline**:
   ```bash
   python3 -m difficulty_direction.run \
       --model_path your/model \
       --subset_datasets my_dataset \
       --evaluation_datasets my_dataset
   ```

## Backwards Compatibility

All existing HuggingFace datasets (E2H-AMC, E2H-GSM8K, E2H-Codeforces, E2H-Lichess) continue to work exactly as before. The system automatically detects the dataset type and uses the appropriate loading method.

## Benefits

1. **No HuggingFace upload required** - Use datasets directly from disk
2. **Multiple file formats supported** - Not just HuggingFace datasets format
3. **Easy to switch between datasets** - Just change the `file_pattern`
4. **Automatic config generation** - Use `generate_dataset_configs.py` helper
5. **Pre-prepared data** - Local datasets skip sampling, using data as-is

## Testing

Tested successfully with:
- Loading local parquet files ✓
- Processing datasets ✓
- Generating formatted prompts ✓
- Full pipeline integration ✓

## Next Steps

To use different success rate datasets from `predicting_learnability/data`:

1. Run the config generator:
   ```bash
   python3 difficulty_direction/generate_dataset_configs.py
   ```

2. Copy the generated configs to `config.py`

3. Use any dataset by name:
   ```bash
   --subset_datasets FineMath-Llama-3B
   ```

## Files Modified

- `difficulty_direction/config.py`
- `difficulty_direction/dataset/load_dataset.py`

## Files Created

- `difficulty_direction/LOCAL_DATASETS_README.md`
- `difficulty_direction/EXAMPLE_ADD_LOCAL_DATASET.py`
- `difficulty_direction/generate_dataset_configs.py`
- `CHANGES_SUMMARY.md` (this file)
