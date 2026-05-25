from huggingface_hub import create_repo
from datasets import Dataset, DatasetDict
import pandas as pd
import os

DATASET_README_TEMPLATE = """# Generations Dataset: {dataset_name}

LLM-generated solutions across train/validation/test splits for multiple models.

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `problem` | str | Problem statement |
| `generated_solutions` | list | Generated solutions with scores |
| `success_rate` | float | Fraction of correct generations |
| `majority_vote_is_correct` | int (0/1) | Whether majority vote is correct |
| `k` | int | Number of samples generated |
| `temperature` | float | Sampling temperature |
| `max_len` | int | Maximum generation length |
| `model_name` | str | Model used for generation |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("CoffeeGitta/difficulty-{dataset_name}-generations", name="<org--model>")
train = dataset["train"]
```

## Citation

```bibtex
@article{{lugoloobi_llms_2026,
    title = {{LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations}},
    url = {{http://arxiv.org/abs/2602.09924}},
    author = {{Lugoloobi, William and Foster, Thomas and Bankes, William and Russell, Chris}},
    year = {{2026}},
}}
```
"""

DATASET_CONFIG = {
    "MATH":           "DigitalLearningGmbH_MATH-lighteval",
    "gsm8k":          "openai_gsm8k",
    "E2H-AMC":        "E2H-AMC",
    "aime_2025":      "opencompass_AIME2025",
    "aime_1983-2024": "gneubig_aime-1983-2024",
}


def enforce_generation_columns(df, k, temperature, max_len):
    """
    Force generation args to exist and be constant.
    """
    df["k"] = k
    df["temperature"] = temperature
    df["max_len"] = max_len
    return df


def enforce_correctness_column(df):
    """
    Ensure majority_vote_is_correct is int8 (0 or 1).
    """
    if "majority_vote_is_correct" in df.columns:
        df["majority_vote_is_correct"] = (
            df["majority_vote_is_correct"]
            .fillna(0)
            .astype("int8")
        )

        # Optional strict check
        unique_vals = set(df["majority_vote_is_correct"].unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"majority_vote_is_correct contains invalid values: {unique_vals}"
            )

    return df


def normalize_schemas(split_dfs):
    """
    Ensure all splits have identical columns.
    """
    all_columns = set()
    for df in split_dfs.values():
        all_columns.update(df.columns)

    for split, df in split_dfs.items():
        missing = all_columns - set(df.columns)
        for col in missing:
            df[col] = pd.NA
        split_dfs[split] = df[sorted(all_columns)]

    return split_dfs


def find_data_file(model_path, split, k=5):
    """
    Auto-detect available parquet file for a given split with specific k.
    Looks for files matching: {split}_maxlen_*_k_{k}_temp_*.parquet
    """
    if not os.path.exists(model_path):
        return None

    import glob
    pattern = os.path.join(model_path, f"{split}_maxlen_*_k_{k}_temp_*.parquet")
    files = glob.glob(pattern)

    if files:
        # Return the first match (or most recent if multiple)
        return sorted(files)[-1]
    return None


def extract_params_from_path(file_path):
    """
    Extract k, temperature, max_len from filename like:
    train_maxlen_3000_k_5_temp_0.7.parquet
    """
    import re
    filename = os.path.basename(file_path)

    # Remove .parquet extension first
    filename = filename.replace('.parquet', '')

    # Extract parameters from filename
    maxlen_match = re.search(r'maxlen_(\d+)', filename)
    k_match = re.search(r'k_(\d+)', filename)
    temp_match = re.search(r'temp_([\d.]+)', filename)

    max_len = int(maxlen_match.group(1)) if maxlen_match else 3000
    k = int(k_match.group(1)) if k_match else 5
    temperature = float(temp_match.group(1)) if temp_match else 0.7

    return k, temperature, max_len


def push_math_dataset(
    base_dir,
    repo_id,
    dataset_name,
    models,
    friendly_name,
    push_readme=True,
):
    """
    models = list of (org, model, k, temperature, max_len) or (org, model)
    If hyperparams not provided, will auto-detect from available files.
    friendly_name is used for the README (e.g. "gsm8k", "MATH").
    """
    from huggingface_hub import HfApi

    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    if push_readme:
        api = HfApi()
        readme_content = DATASET_README_TEMPLATE.format(dataset_name=friendly_name)
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✅ Uploaded README to {repo_id}")

    for model_info in models:
        # Support both (org, model) and (org, model, k, temp, max_len) tuples
        if len(model_info) == 2:
            org, model = model_info
            k, temperature, max_len = 5, None, None  # Default k=5, auto-detect temp and max_len
        elif len(model_info) == 5:
            org, model, k, temperature, max_len = model_info
        else:
            raise ValueError(f"Model info must be (org, model) or (org, model, k, temp, max_len), got {model_info}")

        print(f"\nProcessing {org}/{model}")

        config_name = f"{org}--{model}"
        split_dfs = {}

        for split in ["train", "val", "test"]:
            # Build model path
            model_path = os.path.join(base_dir, org, model, dataset_name)

            # If hyperparams not specified, auto-detect from available files (with k=5 enforced)
            if temperature is None or max_len is None:
                file_path = find_data_file(model_path, split, k=5)
                if not file_path:
                    print(f"  {split} not found for {config_name} (k=5)")
                    continue
                # Extract actual parameters from filename
                actual_k, actual_temp, actual_maxlen = extract_params_from_path(file_path)
            else:
                file_path = os.path.join(
                    model_path,
                    f"{split}_maxlen_{max_len}_k_{k}_temp_{temperature}.parquet"
                )
                actual_k, actual_temp, actual_maxlen = k, temperature, max_len

            if not os.path.exists(file_path):
                print(f"  {split} not found for {config_name}")
                continue

            df = pd.read_parquet(file_path)

            # Enforce numeric consistency
            if "rating" in df.columns:
                df["rating"] = df["rating"].astype("float64")
            else:
                df["rating"] = pd.Series([None] * len(df), dtype="float64")

            # Rename "question" to "problem" if needed
            if "question" in df.columns and "problem" not in df.columns:
                df = df.rename(columns={"question": "problem"})

            if "idx" in df.columns:
                df = df.drop(columns=["idx"])

            # Enforce generation config columns with actual parameters
            df = enforce_generation_columns(df, actual_k, actual_temp, actual_maxlen)

            # Enforce correctness dtype
            df = enforce_correctness_column(df)

            split_name = "validation" if split == "val" else split

            print(f"  {split_name}: {len(df)} rows")

            split_dfs[split_name] = df

        if not split_dfs:
            print(f"  No splits found for {config_name}, skipping")
            continue

        # Normalize schemas across splits
        split_dfs = normalize_schemas(split_dfs)

        # Convert to HF DatasetDict
        dataset_dict = DatasetDict({
            split: Dataset.from_pandas(df, preserve_index=False)
            for split, df in split_dfs.items()
        })

        # Push to hub
        dataset_dict.push_to_hub(
            repo_id,
            config_name=config_name,
            private=False,
        )

        print(f"✅ Uploaded {config_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace API token")
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login
        login(token=args.token)
    else:
        print("Warning: no --token provided and HF_TOKEN not set. Upload may fail.")

    BASE_DIR = "data"
    HF_ORG = "CoffeeGitta"

    MODELS = [
        ("openai",      "gpt-oss-20b_low"),
        ("openai",      "gpt-oss-20b_high"),
        ("openai",      "gpt-oss-20b_medium"),
        ("Qwen",        "Qwen2.5-Math-1.5B-Instruct"),
        ("Qwen",        "Qwen2.5-Math-7B-Instruct"),
        ("Qwen",        "Qwen3-8B"),
        ("deepseek-ai", "DeepSeek-R1-Distill-Qwen-7B"),
    ]

    for friendly_name, dataset_dir in DATASET_CONFIG.items():
        repo_id = f"{HF_ORG}/difficulty-{friendly_name}-generations"
        print(f"\n{'='*60}")
        print(f"Dataset: {friendly_name}  →  {repo_id}")
        print(f"{'='*60}")
        push_math_dataset(
            base_dir=BASE_DIR,
            repo_id=repo_id,
            dataset_name=dataset_dir,
            models=MODELS,
            friendly_name=friendly_name,
        )
