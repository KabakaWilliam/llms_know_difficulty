import os
import json
import random
import pandas as pd
import hashlib
from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_CONFIGS, CODEFORCES_TEMPLATE, PROMPT_TEMPLATES
from utils import fen_to_ascii_grid



# Dataset directory paths
dataset_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
raw_dir = dataset_dir_path / "raw"
processed_dir = dataset_dir_path / "processed"
splits_dir = dataset_dir_path / "splits"

# Create directories if they don't exist
for dir_path in [raw_dir, processed_dir, splits_dir]:
    dir_path.mkdir(exist_ok=True)


def make_cf_prompt(row):
                    return CODEFORCES_TEMPLATE.format(
                        problem_main = row["problem_main"],
                        problem_note = row["problem_note"],
                        input_spec   = row["input_spec"],
                        output_spec  = row["output_spec"],
                        sample_inputs  = row["sample_inputs"],
                        sample_outputs = row["sample_outputs"],
                    )



def generate_uid(problem_text: str, dataset_name: str) -> str:
    """Generate a unique identifier for each problem."""
    combined_text = f"{dataset_name}:{problem_text}"
    return hashlib.md5(combined_text.encode()).hexdigest()


def generate_full_dataset_name(base_name: str, config: Dict[str, Any], raw_cfg=None) -> str:
    """Generate full dataset name including parameters for local datasets.
    
    For local datasets with file patterns, this creates a unique identifier like:
    predicting_learnability_MATH_ModelName-SR_max_3000_k_1_temp_0.0
    """
    if config.get("dataset_type") != "local":
        return base_name
    
    # Extract parameters from file pattern
    file_pattern = config.get("file_pattern", "")
    
    # Build the identifier from the pattern without split
    format_params = {}
    
    if raw_cfg and hasattr(raw_cfg, 'model_alias') and raw_cfg.model_alias:
        format_params["model_alias"] = raw_cfg.model_alias
    
    for param in ["max_tokens", "k", "temperature"]:
        if param in config:
            format_params[param] = config[param]
    
    # Create identifier by formatting pattern without split and removing extension
    if format_params:
        try:
            # Use a placeholder for split to extract the base pattern
            temp_pattern = file_pattern.replace("{split}", "")
            identifier_parts = [base_name]
            
            # Add formatted parts
            if "model_alias" in format_params:
                identifier_parts.append(format_params["model_alias"])
            if "max_tokens" in format_params:
                identifier_parts.append(f"max_{format_params['max_tokens']}")
            if "k" in format_params:
                identifier_parts.append(f"k_{format_params['k']}")
            if "temperature" in format_params:
                identifier_parts.append(f"temp_{format_params['temperature']}")
            
            return "_".join(identifier_parts)
        except Exception as e:
            print(f"Warning: Could not generate full dataset name: {e}")
            return base_name
    
    return base_name


def load_local_dataset(dataset_name: str, split: str, config: Dict[str, Any], raw_cfg=None) -> pd.DataFrame:
    """Load dataset from local files (parquet, json, csv, etc.)."""
    local_path = Path(config["local_path"])
    
    # Add subdirectory if specified
    if "subdirectory" in config:
        local_path = local_path / config["subdirectory"]
    
    file_pattern = config["file_pattern"]
    file_format = config.get("file_format", "parquet")
    
    # Build format parameters dictionary
    format_params = {"split": split}
    
    # Add model_alias if available
    if raw_cfg and hasattr(raw_cfg, 'model_alias') and raw_cfg.model_alias:
        format_params["model_alias"] = raw_cfg.model_alias
    
    # Add success rate parameters if they exist in config
    for param in ["max_tokens", "k", "temperature"]:
        if param in config:
            format_params[param] = config[param]
    
    # Format the filename with all available parameters
    try:
        filename = file_pattern.format(**format_params)
        print(f"✓ Formatted filename: {filename}")
    except KeyError as e:
        raise ValueError(f"Missing required parameter in file_pattern: {e}. Available params: {format_params}")
    
    file_path = local_path / filename
    
    print(f"Loading {dataset_name} from local file: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {file_path}")
    
    # Load based on file format
    if file_format == "parquet":
        df = pd.read_parquet(file_path)
    elif file_format == "json" or file_format == "jsonl":
        df = pd.read_json(file_path, lines=(file_format == "jsonl"))
    elif file_format == "csv":
        df = pd.read_csv(file_path)
    elif file_format == "pkl" or file_format == "pickle":
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Loaded {len(df)} samples from {file_path}")
    return df


def load_raw_dataset(dataset_name: str, split: Optional[str] = None, save_locally: bool = True, raw_cfg=None) -> Union[Dataset, DatasetDict, pd.DataFrame]:
    """Load raw dataset from HuggingFace or local files and save locally."""
    # Use config from raw_cfg if available, otherwise fall back to global DATASET_CONFIGS
    if raw_cfg is not None and hasattr(raw_cfg, 'dataset_config'):
        config = raw_cfg.dataset_config[dataset_name]
    else:
        config = DATASET_CONFIGS[dataset_name]
    dataset_type = config.get("dataset_type", "huggingface")  # Default to huggingface for backward compatibility
    
    if dataset_type == "local":
        # Load from local files
        if split is None:
            raise ValueError(f"Split must be specified for local dataset: {dataset_name}")
        
        dataset = load_local_dataset(dataset_name, split, config, raw_cfg=raw_cfg)
        
        # Generate full dataset name for local datasets
        full_dataset_name = generate_full_dataset_name(dataset_name, config, raw_cfg)
        
        if save_locally:
            # Save raw dataset using full name
            raw_path = raw_dir / full_dataset_name
            raw_path.mkdir(exist_ok=True)
            save_path = raw_path / f"{split}.pkl"
            dataset.to_pickle(save_path)
            print(f"Raw dataset saved to {save_path}")
        
        return dataset
    
    else:  # huggingface
        print(f"Loading {dataset_name} from HuggingFace...")
        if split:
            dataset = load_dataset(
                config["hf_dataset"], 
                config["subset_name"], 
                split=split
            )
        else:
            dataset = load_dataset(
                config["hf_dataset"], 
                config["subset_name"]
            )

        if save_locally:
            # Save raw dataset
            raw_path = raw_dir / dataset_name
            raw_path.mkdir(exist_ok=True)
            
            if isinstance(dataset, Dataset):
                save_path = raw_path / (split if split else "dataset")
                dataset.save_to_disk(save_path)
            elif isinstance(dataset, DatasetDict):
                dataset.save_to_disk(raw_path)
            else:
                # Handle other types by converting to pandas first
                if hasattr(dataset, 'to_pandas'):
                    df = dataset.to_pandas()
                    df.to_pickle(raw_path / f"{split}.pkl" if split else raw_path / "dataset.pkl")
        
            print(f"Raw dataset saved to {raw_path}")
        return dataset


def create_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test split for datasets that don't have natural splits."""
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:split_idx].copy()
    test_df = df_shuffled[split_idx:].copy()
    return train_df, test_df


def apply_prompt_template(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply prompt template to create formatted prompts.
    
    For local datasets with full names (e.g., 'predicting_learnability_Model_max_3000_k_1_temp_0.0'),
    extract the base name to find the template.
    """
    # Extract base name for local datasets (everything before the first underscore after base name)
    base_name = dataset_name
    if dataset_name.startswith("predicting_MATH_learnability"):
        base_name = "predicting_MATH_learnability"
    
    template_config = PROMPT_TEMPLATES.get(base_name)
    
    if not template_config:
        raise ValueError(f"No template found for dataset: {base_name} (original: {dataset_name})")
    
    template = template_config.get("template")
    prompt_column = template_config["prompt_column"]
    
    # If no template is specified (e.g., for predicting_learnability), use prompt as-is
    if template is None:
        print(f"No template for {base_name}. Using prompt column directly")
        df["formatted_prompt"] = df[prompt_column]
    elif base_name == "E2H-Lichess":
        df["formatted_prompt"] = df[prompt_column].apply(
            lambda x: template.format(BOARD=fen_to_ascii_grid(x))
        )
    elif base_name == "E2H-Codeforces":
        df["formatted_prompt"] = df.apply(make_cf_prompt, axis=1)
    else:
        # Create formatted prompts
        df["formatted_prompt"] = df[prompt_column].apply(
            lambda x: template.format(problem=x)
        )

    return df


def get_dataframe_from_dataset(dataset: Union[Dataset, DatasetDict, pd.DataFrame], split_name: Optional[str] = None) -> pd.DataFrame:
    """Convert a dataset to pandas DataFrame safely."""
    if isinstance(dataset, pd.DataFrame):
        return dataset
    elif isinstance(dataset, Dataset):
        return dataset.to_pandas()
    elif isinstance(dataset, DatasetDict):
        if split_name and split_name in dataset:
            return dataset[split_name].to_pandas()
        else:
            # If no split specified, use the first available split
            first_split = list(dataset.keys())[0]
            return dataset[first_split].to_pandas()
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def process_dataset(dataset_name: str, n_train: int = 2000, n_test: int = 500, do_sample: bool = True, save_data: bool = True, raw_cfg=None) -> Dict[str, pd.DataFrame]:
    """
    Process a dataset by loading, formatting, and splitting.
    
    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    # Use config from raw_cfg if available, otherwise fall back to global DATASET_CONFIGS
    if raw_cfg is not None and hasattr(raw_cfg, 'dataset_config'):
        config = raw_cfg.dataset_config[dataset_name]
    else:
        config = DATASET_CONFIGS[dataset_name]
    dataset_type = config.get("dataset_type", "huggingface")
    processed_data = {}
    
    print(f"Processing {dataset_name}...")

    # Load raw data based on whether it has train/test splits
    if config["has_train_split"]:
        # Load train and test/eval splits
        train_dataset = load_raw_dataset(dataset_name, "train", save_locally=save_data, raw_cfg=raw_cfg)
        
        # Determine the test split name (could be 'test' or 'eval')
        test_split_name = "test" if "test" in config["splits"] else "eval"
        test_dataset = load_raw_dataset(dataset_name, test_split_name, save_locally=save_data, raw_cfg=raw_cfg)
        
        # Convert to pandas DataFrames
        train_df = get_dataframe_from_dataset(train_dataset)
        test_df = get_dataframe_from_dataset(test_dataset)
        
    else:
        # Load eval split and create manual train/test split
        eval_dataset = load_raw_dataset(dataset_name, "eval", save_locally=save_data, raw_cfg=raw_cfg)
        
        # Convert to pandas DataFrame
        eval_df = get_dataframe_from_dataset(eval_dataset)
        
        # Create train/test split
        train_ratio = config.get("train_split_ratio", 0.8)
        train_df, test_df = create_train_test_split(eval_df, train_ratio)
  
    # Standardize columns
    if dataset_name == "E2H-Codeforces":
        prompt_column = "problem_main"
    else:
        prompt_column = config["prompt_column"]
    difficulty_column = config["difficulty_column"]
    answer_column = config["answer_column"]
    
    # Process both dataframes
    processed_dfs = []
    for df in [train_df, test_df]:
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Rename columns for standardization
        df_copy["problem"] = df_copy[prompt_column]


        df_copy["difficulty"] = df_copy[difficulty_column]
        df_copy["answer"] = df_copy[answer_column]
        
        # Generate UIDs
        df_copy["uid"] = df_copy["problem"].apply(lambda x: generate_uid(x, dataset_name))
        # Apply prompt template
        df_copy = apply_prompt_template(df_copy, dataset_name)
        # Keep only essential columns
        df_copy = df_copy[["uid", "problem", "formatted_prompt", "difficulty", "answer"]].copy()
        processed_dfs.append(df_copy)
    
    train_df, test_df = processed_dfs
    
    # Apply size limits (either by sampling or truncating)
    if len(train_df) > n_train:
        if do_sample:
            train_df = train_df.sample(n=n_train, random_state=42).reset_index(drop=True)
        else:
            # For local datasets, take first n_train samples
            train_df = train_df.iloc[:n_train].reset_index(drop=True)
    
    if len(test_df) > n_test:
        if do_sample:
            test_df = test_df.sample(n=n_test, random_state=42).reset_index(drop=True)
        else:
            # For local datasets, take first n_test samples
            test_df = test_df.iloc[:n_test].reset_index(drop=True)
    
    processed_data["train"] = train_df
    processed_data["test"] = test_df
    
    if save_data:
        # Save processed data
        processed_path = processed_dir / dataset_name
        processed_path.mkdir(exist_ok=True)
        
        train_df.to_pickle(processed_path / "train.pkl")
        test_df.to_pickle(processed_path / "test.pkl")
        
        # Also save as JSON for easy inspection
        train_df.to_json(processed_path / "train.json", orient="records", indent=2)
        test_df.to_json(processed_path / "test.json", orient="records", indent=2)
        
        print(f"Processed data saved to {processed_path}")
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return processed_data


def load_processed_dataset(dataset_name: str, split: str = "train") -> pd.DataFrame:
    """Load previously processed dataset."""
    processed_path = processed_dir / dataset_name / f"{split}.pkl"
    
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")
    
    return pd.read_pickle(processed_path)


def prepare_all_datasets(dataset_names: List[str], n_train: int = 2000, n_test: int = 500, raw_cfg=None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Process all specified datasets.
    
    Returns a dictionary where keys are full dataset names (including parameters for local datasets).
    For example: 'predicting_learnability_Qwen2.5-Math-1.5B-Instruct_max_3000_k_1_temp_0.0'
    """
    all_datasets = {}
    
    for dataset_name in dataset_names:
        # Use config from raw_cfg if available, otherwise fall back to global DATASET_CONFIGS
        if raw_cfg is not None and hasattr(raw_cfg, 'dataset_config'):
            config = raw_cfg.dataset_config[dataset_name]
        else:
            config = DATASET_CONFIGS[dataset_name]
        dataset_type = config.get("dataset_type", "huggingface")
        
        # Generate full dataset name (includes parameters for local datasets)
        full_dataset_name = generate_full_dataset_name(dataset_name, config, raw_cfg)
        
        # For local datasets, don't sample since they're pre-prepared
        do_sample = (dataset_type != "local")
        
        if not do_sample:
            print(f"{full_dataset_name} (local dataset): train => {n_train}, test => {n_test}")
        
        # Use full dataset name as key
        all_datasets[full_dataset_name] = process_dataset(dataset_name, n_train, n_test, do_sample=do_sample,raw_cfg=raw_cfg)
    
    return all_datasets


# Example usage
if __name__ == "__main__":
    # Process both datasets
    dataset_names = ["E2H-AMC", "E2H-GSM8K"]
    all_data = prepare_all_datasets(dataset_names, n_train=2000, n_test=500)
    
    # Print summary
    for dataset_name, data in all_data.items():
        print(f"\n{dataset_name}:")
        print(f"  Train: {len(data['train'])} samples")
        print(f"  Test: {len(data['test'])} samples")
        print(f"  Sample formatted prompt:")
        print(f"  {data['train']['formatted_prompt'].iloc[0][:200]}...")
        print(f"**Answer**: {data['train']['answer'].iloc[0]}")
