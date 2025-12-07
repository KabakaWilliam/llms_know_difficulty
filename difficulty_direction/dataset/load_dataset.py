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


def load_raw_dataset(dataset_name: str, split: Optional[str] = None, save_locally: bool = True) -> Union[Dataset, DatasetDict]:
    """Load raw dataset from HuggingFace and save locally."""
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"Loading {dataset_name} from HuggingFace...")
    if dataset_name == "predicting_learnability":
        FILE_PATH = f"{config["hf_dataset"]}/{config["subset_name"]}_{split}.parquet"
        print(f"Parquet file path: {FILE_PATH}")
        dataset = pd.read_parquet(FILE_PATH)
    else:
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
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]
    return train_df, test_df


def apply_prompt_template(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply prompt template to create formatted prompts."""
    template_config = PROMPT_TEMPLATES.get(dataset_name)
    if dataset_name != "predicting_learnability":
        if not template_config:
            raise ValueError(f"No template found for dataset: {dataset_name}")
        
        template = template_config["template"]
        prompt_column = template_config["prompt_column"]
        if dataset_name == "E2H-Lichess":
            df["formatted_prompt"] = df[prompt_column].apply(
                lambda x: template.format(BOARD=fen_to_ascii_grid(x))
            )
        elif dataset_name == "E2H-Codeforces":
            df["formatted_prompt"] = df.apply(make_cf_prompt, axis=1)
        else:
            # Create formatted prompts
            df["formatted_prompt"] = df[prompt_column].apply(
                lambda x: template.format(problem=x)
            )
    else:
        prompt_column = template_config["prompt_column"]
        print("No template. Using default prompt")
        df["formatted_prompt"] = df[prompt_column]

    return df


def get_dataframe_from_dataset(dataset: Union[Dataset, DatasetDict], split_name: Optional[str] = None) -> pd.DataFrame:
    """Convert a dataset to pandas DataFrame safely."""
    if isinstance(dataset, Dataset):
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


def process_dataset(dataset_name: str, n_train: int = 2000, n_test: int = 500, do_sample: bool = True, save_data: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Process a dataset by loading, formatting, and splitting.
    
    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    config = DATASET_CONFIGS[dataset_name]
    processed_data = {}
    
    print(f"Processing {dataset_name}...")

    if dataset_name == "predicting_learnability":
        train_df = load_raw_dataset(dataset_name, "train", save_locally=save_data)
        test_df = load_raw_dataset(dataset_name, "test", save_locally=save_data)
    else:
        # Load raw data
        if config["has_train_split"]:
            # Load train and eval splits separately
            train_dataset = load_raw_dataset(dataset_name, "train", save_locally=save_data)
            eval_dataset = load_raw_dataset(dataset_name, "eval", save_locally=save_data)
            
            # Convert to pandas DataFrames
            train_df = get_dataframe_from_dataset(train_dataset, "train")
            test_df = get_dataframe_from_dataset(eval_dataset, "eval")
            
        else:
            # Load eval split and create manual train/test split
            eval_dataset = load_raw_dataset(dataset_name, "eval", save_locally=save_data)
            
            # Convert to pandas DataFrame
            eval_df = get_dataframe_from_dataset(eval_dataset, "eval")
            
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
    
    if do_sample:
        # Sample data if needed
        if len(train_df) > n_train:
            train_df = train_df.sample(n=n_train, random_state=42).reset_index(drop=True)
        
        if len(test_df) > n_test:
            test_df = test_df.sample(n=n_test, random_state=42).reset_index(drop=True)
    
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


def prepare_all_datasets(dataset_names: List[str], n_train: int = 2000, n_test: int = 500) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Process all specified datasets."""
    all_datasets = {}
    
    for dataset_name in dataset_names:
        config = DATASET_CONFIGS[dataset_name]
        if dataset_name == "predicting_learnability":
            print(f"Predicting Learnability data splits: train => {n_train}, test => {n_test} ")
            all_datasets[dataset_name] = process_dataset(dataset_name, n_train, n_test, do_sample=False)
        else:
                all_datasets[dataset_name] = process_dataset(dataset_name, n_train, n_test)
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
