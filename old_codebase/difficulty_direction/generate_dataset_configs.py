"""
Helper script to generate dataset configurations for all success rate datasets
in the predicting_learnability directory.

This automatically creates config entries for all *-SR_{train,test}.parquet files.
"""

import os
from pathlib import Path

# Path to predicting_learnability data
DATA_DIR = Path("/VData/linna4335/llms_know_difficult/predicting_learnability/data")

def generate_configs():
    """Generate dataset configurations for all SR parquet files."""
    
    # Find all unique dataset names (without _train or _test suffix)
    datasets = set()
    for file in os.listdir(DATA_DIR):
        if file.endswith("_train.parquet"):
            # Extract dataset name (remove _train.parquet)
            dataset_name = file.replace("_train.parquet", "")
            datasets.add(dataset_name)
    
    print("# Add these to DATASET_CONFIGS in config.py:\n")
    
    for dataset in sorted(datasets):
        config_name = dataset.replace("MATH_", "").replace("-SR", "")
        
        config = f'''    "{config_name}": {{
        "dataset_type": "local",
        "local_path": "{DATA_DIR}",
        "file_pattern": "{dataset}_{{split}}.parquet",
        "file_format": "parquet",
        "splits": ["train", "test"],
        "prompt_column": "prompt",
        "answer_column": "ground_truth",
        "has_train_split": True,
        "difficulty_column": "success_rate"
    }},'''
        
        print(config)
        print()

def generate_template_configs():
    """Generate prompt template configurations."""
    
    datasets = set()
    for file in os.listdir(DATA_DIR):
        if file.endswith("_train.parquet"):
            dataset_name = file.replace("_train.parquet", "")
            datasets.add(dataset_name)
    
    print("\n# Add these to PROMPT_TEMPLATES in config.py:\n")
    
    for dataset in sorted(datasets):
        config_name = dataset.replace("MATH_", "").replace("-SR", "")
        
        template = f'''    "{config_name}": {{
        "template": None,
        "prompt_column": "prompt"
    }},'''
        
        print(template)

if __name__ == "__main__":
    print("=" * 70)
    print("Dataset Configuration Generator")
    print("=" * 70)
    print(f"\nScanning: {DATA_DIR}\n")
    
    generate_configs()
    generate_template_configs()
    
    print("\n" + "=" * 70)
    print("\nUsage example:")
    print("=" * 70)
    print("""
CUDA_VISIBLE_DEVICES=0 python3 -m difficulty_direction.run \\
    --model_path Qwen/Qwen2.5-Math-1.5B \\
    --use_k_fold \\
    --batch_size 16 \\
    --n_train 1000 \\
    --n_test 500 \\
    --subset_datasets DeepSeek-R1-Distill-Qwen-1.5B \\
    --evaluation_datasets DeepSeek-R1-Distill-Qwen-1.5B \\
    --generation_batch_size 8 \\
    --max_new_tokens 2000
""")
