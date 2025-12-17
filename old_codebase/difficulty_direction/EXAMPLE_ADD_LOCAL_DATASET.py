"""
Example: Adding a New Local Dataset

This example shows how to add a new local dataset to the difficulty_direction project.
"""

# Step 1: Prepare your dataset files
# Your dataset should be in one of these formats: parquet, json, jsonl, csv, or pickle
# Files should follow a naming pattern like: mydata_train.parquet, mydata_test.parquet

# Step 2: Add configuration to config.py
# In the DATASET_CONFIGS dictionary, add:

"""
"my_new_dataset": {
    "dataset_type": "local",
    "local_path": "/path/to/your/dataset/directory",
    "file_pattern": "mydata_{split}.parquet",  # {split} gets replaced with train/test
    "file_format": "parquet",
    "splits": ["train", "test"],
    "prompt_column": "question",  # Column name in your data
    "answer_column": "solution",  # Column name in your data
    "has_train_split": True,
    "difficulty_column": "score"  # Column name in your data
}
"""

# Step 3: (Optional) Add a prompt template if you need to format prompts
# In the PROMPT_TEMPLATES dictionary, add:

"""
"my_new_dataset": {
    "template": "Solve this problem step by step:\\n{problem}\\n\\nProvide your answer in the format: Answer: <your answer>",
    "prompt_column": "question"
}
"""

# If you don't need formatting, you can set template to None:
"""
"my_new_dataset": {
    "template": None,
    "prompt_column": "question"
}
"""

# Step 4: Run with your new dataset

"""
CUDA_VISIBLE_DEVICES=0 python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --use_k_fold \
    --batch_size 16 \
    --n_train 1000 \
    --n_test 500 \
    --subset_datasets my_new_dataset \
    --evaluation_datasets my_new_dataset \
    --generation_batch_size 8 \
    --max_new_tokens 2000
"""

# ============================================================================
# REAL EXAMPLE: Using different success rate data from predicting_learnability
# ============================================================================

# Let's say you have multiple parquet files for different models:
# - MATH_Model-A_train.parquet
# - MATH_Model-A_test.parquet
# - MATH_Model-B_train.parquet
# - MATH_Model-B_test.parquet

# Add these to config.py:

"""
"learnability_modelA": {
    "dataset_type": "local",
    "local_path": "/VData/linna4335/llms_know_difficult/predicting_learnability/data",
    "file_pattern": "MATH_Model-A_{split}.parquet",
    "file_format": "parquet",
    "splits": ["train", "test"],
    "prompt_column": "prompt",
    "answer_column": "ground_truth",
    "has_train_split": True,
    "difficulty_column": "success_rate"
},
"learnability_modelB": {
    "dataset_type": "local",
    "local_path": "/VData/linna4335/llms_know_difficult/predicting_learnability/data",
    "file_pattern": "MATH_Model-B_{split}.parquet",
    "file_format": "parquet",
    "splits": ["train", "test"],
    "prompt_column": "prompt",
    "answer_column": "ground_truth",
    "has_train_split": True,
    "difficulty_column": "success_rate"
}
"""

# Then use them:
"""
CUDA_VISIBLE_DEVICES=0 python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --use_k_fold \
    --subset_datasets learnability_modelA \
    --evaluation_datasets learnability_modelA
"""

# ============================================================================
# Example with CSV format
# ============================================================================

"""
"my_csv_dataset": {
    "dataset_type": "local",
    "local_path": "/home/user/datasets",
    "file_pattern": "problems_{split}.csv",
    "file_format": "csv",
    "splits": ["train", "test"],
    "prompt_column": "problem_text",
    "answer_column": "correct_answer",
    "has_train_split": True,
    "difficulty_column": "difficulty_rating"
}
"""

# ============================================================================
# Example with JSON format
# ============================================================================

"""
"my_json_dataset": {
    "dataset_type": "local",
    "local_path": "/home/user/datasets",
    "file_pattern": "data_{split}.json",
    "file_format": "json",
    "splits": ["train", "test"],
    "prompt_column": "query",
    "answer_column": "response",
    "has_train_split": True,
    "difficulty_column": "complexity"
}
"""

# ============================================================================
# Example with only one file (automatic train/test split)
# ============================================================================

"""
"my_single_file_dataset": {
    "dataset_type": "local",
    "local_path": "/home/user/datasets",
    "file_pattern": "all_data.parquet",  # No {split} placeholder
    "file_format": "parquet",
    "splits": ["eval"],  # Use "eval" as the single split name
    "prompt_column": "question",
    "answer_column": "answer",
    "has_train_split": False,  # System will auto-split
    "train_split_ratio": 0.8,  # 80% for training, 20% for testing
    "difficulty_column": "hardness"
}
"""
