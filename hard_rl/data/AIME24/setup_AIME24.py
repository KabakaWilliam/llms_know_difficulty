# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the AIME 2024 dataset to parquet format
"""

import argparse
import os
import re
import json
import pandas as pd
from datasets import load_dataset

try:
    from verl.utils.hdfs_io import copy, makedirs
    HDFS_AVAILABLE = True
except ImportError:
    print("Warning: verl.utils.hdfs_io not available. HDFS functionality will be disabled.")
    HDFS_AVAILABLE = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/VData/linna4335/difficulty_check/hard_rl/data/AIME24")
    parser.add_argument("--hdfs_dir", default=None)


    args = parser.parse_args()

    data_source = "lighteval/MATH"

    # Load AIME 2024 dataset from Hugging Face
    print("Loading AIME 2024 dataset from Hugging Face...")
    dataset = load_dataset("math-ai/aime24")
    
    # AIME24 only has a test split, so we'll use it as both train and validation
    # In practice, you might want to split this or use it only for evaluation
    test_data = dataset["test"]
    
    # Convert to list of dictionaries
    data_list = [dict(example) for example in test_data]

    
    instruction_following = 'Let\'s think step by step and output the final answer after "\\boxed{}".'
    # instruction_following = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> with \\boxed{{}}, respectively, i.e., <think> reasoning process here </think> <answer> \\boxed{{answer here}} </answer>.\nUser: {question}\nAssistant: <think>"""
    

    # Process AIME 2024 data
    def process_aime_example(example, idx, split):
        """Process a single AIME example"""
        problem_raw = example["problem"]
        question = problem_raw + " " + instruction_following
        
        # Extract answer from solution (should be in \boxed{} format)
        solution_raw = example["solution"]
        # AIME solutions are typically just the boxed answer
        solution = solution_raw.strip()
        
        # Extract numeric answer if it's in \boxed{} format
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            answer = boxed_match.group(1)
        else:
            answer = solution
        
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "solution": solution_raw,
                "problem": problem_raw,
                "aime_id": example.get("id", ""),
                "url": example.get("url", ""),
                "level": "AIME",  # All AIME problems are high level
                "subject": "Number Theory, Algebra, Geometry, Combinatorics",  # AIME covers multiple areas
                "unique_id": f"aime24_{example.get('id', idx)}",
            },
        }
        return data

    # Process the data - use all AIME24 problems for validation only
    processed_val = []
    
    for idx, example in enumerate(data_list):
        processed_val.append(process_aime_example(example, idx, "validation"))

    # Convert to format compatible with datasets
    val_df = pd.DataFrame(processed_val)

    print(f"Processed {len(processed_val)} validation examples")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Save as parquet file (validation only)
    val_parquet_path = os.path.join(local_dir, "validation.parquet")
    
    val_df.to_parquet(val_parquet_path, index=False)
    
    print(f"Saved validation data to: {val_parquet_path}")

    if hdfs_dir is not None:
        if HDFS_AVAILABLE:
            print(f"Copying to HDFS: {hdfs_dir}")
            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
            print("HDFS copy completed")
        else:
            print("Warning: HDFS functionality not available. Skipping HDFS copy.")
