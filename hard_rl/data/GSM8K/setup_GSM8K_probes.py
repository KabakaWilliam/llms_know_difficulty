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
Preprocess the MATH dataset to parquet format
"""

import argparse
import os
import re
import json
import torch
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/VData/linna4335/difficulty_check/hard_rl/data/MATH")
    parser.add_argument("--probe_name", default="E2H-AMC")
    parser.add_argument("--hdfs_dir", default=None)


    args = parser.parse_args()
    CHOSEN_PROBE = args.probe_name

    data_source = "lighteval/MATH"

    # Load JSONL files directly
    train_data = load_jsonl(os.path.join(args.local_dir, f"train_probe_{CHOSEN_PROBE}.jsonl"))
    # val_data = load_jsonl(os.path.join(args.local_dir, "validation.jsonl"))

    instruction_following = 'Let\'s think step by step and output the final answer after "\\boxed{}".'
    # instruction_following = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> with \\boxed{{}}, respectively, i.e., <think> reasoning process here </think> <answer> \\boxed{{answer here}} </answer>.\nUser: {question}\nAssistant: <think>"""
    

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem_raw = example["problem"]

            question = problem_raw + " " + instruction_following
            
            # question = instruction_following.format(question=problem_raw)

            solution_raw = example["solution"]
            # Use the answer field directly
            solution = example["answer"]
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "solution": solution_raw,
                    "problem": problem_raw,
                    "subject": example.get("subject", ""),
                    "level": example.get("probe_levels", ""),
                    "og_level": example.get("level", ""),
                    "unique_id": example.get("unique_id", ""),
                    "level_sigmoid": torch.sigmoid(torch.Tensor([example.get("probe_levels", "")])).item()
                },
            }
            return data

        return process_fn

    # Process the data
    processed_train = []
    for idx, example in enumerate(train_data):
        processed_train.append(make_map_fn("train")(example, idx))
    
    # processed_val = []
    # for idx, example in enumerate(val_data):
    #     processed_val.append(make_map_fn("validation")(example, idx))

    # Convert to format compatible with datasets
    train_df = pd.DataFrame(processed_train)
    # val_df = pd.DataFrame(processed_val)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save as parquet files
    train_df.to_parquet(os.path.join(local_dir, f"train_{CHOSEN_PROBE}.parquet"), index=False)
    # val_df.to_parquet(os.path.join(local_dir, "validation.parquet"), index=False)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
