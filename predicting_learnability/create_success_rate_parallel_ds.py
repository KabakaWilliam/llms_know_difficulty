"""
Success Rate Calculator for Math LLMs
Computes success rates for language models on math datasets using parallel rollouts.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List

from vllm import LLM, SamplingParams
from math_verify import parse, verify
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_SUCCESS_RATE_CONFIGS = {
    "MATH": {
        "loading_strategy": "local",
        "path": "predicting_learnability/data/LOCAL/MATH",
        "splits": ["train", "test"],
        "prompt_col": "problem",
        "answer_col": "answer",
        "prompt_template": "{problem} Let's think step by step and output the final answer after \\boxed{{}}."
    },
    "GSM8K": {
        "loading_strategy": "HF",
        "path": "openai/gsm8k",
        "subset": "main",
        "splits": ["train", "test"],
        "prompt_col": "question",
        "answer_col": "answer",
        "prompt_template": "{question} Let's think step by step and output the final answer after \\boxed{{}}."
    },
    "MATH_X_GSM8K": {
        "loading_strategy": "local",
        "path": "predicting_learnability/data/LOCAL/MATH_X_GSM8K",
        "splits": ["train", "test"],
        "prompt_col": "question",
        "answer_col": "answer",
        "prompt_template": "{question} Let's think step by step and output the final answer after \\boxed{{}}."
    },
}

# Experiment Configuration
CONFIG = {
    # Device settings
    "device": 2,
    
    # Model settings
    # "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "model_name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "memory_util": 0.70,
    
    # Dataset settings
    "datasets": ["MATH_X_GSM8K"],  # List of datasets to process, e.g., ["MATH", "GSM8K"]
    "splits": ["train", "test"],
    
    # Generation settings
    "num_rollouts": 1,
    "max_tokens": 3000, #32768 #3000
    "temperature": 0.0,
    
    # Batch settings (tune for GPU/memory)
    "batch_questions": 64,   # number of distinct questions per batch
    "model_chunk": 3000,     # number of prompts sent to model at once
    
    # Output settings
    "output_base_dir": "data/SR_DATASETS",
    "save_plots": True,
    
    # Parallel processing
    "max_workers": 8,
}


# ============================================================================
# SETUP
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = f"{CONFIG['device']}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

assert CONFIG["model_chunk"] % CONFIG["num_rollouts"] == 0, \
    "model_chunk must be divisible by num_rollouts"

if CONFIG["num_rollouts"] == 1:
    assert CONFIG["temperature"] == 0.0

if CONFIG["num_rollouts"] > 1:
    assert CONFIG["temperature"] != 0.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset_from_config(dataset_name: str) -> Dict[str, pd.DataFrame]:
    """Load dataset based on configuration."""
    config = DATA_SUCCESS_RATE_CONFIGS[dataset_name]
    dataframes = {}
    
    if config["loading_strategy"] == "local":
        # Load from local JSONL files
        for split in config["splits"]:
            file_path = f"{config['path']}/{split}.jsonl"
            # Make path absolute relative to script location
            if not os.path.isabs(file_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(script_dir, "..", file_path)
                file_path = os.path.normpath(file_path)
            
            print(f"Loading {split} from: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_json(file_path, lines=True)
            # Optional: limit to first 5 rows for testing
            # dataframes[split] = df if len(df) > 5 else df
            dataframes[split] = df
            
    elif config["loading_strategy"] == "HF":
        # Load from HuggingFace
        if "subset" in config:
            ds = load_dataset(config["path"], config["subset"])
        else:
            ds = load_dataset(config["path"])
        
        # Special handling for GSM8K to extract numeric answers
        if dataset_name == "GSM8K":
            for ds_split in ds.keys():
                extracted_answers = [
                    item["answer"].split("\n####")[-1].strip() 
                    for item in ds[ds_split]
                ]
                ds[ds_split] = ds[ds_split].map(
                    lambda x, idx: {"answer": extracted_answers[idx]}, 
                    with_indices=True
                )
        
        # Convert splits to DataFrames
        for split in config["splits"]:
            if split in ds:
                dataframes[split] = ds[split].to_pandas()
    
    return dataframes


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_prompt_and_gt(row: pd.Series, config: Dict) -> Tuple[str, str]:
    """Extract prompt and ground truth from a dataset row."""
    prompt_value = row[config["prompt_col"]]
    gt = row[config["answer_col"]]
    
    # Apply prompt template if available
    if "prompt_template" in config:
        template = config["prompt_template"]
        prompt = template.format(**{config["prompt_col"]: prompt_value})
    else:
        prompt = prompt_value
    
    return prompt, gt


def eval_response(item: Tuple[str, str]) -> int:
    """Evaluate a single response against ground truth."""
    gt, resp = item
    try:
        ans = parse(resp)
    except Exception:
        ans = resp
    try:
        return 1 if verify(parse(f"${gt}$"), ans) else 0
    except Exception:
        return 0


def apply_chat_template(prompt: str, llm: LLM) -> str:
    """Apply chat template to prompt if available."""
    try:
        return llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
    except:
        print("Continuing without chat template")
        return prompt


# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("Loading model...")
llm = LLM(model=CONFIG["model_name"], gpu_memory_utilization=CONFIG["memory_util"])

print(f"{'='*70}")
print(f"Success Rate Experiment")
print(f"{'='*70}")
print(f"Model: {CONFIG['model_name']}")
print(f"Datasets: {', '.join(CONFIG['datasets'])}")
print(f"Rollouts per question: {CONFIG['num_rollouts']}")
print(f"Temperature: {CONFIG['temperature']}")
print(f"Max tokens: {CONFIG['max_tokens']}")
print(f"{'='*70}\n")

# Setup sampling parameters
SAMPLING_PARAMS = SamplingParams(
    temperature=CONFIG["temperature"],
    max_tokens=CONFIG["max_tokens"]
)

# Process each dataset
for DATASET_NAME in CONFIG["datasets"]:
    print(f"\n{'#'*70}")
    print(f"# Processing Dataset: {DATASET_NAME}")
    print(f"{'#'*70}\n")
    
    # Load dataset
    print(f"Loading {DATASET_NAME} dataset...")
    DATAFRAMES = load_dataset_from_config(DATASET_NAME)
    DATASET_CONFIG = DATA_SUCCESS_RATE_CONFIGS[DATASET_NAME]
    
    # Process each split
    for SPLIT_NAME in CONFIG["splits"]:
        if SPLIT_NAME not in DATAFRAMES:
            print(f"Warning: Split '{SPLIT_NAME}' not found in dataset. Skipping.")
            continue
        
        df = DATAFRAMES[SPLIT_NAME]
        print(f"\n{'='*70}")
        print(f"Processing {SPLIT_NAME} split ({len(df)} examples)")
        print(f"{'='*70}")
        
        n_q = len(df)
        
        # Pre-extract and format prompts
        print("Preparing prompts...")
        prompts = []
        gts = []
        for _, row in df.iterrows():
            prompt, gt = extract_prompt_and_gt(row, DATASET_CONFIG)
            prompt = apply_chat_template(prompt, llm)
            prompts.append(prompt)
            gts.append(gt)
        
        # Initialize counters
        correct_counts = [0] * n_q
        total_counts = [0] * n_q
        
        # Process in batches
        for q_start in tqdm(
            range(0, n_q, CONFIG["batch_questions"]), 
            desc=f"Generating responses for {SPLIT_NAME}"
        ):
            q_end = min(q_start + CONFIG["batch_questions"], n_q)
            batch_indices = list(range(q_start, q_end))
            
            # Build repeated prompts for rollouts
            batch_prompts = []
            mapping = []
            for qi in batch_indices:
                for _ in range(CONFIG["num_rollouts"]):
                    batch_prompts.append(prompts[qi])
                    mapping.append(qi)
            
            # Generate in chunks to manage memory
            for chunk_start in range(0, len(batch_prompts), CONFIG["model_chunk"]):
                chunk_end = min(chunk_start + CONFIG["model_chunk"], len(batch_prompts))
                chunk_prompts = batch_prompts[chunk_start:chunk_end]
                chunk_mapping = mapping[chunk_start:chunk_end]
                
                # Generate responses
                outputs = llm.generate(chunk_prompts, SAMPLING_PARAMS)
                
                # Extract text responses
                responses = []
                for out in outputs:
                    try:
                        text = out.outputs[0].text
                    except Exception:
                        text = ""
                    responses.append(text)
                
                # Evaluate responses in parallel
                items = [(gts[qidx], resp) for qidx, resp in zip(chunk_mapping, responses)]
                with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
                    for qidx, result in zip(chunk_mapping, executor.map(eval_response, items)):
                        correct_counts[qidx] += result
                        total_counts[qidx] += 1
        
        # Build results DataFrame
        results = {
            i: {
                "correct": correct_counts[i],
                "total": total_counts[i],
                "success_rate": correct_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
            }
            for i in range(n_q)
        }
        
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df["question"] = prompts
        results_df["answer"] = gts
        
        # Save results
        model_alias = CONFIG["model_name"].split("/")[-1]
        output_dir = f"{CONFIG['output_base_dir']}/{DATASET_NAME}"
        os.makedirs(output_dir, exist_ok=True)
        
        filename_base = (
            f"{model_alias}-SR_{SPLIT_NAME}_"
            f"max_{CONFIG['max_tokens']}_"
            f"k_{CONFIG['num_rollouts']}_"
            f"temp_{CONFIG['temperature']}"
        )
        
        parquet_path = f"{output_dir}/{filename_base}.parquet"
        results_df.to_parquet(parquet_path)
        print(f"\nSaved results to: {parquet_path}")
        
        # Save plot
        if CONFIG["save_plots"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            results_df["success_rate"].hist(ax=ax, bins=20)
            ax.set_xlabel("Success rate")
            ax.set_ylabel("Count")
            ax.set_title(f"Success Rate Distribution: {SPLIT_NAME} [{model_alias}]")
            
            plot_path = f"{output_dir}/success_rate_hist_{filename_base}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot to: {plot_path}")
        
        # Print summary statistics
        print(f"\nResults Summary for {SPLIT_NAME}:")
        print(f"  Mean Success Rate: {results_df['success_rate'].mean():.4f}")
        print(f"  Median Success Rate: {results_df['success_rate'].median():.4f}")
        print(f"  Std Dev: {results_df['success_rate'].std():.4f}")
        print(f"  Min: {results_df['success_rate'].min():.4f}")
        print(f"  Max: {results_df['success_rate'].max():.4f}")

        # Send notification after each split
        try:
            notification_msg = (
                f"✅ {DATASET_NAME} {SPLIT_NAME} complete!\n"
                f"📊 Model: {model_alias}\n"
                f"📈 Mean SR: {results_df['success_rate'].mean():.2%} "
                f"(n={len(results_df)})\n"
                f"📂 {parquet_path}"
            )
            requests.post(
                "https://ntfy.sh/training_runs_lugoloobi",
                data=notification_msg.encode(encoding='utf-8'),
                headers={"Title": f"SR Analysis - {DATASET_NAME} {SPLIT_NAME}"}
            )
        except:
            pass  # Don't fail if notification doesn't work

print(f"\n{'='*70}")
print("All experiments complete!")
print(f"{'='*70}")






