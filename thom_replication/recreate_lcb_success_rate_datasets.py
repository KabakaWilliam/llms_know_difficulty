import pandas as pd
import os
from utils.LiveCodeBench.utils import format_prompts_for_eval
from transformers import AutoTokenizer

LCB_DATASET_CONFIGS = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "url": "https://github.com/LiveCodeBench/submissions/raw/refs/heads/main/Qwen2.5-Coder-Ins-7B/Scenario.codegeneration_10_0.2_eval_all.json",
        "cutoff_date": "2024-09-01",
        "k": 1,
        "max_len": 512,
        "temp": 0.7
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "url": "https://github.com/LiveCodeBench/submissions/raw/refs/heads/main/Qwen2.5-Coder-Ins-32B/Scenario.codegeneration_10_0.2_eval_all.json",
        "cutoff_date": "2024-09-01",
        "k": 1,
        "max_len": 512,
        "temp": 0.7
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "url": "https://github.com/LiveCodeBench/submissions/raw/refs/heads/main/Qwen2.5-Ins-7B/Scenario.codegeneration_10_0.2_eval_all.json",
        "cutoff_date": "2024-09-01",
        "k": 1,
        "max_len": 512,
        "temp": 0.7
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "url": "https://github.com/LiveCodeBench/submissions/raw/refs/heads/main/Qwen2.5-Ins-32B/Scenario.codegeneration_10_0.2_eval_all.json",
        "cutoff_date": "2024-09-01",
        "k": 1,
        "max_len": 512,
        "temp": 0.7
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "url": "https://github.com/LiveCodeBench/submissions/raw/refs/heads/main/Qwen2.5-Ins-72B/Scenario.codegeneration_10_0.2_eval_all.json",
        "cutoff_date": "2024-09-01",
        "k": 1,
        "max_len": 512,
        "temp": 0.7
    },
}

# Process all models in the config
for MODEL_NAME, config in LCB_DATASET_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Processing: {MODEL_NAME}")
    print(f"{'='*60}")
    
    MODEL_ALIAS = MODEL_NAME.split("/")[-1]
    url = config["url"]
    cutoff_date_str = config["cutoff_date"]
    k = config["k"]
    max_len = config["max_len"]
    temp = config["temp"]
    
    # Load data
    print(f"Loading from: {url}")
    df = pd.read_json(url)
    
    # Format prompts
    print(f"Formatting prompts with tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df["formatted_prompt"] = df["question_content"].apply(lambda x: format_prompts_for_eval(x, tokenizer))
    df["success_rate"] = df["pass@1"]
    df["idx"] = range(len(df))
    
    # Convert contest_date to datetime and get the range
    df['contest_date'] = pd.to_datetime(df['contest_date'])
    
    min_date = df['contest_date'].min()
    max_date = df['contest_date'].max()
    date_range = max_date - min_date
    
    print(f"Earliest contest: {min_date}")
    print(f"Latest contest: {max_date}")
    print(f"Date range: {date_range}")
    
    # Create train/test split based on date
    cutoff_date = pd.Timestamp(cutoff_date_str)
    
    train_df = df[df['contest_date'] < cutoff_date].copy()
    test_df = df[df['contest_date'] >= cutoff_date].copy()
    
    print(f"Train size: {len(train_df)}, SR: {train_df["success_rate"].mean()} || (dates before {cutoff_date_str})")
    print(f"Test size: {len(test_df)}, SR: {test_df["success_rate"].mean()} || (dates from {cutoff_date_str} onwards)")
    print(f"Train date range: {train_df['contest_date'].min()} to {train_df['contest_date'].max()}")
    print(f"Test date range: {test_df['contest_date'].min()} to {test_df['contest_date'].max()}")
    
    # Save parquet files
    org = MODEL_NAME.split("/")[0]
    DATA_ROOT = f"../data/{org}/{MODEL_ALIAS}/LiveCodeBench"
    os.makedirs(DATA_ROOT, exist_ok=True)
    train_path = f"{DATA_ROOT}/train_maxlen_{max_len}_k_{k}_temp_{temp}.parquet"
    test_path = f"{DATA_ROOT}/test_maxlen_{max_len}_k_{k}_temp_{temp}.parquet"
    
    print(f"Saving train to: {train_path}")
    train_df.to_parquet(train_path)
    print(f"Saving test to: {test_path}")
    test_df.to_parquet(test_path)
    
    print(f"✓ Completed {MODEL_NAME}")
    print("========="*30)