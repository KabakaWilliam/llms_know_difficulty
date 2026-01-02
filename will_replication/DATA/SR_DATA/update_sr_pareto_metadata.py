# we're using this script to update the dataframes with cost and majority 
# vote data if it doesn't exist.
import sys
import pandas as pd
from pathlib import Path

repo_root = Path.cwd().parent.parent.parent
sys.path.insert(0, str(repo_root))

from will_replication.my_utils.utils import majority_vote_from_samples, get_output_tokens, get_output_cost, add_majority_vote_answer, encode_str, decode_str, SIMPLE_MODEL_POOL_CONFIG, TOKENS_PER_MILLION
from thom_replication.utils.verification_math import try_extract_solution, compute_score


TASKS = [
    "DigitalLearningGmbH_MATH-lighteval",
    "openai_gsm8k",
    "gneubig_aime-1983-2024",
    "opencompass_AIME2025",
]
MODEL_SIZES = ["1.5B", "7B", "72B"]
SPLITS = ["train", "test"]

K=1
TEMP=0.0

# Mapping from file model size to full model name
MODEL_NAME_MAP = {
    "1.5B": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "72B": "Qwen/Qwen2.5-Math-72B-Instruct",
}

def calculate_cost_with_pool_config(sr_df, model_size):
    """
    Recalculate costs using SIMPLE_MODEL_POOL_CONFIG instead of stored costs.
    This fixes cost calculations for older data files.
    """
    model_name = MODEL_NAME_MAP.get(model_size)
    if not model_name or model_name not in SIMPLE_MODEL_POOL_CONFIG:
        print(f"  ⚠️  Model {model_name} not found in SIMPLE_MODEL_POOL_CONFIG, using stored costs")
        return sr_df
    
    model_config = SIMPLE_MODEL_POOL_CONFIG[model_name]
    input_cost_per_token = model_config["model_costs"]["input_per_mill"] / TOKENS_PER_MILLION
    output_cost_per_token = model_config["model_costs"]["output_per_mill"] / TOKENS_PER_MILLION
    
    # Recalculate total output cost based on total_output_tokens
    sr_df["total_output_cost_usd"] = sr_df["total_output_tokens"] * output_cost_per_token
    
    # Recalculate total cost
    sr_df["total_cost_usd"] = sr_df["input_cost_usd_once"] + sr_df["total_output_cost_usd"]
    
    return sr_df

for TASK in TASKS:
    for MODEL_SIZE in MODEL_SIZES:
        for SPLIT in SPLITS:
            if "aime-1983" in TASK:
                SR_DATA_PATH = f"{TASK}/train-Qwen-Qwen2.5-Math-{MODEL_SIZE}-Instruct_maxlen_3000_k_{K}_temp_{TEMP}.parquet"
            else:
                SR_DATA_PATH = f"{TASK}/{SPLIT}-Qwen-Qwen2.5-Math-{MODEL_SIZE}-Instruct_maxlen_3000_k_{K}_temp_{TEMP}.parquet"
            
            try:
                sr_df = pd.read_parquet(SR_DATA_PATH)
            except:
                print(f"❌ ERROR: Couldn't load this path: {SR_DATA_PATH}")
                continue
            try:
                sr_df["problem_id"] = sr_df["problem"].apply(encode_str)
                sr_df["total_output_tokens"] = sr_df["generated_solutions"].apply(get_output_tokens)
                
                # Recalculate costs using proper pool config
                sr_df = calculate_cost_with_pool_config(sr_df, MODEL_SIZE)
                
                sr_df["majority_vote_extracted_answer"] = sr_df["generated_solutions"].apply(add_majority_vote_answer)
                sr_df["majority_vote_is_correct"] = sr_df.apply(
                    lambda row: compute_score(solution_str=f"\\boxed{{{row['majority_vote_extracted_answer']}}}", ground_truth=row["ground_truth"]),
                    axis=1
                )
            except Exception as e:
                print(f"❌ ERROR on {SR_DATA_PATH}: {e}")


            sr_df.to_parquet(SR_DATA_PATH)
            print(f"✅ updated {SR_DATA_PATH}")
print("Completed updating metadata.")