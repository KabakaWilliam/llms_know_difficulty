# we're using this script to update the dataframes with cost and majority 
# vote data if it doesn't exist.
import sys
import pandas as pd
from pathlib import Path

repo_root = Path.cwd().parent.parent.parent
sys.path.insert(0, str(repo_root))

from will_replication.my_utils.utils import majority_vote_from_samples, get_output_tokens, get_output_cost, add_majority_vote_answer, encode_str, decode_str
from thom_replication.utils.verification_math import try_extract_solution, compute_score


TASKS = [
    "DigitalLearningGmbH_MATH-lighteval",
    "openai_gsm8k",
    "gneubig_aime-1983-2024",
    "opencompass_AIME2025",
]
MODEL_SIZES = ["1.5B", "7B", "72B"]
SPLITS = ["train", "test"]

K=5
TEMP=0.6

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
                sr_df["total_output_cost_usd"] = sr_df["generated_solutions"].apply(get_output_cost)
                sr_df["total_cost_usd"] = sr_df["input_cost_usd_once"] + sr_df["total_output_cost_usd"]
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