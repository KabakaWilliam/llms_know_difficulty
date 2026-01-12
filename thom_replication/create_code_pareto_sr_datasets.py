# Largely the same as predicting_learnability/rollout_success_rate.py but a bit cleaner
# and uses our own verification functions
# run the apptainer.sh script first
from typing import Dict, Optional
from datetime import datetime

def filter_by_date(example, cutoff_date):
    try:
        # Handle ISO format with timestamp: "2023-08-21T00:00:00"
        date_str = example["contest_date"].split("T")[0]  # Extract just "2023-08-21"
        contest_date = datetime.strptime(date_str, "%Y-%m-%d")
        return contest_date >= cutoff_date
    except:
        return False


def filter_by_date_before(example, cutoff_date):
    """Filter examples with dates BEFORE the cutoff_date (for train split)"""
    try:
        # Handle ISO format with timestamp: "2023-08-21T00:00:00"
        date_str = example["contest_date"].split("T")[0]  # Extract just "2023-08-21"
        contest_date = datetime.strptime(date_str, "%Y-%m-%d")
        return contest_date < cutoff_date
    except:
        return False


def get_task(name):
    import os
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cutoff_date = datetime.strptime("2024-10-01", "%Y-%m-%d") #qwen coder came out 2024/09

    if name == "livecodebench_code_generation_lite":
        from utils.LiveCodeBench.utils import  compute_score
        ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6")
            
        # Create test split (problems >= cutoff_date) and train split (problems < cutoff_date)
        test_split = ds["test"].filter(lambda x: filter_by_date(x, cutoff_date))
        train_split = ds["test"].filter(lambda x: filter_by_date_before(x, cutoff_date))
        
        ds = {
            "train": train_split,
            "test": test_split
        }

        for split in ds:
            ds[split] = ds[split].rename_column("question_content", "problem") 
            ds[split] = ds[split].rename_column("private_test_cases", "extracted_solution") 
        print(f"Loading dataset:\n {ds}")
        return ds, compute_score

def main(
    model_name: str,
    max_response_len: int = 3000,
    temperature: float = 1.0,
    prompt_suffix: str = "",
    top_p: float = 1,
    top_k: int = -1,
    level_reasoning=None,
    gpu_memory_utilization: float = 0.70,
    num_rollouts_per_question: int = 50,
    max_questions_per_split: Optional[int] = None,
    tensor_parallel_size: int = 1,
    pricing_config: Optional[dict] = None,
    output_root: str = "../will_replication/DATA/SR_DATA",
    # batch_size: int = 16,  # generation batching; tune for your GPU
    batch_size_by_model: Optional[dict] = None,
    max_concurrent_API_requests: int = 16,
    sandbox_memory_limit_mb: int = 1024
):
    """
    Consolidated rollout main():
      - loads tasks
      - formats prompts with chat template
      - runs rollouts with vLLM
      - records per-rollout input/output tokens + cost (if pricing_config provided)
      - aggregates per-question totals
      - saves per-split parquet
    """

    import os
    import random
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from utils.LiveCodeBench.utils import format_prompts_for_eval, extract_solution

    # --------- pricing ----------
    TOKENS_PER_MILLION = 1_000_000
    if pricing_config is None:
        pricing_config = {}

    if batch_size_by_model is None:
        batch_size_by_model = {}
    # Default fallbacks
    default_bs = 256

    batch_size = batch_size_by_model.get(model_name, default_bs)


    model_costs = pricing_config.get(model_name, {}).get("model_costs", {})
    in_rate = model_costs.get("input_per_mill", None)
    out_rate = model_costs.get("output_per_mill", None)
    has_pricing = (in_rate is not None) and (out_rate is not None)

    # --------- sampling ----------
    sampling_params = SamplingParams(
        max_tokens=max_response_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=1,  # we do rollouts by repeating prompts
    )

    # --------- tokenizer ----------
    # Use HF tokenizer to apply_chat_template consistently.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # --------- vLLM model ----------
    # (72B special casing)
    # if "72b" in model_name.lower() or "120b" in model_name.lower():
    if  model_name.lower() in["72b", "120b"]:
        print("really large model loaded 🤯❗️")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.92,
            max_model_len=4096,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
        )
    else:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            max_num_batched_tokens=8192
        )

    # --------- tasks ----------
    TASKS = [
        "livecodebench_code_generation_lite"
    ]

    seed = 42
    rng = random.Random(seed)

    def format_prompt(problem_text: str) -> str:
        raw = problem_text + " " + prompt_suffix
        messages = [{"role": "user", "content": raw}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if "gpt-oss" in tokenizer.name_or_path.lower() and level_reasoning!=None:
            formatted_prompt = formatted_prompt.replace("Reasoning: medium", f"Reasoning: {level_reasoning}")
            formatted_prompt = formatted_prompt.replace("Reasoning: easy", f"Reasoning: {level_reasoning}")
        return formatted_prompt


    def count_input_tokens_batch(prompts: list[str]) -> list[int]:
        enc = tokenizer(prompts, add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    for TASK in TASKS:
        TASK_ALIAS = "_".join(TASK.split("/"))
        ds, compute_score = get_task(TASK_ALIAS)

        # --- choose indices per split ---
        selected_indices = {}
        for split in ds:
            n = len(ds[split])
            if max_questions_per_split is None:
                selected_indices[split] = list(range(n))
            else:
                k = min(max_questions_per_split, n)
                selected_indices[split] = rng.sample(range(n), k)

        print(f"\n 🚀 Beginning rollout on TASK={TASK} | model={model_name}\n")

        # We will build:
        # - unique prompts per (split, idx) for input token counting
        # - expanded "inputs" list for rollouts (prompt repeated num_rollouts_per_question times)
        unique_items = []  # list of dicts {split, idx, problem, formatted_prompt}
        inputs = []        # expanded per-rollout list of dicts (same keys)

        for split in ds:
            for idx in selected_indices[split]:
                item = ds[split][idx]
                if "code" in TASK:
                    formatted_prompt = format_prompts_for_eval(item["problem"], tokenizer)
                else:
                    formatted_prompt = format_prompt(item["problem"])
                    
                unique_items.append(
                    {
                        "split": split,
                        "idx": idx,
                        "problem": item["problem"],
                        "formatted_prompt": formatted_prompt,
                    }
                )
                for _ in range(num_rollouts_per_question):
                    inputs.append(
                        {
                            "split": split,
                            "idx": idx,
                            "problem": item["problem"],
                            "formatted_prompt": formatted_prompt,
                        }
                    )

        # --- input token counts (unique prompts) ---
        unique_prompts = [u["formatted_prompt"] for u in unique_items]
        unique_in_tok = count_input_tokens_batch(unique_prompts)

        key_to_in_tok = {
            (u["split"], u["idx"]): int(t)
            for u, t in zip(unique_items, unique_in_tok)
        }

        # --- run generation in batches (important for large rollouts) ---
        outputs_all = []
        for start in tqdm(range(0, len(inputs), batch_size), desc=f"Generating {TASK}"):
            batch = inputs[start : start + batch_size]
            batch_prompts = [b["formatted_prompt"] for b in batch]
            outputs_all.extend(llm.generate(batch_prompts, sampling_params=sampling_params))

        unload_model(llm)
        # --- collect results ---
        results = {split: {} for split in ds}

        # We'll also track per-question totals across rollouts
        print(f"\n🧮 Calculating metrics for: {split} split (n={len(selected_indices[split])}): \n")
        for inp, out in zip(inputs, outputs_all):
            split = inp["split"]
            idx = inp["idx"]

            if idx not in results[split]:
                ground_truth = extract_solution(ds[split][idx]["extracted_solution"])
                results[split][idx] = {
                    "problem": inp["problem"],
                    "formatted_prompt": inp["formatted_prompt"],
                    "ground_truth": ground_truth,
                    "generated_solutions": [],

                    # totals across rollouts
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost_usd": 0.0 if has_pricing else np.nan,
                    "input_cost_usd_once": 0.0 if has_pricing else np.nan,
                }

            generated_text = out.outputs[0].text
            out_tok = len(out.outputs[0].token_ids) if out.outputs[0].token_ids is not None else np.nan
            in_tok = key_to_in_tok[(split, idx)]

            # score
            score = compute_score(generated_text, results[split][idx]["ground_truth"], max_concurrent_API_requests=max_concurrent_API_requests,
            sandbox_memory_limit_mb=sandbox_memory_limit_mb)

            # cost per rollout
            if has_pricing and not np.isnan(out_tok):
                input_cost_once = (float(in_tok) / TOKENS_PER_MILLION) * float(in_rate)
                output_cost = (float(out_tok) / TOKENS_PER_MILLION) * float(out_rate)
            else:
                input_cost_once = np.nan
                output_cost = np.nan

            # ---- record rollout ----
            results[split][idx]["generated_solutions"].append(
                {
                    "text": generated_text,
                    "score": score,
                    "input_tokens": int(in_tok),  # debug (same each rollout)
                    "output_tokens": int(out_tok) if not np.isnan(out_tok) else np.nan,
                    "input_cost_usd_once": float(input_cost_once) if not np.isnan(input_cost_once) else np.nan,  # debug
                    "output_cost_usd": float(output_cost) if not np.isnan(output_cost) else np.nan,
                    # better name: this is output-only for this rollout
                    "rollout_cost_usd": float(output_cost) if not np.isnan(output_cost) else np.nan,
                }
            )

            # update totals
            if results[split][idx]["total_input_tokens"] == 0:
                results[split][idx]["total_input_tokens"] = int(in_tok)
                if has_pricing and not np.isnan(input_cost_once):
                    results[split][idx]["input_cost_usd_once"] = float(input_cost_once)
                    results[split][idx]["total_cost_usd"] += float(input_cost_once)

        # --- compute success rates ---
        for split in results:
            for idx in results[split]:
                gen_sols = results[split][idx]["generated_solutions"]
                correct = sum(gs["score"] for gs in gen_sols)
                total = len(gen_sols)
                results[split][idx]["success_rate"] = correct / total if total > 0 else 0.0

                

        # --- print overall success rates ---
        for split in results:
            total_questions = len(results[split])
            total_success = sum(results[split][idx]["success_rate"] for idx in results[split])
            overall_success_rate = total_success / total_questions if total_questions > 0 else 0.0
            print(f"Overall success rate for split {split}: {overall_success_rate:.4f}")

        # --- save per split ---
        os.makedirs(output_root, exist_ok=True)
        output_dir = os.path.join(output_root, TASK.replace("/", "_"))
        os.makedirs(output_dir, exist_ok=True)

        for split in results:
            results_df = pd.DataFrame.from_dict(results[split], orient="index")
            results_df["model_name"] = model_name
            results_df["task"] = TASK
            results_df["split"] = split

            model_alias = model_name.replace("/", "-")
            if max_questions_per_split is not None:
                filepath = (
                    f"{output_dir}/{split}-{len(results[split])}-{model_alias}"
                    f"_maxlen_{max_response_len}_k_{num_rollouts_per_question}_temp_{temperature}.parquet"
                )
            else:
                filepath = (
                    f"{output_dir}/{split}-{model_alias}"
                    f"_maxlen_{max_response_len}_k_{num_rollouts_per_question}_temp_{temperature}.parquet"
                )
            results_df = results_df.reset_index().rename(columns={"index": "idx"})
            results_df["problem_id"] = results_df["problem"].apply(encode_str)

            if has_pricing:
                # Output tokens/cost accumulate across rollouts
                results_df["total_output_tokens"] = results_df["generated_solutions"].apply(get_output_tokens)
                results_df["total_output_cost_usd"] = results_df["generated_solutions"].apply(get_output_cost)
                results_df["total_cost_usd"] = results_df["input_cost_usd_once"] + results_df["total_output_cost_usd"]

            results_df.to_parquet(filepath)
            print(f"Saved {TASK} / {split} split to: {filepath}")
            
            # Calculate Pass@K
            pass_at_k = results_df["success_rate"].mean()
            
            # Send notification via ntfy.sh
            try:
                ntfy_message = (
                    f"✅ {TASK}\n"
                    f"Split: {split} | Model: {model_name}\n"
                    f"Config: k={num_rollouts_per_question}, τ={temperature}, maxlen={max_response_len}\n"
                    f"Pass@{num_rollouts_per_question}: {pass_at_k:.4f}"
                )
                requests.post("https://ntfy.sh/llms_know_difficulty", data=ntfy_message)
                print('✅ Metrics sent to ntfy.sh')
            except Exception as e:
                print(f"⚠️ ntfy.sh notification failed: {e}")

    unload_model(llm)

if __name__ == "__main__":
    import time
    import gc
    from pathlib import Path
    import sys
    import requests

    repo_root = Path.cwd().parent
    sys.path.insert(0, str(repo_root))

    from will_replication.my_utils.utils import (
        SIMPLE_MODEL_POOL_CONFIG, ModelConfig, unload_model, get_output_cost,
        get_output_tokens, get_output_text, encode_str
    )
    
    MODELS_TO_RUN = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    # "Qwen/Qwen2.5-Coder-3B-Instruct",
    # "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-Coder-32B-Instruct",
    ]

    batch_size_by_model = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": 16,
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
    "Qwen/Qwen2.5-Math-7B-Instruct":  256,
    "Qwen/Qwen2.5-Math-72B-Instruct":  128,
    "Qwen/Qwen2.5-1.5B-Instruct": 256,
    "Qwen/Qwen2.5-7B-Instruct":  256,
    "Qwen/Qwen2.5-72B-Instruct":  128,
    "openai/gpt-oss-20b":  256,
    "openai/gpt-oss-120b":  64,
    }
    
    for i, MODEL_TO_ROLLOUT in enumerate(MODELS_TO_RUN):
        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(MODELS_TO_RUN)}: {MODEL_TO_ROLLOUT}")
        print(f"{'='*60}\n")
        
        main(
            model_name=MODEL_TO_ROLLOUT,
            max_questions_per_split=5,
            # level_reasoning="high",
            tensor_parallel_size=2,
            num_rollouts_per_question=1,
            temperature=0.2,
            top_p=0.95,
            top_k=-1,
            gpu_memory_utilization=0.10, #increase according to VRAM available
            pricing_config=SIMPLE_MODEL_POOL_CONFIG,
            batch_size_by_model=batch_size_by_model,
            max_response_len=4096,
            max_concurrent_API_requests=8
        )
        
        print(f"\nFinished processing {MODEL_TO_ROLLOUT}")
            # Send notification via ntfy.sh
        try:
            requests.post("https://ntfy.sh/llms_know_difficulty", data=f"Finished processing {MODEL_TO_ROLLOUT}")
            print('✅ notification request sent')
        except Exception as e:
            print(f"ntfy.sh notification failed: {e}")
        
        # Clean up and wait for vLLM to die before loading next model
        if i < len(MODELS_TO_RUN) - 1:  # Don't wait after the last model
            print("Cleaning up and waiting for vLLM to release resources...")
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(10)  # Wait 10 seconds for vLLM to fully shutdown
            print(f"Ready to load next model: {MODELS_TO_RUN[i+1]}\n")



