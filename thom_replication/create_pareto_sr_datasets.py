# Largely the same as predicting_learnability/rollout_success_rate.py but a bit cleaner
# and uses our own verification functions
from typing import Dict, Optional

def get_task(name):
    import os
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if name == 'DigitalLearningGmbH_MATH-lighteval':
        from utils.verification_math import compute_score, extract_solution
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")
        # extract the solution from the chain of thought
        def add_extracted_solution(example):
            example['extracted_solution'] = extract_solution(example['solution'])
            return example
        for split in ds:
            ds[split] = ds[split].map(add_extracted_solution)

        return ds, compute_score
    
    elif name == "openai_gsm8k":
        from utils.verification_math import compute_score, extract_gsm8k_solution
        ds = load_dataset("openai/gsm8k", "main")
        # extract the solution from the chain of thought
        def add_extracted_solution(example):
            example['extracted_solution'] = extract_gsm8k_solution(example['answer'])
            return example
        for split in ds:
            ds[split] = ds[split].rename_column("question", "problem") # rename "question" -> "problem"
            ds[split] = ds[split].map(add_extracted_solution)

        return ds, compute_score
    
    elif name == "gneubig_aime-1983-2024":
        from utils.verification_math import compute_score

        ds = load_dataset("gneubig/aime-1983-2024")

        def add_extracted_solution(example):
            example["extracted_solution"] = example["Answer"]
            return example

        for split in ds:
            # rename Question -> problem
            ds[split] = ds[split].rename_column("Question", "problem")
            # add extracted_solution from Answer
            ds[split] = ds[split].map(add_extracted_solution)

        return ds, compute_score
    
    elif name == "opencompass_AIME2025":
        from utils.verification_math import compute_score

        ds = load_dataset("opencompass/AIME2025", "AIME2025-I")

        def add_extracted_solution(example):
            example["extracted_solution"] = example["answer"]
            return example

        for split in ds:
            # rename question -> problem
            ds[split] = ds[split].rename_column("question", "problem")
            # add extracted_solution from answer
            ds[split] = ds[split].map(add_extracted_solution)

        return ds, compute_score
    
    elif name.startswith('Qwen_PolyMath_'):
        from utils.verification_math import compute_score
        from datasets import concatenate_datasets
        
        # Extract language suffix (e.g., "en" from "Qwen_PolyMath_en")
        language = name.split('Qwen_PolyMath_')[1]
        
        ds = load_dataset("Qwen/PolyMath", language)
        
        LEVELS = ["top", "high", "medium", "low"]
        
        # Process each level: rename columns, add level, and create train/test splits in one pass
        train_datasets = []
        test_datasets = []
        
        for level in LEVELS:
            data = ds[level]
            
            # Combined processing: extract solution, rename column, add level
            def process_data(example, level=level):
                example['extracted_solution'] = example['answer']
                example['level'] = level
                return example
            
            processed = data.map(process_data).rename_column("question", "problem")
            
            # Create train/test split (80/20)
            train_test = processed.train_test_split(test_size=0.2, seed=42)
            train_datasets.append(train_test['train'])
            test_datasets.append(train_test['test'])
        
        # Combine all levels into single train and test datasets
        ds_final = {
            'train': concatenate_datasets(train_datasets),
            'test': concatenate_datasets(test_datasets)
        }

        return ds_final, compute_score

    raise ValueError(f"Unknown task {name}")


def main(
    model_name: str,
    max_response_len: int = 3000,
    temperature: float = 1.0,
    prompt_suffix: str = "Let's think step by step and output the final answer within \\boxed{}.",
    level_reasoning=None,
    num_rollouts_per_question: int = 50,
    max_questions_per_split: Optional[int] = None,
    tensor_parallel_size: int = 1,
    pricing_config: Optional[dict] = None,
    output_root: str = "../data/",
    # batch_size: int = 16,  # generation batching; tune for your GPU
    batch_size_by_model: Optional[dict] = None,
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

    # --------- pricing ----------
    TOKENS_PER_MILLION = 1_000_000
    if pricing_config is None:
        pricing_config = {}

    if batch_size_by_model is None:
        batch_size_by_model = {}
    # Default fallbacks
    default_bs = 256
    TOP_P=1
    if "72" in model_name:
        default_bs = 32
    if "qwen2.5-math" in model_name.lower() and num_rollouts_per_question > 1:
        # optimal setting for math https://github.com/QwenLM/Qwen2.5-Math#:~:text=with%20trust_remote_code.-,Warning,-We%20use%20temperature
        TOP_P = 0.8

    batch_size = batch_size_by_model.get(model_name, default_bs)


    model_costs = pricing_config.get(model_name, {}).get("model_costs", {})
    in_rate = model_costs.get("input_per_mill", None)
    out_rate = model_costs.get("output_per_mill", None)
    has_pricing = (in_rate is not None) and (out_rate is not None)

    # --------- sampling ----------
    sampling_params = SamplingParams(
        max_tokens=max_response_len,
        temperature=temperature,
        top_p=TOP_P,
        top_k=-1,
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
            gpu_memory_utilization=0.85,
        )

    LANGUAGE_SUFFIXES = ["en", "sw", "zh", "es", "ar", "fr", "bn", "pt", "ru", "id", "de", "ja", "vi", "it", "te", "ko", "th", "ms"]
    # --------- tasks ----------
    TASKS = [
        # "opencompass_AIME2025",
        # "gneubig_aime-1983-2024",
        "DigitalLearningGmbH_MATH-lighteval",
        # "openai_gsm8k",
    ] 
    # + [f"Qwen_PolyMath_{lang}" for lang in LANGUAGE_SUFFIXES]

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

        # --- collect results ---
        results = {split: {} for split in ds}

        # We'll also track per-question totals across rollouts
        for inp, out in zip(inputs, outputs_all):
            split = inp["split"]
            idx = inp["idx"]

            if idx not in results[split]:
                ground_truth = ds[split][idx]["extracted_solution"]
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
            score = compute_score(generated_text, results[split][idx]["ground_truth"])

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

        # --- compute success rates and pass@k ---
        for split in results:
            for idx in results[split]:
                gen_sols = results[split][idx]["generated_solutions"]
                correct = sum(gs["score"] for gs in gen_sols)
                total = len(gen_sols)
                results[split][idx]["success_rate"] = correct / total if total > 0 else 0.0
                # pass@k: whether at least one solution is correct
                results[split][idx]["pass_at_k"] = 1.0 if correct > 0 else 0.0

                # majority_vote_extracted_answer = add_majority_vote_answer(gen_sols)
                # # majority_vote_extracted_answer = majority_vote_from_samples(gen_sols, extract_answer_fn=try_extract_solution)[1]

                # majority_vote_extracted_answer_boxed = f"\\boxed{{{majority_vote_extracted_answer}}}"
                # majority_vote_score = compute_score(solution_str=majority_vote_extracted_answer_boxed, ground_truth=results[split][idx]["ground_truth"])

                # results[split][idx]["majority_vote_extracted_answer"] = majority_vote_extracted_answer
                # results[split][idx]["majority_vote_is_correct"] = majority_vote_score
                

        # --- print overall success rates and pass@k ---
        for split in results:
            total_questions = len(results[split])
            total_success = sum(results[split][idx]["success_rate"] for idx in results[split])
            overall_success_rate = total_success / total_questions if total_questions > 0 else 0.0
            total_pass_at_k = sum(results[split][idx]["pass_at_k"] for idx in results[split])
            overall_pass_at_k = total_pass_at_k / total_questions if total_questions > 0 else 0.0
            print(f"Overall success rate for split {split}: {overall_success_rate:.4f}")
            print(f"Overall Pass@k for split {split}: {overall_pass_at_k:.4f}")

        # --- save per split ---
        os.makedirs(output_root, exist_ok=True)
        
        # Extract model family and model name for output directory
        # e.g., "Qwen/Qwen2.5-1.5B-Instruct" -> family="Qwen", name="Qwen2.5-1.5B-Instruct"
        if "/" in model_name:
            model_family, model_full_name = model_name.split("/", 1)
        else:
            model_family = model_name.split("-")[0]
            model_full_name = model_name

        if "gpt" in model_full_name.lower():
            model_full_name += f"_{level_reasoning}"
        
        output_dir = os.path.join(output_root, model_family, model_full_name, TASK.replace("/", "_"))
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
                results_df["majority_vote_extracted_answer"] = results_df["generated_solutions"].apply(add_majority_vote_answer)
                results_df["majority_vote_is_correct"] = results_df.apply(
                lambda row: compute_score(solution_str=f"\\boxed{{{row['majority_vote_extracted_answer']}}}", ground_truth=row["ground_truth"]),
                axis=1
                )
                mv_accuracy = results_df["majority_vote_is_correct"].mean()
                print(f"Majority Vote Accuracy : {mv_accuracy}")
            else:
                mv_accuracy = np.nan

            results_df.to_parquet(filepath)
            print(f"Saved {TASK} / {split} split to: {filepath}")
            
            # Calculate Pass@K (fraction of questions with at least one correct solution)
            pass_at_k = results_df["pass_at_k"].mean()
            success_rate = results_df["success_rate"].mean()
            
            # Send notification via ntfy.sh
            try:
                ntfy_message = (
                    f"✅ {TASK}\n"
                    f"Split: {split} | Model: {model_name}\n"
                    f"Config: k={num_rollouts_per_question}, τ={temperature}, maxlen={max_response_len}\n"
                    f"Pass@{num_rollouts_per_question}: {pass_at_k:.4f} | Success Rate: {success_rate:.4f} | MV Acc: {mv_accuracy:.4f}"
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
        get_output_tokens, get_output_text, add_majority_vote_answer, encode_str
    )
    
    MODELS_TO_RUN = [
    #  "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2-1.5B-Instruct",
    # "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "Qwen/Qwen2.5-Math-72B-Instruct",
    "openai/gpt-oss-20b"
    # "openai/gpt-oss-120b"
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct"
    ]

    batch_size_by_model = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 768,
    "Qwen/Qwen2.5-Math-7B-Instruct":  256,
    "Qwen/Qwen2.5-Math-72B-Instruct":  128,
    "Qwen/Qwen2.5-1.5B-Instruct": 128,
    "Qwen/Qwen2.5-7B-Instruct":  256,
    "Qwen/Qwen2.5-72B-Instruct":  128,
    "openai/gpt-oss-20b":  768,
    "openai/gpt-oss-120b":  64,
    }
    TEMPERATURES = [0.2, 0.5, 0.8, 0.9, 1.0, 1.5]
    NUM_ROLLOUTS=10
    
    for i, MODEL_TO_ROLLOUT in enumerate(MODELS_TO_RUN):

        for SELECTED_TEMP in TEMPERATURES:
            print(f"\n{'='*60}")
            print(f"Processing model {i+1}/{len(MODELS_TO_RUN)}: {MODEL_TO_ROLLOUT} at temp {SELECTED_TEMP}")
            print(f"{'='*60}\n")

            if SELECTED_TEMP == 0.0:
                NUM_ROLLOUTS = 1
            
            main(
                model_name=MODEL_TO_ROLLOUT,
                # max_questions_per_split=15,
                level_reasoning="medium",
                tensor_parallel_size=1,
                num_rollouts_per_question=NUM_ROLLOUTS,
                temperature=SELECTED_TEMP,
                pricing_config=SIMPLE_MODEL_POOL_CONFIG,
                batch_size_by_model=batch_size_by_model,
                max_response_len=3000
            )
            
            print(f"\nFinished processing {MODEL_TO_ROLLOUT} at temp {SELECTED_TEMP}")
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



