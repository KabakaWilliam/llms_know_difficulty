"""
Execute and save all routing strategies

This script applies all routing strategies to labeled probe data,
executes inference with vLLM, and saves results with identifiable names
for later visualization in the comparison notebook.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup paths - use script location, not current working directory
script_dir = Path(__file__).parent.parent  # router_code -> repo root
repo_root = script_dir
sys.path.insert(0, str(repo_root))

from will_replication.my_utils.utils import (
    load_labelled_probe_dataset,
    SIMPLE_MODEL_POOL_CONFIG,
    majority_vote_from_samples,
    batch_apply_chat_template,
    count_input_tokens_batch,
    unload_model,
    decode_str,
    get_output_tokens,
    get_output_cost,
    add_majority_vote_answer,
    try_extract_solution,
    encode_str,
    compute_passk_from_json_solutions,
    run_routed_vllm_inference,
    VLLMModelRunCfg
)

from thom_replication.utils.verification_math import (
    extract_gsm8k_solution,
    extract_solution,
    compute_score
)

from router_code.routing_strategies import (
    get_all_strategies,
    apply_routing_strategy,
    ROUTING_STRATEGIES
)

import requests


def execute_routing_strategies(
    labelled_datasets: List[str] = None,
    probing_dataset: str = "DigitalLearningGmbH_MATH-lighteval",
    probe_k: int = 1,
    probe_temp: float = 0.0,
    target_confidence: float = 0.90,
    max_tokens: int = 3000,
    batch_size_by_model: Dict = None,
    save_base_dir: str = None,
    dry_run: bool = False,
    data_path: str = None,
):
    """
    Execute all routing strategies and save results.
    
    Args:
        labelled_datasets: list of labelled dataset names to evaluate on
        probing_dataset: which dataset to use for probe
        probe_k: k for probe generation
        probe_temp: temperature for probe
        target_confidence: confidence threshold for routing
        max_tokens: max tokens for generation
        batch_size_by_model: batch sizes for each model
        save_base_dir: base directory for saving results (defaults to repo_root/pika_cascade_trial)
        dry_run: if True, only prepare data without running inference
        data_path: path to probe data (defaults to repo_root/will_replication/probe_results/DATA/Labelled_SR)
    """
    
    # Set default paths relative to repo root
    if save_base_dir is None:
        save_base_dir = str(repo_root / "pika_cascade_trial")
    if data_path is None:
        data_path = str(repo_root / "will_replication" / "probe_results" / "DATA" / "Labelled_SR")
    
    # Validate paths exist
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(
            f"Probe data directory not found: {data_path}\n"
            f"Expected at: {data_path_obj.resolve()}\n"
            f"Please ensure will_replication/probe_results is available"
        )
    
    save_base_dir_obj = Path(save_base_dir)
    save_base_dir_obj.mkdir(parents=True, exist_ok=True)
    
    print(f"Data path: {data_path_obj.resolve()}")
    print(f"Save base dir: {save_base_dir_obj.resolve()}")
    
    if labelled_datasets is None:
        labelled_datasets = [
            "opencompass/AIME2025",
            "gneubig/aime-1983-2024",
            "openai/gsm8k",
            "DigitalLearningGmbH/MATH-lighteval",
        ]
    
    if batch_size_by_model is None:
        batch_size_by_model = {
            "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
            "Qwen/Qwen2.5-Math-7B-Instruct": 256,
            "Qwen/Qwen2.5-Math-72B-Instruct": 64,
        }
    
    model_pool = list(SIMPLE_MODEL_POOL_CONFIG.keys())
    model_run_cfgs = {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": VLLMModelRunCfg(
            tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=4096
        ),
        "Qwen/Qwen2.5-Math-7B-Instruct": VLLMModelRunCfg(
            tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=4096
        ),
        "Qwen/Qwen2.5-Math-72B-Instruct": VLLMModelRunCfg(
            tensor_parallel_size=2, gpu_memory_utilization=0.90, max_model_len=4096
        ),
    }
    
    TOKENS_PER_MILLION = 1_000_000
    STORE_ALL_SAMPLES_COL = "generated_solutions"
    ORIGINAL_GT_COLUMN = "original_solution"
    FINAL_GT_COLUMN = "extracted_gts"
    
    cost_ratios = {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": 1.0,
        "Qwen/Qwen2.5-Math-7B-Instruct": 2.0,
        "Qwen/Qwen2.5-Math-72B-Instruct": 9.0,
    }
    
    strategies = get_all_strategies()
    
    print("=" * 80)
    print(f"EXECUTING {len(strategies)} ROUTING STRATEGIES")
    print("=" * 80)
    print(f"\nAvailable strategies:")
    for i, strategy_name in enumerate(strategies.keys(), 1):
        print(f"  {i}. {strategy_name}")
    
    # Process each labeled dataset
    for labelled_dataset_full in labelled_datasets:
        dataset_alias = labelled_dataset_full.replace("/", "_")
        
        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_alias}")
        print(f"{'=' * 80}")
        
        # Load probe data for this dataset from both models
        try:
            # Load 1.5B model scores
            probe_df_1_5b = load_labelled_probe_dataset(
                MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct",
                PROBE_SOURCE_DATASET=probing_dataset,
                LABELLED_DATASET=dataset_alias,
                K=probe_k,
                TEMPERATURE=probe_temp,
                DATA_PATH=data_path
            )
            
            # Load 7B model scores
            probe_df_7b = load_labelled_probe_dataset(
                MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct",
                PROBE_SOURCE_DATASET=probing_dataset,
                LABELLED_DATASET=dataset_alias,
                K=probe_k,
                TEMPERATURE=probe_temp,
                DATA_PATH=data_path
            )
            
            # Merge on index (both should have same samples in same order)
            # Start with common columns that should exist
            probe_df = probe_df_1_5b[["idx", "original_solution"]].copy()
            
            # Add problem text - try multiple column names
            if "problem" in probe_df_1_5b.columns:
                probe_df["problem"] = probe_df_1_5b["problem"].values
            elif "formatted_prompt" in probe_df_1_5b.columns:
                probe_df["problem"] = probe_df_1_5b["formatted_prompt"].values
            elif "formatted" in probe_df_1_5b.columns:
                probe_df["problem"] = probe_df_1_5b["formatted"].values
            elif "prompt_scored" in probe_df_1_5b.columns:
                probe_df["problem"] = probe_df_1_5b["prompt_scored"].values
            else:
                # If no problem column found, use empty strings as fallback
                print(f"⚠️ Warning: No problem/prompt column found. Available columns: {probe_df_1_5b.columns.tolist()}")
                probe_df["problem"] = ""
            
            # Use calibrated_score if available, otherwise fall back to sigmoid score
            score_col_1_5b = "calibrated_score" if "calibrated_score" in probe_df_1_5b.columns else "score"
            score_col_7b = "calibrated_score" if "calibrated_score" in probe_df_7b.columns else "score"
            
            probe_df["score_1.5B"] = probe_df_1_5b[score_col_1_5b].values
            probe_df["score_7B"] = probe_df_7b[score_col_7b].values
            
            print(f"✓ Loaded probe data: {len(probe_df)} samples with score_1.5B (from '{score_col_1_5b}') and score_7B (from '{score_col_7b}')")
        except Exception as e:
            print(f"✗ Failed to load probe data: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Ensure we have the required score columns
        if "score_1.5B" not in probe_df.columns or "score_7B" not in probe_df.columns:
            print(f"✗ Missing required score columns (score_1.5B, score_7B)")
            continue
        
        # Apply each strategy
        for strategy_name, strategy_func in strategies.items():
            print(f"\n  → Executing strategy: {strategy_name}")
            
            try:
                # Apply routing strategy
                routes = apply_routing_strategy(
                    probe_df,
                    strategy_name=strategy_name,
                    target_conf=target_confidence,
                    cost_ratios=cost_ratios,
                )
                
                # Create dataframe with routing results
                routing_df = probe_df.copy()
                routing_df["route_to"] = routes
                
                # Parse MV routing strings to extract k and temperature
                # MV routes look like "mv_1.5B_k8_t0.7" or "mv_7B_k8_t0.7"
                # Regular routes are full model names
                routing_df["is_mv"] = routing_df["route_to"].str.contains("mv_", na=False)
                routing_df["sc_n"] = 1
                routing_df["sc_temp"] = 0.0
                
                # For MV routes, extract k and temperature
                for idx, row in routing_df.iterrows():
                    if row["is_mv"]:
                        route_str = row["route_to"]
                        # Parse "mv_1.5B_k8_t0.7" or "mv_72B_k8_t0.7" → k=8, temp=0.7, model=1.5B/7B/72B
                        if "_k" in route_str and "_t" in route_str:
                            parts = route_str.split("_")
                            model_size = parts[1]  # "1.5B", "7B", or "72B"
                            k_val = int(parts[2].replace("k", ""))
                            t_val = float(parts[3].replace("t", ""))
                            
                            # Replace with actual model name
                            if "1.5B" in model_size:
                                routing_df.at[idx, "route_to"] = "Qwen/Qwen2.5-Math-1.5B-Instruct"
                            elif "7B" in model_size:
                                routing_df.at[idx, "route_to"] = "Qwen/Qwen2.5-Math-7B-Instruct"
                            elif "72B" in model_size:
                                routing_df.at[idx, "route_to"] = "Qwen/Qwen2.5-Math-72B-Instruct"
                            
                            routing_df.at[idx, "sc_n"] = k_val
                            routing_df.at[idx, "sc_temp"] = t_val
                
                # Ensure 72B is ALWAYS greedy (never MV) due to memory constraints
                # EXCEPTION: Allow mv_always_72B strategy to use MV on 72B if explicitly requested
                mask_72b = routing_df["route_to"] == "Qwen/Qwen2.5-Math-72B-Instruct"
                # Only override to greedy if NOT using mv_always_72B strategy
                if strategy_name != "mv_always_72B":
                    routing_df.loc[mask_72b, "sc_n"] = 1
                    routing_df.loc[mask_72b, "sc_temp"] = 0.0
                    routing_df.loc[mask_72b, "is_mv"] = False

                
                # Print routing breakdown
                route_counts = routing_df["route_to"].value_counts()
                print(f"    Routing breakdown:")
                for model, count in route_counts.items():
                    # Count how many MV vs greedy for this model
                    model_mask = routing_df["route_to"] == model
                    mv_count = routing_df[model_mask & routing_df["is_mv"]].shape[0]
                    greedy_count = model_mask.sum() - mv_count
                    
                    pct = 100 * count / len(routing_df)
                    if mv_count > 0 and greedy_count > 0:
                        print(f"      {model.split('/')[-1]}: {count} ({pct:.1f}%) - MV×{mv_count} + Greedy×{greedy_count}")
                    elif mv_count > 0:
                        print(f"      {model.split('/')[-1]}: {count} ({pct:.1f}%) - MV (k=8)")
                    else:
                        print(f"      {model.split('/')[-1]}: {count} ({pct:.1f}%) - Greedy")
                
                # Save routing decisions
                save_dir = f"{save_base_dir}/{probing_dataset}_probe/{dataset_alias}_routed"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"routing_{strategy_name}_conf{target_confidence}.parquet")
                routing_df.to_parquet(save_path, index=False)
                print(f"    ✓ Saved routing decisions: {save_path}")
                
                if dry_run:
                    print(f"    (DRY RUN - skipping inference)")
                    continue
                
                # Run inference with routed models
                print(f"    Running vLLM inference...")
                
                routing_df = run_routed_vllm_inference(
                    routing_df,
                    route_col="route_to",
                    prompt_col="problem",
                    out_text_col="routed_response_text",
                    input_cost_col="input_cost_usd_once",
                    gt_col=ORIGINAL_GT_COLUMN,
                    max_tokens=max_tokens,
                    batch_size_by_model=batch_size_by_model,
                    checkpoint_path=None,
                    pricing_config=SIMPLE_MODEL_POOL_CONFIG,
                    model_run_cfgs=model_run_cfgs,
                    n_col="sc_n",
                    temperature_col="sc_temp",
                    majority_vote=True,
                    extract_answer_fn=try_extract_solution,
                    store_all_samples_col=STORE_ALL_SAMPLES_COL,
                    charge_input_per_sample=False,
                )
                
                # Process results
                routing_df["majority_vote_extracted_answer"] = (
                    routing_df[STORE_ALL_SAMPLES_COL].apply(
                        lambda x: add_majority_vote_answer(json.loads(x))
                    )
                )
                
                # Extract ground truth
                if "gsm8k" in dataset_alias.lower():
                    routing_df[FINAL_GT_COLUMN] = routing_df[ORIGINAL_GT_COLUMN].apply(
                        extract_gsm8k_solution
                    )
                elif "aime" in dataset_alias.lower():
                    routing_df[FINAL_GT_COLUMN] = routing_df[ORIGINAL_GT_COLUMN]
                else:
                    routing_df[FINAL_GT_COLUMN] = routing_df[ORIGINAL_GT_COLUMN].apply(
                        extract_solution
                    )
                
                # Compute metrics
                routing_df["majority_vote_is_correct"] = routing_df.apply(
                    lambda row: compute_score(
                        solution_str=f"\\boxed{{{row['majority_vote_extracted_answer']}}}",
                        ground_truth=row[FINAL_GT_COLUMN],
                    ),
                    axis=1,
                )
                routing_df["passk_score"] = routing_df.apply(
                    lambda row: compute_passk_from_json_solutions(
                        generated_sols_obj=row[STORE_ALL_SAMPLES_COL],
                        ground_truth=row[FINAL_GT_COLUMN],
                    ),
                    axis=1,
                )
                
                # Save results
                result_path = os.path.join(
                    save_dir, f"answered_{strategy_name}_conf{target_confidence}.parquet"
                )
                routing_df.to_parquet(result_path, index=True)
                
                # Print metrics
                accuracy = routing_df["majority_vote_is_correct"].mean()
                passk = routing_df["passk_score"].mean()
                cost = routing_df["total_cost_usd"].sum()
                
                print(f"    Results:")
                print(f"      Accuracy: {accuracy:.4f}")
                print(f"      Pass@K: {passk:.4f}")
                print(f"      Total Cost: ${cost:.4f}")
                print(f"    ✓ Saved results: {result_path}")
                
            except Exception as e:
                print(f"    ✗ Error executing strategy: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✓ Completed {len(strategies)} strategies for {dataset_alias}")
    
    print(f"\n{'=' * 80}")
    print("✅ ALL STRATEGIES COMPLETED")
    print(f"{'=' * 80}")
    
    # Send notification
    try:
        requests.post(
            "https://ntfy.sh/llms_know_difficulty",
            data="✅ Finished executing all routing strategies"
        )
        print("✓ Notification sent")
    except Exception as e:
        print(f"✗ Notification failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute and save all routing strategies"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data but skip inference"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of datasets to process"
    )
    parser.add_argument(
        "--target-conf",
        type=float,
        default=0.90,
        help="Target confidence threshold"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to probe data directory"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Base directory for saving results"
    )
    
    args = parser.parse_args()
    
    execute_routing_strategies(
        labelled_datasets=args.datasets,
        target_confidence=args.target_conf,
        dry_run=args.dry_run,
        data_path=args.data_path,
        save_base_dir=args.save_dir,
    )
