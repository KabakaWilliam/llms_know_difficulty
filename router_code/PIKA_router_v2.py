#PIKA_router_v2.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import time
import gc
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

EASY_THRESHOLD = 0.9
MEDIUM_THRESHOLD = 0.4
threshold_str = f"threshE{EASY_THRESHOLD}_M{MEDIUM_THRESHOLD}"


# ----------------------------
# Routing
# ----------------------------
def route_questions(predicted_score: float, model_pool: List[str]) -> str:
    # 0-40, 40-80, 80-100
    if predicted_score >= EASY_THRESHOLD:
        return model_pool[0]  # easy
    elif predicted_score >= MEDIUM_THRESHOLD:
        return model_pool[1]  # medium
    else:
        return model_pool[2]  # hard
    
def add_self_consistency_columns(
    df: pd.DataFrame,
    *,
    route_col: str,
    model_pool: List[str],
    n_by_tier: Dict[str, int],
    temperature: float,
    sc_n_col: str = "sc_n",
    sc_temp_col: str = "sc_temp",
) -> pd.DataFrame:
    easy_model, mid_model, hard_model = model_pool[0], model_pool[1], model_pool[2]

    def tier_from_route(m: str) -> str:
        if m == easy_model:
            return "easy"
        elif m == mid_model:
            return "medium"
        elif m == hard_model:
            return "hard"
        return "hard"

    df[sc_temp_col] = float(temperature)
    df[sc_n_col] = df[route_col].apply(lambda m: int(n_by_tier[tier_from_route(m)]))
    return df


# ----------------------------
# Main
# ----------------------------
def main():
    repo_root = Path.cwd().parent
    sys.path.insert(0, str(repo_root))
    from thom_replication.utils.verification_math import (
        extract_gsm8k_solution,
        extract_solution,
        compute_score)

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
    # ----------------------------
    # Experiment config (MODULAR)
    # ----------------------------
    LABELLED_DATASETS_LIST = [
        "opencompass/AIME2025",
        "gneubig/aime-1983-2024",
        "openai/gsm8k",
        "DigitalLearningGmbH/MATH-lighteval",
    ]

    PROBING_DATASET = "DigitalLearningGmbH_MATH-lighteval"
    PROBE_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

    # Load probe-labelled files from THIS config:
    PROBE_K = 1
    PROBE_TEMP = 0.0

    # Run routed inference with THIS config:
    ROUTING_K = 5          # bookkeeping / output naming
    ROUTING_TEMP = 0.6

    # Baselines: (k,t) used for OG points NOW (you can extend later)
    BASELINE_K = 1
    BASELINE_T = 0.0

    SC_POLICY_NAME = ""


    # Self-consistency by routing tier
    if ROUTING_TEMP == 0.0:
        SC_POLICY = {"easy": 1, "medium": 1, "hard": 1}
        SC_POLICY_NAME = "greedy"
    else:
        SC_POLICY = {"easy": 2, "medium": 4, "hard": 3}
        SC_POLICY_NAME="_".join(f"{k}-{v}" for k, v in sorted(SC_POLICY.items()))

    # Inference params
    MAX_TOKENS = 3000

    # Cost semantics
    CHARGE_INPUT_PER_SAMPLE = False
    TOKENS_PER_MILLION = 1_000_000


    # Optional debug (big!)
    STORE_ALL_SAMPLES_COL = "generated_solutions"  # e.g. "all_samples_json" | None
    ORIGINAL_GT_COLUMN = "original_solution"
    FINAL_GT_COLUMN = "extracted_gts"

    batch_size_by_model = {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
        "Qwen/Qwen2.5-Math-7B-Instruct":  256,
        "Qwen/Qwen2.5-Math-72B-Instruct":  128,
    }

    # Where OG SR_DATA lives
    SR_DATA_BASEDIR = "../will_replication/DATA/SR_DATA"

    # Where to save figures
    FIG_DIR = "pareto_figures"

    # Where to save router_df
    ROUTER_SAVE_DF_DIR = "pika_router_runs"


    PROBE_RESULTS_DIR = "../will_replication/probe_results/DATA"
    LABELLED_SR_PATH = f"{PROBE_RESULTS_DIR}/Labelled_SR"

    PROBE_MODEL_ALIAS = "_".join(PROBE_MODEL_NAME.split("/"))

    model_pool = list(SIMPLE_MODEL_POOL_CONFIG.keys())

    model_run_cfgs = {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": VLLMModelRunCfg(tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=4096),
        "Qwen/Qwen2.5-Math-7B-Instruct":   VLLMModelRunCfg(tensor_parallel_size=1, gpu_memory_utilization=0.70, max_model_len=4096),
        "Qwen/Qwen2.5-Math-72B-Instruct":  VLLMModelRunCfg(tensor_parallel_size=2, gpu_memory_utilization=0.92, max_model_len=4096),
    }

    for LABELLED_DATASET_FULL_NAME in LABELLED_DATASETS_LIST:
        LABELLED_DS_ALIAS = LABELLED_DATASET_FULL_NAME.replace("/", "_")

        # --------- LOAD LABELLED PROBE FILES (probe temp/k) ----------
        try:
            routing_dataset_df = load_labelled_probe_dataset(
                MODEL_NAME=PROBE_MODEL_NAME,
                PROBE_SOURCE_DATASET=PROBING_DATASET,
                LABELLED_DATASET=LABELLED_DS_ALIAS,
                K=PROBE_K,
                TEMPERATURE=PROBE_TEMP,
                DATA_PATH=LABELLED_SR_PATH,
            )
        except Exception:
            print(f"❌ dataset config doesn't exist for {LABELLED_DS_ALIAS} @ (K={PROBE_K}, T={PROBE_TEMP})")
            continue
        
        # routing_dataset_df = routing_dataset_df.sample(n=10, random_state=42)
        print("\n\n==============================")
        print(f"Dataset: {LABELLED_DS_ALIAS}")
        print(f"Probe config    : K={PROBE_K} T={PROBE_TEMP}")
        print(f"Routing config  : K={ROUTING_K} T={ROUTING_TEMP}")
        print(f"SC policy       : {SC_POLICY}")

        score_col = "calibrated_score" if PROBE_K > 1 else "score"
        if score_col not in routing_dataset_df.columns:
            raise KeyError(f"Expected {score_col} in labelled probe df columns, got: {list(routing_dataset_df.columns)}")

        # --------- ROUTE USING PROBE SCORES ----------
        routing_dataset_df["route_to"] = routing_dataset_df[score_col].astype(float).apply(
            lambda s: route_questions(s, model_pool)
        )

        print("Routing breakdown:")
        print(routing_dataset_df["route_to"].value_counts())

        # --------- ASSIGN SELF-CONSISTENCY + ROUTING TEMP ----------
        routing_dataset_df = add_self_consistency_columns(
                routing_dataset_df,
                route_col="route_to",
                model_pool=model_pool,
                n_by_tier=SC_POLICY,
                temperature=ROUTING_TEMP,
                sc_n_col="sc_n",
                sc_temp_col="sc_temp",
                )
        
        print("Self-consistency n breakdown:")
        print(routing_dataset_df["sc_n"].value_counts())

        ckpt_path = f"{LABELLED_DS_ALIAS}_routed_probeK_{PROBE_K}_probeT_{PROBE_TEMP}_routeK{ROUTING_K}_routeT_{ROUTING_TEMP}_ckpt.parquet"

        # --------- RUN ROUTED INFERENCE ----------
        routing_dataset_df = run_routed_vllm_inference(
            routing_dataset_df,
            route_col="route_to",
            prompt_col="problem",
            out_text_col="routed_response_text",
            input_cost_col="input_cost_usd_once",
            gt_col=ORIGINAL_GT_COLUMN,
            max_tokens=MAX_TOKENS,
            batch_size_by_model=batch_size_by_model,
            checkpoint_path=ckpt_path,
            pricing_config=SIMPLE_MODEL_POOL_CONFIG,
            model_run_cfgs=model_run_cfgs,
            n_col="sc_n",
            temperature_col="sc_temp",
            majority_vote=True,
            extract_answer_fn=try_extract_solution,
            store_all_samples_col=STORE_ALL_SAMPLES_COL,
            charge_input_per_sample=CHARGE_INPUT_PER_SAMPLE,
        )
        routing_dataset_df["majority_vote_extracted_answer"] = routing_dataset_df[STORE_ALL_SAMPLES_COL].apply( lambda x: add_majority_vote_answer(json.loads(x)))

        # add ground truth
        if "gsm8k" in LABELLED_DS_ALIAS.lower():
            routing_dataset_df[FINAL_GT_COLUMN] = routing_dataset_df[ORIGINAL_GT_COLUMN].apply(extract_gsm8k_solution)
        elif "aime" in LABELLED_DS_ALIAS.lower():
            routing_dataset_df[FINAL_GT_COLUMN] = routing_dataset_df[ORIGINAL_GT_COLUMN]
        else:
            routing_dataset_df[FINAL_GT_COLUMN] = routing_dataset_df[ORIGINAL_GT_COLUMN].apply(extract_solution)

        routing_dataset_df["majority_vote_is_correct"] = routing_dataset_df.apply(
        lambda row: compute_score(solution_str=f"\\boxed{{{row['majority_vote_extracted_answer']}}}", ground_truth=row[FINAL_GT_COLUMN]),
        axis=1
        )
        routing_dataset_df["passk_score"] = routing_dataset_df.apply(
        lambda row: compute_passk_from_json_solutions(generated_sols_obj=row[STORE_ALL_SAMPLES_COL], ground_truth=row[FINAL_GT_COLUMN]),
        axis=1
    )
        os.makedirs(ROUTER_SAVE_DF_DIR, exist_ok=True)
        out_path = (
            f"{ROUTER_SAVE_DF_DIR}/{LABELLED_DS_ALIAS}_routed_by_{PROBING_DATASET}_{PROBE_MODEL_ALIAS}"
            f"_probeK{PROBE_K}_probeT{PROBE_TEMP}"
            f"_routeK{ROUTING_K}_routeT{ROUTING_TEMP}"
            f"_sc_{SC_POLICY_NAME}_{threshold_str}.parquet"
            )
        
        print(f"Router Accuracy (Majority Vote): {routing_dataset_df["majority_vote_is_correct"].mean()}")
        print(f"Router Pass@K: {routing_dataset_df["passk_score"].mean()}")
        print(f"Router Cost: {routing_dataset_df["total_cost_usd"].sum()}")
        
        routing_dataset_df.to_parquet(out_path, index=True)
        print(f"✅ Done. Saved routed df: {out_path}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()