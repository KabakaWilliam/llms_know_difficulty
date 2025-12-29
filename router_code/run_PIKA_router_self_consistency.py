"""
run_router_modular.py  (with Pareto plotting + figure saving)

Adds:
- After routing inference for each dataset, compute:
    * ROUTER point: (cost=sum(total_cost_usd), score=mean(routed_is_correct)) from routed parquet.
    * Baseline points: other model performances at greedy k=1, t=0.0 (placeholder metadata so we can extend later)
- Plot Pareto frontier per dataset and save as PNG/PDF.

Important:
- Baseline OG parquet format you described earlier often has only one row (aggregate), not per-example idx.
  We therefore aggregate directly from those files:
      score = mean(success_rate)   (usually same as the scalar)
      cost  = sum(total_cost_usd)
  This matches your “7B cost should be 0.78158…” expectation.

Extensibility:
- Each point includes meta: k, t (currently baselines fixed to k=1,t=0.0; router uses ROUTING_K/ROUTING_TEMP)
- Later you can add baseline points for other (k,t) and they’ll just appear.
"""

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


# ----------------------------
# Experiment config (MODULAR)
# ----------------------------
LABELLED_DATASETS_LIST = [
    # "opencompass/AIME2025",
    # "gneubig/aime-1983-2024",
    # "openai/gsm8k",
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


# Self-consistency by routing tier
if ROUTING_TEMP == 0.0:
    SC_POLICY = {"easy": 1, "medium": 1, "hard": 1}
else:
    SC_POLICY = {"easy": 3, "medium": 5, "hard": 1}

# Inference params
MAX_TOKENS = 3000

# Cost semantics
CHARGE_INPUT_PER_SAMPLE = False

# Optional debug (big!)
STORE_ALL_SAMPLES_COL = None  # e.g. "all_samples_json"

batch_size_by_model = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
    "Qwen/Qwen2.5-Math-7B-Instruct":  128,
    "Qwen/Qwen2.5-Math-72B-Instruct":  64,
}


# Where OG SR_DATA lives
SR_DATA_BASEDIR = "../will_replication/DATA/SR_DATA"

# Where to save figures
FIG_DIR = "pareto_figures"


# ----------------------------
# vLLM helpers
# ----------------------------
@dataclass
class VLLMModelRunCfg:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096


def unload_model(llm: LLM) -> None:
    try:
        if hasattr(llm, "llm_engine"):
            del llm.llm_engine
    except Exception:
        pass
    del llm
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


def batch_apply_chat_template(problems, tokenizer):
    prompt_store = []
    for problem in problems:
        messages = [{"role": "user", "content": problem}]
        prompt_store.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompt_store


def count_input_tokens_batch(prompts: List[str], tokenizer) -> List[int]:
    enc = tokenizer(prompts, add_special_tokens=False)
    return [len(ids) for ids in enc["input_ids"]]


TOKENS_PER_MILLION = 1_000_000


# ----------------------------
# Routing
# ----------------------------
def route_questions(predicted_score: float, model_pool: List[str]) -> str:
    if predicted_score >= 0.9:
        return model_pool[0]  # easy
    elif predicted_score >= 0.4:
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
# Self-consistency vote (NO LABELS)
# ----------------------------
def _normalize_answer_for_vote(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    ans = str(ans).strip()
    return ans if ans else None


def majority_vote_from_samples(
    sample_texts: List[str],
    *,
    extract_answer_fn: Callable[[str], Optional[str]],
) -> Tuple[str, Optional[str], int]:
    answers = [_normalize_answer_for_vote(extract_answer_fn(t)) for t in sample_texts]

    if any(a is not None for a in answers):
        keys = [a if a is not None else "__NO_ANSWER__" for a in answers]
        winner_key, _ = Counter(keys).most_common(1)[0]
        winner_idx = next(i for i, k in enumerate(keys) if k == winner_key)
        chosen_text = sample_texts[winner_idx]
        chosen_answer = None if winner_key == "__NO_ANSWER__" else winner_key
        return chosen_text, chosen_answer, winner_idx

    norm_texts = [t.strip() for t in sample_texts]
    winner_text, _ = Counter(norm_texts).most_common(1)[0]
    winner_idx = next(i for i, nt in enumerate(norm_texts) if nt == winner_text)
    return sample_texts[winner_idx], None, winner_idx


# ----------------------------
# Routed inference with per-row n/temp
# ----------------------------
def run_routed_vllm_inference(
    df: pd.DataFrame,
    *,
    route_col: str,
    prompt_col: str,
    out_text_col: str = "routed_response_text",
    out_model_col: str = "response_model",
    prompt_text_col: str = "prompt_text",
    input_num_tokens_col: str = "input_num_tokens",
    out_tok_col: str = "response_num_tokens",          # TOTAL output tokens across all samples
    out_latency_col: str = "response_latency_s",
    out_err_col: str = "response_error",
    input_cost_col: str = "input_cost_usd",
    output_cost_col: str = "output_cost_usd",
    total_cost_col: str = "total_cost_usd",
    pricing_config: Optional[dict] = None,
    max_tokens: int = 3000,
    batch_size_by_model: Optional[dict] = None,
    checkpoint_path: Optional[str] = None,
    model_run_cfgs: Optional[Dict[str, VLLMModelRunCfg]] = None,
    n_col: str = "sc_n",
    temperature_col: str = "sc_temp",
    majority_vote: bool = True,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    store_all_samples_col: Optional[str] = None,
    charge_input_per_sample: bool = False,
) -> pd.DataFrame:
    if model_run_cfgs is None:
        model_run_cfgs = {}
    if pricing_config is None:
        pricing_config = {}
    if majority_vote and extract_answer_fn is None:
        raise ValueError("majority_vote=True requires extract_answer_fn")
    if batch_size_by_model is None:
        batch_size_by_model = {}

    for col, default in [
        (prompt_text_col, None),
        (out_text_col, None),
        (out_model_col, None),
        (input_num_tokens_col, np.nan),
        (out_tok_col, np.nan),
        (out_latency_col, np.nan),
        (out_err_col, None),
        (input_cost_col, np.nan),
        (output_cost_col, np.nan),
        (total_cost_col, np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    if store_all_samples_col is not None and store_all_samples_col not in df.columns:
        df[store_all_samples_col] = None

    pending_mask = df[out_text_col].isna()
    if pending_mask.sum() == 0:
        print("✅ Nothing to do: all rows already have responses.")
        return df

    routes = df.loc[pending_mask, route_col].dropna().unique().tolist()
    print(f"Routes to run: {routes}")

    for model_name in routes:
        default_bs = 256
        if "72" in model_name:
            default_bs = 32
        batch_size = batch_size_by_model.get(model_name, default_bs)

        model_mask = pending_mask & (df[route_col] == model_name)
        idxs = df.index[model_mask].tolist()
        if not idxs:
            continue

        cfg = model_run_cfgs.get(model_name, VLLMModelRunCfg())
        print(f"\n=== Running model: {model_name} | rows: {len(idxs)} | batch_size={batch_size} ===")
        print(f"vLLM cfg: {cfg}")

        model_costs = pricing_config.get(model_name, {}).get("model_costs", {})
        in_rate = model_costs.get("input_per_mill", None)
        out_rate = model_costs.get("output_per_mill", None)
        has_pricing = (in_rate is not None) and (out_rate is not None)

        if "72" in model_name:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_model_len=cfg.max_model_len,
                max_num_seqs=64,
                max_num_batched_tokens=8192,
            )
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_model_len=cfg.max_model_len,
            )

        tokenizer = llm.llm_engine.tokenizer.tokenizer

        sub = df.loc[idxs, [n_col, temperature_col]].copy()
        sub[n_col] = sub[n_col].astype(int)
        sub[temperature_col] = sub[temperature_col].astype(float)

        for (n_val, temp_val), sub_idx_df in sub.groupby([n_col, temperature_col]):
            group_idxs = sub_idx_df.index.tolist()
            sampling = SamplingParams(
                temperature=float(temp_val),
                max_tokens=max_tokens,
                n=int(n_val),
            )

            print(f"  -> group n={n_val} temp={temp_val} rows={len(group_idxs)}")

            for start in tqdm(range(0, len(group_idxs), batch_size), desc=f"{model_name} n={n_val}"):
                batch_idxs = group_idxs[start : start + batch_size]
                problems = df.loc[batch_idxs, prompt_col].tolist()

                prompts = batch_apply_chat_template(problems, tokenizer)
                input_tok_counts = count_input_tokens_batch(prompts, tokenizer)
                df.loc[batch_idxs, prompt_text_col] = prompts

                t0 = time.time()
                try:
                    outputs = llm.generate(prompts, sampling_params=sampling)
                    latency = time.time() - t0

                    chosen_texts: List[str] = []
                    total_out_tok_counts: List[int] = []
                    all_samples_json: List[str] = []
                    errs = [None] * len(outputs)

                    for req_out in outputs:
                        sample_texts = [o.text for o in req_out.outputs]
                        sample_tok_counts = [
                            (len(o.token_ids) if o.token_ids is not None else 0) for o in req_out.outputs
                        ]
                        total_out_tok_counts.append(int(np.sum(sample_tok_counts)))

                        if store_all_samples_col is not None:
                            all_samples_json.append(json.dumps(sample_texts, ensure_ascii=False))

                        if majority_vote and n_val > 1:
                            chosen_text, _, _ = majority_vote_from_samples(
                                sample_texts, extract_answer_fn=extract_answer_fn
                            )
                            chosen_texts.append(chosen_text)
                        else:
                            chosen_texts.append(sample_texts[0])

                    df.loc[batch_idxs, out_text_col] = chosen_texts
                    df.loc[batch_idxs, out_model_col] = model_name
                    df.loc[batch_idxs, input_num_tokens_col] = input_tok_counts
                    df.loc[batch_idxs, out_tok_col] = total_out_tok_counts
                    df.loc[batch_idxs, out_latency_col] = latency
                    df.loc[batch_idxs, out_err_col] = errs
                    if store_all_samples_col is not None:
                        df.loc[batch_idxs, store_all_samples_col] = all_samples_json

                    if has_pricing:
                        in_arr = np.array(input_tok_counts, dtype=float)
                        out_arr = np.array(total_out_tok_counts, dtype=float)
                        if charge_input_per_sample and n_val > 1:
                            in_arr = in_arr * float(n_val)

                        input_costs = (in_arr / TOKENS_PER_MILLION) * float(in_rate)
                        output_costs = (out_arr / TOKENS_PER_MILLION) * float(out_rate)
                        df.loc[batch_idxs, input_cost_col] = input_costs
                        df.loc[batch_idxs, output_cost_col] = output_costs
                        df.loc[batch_idxs, total_cost_col] = input_costs + output_costs

                except Exception as e:
                    latency = time.time() - t0
                    df.loc[batch_idxs, out_text_col] = None
                    df.loc[batch_idxs, out_model_col] = model_name
                    df.loc[batch_idxs, input_num_tokens_col] = np.nan
                    df.loc[batch_idxs, out_tok_col] = np.nan
                    df.loc[batch_idxs, out_latency_col] = latency
                    df.loc[batch_idxs, out_err_col] = repr(e)
                    df.loc[batch_idxs, input_cost_col] = np.nan
                    df.loc[batch_idxs, output_cost_col] = np.nan
                    df.loc[batch_idxs, total_cost_col] = np.nan
                    if store_all_samples_col is not None:
                        df.loc[batch_idxs, store_all_samples_col] = None

                if checkpoint_path is not None:
                    df.to_parquet(checkpoint_path, index=True)

        unload_model(llm)
        pending_mask = df[out_text_col].isna()
        if checkpoint_path is not None:
            df.to_parquet(checkpoint_path, index=True)

    return df


# ----------------------------
# Pareto plotting helpers
# ----------------------------
def pareto_efficient_min_cost_max_score(costs, scores, eps=1e-12):
    costs = np.asarray(costs, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(costs)
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_eff[i]:
            continue
        dominated = (costs <= costs[i] + eps) & (scores >= scores[i] - eps) & (
            (costs < costs[i] - eps) | (scores > scores[i] + eps)
        )
        if np.any(dominated):
            is_eff[i] = False
    return is_eff


def plot_pareto_frontier_and_save(
    points: List[dict],
    *,
    title: str,
    out_png: str,
    out_pdf: Optional[str] = None,
    x_label: str = "Total eval cost (USD)",
    y_label: str = "Pass@1",
):
    """
    points: list of dicts:
      {
        "name": str,
        "cost": float,
        "score": float,
        "kind": "model"|"router",
        "k": int|None,
        "t": float|None
      }
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    names = [p["name"] for p in points]
    costs = np.array([p["cost"] for p in points], dtype=float)
    scores = np.array([p["score"] for p in points], dtype=float)

    mask = pareto_efficient_min_cost_max_score(costs, scores)
    pareto_pts = sorted([p for p, m in zip(points, mask) if m], key=lambda d: d["cost"])

    uniq_names = list(dict.fromkeys(names))
    cmap = plt.get_cmap("tab20")
    color_map = {n: cmap(i % cmap.N) for i, n in enumerate(uniq_names)}

    fig, ax = plt.subplots(figsize=(9, 5))

    for n in uniq_names:
        pts_n = [p for p in points if p["name"] == n]
        kind = pts_n[0].get("kind", "model")
        marker = "*" if kind == "router" else "o"
        size = 180 if kind == "router" else 70
        edge = "black" if kind == "router" else None

        ax.scatter(
            [p["cost"] for p in pts_n],
            [p["score"] for p in pts_n],
            label=n,
            marker=marker,
            s=size,
            c=[color_map[n]],
            edgecolors=edge,
            linewidths=1.0 if kind == "router" else 0.0,
            alpha=0.9,
            zorder=3 if kind == "router" else 2,
        )

    if len(pareto_pts) >= 2:
        ax.plot([p["cost"] for p in pareto_pts], [p["score"] for p in pareto_pts], linewidth=1.5, zorder=1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)

    # keep costs readable
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def find_og_parquet_path(ds_alias: str, model_alias: str, base_dir: str = SR_DATA_BASEDIR) -> str:
    """
    Current baselines only: greedy k=1 temp=0.0
    Your existing naming uses:
      test-{alias}_maxlen_3000_k_1_temp_0.0.parquet
      train-{alias}_maxlen_3000_k_1_temp_0.0.parquet  (for some AIME)
    """
    candidates = []
    if "aime2025" in ds_alias.lower():
        candidates.append(f"{base_dir}/{ds_alias}/test-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")
    elif "aime" in ds_alias.lower():
        candidates.append(f"{base_dir}/{ds_alias}/train-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")
        candidates.append(f"{base_dir}/{ds_alias}/test-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")
    elif "math" in ds_alias.lower():
        candidates.append(f"{base_dir}/MATH/test-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")
        candidates.append(f"{base_dir}/MATH/train-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")
    else:
        candidates.append(f"{base_dir}/{ds_alias}/test-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")
        candidates.append(f"{base_dir}/{ds_alias}/train-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet")

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing OG parquet for ds={ds_alias} model={model_alias}. Tried:\n" + "\n".join(candidates))


def aggregate_og_benchmark(og_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Aggregate using the OG parquet's columns.
    Works whether it's per-example or already aggregated (single row).
    """
    if "success_rate" not in og_df.columns or "total_cost_usd" not in og_df.columns:
        raise KeyError(f"OG df missing required columns. Have={list(og_df.columns)}")
    score = float(og_df["success_rate"].mean())
    cost = float(og_df["total_cost_usd"].sum())
    return score, cost


# ----------------------------
# Main
# ----------------------------
def main():
    repo_root = Path.cwd().parent
    sys.path.insert(0, str(repo_root))

    from will_replication.my_utils.utils import (
        load_labelled_probe_dataset,
        SIMPLE_MODEL_POOL_CONFIG,
    )
    from thom_replication.utils.verification_math import (
        try_extract_solution,
        extract_solution,
        extract_gsm8k_solution,
        compute_score,
    )

    os.makedirs(FIG_DIR, exist_ok=True)

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
        LABELLED_DATASET_NAME = "_".join(LABELLED_DATASET_FULL_NAME.split("/"))

        # --------- LOAD LABELLED PROBE FILES (probe temp/k) ----------
        try:
            routing_dataset_df = load_labelled_probe_dataset(
                MODEL_NAME=PROBE_MODEL_NAME,
                PROBE_SOURCE_DATASET=PROBING_DATASET,
                LABELLED_DATASET=LABELLED_DATASET_NAME,
                K=PROBE_K,
                TEMPERATURE=PROBE_TEMP,
                DATA_PATH=LABELLED_SR_PATH,
            )
        except Exception:
            print(f"❌ dataset config doesn't exist for {LABELLED_DATASET_NAME} @ (K={PROBE_K}, T={PROBE_TEMP})")
            continue

        print("\n\n==============================")
        print(f"Dataset: {LABELLED_DATASET_NAME}")
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

        ckpt_path = f"{LABELLED_DATASET_NAME}_routed_probeK{PROBE_K}_probeT{PROBE_TEMP}_routeK{ROUTING_K}_routeT{ROUTING_TEMP}_ckpt.parquet"

        # --------- RUN ROUTED INFERENCE ----------
        routing_dataset_df = run_routed_vllm_inference(
            routing_dataset_df,
            route_col="route_to",
            prompt_col="prompt_scored",
            out_text_col="routed_response_text",
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

        # save routed parquet
        out_path = (
            f"{LABELLED_DATASET_NAME}_routed_by_{PROBING_DATASET}_{PROBE_MODEL_ALIAS}"
            f"_probeK{PROBE_K}_probeT{PROBE_TEMP}"
            f"_routeK{ROUTING_K}_routeT{ROUTING_TEMP}"
            f"_sc_final.parquet"
        )
        routing_dataset_df.to_parquet(out_path, index=True)
        print(f"✅ Done. Saved routed df: {out_path}")

        # ----------------------------
        # Compute ROUTER performance for Pareto
        # ----------------------------
        # IMPORTANT: evaluate routed correctness here (uses labels AFTER inference, not during)
        # This does not leak into routing since it happens post-hoc.
        if "gsm8k" in LABELLED_DATASET_FULL_NAME.lower():
            gts = routing_dataset_df["original_solution"].apply(extract_gsm8k_solution).tolist()
        elif "aime" in LABELLED_DATASET_FULL_NAME.lower():
            gts = routing_dataset_df["original_solution"].tolist()
        else:
            gts = routing_dataset_df["original_solution"].apply(extract_solution).tolist()

        preds = routing_dataset_df["routed_response_text"].tolist()
        routed_is_correct = [compute_score(solution_str=p, ground_truth=g) for p, g in zip(preds, gts)]
        routing_dataset_df["routed_is_correct"] = routed_is_correct

        router_score = float(np.mean(routed_is_correct))
        router_cost = float(routing_dataset_df["total_cost_usd"].sum())

        # ----------------------------
        # Collect baseline points (greedy k=1, t=0.0) + router point
        # ----------------------------
        ds_alias = LABELLED_DATASET_NAME  # already "_" joined
        points: List[dict] = []

        # Baselines from SR_DATA
        for og_model_name in model_pool:
            og_model_alias = "-".join(og_model_name.split("/"))
            og_path = find_og_parquet_path(ds_alias, og_model_alias, base_dir=SR_DATA_BASEDIR)
            og_df = pd.read_parquet(og_path)

            og_score, og_cost = aggregate_og_benchmark(og_df)
            points.append({
                "name": og_model_alias,
                "kind": "model",
                "cost": og_cost,
                "score": og_score,
                "k": BASELINE_K,
                "t": BASELINE_T,
                "path": og_path,
            })

        # Router point (with metadata)
        points.append({
            "name": "ROUTER",
            "kind": "router",
            "cost": router_cost,
            "score": router_score,
            "k": ROUTING_K,
            "t": ROUTING_TEMP,
            "path": out_path,
        })

        # Print points for sanity
        print("\nPareto points (what will be plotted):")
        print(pd.DataFrame(points)[["name", "kind", "cost", "score", "k", "t"]].sort_values("cost").to_string(index=False))

        # ----------------------------
        # Plot + save
        # ----------------------------
        title = f"Pareto frontier — {LABELLED_DATASET_FULL_NAME} (cost↓, pass@1↑)"
        stem = (
            f"{ds_alias}"
            f"_probeK{PROBE_K}_probeT{PROBE_TEMP}"
            f"_routeK{ROUTING_K}_routeT{ROUTING_TEMP}"
        )
        out_png = os.path.join(FIG_DIR, f"{stem}.png")
        # out_pdf = os.path.join(FIG_DIR, f"{stem}.pdf")
        out_pdf = None

        plot_pareto_frontier_and_save(
            points,
            title=title,
            out_png=out_png,
            out_pdf=out_pdf,
            x_label="Total eval cost (USD)",
            y_label="Pass@1",
        )
        print(f"🖼️ Saved Pareto figures: {out_png} and {out_pdf}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
