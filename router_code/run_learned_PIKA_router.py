"""
run_router_learned.py

End-to-end script to run a *learned* router on MATH (or other datasets) where:
- You have a probe-labelled dataset with predicted SR@K (e.g., SR@50) in column `calibrated_score`
- You want to route examples to actions (model, n, temp) to match a target baseline
  (here: 7B greedy k=1, t=0.0) at minimum expected cost.
- No label leakage: routing + majority-vote selection uses NO ground-truth labels.
- Labels are only used for *post-hoc evaluation* (Pass@1).

Key idea:
1) Fit per-model calibrators:  p50_hat  ->  P(correct | model, greedy)
   (isotonic regression on aligned prompts vs OG greedy outputs)
2) Define actions (model, n, temp) with expected cost (from OG token stats + pricing).
3) Per example, pick cheapest action s.t. predicted pass@n >= baseline_acc - slack.
4) Run vLLM grouped by (model, n, temp) + majority vote on extracted answer.
5) Save results + Pareto plots.

Assumptions:
- `load_labelled_probe_dataset` available at will_replication.my_utils.utils
- `SIMPLE_MODEL_POOL_CONFIG` provides per-model costs {input_per_mill, output_per_mill}
- `try_extract_solution`, `extract_solution`, `extract_gsm8k_solution`, `compute_score` exist
  (use your existing utilities)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import gc
import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.isotonic import IsotonicRegression


TOKENS_PER_MILLION = 1_000_000


# =========================
# Config (edit me)
# =========================
LABELLED_DATASETS_LIST = [
    "DigitalLearningGmbH/MATH-lighteval",
    # "opencompass/AIME2025",
    # "gneubig/aime-1983-2024",
    # "openai/gsm8k",
]

PROBING_DATASET = "MATH"
PROBE_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

# Probe-labelled file config (this selects which labelled parquet to load)
# For your SR@50 @ temp=0.6:
PROBE_K = 50
PROBE_TEMP = 0.6

# Routing inference config (what vLLM uses)
ROUTING_TEMP = 0.6
MAX_TOKENS = 3000

# Target baseline to match: 7B greedy k=1, t=0.0
BASELINE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
BASELINE_K = 1
BASELINE_TEMP = 0.0
SLACK = 0.005  # router targets baseline_acc - slack

# Candidate self-consistency n values per model (extend later)
N_CHOICES_BY_MODEL = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": [1, 3],
    "Qwen/Qwen2.5-Math-7B-Instruct":   [1, 3, 5],
    "Qwen/Qwen2.5-Math-72B-Instruct":  [1, 5],
}

# Cost semantics
CHARGE_INPUT_PER_SAMPLE = False  # default: input tokens charged once, output scales with n

# vLLM execution
BATCH_SIZE_BY_MODEL = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
    "Qwen/Qwen2.5-Math-7B-Instruct":   128,
    "Qwen/Qwen2.5-Math-72B-Instruct":   64,
}
MODEL_RUN_CFGS = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": {"tp": 1, "mem": 0.60, "maxlen": 4096},
    "Qwen/Qwen2.5-Math-7B-Instruct":   {"tp": 1, "mem": 0.70, "maxlen": 4096},
    "Qwen/Qwen2.5-Math-72B-Instruct":  {"tp": 2, "mem": 0.92, "maxlen": 4096},
}

# Paths
SR_DATA_BASEDIR = "../will_replication/DATA/SR_DATA"
PROBE_RESULTS_DIR = "../will_replication/probe_results/DATA"
LABELLED_SR_PATH = f"{PROBE_RESULTS_DIR}/Labelled_SR"
OUT_DIR = "router_runs"
FIG_DIR = "router_figs"


# =========================
# vLLM helpers
# =========================
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


def run_routed_vllm_inference(
    df: pd.DataFrame,
    *,
    route_col: str,
    prompt_col: str,
    out_text_col: str = "routed_response_text",
    out_model_col: str = "response_model",
    prompt_text_col: str = "prompt_text",
    input_num_tokens_col: str = "input_num_tokens",
    out_tok_col: str = "response_num_tokens",      # TOTAL output tokens across all samples
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
    if batch_size_by_model is None:
        batch_size_by_model = {}
    if majority_vote and extract_answer_fn is None:
        raise ValueError("majority_vote=True requires extract_answer_fn")

    # ensure output columns exist
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
        default_bs = 256 if "72" not in model_name else 32
        batch_size = int(batch_size_by_model.get(model_name, default_bs))

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

        # init LLM once per model
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

        # group by (n, temperature)
        sub = df.loc[idxs, [n_col, temperature_col]].copy()
        sub[n_col] = sub[n_col].astype(int)
        sub[temperature_col] = sub[temperature_col].astype(float)

        for (n_val, temp_val), sub_idx_df in sub.groupby([n_col, temperature_col]):
            group_idxs = sub_idx_df.index.tolist()
            sampling = SamplingParams(temperature=float(temp_val), max_tokens=max_tokens, n=int(n_val))
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
                    errs = [None] * len(outputs)
                    all_samples_json: List[str] = []

                    for req_out in outputs:
                        sample_texts = [o.text for o in req_out.outputs]
                        sample_tok_counts = [(len(o.token_ids) if o.token_ids is not None else 0) for o in req_out.outputs]
                        total_out_tok_counts.append(int(np.sum(sample_tok_counts)))

                        if store_all_samples_col is not None:
                            all_samples_json.append(json.dumps(sample_texts, ensure_ascii=False))

                        if majority_vote and n_val > 1:
                            chosen_text, _, _ = majority_vote_from_samples(sample_texts, extract_answer_fn=extract_answer_fn)
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


# =========================
# Learned router: calibrate p50 -> pass@1, then choose cheapest action meeting baseline
# =========================
def _norm_text(x: str) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())

def make_key_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(_norm_text).map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())

def make_train_val_split(df: pd.DataFrame, frac_train: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(frac_train * len(df))
    tr = df.iloc[idx[:cut]]
    va = df.iloc[idx[cut:]]
    return tr, va

def fit_isotonic_calibrator(df_join: pd.DataFrame, p_col: str, y_col: str):
    x = np.clip(df_join[p_col].astype(float).to_numpy(), 0.0, 1.0)
    y = np.clip(df_join[y_col].astype(float).to_numpy(), 0.0, 1.0)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)
    return iso

def find_og_greedy_path(ds_alias: str, model_alias: str, base_dir: str = SR_DATA_BASEDIR):
    # Your SR_DATA convention
    if ds_alias.lower().startswith("digitallearninggmbh_math-lighteval") or "math" in ds_alias.lower():
        candidates = [
            f"{base_dir}/MATH/test-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet",
            f"{base_dir}/MATH/train-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet",
        ]
    else:
        candidates = [
            f"{base_dir}/{ds_alias}/test-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet",
            f"{base_dir}/{ds_alias}/train-{model_alias}_maxlen_3000_k_1_temp_0.0.parquet",
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing OG greedy parquet for ds={ds_alias}, model={model_alias}. Tried:\n" + "\n".join(candidates))

def load_og_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # prefer raw problem text if available
    prompt_col = None
    # for c in ["problem", "formatted_prompt", "prompt", "prompt_text"]:
    for c in ["formatted_prompt"]:
        if c in df.columns:
            prompt_col = c
            break
    if prompt_col is None:
        raise KeyError(f"OG df missing prompt col. cols={list(df.columns)}")
    df = df.copy()
    df["_key"] = make_key_series(df[prompt_col])
    if "success_rate" not in df.columns:
        raise KeyError(f"OG df missing success_rate. cols={list(df.columns)}")
    df["_y"] = df["success_rate"].astype(float).clip(0.0, 1.0)
    if "total_input_tokens" not in df.columns or "total_output_tokens" not in df.columns:
        raise KeyError(f"Need total_input_tokens and total_output_tokens in OG df for cost modeling. cols={list(df.columns)}")
    return df

def estimate_token_stats(og_df: pd.DataFrame) -> Tuple[float, float]:
    return float(og_df["total_input_tokens"].mean()), float(og_df["total_output_tokens"].mean())

def action_cost_per_example(model_name: str, n: int, pricing_config: dict, avg_in_tok: float, avg_out_tok: float, charge_input_per_sample: bool):
    rates = pricing_config[model_name]["model_costs"]
    in_rate = float(rates["input_per_mill"])
    out_rate = float(rates["output_per_mill"])
    in_tok = avg_in_tok * (n if charge_input_per_sample else 1.0)
    out_tok = avg_out_tok * float(n)
    return float((in_tok / TOKENS_PER_MILLION) * in_rate + (out_tok / TOKENS_PER_MILLION) * out_rate)

def build_actions(model_pool: list[str], pricing_config: dict, routing_temp: float, n_choices_by_model: dict, token_stats: dict, charge_input_per_sample: bool):
    actions = []
    for m in model_pool:
        avg_in_tok, avg_out_tok = token_stats[m]
        for n in n_choices_by_model[m]:
            actions.append({
                "model": m,
                "n": int(n),
                "temp": float(routing_temp),
                "cost_per_ex": action_cost_per_example(m, int(n), pricing_config, avg_in_tok, avg_out_tok, charge_input_per_sample)
            })
    return sorted(actions, key=lambda a: a["cost_per_ex"])

def route_min_cost_to_hit_target(
    df: pd.DataFrame,
    *,
    p50_col: str,
    baseline_acc: float,
    slack: float,
    actions: list[dict],
    calibrators_by_model: dict[str, IsotonicRegression],
):
    p_target = float(baseline_acc - slack)
    p50 = np.clip(df[p50_col].astype(float).to_numpy(), 0.0, 1.0)

    routes, ns, ts, exp_costs, phats = [], [], [], [], []
    for p in p50:
        chosen = None
        chosen_phat = None
        for a in actions:
            m, n = a["model"], a["n"]
            p1 = float(calibrators_by_model[m].predict([p])[0])
            pn = 1.0 - (1.0 - p1) ** float(n)  # proxy uplift for SC
            if pn >= p_target:
                chosen = a
                chosen_phat = pn
                break
        if chosen is None:
            chosen = actions[-1]
            m, n = chosen["model"], chosen["n"]
            p1 = float(calibrators_by_model[m].predict([p])[0])
            chosen_phat = 1.0 - (1.0 - p1) ** float(n)

        routes.append(chosen["model"])
        ns.append(int(chosen["n"]))
        ts.append(float(chosen["temp"]))
        exp_costs.append(float(chosen["cost_per_ex"]))
        phats.append(float(chosen_phat))

    out = df.copy()
    out["route_to"] = routes
    out["sc_n"] = ns
    out["sc_temp"] = ts
    out["route_expected_cost_usd"] = exp_costs
    out["route_predicted_pass_at_n"] = phats
    out["baseline_target_acc"] = float(baseline_acc)
    out["routing_slack"] = float(slack)
    return out


# =========================
# Pareto plotting + save
# =========================
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

def plot_pareto_frontier(points, title, save_path: str):
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
        ax.scatter([p["cost"] for p in pts_n],
                   [p["score"] for p in pts_n],
                   label=n,
                   marker=marker,
                   s=size,
                   c=[color_map[n]],
                   edgecolors=edge,
                   linewidths=1.0 if kind == "router" else 0.0,
                   alpha=0.9)

    if len(pareto_pts) >= 2:
        ax.plot([p["cost"] for p in pareto_pts], [p["score"] for p in pareto_pts], linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Total eval cost (USD)")
    ax.set_ylabel("Pass@1")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    repo_root = Path.cwd().parent
    import sys
    sys.path.insert(0, str(repo_root))

    from will_replication.my_utils.utils import load_labelled_probe_dataset, SIMPLE_MODEL_POOL_CONFIG
    from thom_replication.utils.verification_math import (
        try_extract_solution,
        extract_solution,
        extract_gsm8k_solution,
        compute_score,
    )

    model_pool = list(SIMPLE_MODEL_POOL_CONFIG.keys())
    assert BASELINE_MODEL in model_pool, "BASELINE_MODEL must be in SIMPLE_MODEL_POOL_CONFIG"

    model_run_cfgs = {
        m: VLLMModelRunCfg(
            tensor_parallel_size=int(MODEL_RUN_CFGS[m]["tp"]),
            gpu_memory_utilization=float(MODEL_RUN_CFGS[m]["mem"]),
            max_model_len=int(MODEL_RUN_CFGS[m]["maxlen"]),
        )
        for m in model_pool
        if m in MODEL_RUN_CFGS
    }

    for ds_full in LABELLED_DATASETS_LIST:
        ds_alias = "_".join(ds_full.split("/"))
        print("\n" + "=" * 90)
        print(f"DATASET: {ds_full} ({ds_alias})")
        print(f"Probe config  : K={PROBE_K} T={PROBE_TEMP}")
        print(f"Route config  : temp={ROUTING_TEMP} (n chosen per-example)")
        print(f"Baseline      : {BASELINE_MODEL} greedy (k={BASELINE_K}, t={BASELINE_TEMP})")
        print("=" * 90)

        # ----- load probe-labelled dataset (contains predicted SR@K) -----
        probe_df = load_labelled_probe_dataset(
            MODEL_NAME=PROBE_MODEL_NAME,
            PROBE_SOURCE_DATASET=PROBING_DATASET,
            LABELLED_DATASET=ds_alias,
            K=PROBE_K,
            TEMPERATURE=PROBE_TEMP,
            DATA_PATH=LABELLED_SR_PATH,
        ).copy()
        print("Probe columns:\n")
        print(probe_df.columns)

        # choose p50 column
        p_col = "calibrated_score" if PROBE_K > 1 else "score"
        if p_col not in probe_df.columns:
            raise KeyError(f"Expected {p_col} in probe df columns. got={list(probe_df.columns)}")

        # prompt column for keying
        prompt_col = None
        # for c in ["problem", "prompt_scored", "prompt", "prompt_text"]:
        for c in ["formatted"]:
            if c in probe_df.columns:
                prompt_col = c
                break
        if prompt_col is None:
            raise KeyError(f"Probe df missing prompt col. cols={list(probe_df.columns)}")

        probe_df["_key"] = make_key_series(probe_df[prompt_col])

        # ----- fit calibrators p50 -> P(correct|model greedy) using OG greedy outputs -----
        calibrators: Dict[str, IsotonicRegression] = {}
        token_stats: Dict[str, Tuple[float, float]] = {}
        baseline_acc = None

        # Load OG greedy df per model and join via key
        for m in model_pool:
            m_alias = "-".join(m.split("/"))
            og_path = find_og_greedy_path(ds_alias, m_alias, base_dir=SR_DATA_BASEDIR)
            og_df = load_og_df(og_path)

            print("og df cols:\n")
            print(og_df.columns)

            print(og_df.head())

            token_stats[m] = estimate_token_stats(og_df)

            joined = probe_df[["_key", p_col]].merge(og_df[["_key", "_y"]], on="_key", how="inner")
            if len(joined) < 1000:
                print(f"⚠️  Low join count for {m_alias}: matched={len(joined)}. Check prompt formatting alignment.")
            tr, va = make_train_val_split(joined, frac_train=0.7, seed=0)
            iso = fit_isotonic_calibrator(tr, p_col=p_col, y_col="_y")
            calibrators[m] = iso

            va_pred = iso.predict(np.clip(va[p_col].astype(float).to_numpy(), 0, 1))
            print(f"[calib] {m_alias:30s} val_pred_mean={va_pred.mean():.3f} val_y_mean={va['_y'].mean():.3f} n={len(va)}")

            if m == BASELINE_MODEL:
                # baseline accuracy measured on the joined set for alignment
                baseline_acc = float(joined["_y"].mean())

        if baseline_acc is None:
            raise RuntimeError("Failed to compute baseline_acc for BASELINE_MODEL (join may be broken).")

        print(f"\nBaseline accuracy (aligned): {baseline_acc:.4f}  | target >= {baseline_acc - SLACK:.4f}")

        # ----- build action set (model, n, temp) with expected cost -----
        actions = build_actions(
            model_pool=model_pool,
            pricing_config=SIMPLE_MODEL_POOL_CONFIG,
            routing_temp=ROUTING_TEMP,
            n_choices_by_model=N_CHOICES_BY_MODEL,
            token_stats=token_stats,
            charge_input_per_sample=CHARGE_INPUT_PER_SAMPLE,
        )
        print("Cheapest actions (top 6):")
        for a in actions[:6]:
            print(a)

        # ----- route: per-example choose min-cost action meeting baseline -----
        routed_df = route_min_cost_to_hit_target(
            probe_df,
            p50_col=p_col,
            baseline_acc=baseline_acc,
            slack=SLACK,
            actions=actions,
            calibrators_by_model=calibrators,
        )

        print("\nRoute_to breakdown:")
        print(routed_df["route_to"].value_counts())
        print("\nsc_n breakdown:")
        print(routed_df["sc_n"].value_counts())
        print(f"\nExpected cost (sum): {routed_df['route_expected_cost_usd'].sum():.4f}")

        # ----- run vLLM inference per (model, n, temp) -----
        ckpt_path = os.path.join(
            OUT_DIR,
            f"{ds_alias}_learnedrouter_probeK{PROBE_K}_probeT{PROBE_TEMP}_routeT{ROUTING_TEMP}_ckpt.parquet",
        )
        routed_df["routed_response_text"] = np.nan  # ensure pending
        routed_df = run_routed_vllm_inference(
            routed_df,
            route_col="route_to",
            prompt_col=prompt_col,
            out_text_col="routed_response_text",
            max_tokens=MAX_TOKENS,
            batch_size_by_model=BATCH_SIZE_BY_MODEL,
            checkpoint_path=ckpt_path,
            pricing_config=SIMPLE_MODEL_POOL_CONFIG,
            model_run_cfgs=model_run_cfgs,
            n_col="sc_n",
            temperature_col="sc_temp",
            majority_vote=True,
            extract_answer_fn=try_extract_solution,  # NO LABELS used here
            store_all_samples_col=None,
            charge_input_per_sample=CHARGE_INPUT_PER_SAMPLE,
        )

        # ----- post-hoc evaluation (labels OK) -----
        # Build extracted GTs depending on dataset type
        if "gsm8k" in ds_full.lower():
            routed_df["extracted_gts"] = routed_df["original_solution"].apply(extract_gsm8k_solution)
            extract_fn_for_pred = try_extract_solution
        elif "aime" in ds_full.lower():
            routed_df["extracted_gts"] = routed_df["original_solution"]
            extract_fn_for_pred = try_extract_solution
        else:
            routed_df["extracted_gts"] = routed_df["original_solution"].apply(extract_solution)
            extract_fn_for_pred = try_extract_solution

        # score correctness
        resp_list = routed_df["routed_response_text"].tolist()
        gt_list = routed_df["extracted_gts"].tolist()
        is_corr = []
        for r, g in zip(resp_list, gt_list):
            is_corr.append(compute_score(solution_str=r, ground_truth=g))
        routed_df["routed_is_correct"] = is_corr

        routed_pass1 = float(np.mean(is_corr))
        routed_cost = float(routed_df["total_cost_usd"].sum())
        print(f"\nLEARNED ROUTER results: pass@1={routed_pass1:.4f} cost={routed_cost:.4f}")

        # save parquet
        out_path = os.path.join(
            OUT_DIR,
            f"{ds_alias}_learnedrouter_probeK{PROBE_K}_probeT{PROBE_TEMP}_routeT{ROUTING_TEMP}_final.parquet",
        )
        routed_df.to_parquet(out_path, index=True)
        print(f"✅ Saved routed df: {out_path}")

        # ----- build Pareto points: OG greedy (k=1,t=0) + router -----
        points = []
        for m in model_pool:
            m_alias = "-".join(m.split("/"))
            og_path = find_og_greedy_path(ds_alias, m_alias, base_dir=SR_DATA_BASEDIR)
            og_df = pd.read_parquet(og_path)
            og_score = float(og_df["success_rate"].mean())
            og_cost = float(og_df["total_cost_usd"].sum())
            points.append({"name": m_alias, "cost": og_cost, "score": og_score, "kind": "model"})

        points.append({"name": "ROUTER", "cost": routed_cost, "score": routed_pass1, "kind": "router"})

        fig_path = os.path.join(
            FIG_DIR,
            f"pareto_{ds_alias}_learnedrouter_probeK{PROBE_K}_probeT{PROBE_TEMP}_routeT{ROUTING_TEMP}.png",
        )
        plot_pareto_frontier(points, title=f"Pareto frontier — {ds_full} (cost↓, pass@1↑)", save_path=fig_path)
        print(f"🖼️  Saved Pareto plot: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
