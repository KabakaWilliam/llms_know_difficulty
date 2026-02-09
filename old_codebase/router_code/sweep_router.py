import os, time, json, gc, traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ----------------------------
# ntfy.sh helper (stdlib only)
# ----------------------------
def ntfy_send(message: str, *, title: str = "PIKA sweep", priority: str = "default", tags: str = "rocket"):
    """
    Configure via env vars:
      NTFY_SERVER=https://ntfy.sh
      NTFY_TOPIC=pika_router_sweep
    """
    import urllib.request

    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    topic = os.environ.get("NTFY_TOPIC", "pika_router_sweep")
    url = f"{server}/{topic}"

    data = message.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", priority)
    req.add_header("Tags", tags)
    req.add_header("Content-Type", "text/plain; charset=utf-8")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            _ = resp.read()
    except Exception as e:
        print(f"[ntfy] failed: {e}")

# ----------------------------
# Sweep config
# ----------------------------
@dataclass(frozen=True)
class SweepCfg:
    probe_k: int
    probe_temp: float
    route_temp: float          # {0.0, 0.6}
    k: int                     # 1..5
    easy_thr: float            # {0.9, 0.8}
    med_thr: float = 0.4

def sc_policy_from_cfg(cfg: SweepCfg) -> tuple[dict[str, int], str]:
    if cfg.route_temp == 0.0:
        return {"easy": 1, "medium": 1, "hard": 1}, "greedy"
    # make k actually matter (uniform self-consistency)
    return {"easy": cfg.k, "medium": cfg.k, "hard": cfg.k}, f"scK{cfg.k}"

def route_questions_cfg(score: float, model_pool: list[str], cfg: SweepCfg) -> str:
    if score >= cfg.easy_thr:
        return model_pool[0]
    elif score >= cfg.med_thr:
        return model_pool[1]
    else:
        return model_pool[2]

# ----------------------------
# Main sweep driver
# ----------------------------
def main():
    repo_root = Path.cwd().parent
    import sys
    sys.path.insert(0, str(repo_root))

    from thom_replication.utils.verification_math import (
        extract_gsm8k_solution,
        extract_solution,
        compute_score,
    )

    from will_replication.my_utils.utils import (
        load_labelled_probe_dataset,
        SIMPLE_MODEL_POOL_CONFIG,
        batch_apply_chat_template,
        count_input_tokens_batch,
        unload_model,
        majority_vote_from_samples,
        add_majority_vote_answer,
        try_extract_solution,
        compute_passk_from_json_solutions,
        run_routed_vllm_inference,
        VLLMModelRunCfg,
    )

    # ----------------------------
    # Fixed experiment config
    # ----------------------------
    LABELLED_DATASETS_LIST = [
        "opencompass/AIME2025",
        "gneubig/aime-1983-2024",
        "openai/gsm8k",
        "DigitalLearningGmbH/MATH-lighteval",
    ]

    PROBING_DATASET = "DigitalLearningGmbH_MATH-lighteval"
    PROBE_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

    # Probe-labelled files (fixed for sweep)
    # PROBE_K = 1
    # PROBE_TEMP = 0.0

    MAX_TOKENS = 3000
    STORE_ALL_SAMPLES_COL = "generated_solutions"
    ORIGINAL_GT_COLUMN = "original_solution"
    FINAL_GT_COLUMN = "extracted_gts"
    CHARGE_INPUT_PER_SAMPLE = False
    TOKENS_PER_MILLION = 1_000_000

    batch_size_by_model = {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
        "Qwen/Qwen2.5-Math-7B-Instruct":  256,
        "Qwen/Qwen2.5-Math-72B-Instruct": 128,
    }

    model_run_cfgs = {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": VLLMModelRunCfg(tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=4096),
        "Qwen/Qwen2.5-Math-7B-Instruct":   VLLMModelRunCfg(tensor_parallel_size=1, gpu_memory_utilization=0.70, max_model_len=4096),
        "Qwen/Qwen2.5-Math-72B-Instruct":  VLLMModelRunCfg(tensor_parallel_size=2, gpu_memory_utilization=0.92, max_model_len=4096),
    }

    PROBE_RESULTS_DIR = "../will_replication/probe_results/DATA"
    LABELLED_SR_PATH = f"{PROBE_RESULTS_DIR}/Labelled_SR"

    ROUTER_SAVE_DF_DIR = "pika_router_sweep"
    os.makedirs(ROUTER_SAVE_DF_DIR, exist_ok=True)

    PROBE_MODEL_ALIAS = "_".join(PROBE_MODEL_NAME.split("/"))
    model_pool = list(SIMPLE_MODEL_POOL_CONFIG.keys())

    # ----------------------------
    # Sweep grid
    # ----------------------------
    PROBE_CFGS = [(1, 0.0), (5, 0.6)] #(K, Temp)
    ROUTE_TEMPS = [0.0, 0.6]
    KS = [1, 2, 3, 4, 5]
    EASY_THRS = [0.9, 0.8]
    MED_THR = 0.4

    sweep_cfgs: list[SweepCfg] = []
    for (pk, pt), easy_thr, rt, k in product(PROBE_CFGS, EASY_THRS, ROUTE_TEMPS, KS):
        if rt == 0.0 and k != 1:
            continue  # optional: avoid redundant greedy repeats
        sweep_cfgs.append(SweepCfg(probe_k=pk, probe_temp=pt, route_temp=rt, k=k, easy_thr=easy_thr, med_thr=MED_THR))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    results_rows = []

    ntfy_send(
        f"Starting PIKA sweep run_id={run_id}\n"
        f"Probe Grid: K={PROBE_CFGS[0][0], PROBE_CFGS[1][0]} T={PROBE_CFGS[0][1], PROBE_CFGS[1][1]} | ProbeModel={PROBE_MODEL_NAME}\n"
        f"Grid: easy_thr={EASY_THRS}, temp={ROUTE_TEMPS}, k={KS}\n"
        f"Datasets: {len(LABELLED_DATASETS_LIST)}",
        title="PIKA sweep started",
        priority="high",
        tags="hourglass_flowing_sand",
    )

    # ----------------------------
    # Sweep execution
    # ----------------------------
    for cfg in sweep_cfgs:
        sc_policy, sc_name = sc_policy_from_cfg(cfg)
        threshold_str = f"threshE{cfg.easy_thr}_M{cfg.med_thr}"

        for ds_full in LABELLED_DATASETS_LIST:
            ds_alias = ds_full.replace("/", "_")

            t_start = time.time()
            try:
                # Load once per (cfg, dataset) run (safe + simple)
                base_df = load_labelled_probe_dataset(
                    MODEL_NAME=PROBE_MODEL_NAME,
                    PROBE_SOURCE_DATASET=PROBING_DATASET,
                    LABELLED_DATASET=ds_alias,
                    K=cfg.probe_k,
                    TEMPERATURE=cfg.probe_temp,
                    DATA_PATH=LABELLED_SR_PATH,
                )
                base_df = base_df.sample(n=15, random_state=42)
                df = base_df.copy(deep=True)

                score_col = "calibrated_score" if cfg.probe_k > 1 else "score"
                if score_col not in df.columns:
                    raise KeyError(f"Expected {score_col} for probe_k={cfg.probe_k}, got columns: {list(base_df.columns)}")

                # Route using sweep thresholds
                df["route_to"] = df[score_col].astype(float).apply(lambda s: route_questions_cfg(s, model_pool, cfg))

                # Set SC params on df for run_routed_vllm_inference
                df["sc_temp"] = float(cfg.route_temp)

                # Convert route_to -> tier -> sc_n
                easy_model, mid_model, hard_model = model_pool[0], model_pool[1], model_pool[2]
                def tier_from_route(m: str) -> str:
                    if m == easy_model: return "easy"
                    if m == mid_model:  return "medium"
                    return "hard"
                df["sc_n"] = df["route_to"].apply(lambda m: int(sc_policy[tier_from_route(m)]))

                # checkpoint unique per run (prevents collisions + allows resume)
                ckpt_path = (
                    f"{ROUTER_SAVE_DF_DIR}/CKPT_{run_id}_{ds_alias}"
                    f"_probeK{cfg.probe_k}_probeT{cfg.probe_temp}"
                    f"_routeT{cfg.route_temp}_k{cfg.k}"
                    f"_{sc_name}_{threshold_str}.parquet"
                )

                # Run inference
                df = run_routed_vllm_inference(
                    df,
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

                # Majority vote extracted answer
                df["majority_vote_extracted_answer"] = df[STORE_ALL_SAMPLES_COL].apply(
                    lambda x: add_majority_vote_answer(json.loads(x)) if isinstance(x, str) else None
                )

                # Ground truth extraction
                if "gsm8k" in ds_alias.lower():
                    df[FINAL_GT_COLUMN] = df[ORIGINAL_GT_COLUMN].apply(extract_gsm8k_solution)
                elif "aime" in ds_alias.lower():
                    df[FINAL_GT_COLUMN] = df[ORIGINAL_GT_COLUMN]
                else:
                    df[FINAL_GT_COLUMN] = df[ORIGINAL_GT_COLUMN].apply(extract_solution)

                # Metrics
                df["majority_vote_is_correct"] = df.apply(
                    lambda row: compute_score(
                        solution_str=f"\\boxed{{{row['majority_vote_extracted_answer']}}}",
                        ground_truth=row[FINAL_GT_COLUMN],
                    ),
                    axis=1,
                )
                df["passk_score"] = df.apply(
                    lambda row: compute_passk_from_json_solutions(
                        generated_sols_obj=row[STORE_ALL_SAMPLES_COL],
                        ground_truth=row[FINAL_GT_COLUMN],
                    ),
                    axis=1,
                )

                acc = float(df["majority_vote_is_correct"].mean())
                passk = float(df["passk_score"].mean())
                cost = float(df["total_cost_usd"].sum()) if "total_cost_usd" in df.columns else float("nan")
                nrows = int(len(df))
                route_counts = df["route_to"].value_counts().to_dict()
                elapsed = time.time() - t_start

                out_path = (
                    f"{ROUTER_SAVE_DF_DIR}/{run_id}_{ds_alias}_routed_by_{PROBING_DATASET}_{PROBE_MODEL_ALIAS}"
                    f"_probeK{cfg.probe_k}_probeT{cfg.probe_temp}"
                    f"_routeT{cfg.route_temp}_k{cfg.k}"
                    f"_sc_{sc_name}_{threshold_str}.parquet"
                )
                df.to_parquet(out_path, index=True)

                results_rows.append({
                    "run_id": run_id,
                    "dataset": ds_alias,
                    "probe_k": cfg.probe_k,
                    "probe_temp": cfg.probe_temp,
                    "route_temp": cfg.route_temp,
                    "k": cfg.k,
                    "easy_thr": cfg.easy_thr,
                    "med_thr": cfg.med_thr,
                    "sc_name": sc_name,
                    "acc_majority": acc,
                    "passk": passk,
                    "total_cost_usd": cost,
                    "nrows": nrows,
                    "elapsed_s": elapsed,
                    "out_path": out_path,
                })

                # Persist a rolling CSV so you can monitor while it runs
                results_csv = f"{ROUTER_SAVE_DF_DIR}/{run_id}_sweep_results.csv"
                pd.DataFrame(results_rows).to_csv(results_csv, index=False)

                # Notify
                ntfy_send(
                    f"✅ DONE\n"
                    f"run_id={run_id}\n"
                    f"dataset={ds_alias}\n"
                    f"probe: K={cfg.probe_k} T={cfg.probe_temp}\n"
                    f"router: easy>={cfg.easy_thr}, med>={cfg.med_thr}, temp={cfg.route_temp}, k={cfg.k}, sc={sc_name}\n"
                    f"rows={nrows} | acc={acc:.4f} | passk={passk:.4f} | cost=${cost:.4f} | t={elapsed/60:.1f}m\n"
                    f"routes={route_counts}\n"
                    f"saved={out_path}",
                    title="PIKA sweep finished a run",
                    priority="default",
                    tags="white_check_mark",
                )

                # Aggressive cleanup between runs (helps long sweeps)
                del df, base_df
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                elapsed = time.time() - t_start
                err_txt = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]
                ntfy_send(
                    f"❌ FAILED\n"
                    f"run_id={run_id}\n"
                    f"dataset={ds_alias}\n"
                    f"cfg: easy>={cfg.easy_thr}, med>={cfg.med_thr}, temp={cfg.route_temp}, k={cfg.k}\n"
                    f"t={elapsed/60:.1f}m\n"
                    f"error:\n{err_txt}",
                    title="PIKA sweep run failed",
                    priority="high",
                    tags="rotating_light",
                )
                # continue sweep
                gc.collect()
                torch.cuda.empty_cache()
                continue

    # Final summary notification
    if results_rows:
        res_df = pd.DataFrame(results_rows)
        best = res_df.sort_values(["acc_majority", "passk", "total_cost_usd"], ascending=[False, False, True]).head(5)
        summary_lines = []
        for _, r in best.iterrows():
            summary_lines.append(
                f"{r['dataset']} | acc={r['acc_majority']:.4f} passk={r['passk']:.4f} cost=${r['total_cost_usd']:.4f} "
                f"| easy>={r['easy_thr']} temp={r['route_temp']} k={r['k']} sc={r['sc_name']}"
            )
        ntfy_send(
            "🏁 Sweep complete\n"
            f"run_id={run_id}\n"
            f"total_runs={len(results_rows)}\n\n"
            "Top 5:\n" + "\n".join(summary_lines),
            title="PIKA sweep completed",
            priority="high",
            tags="checkered_flag",
        )
    else:
        ntfy_send(
            f"🏁 Sweep complete but no successful runs (run_id={run_id}).",
            title="PIKA sweep completed (no successes)",
            priority="high",
            tags="warning",
        )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
