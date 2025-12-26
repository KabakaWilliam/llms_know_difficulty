import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import time
import gc
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path
import sys


# ----------------------------
# Helpers (safe at import time)
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


def route_questions(predicted_score: float, model_pool: list[str]) -> str:
    if predicted_score >= 0.8:
        return model_pool[0]
    elif predicted_score >= 0.5:
        return model_pool[1]
    else:
        return model_pool[2]


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


def count_input_tokens_batch(prompts: list[str], tokenizer) -> list[int]:
    enc = tokenizer(prompts, add_special_tokens=False)
    return [len(ids) for ids in enc["input_ids"]]


TOKENS_PER_MILLION = 1_000_000


def run_routed_vllm_inference(
    df: pd.DataFrame,
    *,
    route_col: str,
    prompt_col: str,
    prompt_text_col: str = "prompt_text",
    out_text_col: str = "response_text",
    out_model_col: str = "response_model",
    input_num_tokens_col: str = "input_num_tokens",
    out_tok_col: str = "response_num_tokens",
    out_latency_col: str = "response_latency_s",
    out_err_col: str = "response_error",
    input_cost_col: str = "input_cost_usd",
    output_cost_col: str = "output_cost_usd",
    total_cost_col: str = "total_cost_usd",
    pricing_config: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    n: int = 1,
    batch_size: int = 32,
    checkpoint_path: Optional[str] = None,
    model_run_cfgs: Optional[Dict[str, VLLMModelRunCfg]] = None,
) -> pd.DataFrame:

    if model_run_cfgs is None:
        model_run_cfgs = {}
    if pricing_config is None:
        pricing_config = {}

    # Ensure output columns exist
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

    pending_mask = df[out_text_col].isna()
    if pending_mask.sum() == 0:
        print("✅ Nothing to do: all rows already have responses.")
        return df

    routes = df.loc[pending_mask, route_col].dropna().unique().tolist()
    print(f"Routes to run: {routes}")

    for model_name in routes:
        model_mask = pending_mask & (df[route_col] == model_name)
        idxs = df.index[model_mask].tolist()
        if not idxs:
            continue

        cfg = model_run_cfgs.get(model_name, VLLMModelRunCfg())
        print(f"\n=== Running model: {model_name} | rows: {len(idxs)} ===")
        print(f"vLLM cfg: {cfg}")

        model_costs = pricing_config.get(model_name, {}).get("model_costs", {})
        in_rate = model_costs.get("input_per_mill", None)
        out_rate = model_costs.get("output_per_mill", None)
        has_pricing = (in_rate is not None) and (out_rate is not None)

        # NOTE: don’t rely on "72" substring long-term, but fine for now.
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

        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
        tokenizer = llm.llm_engine.tokenizer.tokenizer

        for start in tqdm(range(0, len(idxs), batch_size), desc=f"Inferencing {model_name}"):
            batch_idxs = idxs[start : start + batch_size]
            problems = df.loc[batch_idxs, prompt_col].tolist()

            prompts = batch_apply_chat_template(problems, tokenizer)
            input_tok_counts = count_input_tokens_batch(prompts, tokenizer)
            df.loc[batch_idxs, prompt_text_col] = prompts

            t0 = time.time()
            try:
                outputs = llm.generate(prompts, sampling_params=sampling)
                latency = time.time() - t0

                texts, out_tok_counts = [], []
                errs = [None] * len(outputs)

                for out in outputs:
                    comp = out.outputs[0]
                    texts.append(comp.text)
                    out_tok_counts.append(len(comp.token_ids) if comp.token_ids is not None else np.nan)

                df.loc[batch_idxs, out_text_col] = texts
                df.loc[batch_idxs, out_model_col] = model_name
                df.loc[batch_idxs, input_num_tokens_col] = input_tok_counts
                df.loc[batch_idxs, out_tok_col] = out_tok_counts
                df.loc[batch_idxs, out_latency_col] = latency
                df.loc[batch_idxs, out_err_col] = errs

                if has_pricing:
                    in_arr = np.array(input_tok_counts, dtype=float)
                    out_arr = np.array(out_tok_counts, dtype=float)
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

            if checkpoint_path is not None:
                df.to_parquet(checkpoint_path, index=True)

        unload_model(llm)
        pending_mask = df[out_text_col].isna()
        if checkpoint_path is not None:
            df.to_parquet(checkpoint_path, index=True)

    return df


# ----------------------------
# Main (ONLY runs in parent)
# ----------------------------

def main():
    repo_root = Path.cwd().parent.parent
    sys.path.insert(0, str(repo_root))

    from will_replication.my_utils.utils import (
        load_probe_data, sigmoid_np, load_labelled_probe_dataset,
        SIMPLE_MODEL_POOL_CONFIG, ModelConfig
    )

    LABELLED_DATASETS_LIST = ["opencompass/AIME2025","gneubig/aime-1983-2024", "DigitalLearningGmbH/MATH-lighteval", "openai/gsm8k"]


    PROBE_RESULTS_DIR = "../../will_replication/probe_results/DATA"
    LABELLED_SR_PATH = f"{PROBE_RESULTS_DIR}/Labelled_SR"
    PROBE_DATA_PATH = f"{PROBE_RESULTS_DIR}/SR_DATA"
    PROBING_DATASET = "MATH"

    for LABELLED_DATASET_FULL_NAME in LABELLED_DATASETS_LIST:
        # LABELLED_DATASET_FULL_NAME = "gneubig/aime-1983-2024"
        LABELLED_DATASET_NAME = "_".join(LABELLED_DATASET_FULL_NAME.split("/"))

        PROBE_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        K = 1
        TEMPERATURE = 0.0

        # _ = load_probe_data(
        #     MODEL_NAME=PROBE_MODEL_NAME,
        #     PROBING_DATASET=PROBING_DATASET,
        #     K=K,
        #     TEMPERATURE=TEMPERATURE,
        #     DATA_PATH=PROBE_DATA_PATH,
        # )
        try:
            routing_dataset_df = load_labelled_probe_dataset(
                MODEL_NAME=PROBE_MODEL_NAME,
                PROBE_SOURCE_DATASET=PROBING_DATASET,
                LABELLED_DATASET=LABELLED_DATASET_NAME,
                K=K,
                TEMPERATURE=TEMPERATURE,
                DATA_PATH=LABELLED_SR_PATH,
            )
        except:
            print("this dataset config doesn't exist")
            continue

        print(f"testing PIKA route on : {LABELLED_DATASET_NAME}")

        model_pool = list(SIMPLE_MODEL_POOL_CONFIG.keys())
        routing_dataset_df["route_to"] = routing_dataset_df["score"].astype(float).apply(
            lambda s: route_questions(s, model_pool)
        )

        print("Routing Dataset Breakdown:")
        print(routing_dataset_df["route_to"].value_counts())

        model_run_cfgs = {
            "Qwen/Qwen2.5-Math-1.5B-Instruct": VLLMModelRunCfg(tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=4096),
            "Qwen/Qwen2.5-Math-7B-Instruct":   VLLMModelRunCfg(tensor_parallel_size=1, gpu_memory_utilization=0.70, max_model_len=4096),
            "Qwen/Qwen2.5-Math-72B-Instruct":  VLLMModelRunCfg(tensor_parallel_size=2, gpu_memory_utilization=0.92, max_model_len=4096),
        }

        routing_dataset_df = run_routed_vllm_inference(
            routing_dataset_df,
            route_col="route_to",
            prompt_col="prompt_scored",
            out_text_col="routed_response_text",
            temperature=TEMPERATURE,
            max_tokens=3000,
            batch_size=16,
            checkpoint_path=f"{LABELLED_DATASET_NAME}_routed.parquet",
            pricing_config=SIMPLE_MODEL_POOL_CONFIG,
            model_run_cfgs=model_run_cfgs,
        )

        routing_dataset_df.to_parquet(f"{LABELLED_DATASET_NAME}_routed_final.parquet", index=True)
        print(f"✅ Done. Saved {LABELLED_DATASET_NAME}_routed_final.parquet")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # safe with CUDA + vLLM
    main()
