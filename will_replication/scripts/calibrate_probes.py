import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt

from pathlib import Path


repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))

from my_utils.utils import load_probe_data, sigmoid_np, fit_platt_binomial, apply_platt, compute_ece_soft


# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME = "openai/gpt-oss-20b"
MODEL_ALIAS = "-".join(MODEL_NAME.split("/"))
K=5
TEMPERATURE=1.0
PROBING_DATASET="gneubig_aime-1983-2024"
# PROBING_DATASET="DigitalLearningGmbH_MATH-lighteval"
MAXLEN=32678 #32678
IS_MAJORITY_VOTE=True

data = load_probe_data(MODEL_NAME=MODEL_NAME, PROBING_DATASET=PROBING_DATASET, K=K, TEMPERATURE=TEMPERATURE, MAXLEN=MAXLEN, IS_MAJORITY_VOTE=IS_MAJORITY_VOTE)


# -------- Usage (calibrate on train-split, evaluate on test) --------
logits_train = np.asarray(data["train_predictions"])
sr50_train   = np.asarray(data["train_actual"])

logits_test  = np.asarray(data["test_predictions"])
sr50_test    = np.asarray(data["test_actual"])

# Option A (recommended): hold out calibration subset from TRAIN
rng = np.random.default_rng(42)
idx = np.arange(len(logits_train))
rng.shuffle(idx)
cal_frac = 0.2
n_cal = int(round(cal_frac * len(idx)))
cal_idx = idx[:n_cal]

logits_cal = logits_train[cal_idx]
sr50_cal   = sr50_train[cal_idx]

n_bins=10
print(f"Num bins: {n_bins}" )

a, b = fit_platt_binomial(logits_cal, sr50_cal, n_trials=50, device="cpu")
print("Platt params:", {"a": a, "b": b})

# Calibrated test probabilities
probs_test_platt = apply_platt(logits_test, a, b)

ece_before = compute_ece_soft(sigmoid_np(logits_test), sr50_test, n_bins=n_bins)
print("ECE before:", ece_before)

# Example: compute soft-label ECE with your function
ece_after = compute_ece_soft(probs_test_platt, sr50_test, n_bins=n_bins)
print(f"Test ECE after Platt scaling: {ece_after:.4f}")


calibrated_probe = {
    "best_layer": data["best_layer"],
    "best_position": data["best_position"],
    "platt_a": a,
    "platt_b": b,
    "K": K,
}

def add_calibrated_probe(
    MODEL_NAME: str,
    PROBING_DATASET: str,
    K: int,
    TEMPERATURE: float,
    platt_a: float,
    platt_b: float,
) -> dict:
    """
    Load best_probe_predictions.json, attach a `calibrated_probe` field, and save.

    calibrated_probe = {
        "best_layer": <from existing probe_data>,
        "best_position": <from existing probe_data>,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "K": K,
    }
    """
    MODEL_ALIAS = "-".join(MODEL_NAME.split("/"))
    GEN_STR = f"maxlen_{MAXLEN}_k_{K}_temp_{TEMPERATURE}_labelcol_majority_vote_is_correct"
    probe_path = Path(f"../probe_results/DATA/SR_DATA/{PROBING_DATASET}/{MODEL_ALIAS}_{GEN_STR}/best_probe_predictions.json")

    # 1) Load existing probe data
    with probe_path.open("r") as f:
        data = json.load(f)

    # 2) Build calibrated probe block
    calibrated_probe = {
        "best_layer": data["best_layer"],
        "best_position": data["best_position"],
        "platt_a": float(platt_a),
        "platt_b": float(platt_b),
        "K": int(K),
    }

    # 3) Attach to dict
    data["calibrated_probe"] = calibrated_probe

    # 4) Save back to disk
    with probe_path.open("w") as f:
        json.dump(data, f, indent=2)

    return data

DATASETS = ["opencompass/AIME2025",
            "gneubig/aime-1983-2024",
              "DigitalLearningGmbH/MATH-lighteval",
                "openai/gsm8k"
                ]

print(f"Calibrating predictions for {MODEL_ALIAS}: \n")
print(f"MATH Benchmark score: {data["avg_benchmark_score"]}")

for DATASET_NAME in DATASETS:
    DATASET_NAME = "_".join(DATASET_NAME.split("/"))
    print(f"====================================\n📜 LOADING {DATASET_NAME} 📜\n====================================")
    PROBE_PREDICTIONS_PATH = f"../probe_results/DATA/Labelled_SR/{PROBING_DATASET}_probe/{DATASET_NAME}/{MODEL_ALIAS}_maxlen_{MAXLEN}_k_{K}_temp_{TEMPERATURE}_labelcol_majority_vote_is_correct/scored.parquet"
    

    labelled_df = pd.read_parquet(PROBE_PREDICTIONS_PATH)

    print("📊 Raw score: \n")
    print(labelled_df["score_raw"].describe())
    print("**************** \n")

    print("📊 Sigmoid score: \n")
    print(labelled_df["score"].describe())
    print("**************** \n")

    print("📊 Calibrated Raw score: \n")
    labelled_df["calibrated_raw_score"] = (labelled_df["score_raw"] * a) + b
    print(labelled_df["calibrated_raw_score"].describe())
    print("**************** \n")

    print("📊 Calibrated Sigmoid score: \n")
    labelled_df["calibrated_score"] = sigmoid_np(labelled_df["calibrated_raw_score"])
    print(labelled_df["calibrated_score"].describe())
    print("**************** \n")
    print("======================================\n")

    labelled_df.to_parquet(PROBE_PREDICTIONS_PATH)
    print(f"✅ Completed calibrating predicitons in {PROBE_PREDICTIONS_PATH}\n")