"""
Utility functions for the probe-based utility router notebook.

Contains:
  - Model name helpers and generation string configs
  - Data loading (probe predictions, merge logic)
  - Routing strategies (probe-based, oracle, oracle-utility)
  - Plotting (Pareto frontier, cost-vs-accuracy figure)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────
# Generation-string configs
# ──────────────────────────────────────────────
GEN_STR_CONFIGS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-20b_low":                     {"maxlen": 131072, "k": 5, "temp": 1.0},
    "openai/gpt-oss-20b_medium":                   {"maxlen": 131072, "k": 5, "temp": 1.0},
    "openai/gpt-oss-20b_high":                     {"maxlen": 131072, "k": 5, "temp": 1.0},
    "Qwen/Qwen2.5-1.5B-Instruct":                 {"maxlen": 3000,   "k": 5, "temp": 0.7},
    "Qwen/Qwen2.5-Math-1.5B-Instruct":            {"maxlen": 3000,   "k": 5, "temp": 0.7},
    "Qwen/Qwen2.5-Math-7B-Instruct":              {"maxlen": 3000,   "k": 5, "temp": 0.7},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct":           {"maxlen": 4096,   "k": 1, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-3B-Instruct":             {"maxlen": 4096,   "k": 5, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-7B-Instruct":             {"maxlen": 4096,   "k": 5, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-14B-Instruct":            {"maxlen": 4096,   "k": 1, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-32B-Instruct":            {"maxlen": 4096,   "k": 1, "temp": 0.2},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":    {"maxlen": 32768,  "k": 5, "temp": 0.6},
}


def get_gen_str(model_name: str) -> str:
    """Return the generation string for *model_name* (falls back to GPT-Low defaults)."""
    cfg = GEN_STR_CONFIGS.get(model_name, GEN_STR_CONFIGS["openai/gpt-oss-20b_low"])
    return f"maxlen_{cfg['maxlen']}_k_{cfg['k']}_temp_{cfg['temp']}"


# ──────────────────────────────────────────────
# Model-name formatting
# ──────────────────────────────────────────────
def short_model_name(m: str) -> str:
    """Shorten a full model / strategy name for plot labels."""
    for prefix in ("Always ", "openai/", "Qwen/", "deepseek-ai/"):
        m = m.replace(prefix, "")
    m = m.replace("DeepSeek-R1-Distill-", "R1-")
    m = m.replace("Qwen2.5-", "")
    m = m.replace("-Instruct", "")
    m = m.replace("gpt-oss-20b-low", "GPT-Low")
    m = m.replace("gpt-oss-20b-medium", "GPT-Med")
    m = m.replace("gpt-oss-20b-high", "GPT-High")
    m = m.replace("_", "-")
    return m


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_model_data(
    model: str,
    dataset: str,
    data_dir: str = "../data",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load test / train parquet files for one model+dataset. Returns ``(test_df, train_df)``."""
    gen_str = get_gen_str(model)
    test_path = f"{data_dir}/{model}/{dataset}/test_{gen_str}.parquet"
    train_path = f"{data_dir}/{model}/{dataset}/train_{gen_str}.parquet"
    try:
        test_df = pd.read_parquet(test_path)
        train_df = pd.read_parquet(train_path)
        return test_df, train_df
    except FileNotFoundError as exc:
        print(f"⚠ Could not load {model}: {exc}")
        return None, None


def get_latest_probe_preds(
    dataset: str,
    model_name: str,
    gen_str: str,
    probe_model_type: str = "linear_eoi_probe",
    chosen_metric: str = "majority_vote_is_correct",
    data_dir: str = "../data",
) -> pd.DataFrame | None:
    """Find and load the most recent probe predictions for a model."""
    base_dir = Path(
        f"{data_dir}/results/{model_name}/{dataset}/{probe_model_type}/{gen_str}/label_{chosen_metric}"
    )
    if not base_dir.exists():
        print(f"⚠ Directory not found: {base_dir}")
        return None

    timestamp_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    if not timestamp_dirs:
        print(f"⚠ No probe directories found in {base_dir}")
        return None

    latest_dir = timestamp_dirs[-1]
    probe_path = latest_dir / "probe_preds.parquet"
    if not probe_path.exists():
        print(f"⚠ Probe file not found: {probe_path}")
        return None

    print(f"✓ Loading probes for {model_name} from: {latest_dir.name}")
    return pd.read_parquet(probe_path)


def merge_probe_predictions(
    test_df: pd.DataFrame,
    probe_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Merge probe predictions into *test_df* using positional indices."""
    result = test_df.loc[test_df.index[probe_preds["idx"].values]].copy()
    result["probe_pred"] = probe_preds["pred"].values
    result["probe_pred_class"] = (result["probe_pred"] > 0.5).astype(int)
    return result


# ──────────────────────────────────────────────
# Tier-cost helpers
# ──────────────────────────────────────────────
def compute_tier_costs(
    models: list[str],
    train_dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Return ``(raw_costs, normalised_costs)`` dicts from the training data.

    Normalised costs are min-max scaled to [0, 1].
    """
    raw = {m: train_dfs[m]["total_output_cost_usd"].mean() for m in models}
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo if hi != lo else 1.0
    normed = {m: (c - lo) / span for m, c in raw.items()}
    return raw, normed


# ──────────────────────────────────────────────
# Routing strategies
# ──────────────────────────────────────────────
def route_problems_by_probe_pred(
    test_dfs_dict: dict[str, pd.DataFrame],
    model_list: list[str],
    chosen_metric: str,
    *,
    strategy: str = "max_probe_pred",
    random_seed: int | None = None,
    value: float = 1.0,
    tier_costs_dict: dict[str, float] | None = None,
    utility_variant: str = "original",
) -> pd.DataFrame:
    """
    Route each problem to the best model based on probe predictions.

    Parameters
    ----------
    strategy : str
        ``"max_probe_pred"`` | ``"min_cost"`` | ``"random"`` | ``"max_utility"``
    utility_variant : str
        For ``max_utility``: ``"original"`` | ``"normalized"`` | ``"ratio"``
        | ``"threshold"`` | ``"sigmoid"`` | ``"success_per_dollar"``
        | ``"balance_costs"`` — or any other string (falls back to
        ``probe_pred - tier_cost * value``).
    """
    available = [m for m in model_list if m in test_dfs_dict]
    if not available:
        raise ValueError(f"No models found in test_dfs_dict ({list(test_dfs_dict)})")

    tier_costs = tier_costs_dict or {m: i + 1 for i, m in enumerate(available)}

    if strategy == "random" and random_seed is not None:
        np.random.seed(random_seed)

    all_indices = sorted(test_dfs_dict[available[0]].index)
    rows: list[dict] = []

    for idx in all_indices:
        cands = pd.DataFrame([
            {
                "model_name": m,
                "probe_pred": test_dfs_dict[m].loc[idx, "probe_pred"],
                chosen_metric: test_dfs_dict[m].loc[idx, chosen_metric],
                "total_output_cost_usd": test_dfs_dict[m].loc[idx, "total_output_cost_usd"],
                "tier_cost": tier_costs[m],
            }
            for m in available
        ])

        if strategy == "max_probe_pred":
            sel = cands.loc[cands["probe_pred"].idxmax()]
            util = sel["probe_pred"]

        elif strategy == "min_cost":
            sel = cands.loc[cands["tier_cost"].idxmin()]
            util = -sel["tier_cost"]

        elif strategy == "random":
            sel = cands.sample(n=1).iloc[0]
            util = np.nan

        elif strategy == "max_utility":
            cands["utility"] = _compute_utility(cands, value, utility_variant)
            best = cands["utility"].idxmax()
            sel = cands.loc[best]
            util = sel["utility"]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rows.append({
            "idx": idx,
            "model_name": sel["model_name"],
            chosen_metric: sel[chosen_metric],
            "probe_pred_class": sel["probe_pred"],
            "total_output_cost_usd": sel["total_output_cost_usd"],
            "utility": util,
        })

    return pd.DataFrame(rows)


def _compute_utility(cands: pd.DataFrame, value: float, variant: str) -> pd.Series:
    """Vectorised utility calculation for a single problem's candidate set."""
    pp = cands["probe_pred"]
    tc = cands["tier_cost"]

    if variant == "original":
        return pp * value - tc
    if variant == "normalized":
        return (pp * 100 * value) - (tc / tc.max() * value)
    if variant == "ratio":
        return pp / tc
    if variant == "threshold":
        return np.where(pp >= 0.6, -tc, pp)
    if variant == "sigmoid":
        return pp * value - (2 / (1 + np.exp(-tc)) - 1)
    if variant == "success_per_dollar":
        return pp ** 2 / tc
    if variant == "balance_costs":
        return pp - tc * value
    # Default fallback (covers og_<lambda> names, etc.)
    return pp - tc * value


def oracle_router(
    test_dfs_dict: dict[str, pd.DataFrame],
    model_list: list[str],
    chosen_metric: str,
    *,
    tier_costs_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Oracle router — routes each problem to the *cheapest* model that solves it
    correctly.  If none solves it, picks the cheapest overall.
    """
    tier_costs = tier_costs_dict or {m: i + 1 for i, m in enumerate(model_list)}
    available = [m for m in model_list if m in test_dfs_dict and m in tier_costs]
    if not available:
        raise ValueError("No available models")

    print(f"Oracle router: {len(available)} models — {[m.split('/')[-1] for m in available]}")

    all_indices = sorted(test_dfs_dict[available[0]].index)
    rows: list[dict] = []

    for idx in all_indices:
        success = [
            {"model": m, "cost": tier_costs[m], "usd": test_dfs_dict[m].loc[idx, "total_output_cost_usd"]}
            for m in available
            if test_dfs_dict[m].loc[idx, chosen_metric] == 1
        ]
        sel_model = (
            min(success, key=lambda s: s["cost"])["model"]
            if success
            else min(available, key=lambda m: tier_costs[m])
        )
        sel_row = test_dfs_dict[sel_model].loc[idx]
        rows.append({
            "idx": idx,
            "model_name": sel_model,
            chosen_metric: sel_row[chosen_metric],
            "total_output_cost_usd": sel_row["total_output_cost_usd"],
        })

    return pd.DataFrame(rows)


def route_problems_oracle_utility(
    test_dfs_dict: dict[str, pd.DataFrame],
    model_list: list[str],
    chosen_metric: str,
    lambda_val: float,
    *,
    tier_costs_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Oracle *utility* router — uses actual success (0/1) instead of probe
    predictions.  Theoretical upper bound for utility-based routing.
    """
    tier_costs = tier_costs_dict or {m: i + 1 for i, m in enumerate(model_list)}
    available = [m for m in model_list if m in test_dfs_dict and m in tier_costs]
    if not available:
        raise ValueError("No available models")

    all_indices = sorted(test_dfs_dict[available[0]].index)
    rows: list[dict] = []

    for idx in all_indices:
        cands = pd.DataFrame([
            {
                "model_name": m,
                chosen_metric: test_dfs_dict[m].loc[idx, chosen_metric],
                "total_output_cost_usd": test_dfs_dict[m].loc[idx, "total_output_cost_usd"],
                "utility": test_dfs_dict[m].loc[idx, chosen_metric] - lambda_val * tier_costs[m],
            }
            for m in available
        ])
        best = cands.loc[cands["utility"].idxmax()]
        rows.append({
            "idx": idx,
            "model_name": best["model_name"],
            chosen_metric: best[chosen_metric],
            "total_output_cost_usd": best["total_output_cost_usd"],
            "utility": best["utility"],
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Comparison-data builder
# ──────────────────────────────────────────────
def build_comparison_data(
    test_dfs_with_probes: dict[str, pd.DataFrame],
    models: list[str],
    chosen_metric: str,
    *,
    routed_random: pd.DataFrame,
    routed_oracle: pd.DataFrame,
    routed_variants: dict[str, pd.DataFrame],
    oracle_utility_sweep: dict[str, pd.DataFrame],
) -> dict[str, list]:
    """Assemble the dict consumed by ``plot_cost_vs_accuracy``."""
    data: dict[str, list] = {"Strategy": [], "Cost": [], chosen_metric: []}

    # "Always model X" baselines
    for m in models:
        df = test_dfs_with_probes[m]
        data["Strategy"].append(f"Always {m}")
        data["Cost"].append(df["total_output_cost_usd"].sum())
        data[chosen_metric].append(df[chosen_metric].mean())

    # Random
    data["Strategy"].append("Random Routing")
    data["Cost"].append(routed_random["total_output_cost_usd"].sum())
    data[chosen_metric].append(routed_random[chosen_metric].mean())

    # Oracle
    data["Strategy"].append("Oracle (Perfect Knowledge)")
    data["Cost"].append(routed_oracle["total_output_cost_usd"].sum())
    data[chosen_metric].append(routed_oracle[chosen_metric].mean())

    # Probe-based router sweep
    for name, df in routed_variants.items():
        data["Strategy"].append(name)
        data["Cost"].append(df["total_output_cost_usd"].sum())
        data[chosen_metric].append(df[chosen_metric].mean())

    # Oracle utility sweep
    for name, df in oracle_utility_sweep.items():
        data["Strategy"].append(name)
        data["Cost"].append(df["total_output_cost_usd"].sum())
        data[chosen_metric].append(df[chosen_metric].mean())

    return data


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def _place_label(ax, x, y, text, *, fontsize=8, dx=3, dy=2, ha="left"):
    ax.annotate(
        text, (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        ha=ha, va="bottom",
        clip_on=True,
        annotation_clip=True,
        zorder=6,
    )


def pareto_frontier(costs, accs, eps=1e-12):
    """Pareto-efficient indices (minimise cost, maximise accuracy)."""
    order = np.argsort(costs)
    best_acc, keep = -np.inf, []
    for i in order:
        if accs[i] > best_acc + eps:
            keep.append(i)
            best_acc = accs[i]
    return np.array(keep, dtype=int)


def _downsample_by_cost(costs, idxs, k=7):
    """Select ~k points along frontier evenly spaced in cost (keeps endpoints)."""
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) <= k:
        return idxs
    c = costs[idxs]
    order = np.argsort(c)
    idxs, c = idxs[order], c[order]
    targets = np.linspace(c.min(), c.max(), k)
    chosen = np.unique([idxs[np.argmin(np.abs(c - t))] for t in targets])
    return chosen[np.argsort(costs[chosen])]


def _split_strategies(
    comparison_data,
    *,
    cost_col="Cost",
    strategy_col="Strategy",
    random_strategy_name="Random Routing",
    baseline_prefix="Always",
    router_regex=r"^og_\d+(\.\d+)?$",
    oracle_util_regex=r"^oracle_util_\d+(\.\d+)?$",
):
    df = pd.DataFrame(comparison_data).copy()
    baseline_df = df[df[strategy_col].astype(str).str.startswith(baseline_prefix)].copy()
    random_df = df[df[strategy_col].astype(str).eq(random_strategy_name)].copy()

    router_mask = df[strategy_col].astype(str).str.match(router_regex, na=False)
    router_df = df[router_mask].copy()
    if len(router_df):
        router_df["lambda"] = router_df[strategy_col].str.replace("og_", "", regex=False).astype(float)
        router_df = router_df.sort_values(cost_col).reset_index(drop=True)

    oracle_mask = df[strategy_col].astype(str).str.match(oracle_util_regex, na=False)
    oracle_util_df = df[oracle_mask].copy()
    if len(oracle_util_df):
        oracle_util_df["lambda"] = oracle_util_df[strategy_col].str.replace("oracle_util_", "", regex=False).astype(float)
        oracle_util_df = oracle_util_df.sort_values(cost_col).reset_index(drop=True)

    return baseline_df, random_df, router_df, oracle_util_df, df


# ──────────────────────────────────────────────
# Main figure
# ──────────────────────────────────────────────
_DEFAULT_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B79A20", "#E45756", "#72B7B2"]


def plot_cost_vs_accuracy(
    comparison_data,
    *,
    metric_col: str,
    title: str = "Cost vs Accuracy",
    cost_col: str = "Cost",
    strategy_col: str = "Strategy",
    figsize: tuple[float, float] = (3.4, 3.2),
    xlim=None,
    ylim=None,
    xscale: str = "auto",
    symlog_linthresh: float = 0.5,
    robust_xlim: bool = True,
    frontier_highlight_k: int = 7,
    show_router_cloud: bool = True,
    show_router_frontier: bool = True,
    show_router_highlights: bool = True,
    label_tweaks: dict | None = None,
    baseline_palette: list[str] | None = None,
    legend_loc=("upper center", (0.5, -0.22), 3),
):
    """Plot the cost-vs-accuracy Pareto figure."""
    label_tweaks = label_tweaks or {}
    palette = baseline_palette or _DEFAULT_PALETTE

    baseline_df, random_df, router_df, oracle_util_df, df = _split_strategies(
        comparison_data, cost_col=cost_col, strategy_col=strategy_col,
    )
    oracle_df = df[df[strategy_col].astype(str).str.contains("Oracle", na=False)].copy()

    if metric_col not in df.columns:
        raise ValueError(f"{metric_col!r} not in {list(df.columns)}")

    fig, ax = plt.subplots(figsize=figsize)

    # ---- router / oracle-util processing ----
    def _process_sweep(sweep_df):
        if sweep_df.empty:
            return None, None, None, None
        x = sweep_df[cost_col].to_numpy(float)
        y = np.maximum.accumulate(sweep_df[metric_col].to_numpy(float))
        fi = pareto_frontier(x, y)
        fi = fi[np.argsort(x[fi])]
        hi = _downsample_by_cost(x, fi, k=frontier_highlight_k)
        return x, y, fi, hi

    rx, ry, rfi, rhi = _process_sweep(router_df)
    ox, oy, ofi, ohi = _process_sweep(oracle_util_df)

    # ---- axis limits ----
    all_costs = df[cost_col].to_numpy(float)
    all_costs = all_costs[np.isfinite(all_costs)]
    if not len(all_costs):
        all_costs = np.array([0.0, 1.0])
    cmax, c95 = float(all_costs.max()), float(np.percentile(all_costs, 95))

    xscale_use = xscale
    if xscale == "auto":
        xscale_use = "symlog" if (cmax / max(1e-9, c95)) >= 6 else "linear"
    if xscale_use == "symlog":
        ax.set_xscale("symlog", linthresh=symlog_linthresh, linscale=1.0)
    if xlim is None:
        xmax = c95 if robust_xlim else cmax
        xmax = max(1.0, xmax * 1.10)
        if xscale_use == "symlog":
            xmax = max(xmax, min(cmax * 1.05, xmax * 5))
        ax.set_xlim(0, xmax)
    else:
        ax.set_xlim(*xlim)

    if ylim is None:
        parts = []
        if not baseline_df.empty:
            parts.append(baseline_df[metric_col].to_numpy(float))
        if ry is not None:
            parts.append(ry)
        if not random_df.empty:
            parts.append(random_df[metric_col].to_numpy(float))
        all_y = np.concatenate(parts) if parts else np.array([0.0, 1.0])
        all_y = all_y[np.isfinite(all_y)]
        if not len(all_y):
            all_y = np.array([0.0, 1.0])
        ax.set_ylim(float(all_y.min()) - 0.02, float(all_y.max()) + 0.04)
    else:
        ax.set_ylim(*ylim)

    # ---- styling ----
    ax.grid(True, alpha=0.12, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cost ($)", fontsize=9, labelpad=1)
    ax.set_ylabel("Accuracy", fontsize=9, labelpad=1)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(title, fontsize=9.5, pad=1)

    # ---- baselines ----
    for i, (_, row) in enumerate(baseline_df.iterrows()):
        x, y = float(row[cost_col]), float(row[metric_col])
        name = short_model_name(str(row[strategy_col]))
        ax.scatter(x, y, s=55, c=palette[i % len(palette)], edgecolors="black", linewidth=0.8, zorder=3)
        if name in label_tweaks:
            _place_label(ax, x, y, name, fontsize=8, **label_tweaks[name])
        else:
            xlim_hi = ax.get_xlim()[1]
            ylim_hi = ax.get_ylim()[1]
            dx = 5 if x < xlim_hi * 0.7 else -5
            dy = 5 if y < ylim_hi * 0.5 else -10
            ha = "left" if x < xlim_hi * 0.7 else "right"
            _place_label(ax, x, y, name, fontsize=8, dx=dx, dy=dy, ha=ha)

    # ---- random ----
    if not random_df.empty:
        r = random_df.iloc[0]
        x, y = float(r[cost_col]), float(r[metric_col])
        ax.scatter(x, y, s=55, c="#7f7f7f", marker="D", edgecolors="black", linewidth=0.8, zorder=3)
        dx = 8 if x < ax.get_xlim()[1] * 0.5 else -5
        ha = "left" if x < ax.get_xlim()[1] * 0.5 else "right"
        _place_label(ax, x, y, "Random", fontsize=8, dx=dx, dy=0, ha=ha)

    # ---- oracle point ----
    if not oracle_df.empty:
        o = oracle_df.iloc[0]
        ax.scatter(
            float(o[cost_col]), float(o[metric_col]),
            s=120, c="#FFD700", marker="*", edgecolors="darkgoldenrod", linewidth=1.2, zorder=6,
        )

    # ---- sweep curves ----
    def _draw_sweep(sx, sy, sfi, *, color, marker, ls="-"):
        if sx is None:
            return
        if show_router_cloud:
            alpha = max(0.15, min(0.4, 20 / len(sx)))
            ax.scatter(sx, sy, s=16, color=color, marker=marker, linewidths=0.9, alpha=alpha, zorder=1)
        if show_router_frontier and sfi is not None and len(sfi):
            ax.plot(sx[sfi], sy[sfi], color=color, linewidth=2.4, alpha=0.95, zorder=2, linestyle=ls)
        if show_router_highlights and sfi is not None and len(sfi):
            ax.scatter(sx[sfi], sy[sfi], s=40, color=color, marker=marker, linewidths=1.6, alpha=0.95, zorder=4)

    _draw_sweep(rx, ry, rfi, color="red", marker="x")
    _draw_sweep(ox, oy, ofi, color="blue", marker="+", ls="--")

    # ---- legend ----
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=5, label="Single"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7f7f7f", markeredgecolor="black", markersize=5, label="Random"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700", markeredgecolor="darkgoldenrod", markersize=9, markeredgewidth=1.2, label="Oracle"),
        Line2D([0], [0], color="red", marker="x", markeredgecolor="red", markersize=6, linewidth=2.4, label="Router (λ sweep)"),
        Line2D([0], [0], color="blue", marker="+", markeredgecolor="blue", markersize=6, linewidth=2.4, linestyle="--", label="Oracle Util (λ sweep)"),
    ]
    loc, anchor, ncol = legend_loc
    ax.legend(handles=legend_elements, loc=loc, bbox_to_anchor=anchor, ncol=ncol, fontsize=8, frameon=False, handlelength=1.3, columnspacing=1.1)
    plt.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.30)
    return fig, ax


# ──────────────────────────────────────────────
# Probe-distribution diagnostics
# ──────────────────────────────────────────────
def plot_probe_histograms(
    test_dfs_with_probes: dict[str, pd.DataFrame],
    models: list[str],
    chosen_metric: str,
    dataset_name: str = "",
):
    """Grid of per-model probe-prediction histograms with AUROC."""
    available = [m for m in models if m in test_dfs_with_probes]
    n = len(available)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, model in enumerate(available):
        ax = axes_flat[i]
        preds = test_dfs_with_probes[model]["probe_pred"].values
        labels = test_dfs_with_probes[model][chosen_metric].values
        try:
            auroc = roc_auc_score(labels, preds)
        except Exception:
            auroc = np.nan

        counts, _, _ = ax.hist(preds, bins=25, alpha=0.75, color="steelblue", edgecolor="black", linewidth=1.0)
        mean_p, med_p, std_p = preds.mean(), np.median(preds), preds.std()
        ax.axvline(mean_p, color="red", ls="--", lw=2.5, label=f"Mean: {mean_p:.3f}", alpha=0.8)
        ax.axvline(med_p, color="green", ls="--", lw=2.5, label=f"Median: {med_p:.3f}", alpha=0.8)

        stats_txt = f"Std: {std_p:.3f}\nMin: {preds.min():.3f}\nMax: {preds.max():.3f}"
        ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        short = short_model_name(model)
        ax.set_title(f"{short}\n(n={len(preds)}, AUROC={auroc:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Probe prediction", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(loc="upper center", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, counts.max() * 1.15)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Probe Prediction Distributions ({dataset_name})", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Dataset title mapping
# ──────────────────────────────────────────────
DATASET_TITLES = {
    "openai_gsm8k": "GSM8K",
    "gneubig_aime-1983-2024": "AIME 25",
    "DigitalLearningGmbH_MATH-lighteval": "MATH",
    "livecodebench_code_generation_lite": "Live Code Bench",
}


def dataset_title(dataset_id: str) -> str:
    return DATASET_TITLES.get(dataset_id, dataset_id)


# ──────────────────────────────────────────────
# Cascade (threshold) routing
# ──────────────────────────────────────────────
def route_cascade_threshold(
    test_dfs_dict: dict[str, pd.DataFrame],
    cascade_models: list[str],
    chosen_metric: str,
    thresholds: tuple[float, ...],
    *,
    tier_costs_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Route problems through a cascade of models, escalating when probe
    confidence is below the corresponding threshold.

    Parameters
    ----------
    cascade_models : list[str]
        Models ordered cheapest → most expensive.  The **last** model is the
        final fallback and needs no probe.
    thresholds : tuple[float, ...]
        One threshold per escalation point (``len(cascade_models) - 1``).
        If the probe prediction for the current model is ≤ the threshold,
        the problem is escalated to the next model.
    """
    tier_costs = tier_costs_dict or {m: i + 1 for i, m in enumerate(cascade_models)}
    last_idx = len(cascade_models) - 1

    all_indices = sorted(test_dfs_dict[cascade_models[0]].index)
    rows: list[dict] = []

    for idx in all_indices:
        cur = 0
        row = test_dfs_dict[cascade_models[cur]].loc[idx]
        escalations = 0

        for th_i, th in enumerate(thresholds):
            if cur >= last_idx:
                break
            if "probe_pred" in row.index and row["probe_pred"] <= th:
                cur += 1
                row = test_dfs_dict[cascade_models[cur]].loc[idx]
                escalations += 1
            else:
                break

        rows.append({
            "idx": idx,
            "model_name": cascade_models[cur],
            chosen_metric: row[chosen_metric],
            "total_output_cost_usd": row["total_output_cost_usd"],
            "probe_pred": row.get("probe_pred"),
            "escalations": escalations,
        })

    return pd.DataFrame(rows)


def route_random(
    test_dfs_dict: dict[str, pd.DataFrame],
    models: list[str],
    chosen_metric: str,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Route every problem to a uniformly random model."""
    rng = np.random.RandomState(random_seed)
    all_indices = sorted(test_dfs_dict[models[0]].index)
    rows: list[dict] = []
    for idx in all_indices:
        m = models[rng.randint(len(models))]
        r = test_dfs_dict[m].loc[idx]
        rows.append({
            "idx": idx,
            "model_name": m,
            chosen_metric: r[chosen_metric],
            "total_output_cost_usd": r["total_output_cost_usd"],
        })
    return pd.DataFrame(rows)


def route_oracle_cascade(
    test_dfs_dict: dict[str, pd.DataFrame],
    cascade_models: list[str],
    chosen_metric: str,
    *,
    tier_costs_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Oracle cascade — for each problem, pick the **cheapest** model (in
    cascade order) that solves it correctly.  Falls back to the cheapest
    if none succeed.
    """
    tier_costs = tier_costs_dict or {m: i + 1 for i, m in enumerate(cascade_models)}
    ordered = sorted(cascade_models, key=lambda m: tier_costs[m])

    all_indices = sorted(test_dfs_dict[cascade_models[0]].index)
    rows: list[dict] = []

    for idx in all_indices:
        selected = ordered[0]  # default cheapest
        for m in ordered:
            if test_dfs_dict[m].loc[idx][chosen_metric] > 0.5:
                selected = m
                break
        r = test_dfs_dict[selected].loc[idx]
        rows.append({
            "idx": idx,
            "model_name": selected,
            chosen_metric: r[chosen_metric],
            "total_output_cost_usd": r["total_output_cost_usd"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Cascade comparison-data builder
# ──────────────────────────────────────────────
def build_cascade_comparison_data(
    test_dfs_with_probes: dict[str, pd.DataFrame],
    models: list[str],
    chosen_metric: str,
    *,
    cascade_results: dict[tuple, pd.DataFrame],
    routed_random: pd.DataFrame,
    routed_oracle: pd.DataFrame,
) -> dict[str, list]:
    """Assemble the dict consumed by ``plot_cost_vs_accuracy_cascade``."""
    data: dict[str, list] = {"Strategy": [], "Cost": [], chosen_metric: []}

    # "Always model X" baselines
    for m in models:
        df = test_dfs_with_probes[m]
        data["Strategy"].append(f"Always {m}")
        data["Cost"].append(df["total_output_cost_usd"].sum())
        data[chosen_metric].append(df[chosen_metric].mean())

    # Random
    data["Strategy"].append("Random Routing")
    data["Cost"].append(routed_random["total_output_cost_usd"].sum())
    data[chosen_metric].append(routed_random[chosen_metric].mean())

    # Oracle
    data["Strategy"].append("Oracle")
    data["Cost"].append(routed_oracle["total_output_cost_usd"].sum())
    data[chosen_metric].append(routed_oracle[chosen_metric].mean())

    # Cascade threshold sweep
    for thresholds, df in cascade_results.items():
        data["Strategy"].append(f"Cascade {thresholds}")
        data["Cost"].append(df["total_output_cost_usd"].sum())
        data[chosen_metric].append(df[chosen_metric].mean())

    return data


# ──────────────────────────────────────────────
# Cascade Pareto plot
# ──────────────────────────────────────────────
def _split_strategies_cascade(
    comparison_data,
    *,
    cost_col="Cost",
    strategy_col="Strategy",
    baseline_prefix="Always",
):
    """Parse comparison dict into baseline / random / oracle / cascade frames."""
    import re

    df = pd.DataFrame(comparison_data).copy()
    baseline_df = df[df[strategy_col].str.startswith(baseline_prefix)].copy()
    random_df = df[df[strategy_col].eq("Random Routing")].copy()
    oracle_df = df[df[strategy_col].eq("Oracle")].copy()

    cascade_mask = df[strategy_col].str.startswith("Cascade")
    cascade_df = df[cascade_mask].copy()
    if not cascade_df.empty:
        # Extract first float from "Cascade (0.7,)" etc.
        cascade_df["threshold"] = cascade_df[strategy_col].apply(
            lambda s: float(m.group(1)) if (m := re.search(r"([0-9]*\.?[0-9]+)", str(s))) else np.nan
        )
        cascade_df = cascade_df.sort_values(cost_col).reset_index(drop=True)

    return baseline_df, random_df, oracle_df, cascade_df, df


def plot_cost_vs_accuracy_cascade(
    comparison_data,
    *,
    metric_col: str,
    title: str = "Cost vs Accuracy (Cascade)",
    cost_col: str = "Cost",
    strategy_col: str = "Strategy",
    figsize: tuple[float, float] = (3.4, 3.2),
    xlim=None,
    ylim=None,
    xscale: str = "auto",
    symlog_linthresh: float = 0.5,
    robust_xlim: bool = True,
    frontier_highlight_k: int = 7,
    show_cascade_cloud: bool = True,
    show_cascade_frontier: bool = True,
    show_cascade_highlights: bool = True,
    label_tweaks: dict | None = None,
    baseline_palette: list[str] | None = None,
    cascade_color: str = "red",
    cascade_marker: str = "x",
    legend_loc=("upper center", (0.5, -0.22), 3),
):
    """Plot cost-vs-accuracy with cascade threshold sweep."""
    label_tweaks = label_tweaks or {}
    palette = baseline_palette or _DEFAULT_PALETTE

    baseline_df, random_df, oracle_df, cascade_df, df = _split_strategies_cascade(
        comparison_data, cost_col=cost_col, strategy_col=strategy_col,
    )

    if metric_col not in df.columns:
        raise ValueError(f"{metric_col!r} not in {list(df.columns)}")

    fig, ax = plt.subplots(figsize=figsize)

    # ---- cascade processing ----
    cx = cy = cfi = chi = None
    if not cascade_df.empty:
        cx = cascade_df[cost_col].to_numpy(float)
        cy = cascade_df[metric_col].to_numpy(float)
        cfi = pareto_frontier(cx, cy)
        cfi = cfi[np.argsort(cx[cfi])]
        chi = _downsample_by_cost(cx, cfi, k=frontier_highlight_k)

    # ---- axis limits ----
    all_costs = df[cost_col].to_numpy(float)
    all_costs = all_costs[np.isfinite(all_costs)]
    if not len(all_costs):
        all_costs = np.array([0.0, 1.0])
    cmax, c95 = float(all_costs.max()), float(np.percentile(all_costs, 95))

    xscale_use = xscale
    if xscale == "auto":
        xscale_use = "symlog" if (cmax / max(1e-9, c95)) >= 6 else "linear"
    if xscale_use == "symlog":
        ax.set_xscale("symlog", linthresh=symlog_linthresh, linscale=1.0)
    if xlim is None:
        xmax = c95 if robust_xlim else cmax
        xmax = max(1.0, xmax * 1.10)
        if xscale_use == "symlog":
            xmax = max(xmax, min(cmax * 1.05, xmax * 5))
        ax.set_xlim(0, xmax)
    else:
        ax.set_xlim(*xlim)

    if ylim is None:
        parts = []
        if not baseline_df.empty:
            parts.append(baseline_df[metric_col].to_numpy(float))
        if cy is not None:
            parts.append(cy)
        if not random_df.empty:
            parts.append(random_df[metric_col].to_numpy(float))
        if not oracle_df.empty:
            parts.append(oracle_df[metric_col].to_numpy(float))
        all_y = np.concatenate(parts) if parts else np.array([0.0, 1.0])
        all_y = all_y[np.isfinite(all_y)]
        if not len(all_y):
            all_y = np.array([0.0, 1.0])
        ax.set_ylim(float(all_y.min()) - 0.02, float(all_y.max()) + 0.015)
    else:
        ax.set_ylim(*ylim)

    # ---- styling ----
    ax.grid(True, alpha=0.12, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cost ($)", fontsize=9, labelpad=1)
    ax.set_ylabel("Accuracy", fontsize=9, labelpad=1)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(title, fontsize=9.5, pad=1)

    # ---- baselines ----
    for i, (_, row) in enumerate(baseline_df.iterrows()):
        x, y = float(row[cost_col]), float(row[metric_col])
        name = short_model_name(str(row[strategy_col]))
        ax.scatter(x, y, s=55, c=palette[i % len(palette)], edgecolors="black", linewidth=0.8, zorder=3)
        if name in label_tweaks:
            _place_label(ax, x, y, name, fontsize=8, **label_tweaks[name])
        else:
            xlim_hi = ax.get_xlim()[1]
            ylim_hi = ax.get_ylim()[1]
            dx = 5 if x < xlim_hi * 0.7 else -5
            dy = 5 if y < ylim_hi * 0.5 else -10
            ha = "left" if x < xlim_hi * 0.7 else "right"
            _place_label(ax, x, y, name, fontsize=8, dx=dx, dy=dy, ha=ha)

    # ---- random ----
    if not random_df.empty:
        r = random_df.iloc[0]
        x, y = float(r[cost_col]), float(r[metric_col])
        ax.scatter(x, y, s=55, c="#7f7f7f", marker="D", edgecolors="black", linewidth=0.8, zorder=3)
        dx = 8 if x < ax.get_xlim()[1] * 0.5 else -5
        ha = "left" if x < ax.get_xlim()[1] * 0.5 else "right"
        _place_label(ax, x, y, "Random", fontsize=8, dx=dx, dy=0, ha=ha)

    # ---- oracle ----
    if not oracle_df.empty:
        o = oracle_df.iloc[0]
        ax.scatter(
            float(o[cost_col]), float(o[metric_col]),
            s=120, c="#FFD700", marker="*", edgecolors="darkgoldenrod", linewidth=1.2, zorder=6,
        )
        _place_label(ax, float(o[cost_col]), float(o[metric_col]), "Oracle", fontsize=8, dx=3, dy=2)

    # ---- cascade cloud + frontier ----
    if cx is not None and len(cx):
        if show_cascade_cloud:
            alpha = max(0.15, min(0.6, 20 / len(cx)))
            ax.scatter(cx, cy, s=28, color=cascade_color, marker=cascade_marker,
                       linewidths=1.2, alpha=alpha, zorder=2)
        if show_cascade_frontier and cfi is not None and len(cfi):
            ax.plot(cx[cfi], cy[cfi], color=cascade_color, linewidth=2.4, alpha=0.95, zorder=3)
        if show_cascade_highlights and cfi is not None and len(cfi):
            ax.scatter(cx[cfi], cy[cfi], s=50, color=cascade_color, marker=cascade_marker,
                       linewidths=1.8, alpha=0.95, zorder=4)

    # ---- legend ----
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=5, label="Single model"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7f7f7f", markeredgecolor="black", markersize=5, label="Random"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700", markeredgecolor="darkgoldenrod", markersize=9, markeredgewidth=1.2, label="Oracle"),
        Line2D([0], [0], color=cascade_color, marker=cascade_marker, markeredgecolor=cascade_color, markersize=6, linewidth=2.4, label="Cascade (threshold sweep)"),
    ]
    loc, anchor, ncol = legend_loc
    ax.legend(handles=legend_elements, loc=loc, bbox_to_anchor=anchor, ncol=ncol, fontsize=8, frameon=False, handlelength=1.3, columnspacing=1.1)
    plt.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.30)
    return fig, ax
