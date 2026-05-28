#!/usr/bin/env python3
"""
Plot probe test performance vs layer number, one figure per (model, dataset).

Walks the results directory tree, extracts the test score from each run's
probe_metadata.json, and emits one PNG per (model, dataset) combo with
one line per probe.

Expected results layout (from src/pika/utils.py:create_results_path):
    {results_root}/{model_family}/{model_name}/{dataset}/{probe}/
        {gen_str}/label_{label_column}/{timestamp}/
            probe_metadata.json

Where gen_str looks like: maxlen_3000_k_5_temp_0.7_layers_1

Usage:
    python3 plot_layer_sweep.py
    python3 plot_layer_sweep.py --results-dir /path/to/data/results
    python3 plot_layer_sweep.py --label majority_vote_is_correct
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# gen_str = "maxlen_{ml}_k_{k}_temp_{t}_layers_{L}"
# The trailing _layers_{L} is optional so the parser still works if the
# pipeline is ever changed to drop it.
GEN_STR_RE = re.compile(
    r"^maxlen_(?P<max_len>\d+)"
    r"_k_(?P<k>\d+)"
    r"_temp_(?P<temperature>[\d.]+)"
    r"(?:_layers_(?P<layer>\d+))?$"
)


def parse_gen_str(gen_str: str) -> dict | None:
    m = GEN_STR_RE.match(gen_str)
    if not m:
        return None
    g = m.groupdict()
    return {
        "max_len": int(g["max_len"]),
        "k": int(g["k"]),
        "temperature": float(g["temperature"]),
        "layer_from_path": int(g["layer"]) if g["layer"] is not None else None,
    }


def extract_test_score(meta: dict) -> tuple[float | None, str]:
    """Return (score, metric_name). Prefers task_type to pick the metric label."""
    task = meta.get("task_type", "")
    metric_name = "AUC" if task == "classification" else (
        "Spearman" if task == "regression" else "test_score"
    )

    # main.py sets `test_score` to the primary metric; fall back to raw fields.
    score = meta.get("test_score")
    if score is None:
        score = meta.get("auc")
    if score is None:
        score = meta.get("spearman")

    if score is None:
        return None, metric_name
    try:
        return float(score), metric_name
    except (TypeError, ValueError):
        return None, metric_name


def collect_runs(results_root: Path) -> pd.DataFrame:
    """Find every probe_metadata.json under results_root and assemble a tidy table."""
    rows = []
    n_skipped = 0
    for meta_path in results_root.rglob("probe_metadata.json"):
        try:
            rel_parts = meta_path.relative_to(results_root).parts
        except ValueError:
            n_skipped += 1
            continue

        # Expect 8 parts: family / model / dataset / probe / gen_str / label_X / ts / probe_metadata.json
        if len(rel_parts) < 8:
            n_skipped += 1
            continue
        family, model_name, dataset, probe, gen_str, label_part, timestamp = rel_parts[:7]

        if not label_part.startswith("label_"):
            n_skipped += 1
            continue
        label = label_part[len("label_"):]

        gen = parse_gen_str(gen_str)
        if gen is None:
            n_skipped += 1
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue

        # Prefer best_layer_idx from metadata; fall back to layer parsed from path.
        layer = meta.get("best_layer_idx")
        if layer is None:
            layer = gen["layer_from_path"]
        if layer is None:
            n_skipped += 1
            continue
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            n_skipped += 1
            continue

        score, metric_name = extract_test_score(meta)
        if score is None:
            n_skipped += 1
            continue

        rows.append({
            "model": f"{family}/{model_name}",
            "dataset": dataset,
            "probe": probe,
            "label": label,
            "layer": layer,
            "test_score": score,
            "metric": metric_name,
            "max_len": gen["max_len"],
            "k": gen["k"],
            "temperature": gen["temperature"],
            "timestamp": timestamp,
            "meta_path": str(meta_path),
        })

    if n_skipped:
        print(f"  (skipped {n_skipped} non-matching / unreadable paths)")
    return pd.DataFrame(rows)


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Within (model, dataset, probe, label, layer, gen_config), keep the latest timestamp."""
    if df.empty:
        return df
    key = ["model", "dataset", "probe", "label", "layer",
           "max_len", "k", "temperature"]
    df = df.sort_values("timestamp")
    deduped = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    n_dropped = len(df) - len(deduped)
    if n_dropped:
        print(f"  (deduped {n_dropped} older timestamps)")
    return deduped


def format_markdown_tables(df: pd.DataFrame) -> str:
    """Render one markdown table per (dataset, probe, label) group.

    Columns are models, rows are layers, cells are test scores.
    """
    if df.empty:
        return ""

    blocks = []
    # Sort groups for stable ordering.
    grouped = df.groupby(["dataset", "probe", "label"], sort=True)
    for (dataset, probe, label), group in grouped:
        metric = group["metric"].iloc[0]

        # Pivot to model × layer. Use mean as the aggregator just in case
        # any duplicate slipped past dedup_latest — dedup should already
        # guarantee one row per (model, layer) for this group.
        pivot = group.pivot_table(
            index="layer",
            columns="model",
            values="test_score",
            aggfunc="mean",
        ).sort_index()

        models = sorted(pivot.columns)

        header = f"### {dataset} · {probe} · label={label} · Test {metric}"
        lines = [header, ""]
        lines.append("| Layer | " + " | ".join(models) + " |")
        lines.append("|---:|" + "---:|" * len(models))
        for layer in pivot.index:
            cells = []
            for m in models:
                v = pivot.loc[layer, m]
                cells.append("—" if pd.isna(v) else f"{v:.4f}")
            lines.append(f"| {layer} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def emit_markdown_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """Print markdown tables to stdout and persist them to out_dir."""
    md = format_markdown_tables(df)
    if not md:
        return

    print()
    print("=" * 78)
    print("Results — markdown")
    print("=" * 78)
    print(md)
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "results_table.md"
    md_path.write_text(md + "\n")
    print(f"Markdown table -> {md_path}")


def plot_per_model_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("No runs found — nothing to plot.")
        return

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for probe, sub in group.groupby("probe"):
            sub = sub.sort_values("layer")
            ax.plot(
                sub["layer"], sub["test_score"],
                marker="o", linewidth=1.8, markersize=6,
                label=probe,
            )

        metric = group["metric"].iloc[0]
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Test {metric}")
        ax.set_title(f"{model}  —  {dataset}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Probe", loc="best")

        # Force integer ticks on layers and put a guideline at chance (0.5) for classification.
        layers_present = sorted(group["layer"].unique())
        if layers_present:
            ax.set_xticks(layers_present)
        if metric == "AUC":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
                       alpha=0.6, label="_chance")

        fig.tight_layout()
        safe_model = model.replace("/", "_")
        out_path = out_dir / f"{safe_model}__{dataset}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}  ({len(group)} points across "
              f"{group['probe'].nunique()} probe(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Plot probe test performance vs layer.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("data/results"),
        help="Root of the results tree (default: data/results)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/plots/layer_sweep"),
        help="Where to write the figures (default: data/plots/layer_sweep)")
    parser.add_argument(
        "--label", type=str, default=None,
        help="Filter to a specific label column (e.g. majority_vote_is_correct). "
             "If unset, every label column found is plotted as its own figure set.")
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {args.results_dir}")

    print(f"Scanning {args.results_dir} ...")
    df = collect_runs(args.results_dir)

    if args.label is not None:
        before = len(df)
        df = df[df["label"] == args.label].copy()
        print(f"Filtered to label={args.label!r}: {len(df)}/{before} runs")

    df = dedupe_latest(df)

    print(
        f"Collected {len(df)} runs: "
        f"{df['model'].nunique()} model(s), "
        f"{df['dataset'].nunique()} dataset(s), "
        f"{df['probe'].nunique()} probe(s), "
        f"{df['label'].nunique()} label(s)"
    )

    if df.empty:
        return

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        print(
            f"  {model} | {dataset}: "
            f"{group['probe'].nunique()} probe(s) × {group['layer'].nunique()} layer(s)"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Tidy summary -> {csv_path}")

    emit_markdown_tables(df, args.out_dir)

    # If the user didn't pass --label, plot each label column separately so the
    # y-axis stays meaningful within a single figure.
    if args.label is None and df["label"].nunique() > 1:
        for label, ldf in df.groupby("label"):
            sub_out = args.out_dir / f"label_{label}"
            print(f"Plotting label={label} -> {sub_out}")
            plot_per_model_dataset(ldf, sub_out)
    else:
        plot_per_model_dataset(df, args.out_dir)


if __name__ == "__main__":
    main()