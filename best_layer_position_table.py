#!/usr/bin/env python3
"""Walk a probe-results directory, read each probe_metadata.json, and build a
table mapping (model, dataset, label) -> (best_layer_idx, best_pos_idx).

Expected directory layout:
    <root>/<org>/<model>/<dataset>/<probe_type>/<config>/<label>/<timestamp>/probe_metadata.json

Usage:
    python summarize_probes.py /path/to/root
    python summarize_probes.py /path/to/root --pivot --latest-only
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_path(p: Path, root: Path):
    """Pull (org, model, dataset, probe_type, config, label, timestamp) out of the
    path relative to the root. Returns None if the layout doesn't match."""
    rel = p.relative_to(root).parts
    # ..., org, model, dataset, probe_type, config, label, timestamp, probe_metadata.json
    if len(rel) < 8:
        return None
    org, model, dataset, probe_type, config, label, timestamp, _ = rel[-8:]
    return {
        "model": f"{org}/{model}",
        "dataset": dataset,
        "probe_type": probe_type,
        "config": config,
        "label": label,
        "timestamp": timestamp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root directory containing model dirs")
    ap.add_argument("--out", type=Path, default=Path("probe_summary.csv"),
                    help="Output CSV for the long-format table")
    ap.add_argument("--pivot", action="store_true",
                    help="Also write a pivot table (rows=model, cols=dataset/label)")
    ap.add_argument("--latest-only", action="store_true",
                    help="If multiple timestamps exist for the same "
                         "(model, dataset, label), keep only the latest")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Only include these label types. Accepts names with or "
                         "without the 'label_' prefix, e.g. "
                         "--labels success_rate majority_vote_is_correct")
    args = ap.parse_args()

    # Normalize the user's label filter to the on-disk form (label_*).
    label_filter = None
    if args.labels:
        label_filter = {
            lbl if lbl.startswith("label_") else f"label_{lbl}"
            for lbl in args.labels
        }

    rows = []
    for f in args.root.rglob("probe_metadata.json"):
        meta = parse_path(f, args.root)
        if meta is None:
            print(f"skip (unexpected layout): {f}")
            continue
        if label_filter is not None and meta["label"] not in label_filter:
            continue
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip (could not read): {f} ({e})")
            continue
        rows.append({
            **meta,
            "best_layer_idx": data.get("best_layer_idx"),
            "best_pos_idx": data.get("best_pos_idx"),
            "best_val_score": data.get("best_val_score"),
            "test_score": data.get("test_score"),
            "path": str(f),
        })

    if not rows:
        if label_filter is not None:
            print(f"No probe_metadata.json files matched labels {sorted(label_filter)}.")
        else:
            print("No probe_metadata.json files found.")
        return

    df = pd.DataFrame(rows)

    if args.latest_only:
        df = (df.sort_values("timestamp")
                .groupby(["model", "dataset", "label"], as_index=False)
                .tail(1)
                .reset_index(drop=True))

    df["best_layer_pos"] = df.apply(
        lambda r: f"L{r['best_layer_idx']}/P{r['best_pos_idx']}", axis=1
    )

    long_cols = ["model", "dataset", "label",
                 "best_layer_idx", "best_pos_idx",
                 "best_val_score", "test_score", "timestamp"]
    long_df = df[long_cols].sort_values(["model", "dataset", "label"]).reset_index(drop=True)

    print(long_df.to_string(index=False))
    long_df.to_csv(args.out, index=False)
    print(f"\nSaved long-format table to {args.out}")

    if args.pivot:
        pivot = df.pivot_table(
            index="model",
            columns=["dataset", "label"],
            values="best_layer_pos",
            aggfunc="last",
        )
        pivot_path = args.out.with_name(args.out.stem + "_pivot.csv")
        pivot.to_csv(pivot_path)
        print(f"\nSaved pivot table to {pivot_path}")
        print(pivot)



if __name__ == "__main__":
    main()