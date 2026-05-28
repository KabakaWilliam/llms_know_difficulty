"""
Post-hoc patch: compute and fill pricing columns for parquet files generated
without a pricing config (so costs were left as nan/0).

Fills / overwrites:
  - total_output_tokens       (sum of output_tokens across rollouts)
  - input_cost_usd_once       (total_input_tokens * in_rate / 1M)
  - total_output_cost_usd     (total_output_tokens * out_rate / 1M)
  - total_cost_usd            (input_cost_usd_once + total_output_cost_usd)

Run from the repo root:
    /opt/anaconda/envs/pika/bin/python3 scripts/patch_pricing.py --models "Qwen/Qwen2.5-7B"
    /opt/anaconda/envs/pika/bin/python3 scripts/patch_pricing.py --models "Qwen/Qwen3-4B-Thinking-2507"
    /opt/anaconda/envs/pika/bin/python3 scripts/patch_pricing.py --dry_run --models "Qwen/Qwen2.5-7B"
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "create_sr_datasets"))

import numpy as np
import pandas as pd

TOKENS_PER_MILLION = 1_000_000

# Pricing per model: (input_per_mill, output_per_mill)
PRICING = {
    "Qwen/Qwen2.5-7B":             (0.20, 0.20),
    "Qwen/Qwen3-4B-Thinking-2507": (0.20, 0.20),
}


def get_total_output_tokens(solutions) -> int:
    return sum(int(s["output_tokens"]) for s in solutions if s.get("output_tokens") is not None)


def patch_file(path: Path, in_rate: float, out_rate: float, dry_run: bool) -> str:
    df = pd.read_parquet(path)

    if "total_input_tokens" not in df.columns:
        return "skip (no total_input_tokens)"

    df["total_output_tokens"] = df["generated_solutions"].apply(get_total_output_tokens)
    df["input_cost_usd_once"]  = df["total_input_tokens"] * in_rate  / TOKENS_PER_MILLION
    df["total_output_cost_usd"] = df["total_output_tokens"] * out_rate / TOKENS_PER_MILLION
    df["total_cost_usd"]       = df["input_cost_usd_once"] + df["total_output_cost_usd"]

    total_cost = df["total_cost_usd"].sum()
    avg_out_tok = df["total_output_tokens"].mean()

    if not dry_run:
        df.to_parquet(path, index=False)

    return (
        f"{'(dry run) ' if dry_run else ''}patched  "
        f"avg_out_tok={avg_out_tok:.0f}  total_cost=${total_cost:.4f}  n={len(df)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--models", nargs="+", default=list(PRICING.keys()),
                        help="Model subdirs to patch (e.g. Qwen/Qwen2.5-7B)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_dir)

    for model in args.models:
        if model not in PRICING:
            print(f"WARNING: no pricing entry for {model} — skipping")
            continue

        in_rate, out_rate = PRICING[model]
        print(f"\n=== {model}  (in=${in_rate}/M  out=${out_rate}/M) ===")

        files = sorted(f for f in data_root.glob(f"{model}/**/*.parquet")
                       if "_backup" not in f.name)

        if not files:
            print("  no parquet files found")
            continue

        for path in files:
            result = patch_file(path, in_rate, out_rate, dry_run=args.dry_run)
            print(f"  {result:70s}  {path.relative_to(data_root)}")


if __name__ == "__main__":
    main()
