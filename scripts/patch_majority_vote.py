"""
Post-hoc patch: add majority_vote_is_correct to parquet files that were generated
without pricing config (so the column was never computed).

Run from the repo root:
    cd /VData/linna4335/llms_know_difficult
    /opt/anaconda/envs/pika/bin/python3 scripts/patch_majority_vote.py

By default patches all parquet files under data/ that are missing the column.
Pass --dry_run to preview without writing.
Pass --models to restrict to specific model dirs (e.g. --models Qwen/Qwen2.5-7B).
"""

import argparse
import sys
from pathlib import Path

# Needed so create_sr_datasets/utils is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "create_sr_datasets"))

import pandas as pd
from utils.my_utils import add_majority_vote_answer
from utils.verification_math import compute_score


def patch_file(path: Path, dry_run: bool) -> str:
    df = pd.read_parquet(path)
    if "majority_vote_is_correct" in df.columns:
        return "skip (already has column)"

    if "generated_solutions" not in df.columns or "ground_truth" not in df.columns:
        return "skip (missing generated_solutions or ground_truth)"

    df["majority_vote_extracted_answer"] = df["generated_solutions"].apply(
        add_majority_vote_answer
    )
    df["majority_vote_is_correct"] = df.apply(
        lambda row: compute_score(
            solution_str=f"\\boxed{{{row['majority_vote_extracted_answer']}}}",
            ground_truth=row["ground_truth"],
        ),
        axis=1,
    )

    mv_acc = df["majority_vote_is_correct"].mean()

    if not dry_run:
        df.to_parquet(path, index=False)

    return f"{'(dry run) ' if dry_run else ''}patched  mv_acc={mv_acc:.4f}  n={len(df)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these model subdirs (e.g. Qwen/Qwen2.5-7B)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_dir)

    if args.models:
        # Expand org/model into two-level glob
        parquet_files = []
        for m in args.models:
            parquet_files.extend(data_root.glob(f"{m}/**/*.parquet"))
    else:
        parquet_files = list(data_root.glob("**/*.parquet"))

    parquet_files = sorted(f for f in parquet_files if "_backup" not in f.name)

    for path in parquet_files:
        result = patch_file(path, dry_run=args.dry_run)
        print(f"{result:60s}  {path.relative_to(data_root)}")


if __name__ == "__main__":
    main()
