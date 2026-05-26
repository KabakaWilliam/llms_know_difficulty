"""Print row counts per split for each dataset using data/openai/gpt-oss-20b_high.

Usage:
    python scripts/dataset_split_counts.py          # plain table (default)
    python scripts/dataset_split_counts.py --latex  # LaTeX tabular
"""
import argparse
import glob
import os
import pyarrow.parquet as pq

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "openai", "gpt-oss-20b_high")

DATASETS = [
    "DigitalLearningGmbH_MATH-lighteval",
    "E2H-AMC",
    "gneubig_aime-1983-2024",
    "opencompass_AIME2025",
    "openai_gsm8k",
    "livecodebench_code_generation_lite",
]

SPLITS = ["train", "val", "test"]


def find_split_file(dataset_path, split, k=5):
    pattern = os.path.join(dataset_path, f"{split}_maxlen_*_k_{k}_temp_*.parquet")
    files = [f for f in glob.glob(pattern) if "_backup" not in f]
    return sorted(files)[-1] if files else None


def row_count(path):
    return pq.read_metadata(path).num_rows


rows = []
for dataset in DATASETS:
    dataset_path = os.path.join(BASE_DIR, dataset)
    counts = {}
    for split in SPLITS:
        f = find_split_file(dataset_path, split)
        counts[split] = row_count(f) if f else None
    rows.append((dataset, counts))

parser = argparse.ArgumentParser()
parser.add_argument("--latex", action="store_true", help="Output as LaTeX tabular")
args = parser.parse_args()


def fmt_plain(v):
    return f"{v:,}" if v is not None else "—"


def fmt_latex(v):
    return f"{v:,}" if v is not None else "---"


if args.latex:
    print(r"\begin{tabular}{lrrr}")
    print(r"  \toprule")
    print(r"  Dataset & Train & Val & Test \\")
    print(r"  \midrule")
    for dataset, counts in rows:
        f = fmt_latex
        print(f"  {dataset} & {f(counts['train'])} & {f(counts['val'])} & {f(counts['test'])} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")
else:
    col_w = max(len(d) for d, _ in rows)
    header = f"{'Dataset':<{col_w}}  {'train':>8}  {'val':>8}  {'test':>8}"
    print(header)
    print("-" * len(header))
    for dataset, counts in rows:
        f = fmt_plain
        print(f"{dataset:<{col_w}}  {f(counts['train']):>8}  {f(counts['val']):>8}  {f(counts['test']):>8}")
