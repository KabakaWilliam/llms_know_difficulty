"""
Build a cross-dataset AUROC (maj@5, linear_eoi_probe) table.

Single combined table across all models:
  col 0  = Model
  col 1  = Probe trained on  (source dataset)
  col 2+ = Evaluated on      (target datasets)
  diagonal (bold in LaTeX)   = in-dataset probe
  off-diagonal               = cross-dataset transfer

Usage:
    python scripts/cross_dataset_auroc_table.py
    python scripts/cross_dataset_auroc_table.py --latex
    python scripts/cross_dataset_auroc_table.py --models "Qwen/Qwen2.5-Math-1.5B-Instruct|3000|5|0.7"
"""

import argparse
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_CONFIGS = [
    "Qwen/Qwen2.5-1.5B-Instruct|3000|5|0.7",
    "Qwen/Qwen2.5-Math-1.5B-Instruct|3000|5|0.7",
    "Qwen/Qwen2.5-Math-7B-Instruct|3000|5|0.7",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|32768|5|0.6",
    "openai/gpt-oss-20b_low|131072|5|1.0",
    "openai/gpt-oss-20b_high|131072|5|1.0",
    "openai/gpt-oss-20b_medium|131072|5|1.0",
]

# Internal dataset name → friendly display label
DATASET_LABELS: dict[str, str] = {
    "DigitalLearningGmbH_MATH-lighteval": "MATH",
    "openai_gsm8k":                        "GSM8K",
    "gneubig_aime-1983-2024":              "AIME",
    # "E2H-AMC":                           "E2H-AMC",
}

DEFAULT_DATASETS = list(DATASET_LABELS.keys())

PROBE_TYPE = "linear_eoi_probe"
LABEL      = "majority_vote_is_correct"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_model_config(cfg: str) -> tuple[str, str, str, str]:
    model_part, maxlen, k, temp = cfg.split("|")
    org, model = model_part.split("/", 1)
    gen_str = f"maxlen_{maxlen}_k_{k}_temp_{temp}"
    return org, model, gen_str, model   # last item = display name


def latest_dir(base: Path) -> Path | None:
    """Return the most recent timestamp subdirectory (YYYYMMDD_*), or None."""
    if not base.exists():
        return None
    dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name[0].isdigit())
    return dirs[-1] if dirs else None


def in_dataset_auc(results_dir: Path, org: str, model: str, dataset: str,
                   gen_str: str) -> float | None:
    base = results_dir / org / model / dataset / PROBE_TYPE / gen_str / f"label_{LABEL}"
    ts_dir = latest_dir(base)
    if ts_dir is None:
        return None
    meta = ts_dir / "probe_metadata.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("auc")


def cross_dataset_auc(results_dir: Path, org: str, model: str,
                      source: str, target: str, gen_str: str) -> float | None:
    base = (results_dir / org / model / target / PROBE_TYPE / gen_str
            / f"label_{LABEL}" / f"probe_from_{source}")
    ts_dir = latest_dir(base)
    if ts_dir is None:
        return None
    preds = ts_dir / "predictions.json"
    if not preds.exists():
        return None
    data = json.loads(preds.read_text())
    return data.get("metrics", {}).get("auc")


def get_auc(results_dir: Path, org: str, model: str,
            source: str, target: str, gen_str: str) -> float | None:
    if source == target:
        return in_dataset_auc(results_dir, org, model, source, gen_str)
    return cross_dataset_auc(results_dir, org, model, source, target, gen_str)


def fmt_val(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "---"


# ── Plain text ────────────────────────────────────────────────────────────────

def print_plain(all_data: list[tuple[str, dict]], datasets: list[str]) -> None:
    """all_data = [(model_name, matrix), ...]"""
    ds_labels  = [DATASET_LABELS[d] for d in datasets]
    col_w      = max(len(l) for l in ds_labels)
    src_w      = max(len(DATASET_LABELS[d]) for d in datasets)
    model_w    = max(len(m) for m, _ in all_data)

    data_w = len("  ".join(f"{l:>{col_w}}" for l in ds_labels))
    indent = model_w + 2 + src_w + 2

    print()
    print(f"{'':{indent}}  {'Evaluated on →':>{data_w}}")
    header = (f"{'Model':<{model_w}}  {'↓ Trained on':<{src_w}}  "
              + "  ".join(f"{l:>{col_w}}" for l in ds_labels))
    print(header)
    print("-" * len(header))

    for i, (model_name, matrix) in enumerate(all_data):
        if i > 0:
            print()   # blank line between model groups
        for j, src in enumerate(datasets):
            src_label = DATASET_LABELS[src]
            # Print model name only on first source row of each group
            mdl_col = model_name if j == 0 else ""
            row_vals = [fmt_val(matrix.get((src, tgt))) for tgt in datasets]
            for k, tgt in enumerate(datasets):
                row_vals[k] = f"*{row_vals[k]}*" if src == tgt else f" {row_vals[k]} "
            print(f"{mdl_col:<{model_w}}  {src_label:<{src_w}}  "
                  + "  ".join(f"{v:>{col_w + 2}}" for v in row_vals))


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def build_latex(all_data: list[tuple[str, dict]], datasets: list[str]) -> str:
    ds_labels = [DATASET_LABELS[d] for d in datasets]
    n         = len(ds_labels)
    col_spec  = "ll" + "r" * n   # Model | Trained on | data cols

    lines = [
        r"\begin{tabular}{" + col_spec + r"}",
        r"  \toprule",
        f"  & & \\multicolumn{{{n}}}{{c}}{{\\textbf{{Evaluated on}}}} \\\\",
        r"  \cmidrule(l){3-" + str(n + 2) + r"}",
        r"  \textbf{Model} & \textbf{Trained on} & "
        + " & ".join(f"\\textbf{{{l}}}" for l in ds_labels) + r" \\",
        r"  \midrule",
    ]

    for i, (model_name, matrix) in enumerate(all_data):
        if i > 0:
            lines.append(r"  \midrule")
        n_src = len(datasets)
        for j, src in enumerate(datasets):
            src_label = DATASET_LABELS[src]
            cells = []
            for tgt in datasets:
                v = matrix.get((src, tgt))
                s = fmt_val(v)
                cells.append(f"\\textbf{{{s}}}" if src == tgt else s)
            # Use \multirow for model name, blank on subsequent rows
            if j == 0:
                mdl_cell = f"\\multirow{{{n_src}}}{{*}}{{{model_name}}}"
            else:
                mdl_cell = ""
            lines.append(f"  {mdl_cell} & {src_label} & " + " & ".join(cells) + r" \\")

    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_CONFIGS,
                        metavar="MODEL_CFG",
                        help='Model configs as "org/model|maxlen|k|temp"')
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        metavar="DATASET",
                        help="Internal dataset names to include")
    parser.add_argument("--results_dir", default="data/results")
    parser.add_argument("--out_dir", default="data/results",
                        help="Directory to save cross_dataset_auroc_table.tex (default: data/results)")
    parser.add_argument("--latex", action="store_true", default=False,
                        help="Print only LaTeX (omit plain table)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    datasets    = args.datasets

    # Gather all model data first
    all_data: list[tuple[str, dict]] = []
    for cfg in args.models:
        org, model, gen_str, _ = parse_model_config(cfg)
        matrix: dict[tuple[str, str], float | None] = {}
        for src in datasets:
            for tgt in datasets:
                matrix[(src, tgt)] = get_auc(results_dir, org, model, src, tgt, gen_str)
        all_data.append((model, matrix))

    if not args.latex:
        print_plain(all_data, datasets)

    latex = build_latex(all_data, datasets)
    print("\n" + latex)

    # Save .tex file
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "cross_dataset_auroc_table.tex"
    tex_path.write_text(latex + "\n")
    print(f"\n✅ LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()
