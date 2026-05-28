"""
Collect the latest probe_metadata.json for each (model, dataset, probe_type, gen_str, label)
combination into a single share directory.

Output structure:
  probe_metadata_2_share/{org}/{model}/{dataset}/{probe_type}/{gen_str}/label_{metric}/{timestamp}/probe_metadata.json

Only the most recent timestamp per combination is copied.

Usage:
    python scripts/collect_probe_metadata.py
    python scripts/collect_probe_metadata.py --probe_type tfidf_probe
    python scripts/collect_probe_metadata.py --models "Qwen/Qwen2.5-Math-7B-Instruct|3000|5|0.7" "openai/gpt-oss-20b_high|131072|5|1.0"
    python scripts/collect_probe_metadata.py --results_dir data/results --out_dir probe_metadata_2_share
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

# Default model configs — same format as MODEL_CONFIGS in run_cross_dataset_probe_sweep.sh:
#   "org/model|maxlen|k|temp"
DEFAULT_MODEL_CONFIGS = [
    "Qwen/Qwen2.5-Math-7B-Instruct|3000|5|0.7",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|32768|5|0.6",
    "openai/gpt-oss-20b_low|131072|5|1.0",
    "openai/gpt-oss-20b_high|131072|5|1.0",
    "openai/gpt-oss-20b_medium|131072|5|1.0",
]


def parse_model_config(cfg: str) -> tuple[str, str, str]:
    """Parse 'org/model|maxlen|k|temp' into (org, model, gen_str)."""
    model_part, maxlen, k, temp = cfg.split("|")
    org, model = model_part.split("/", 1)
    gen_str = f"maxlen_{maxlen}_k_{k}_temp_{temp}"
    return org, model, gen_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="data/results")
    parser.add_argument("--out_dir", default="probe_metadata_2_share")
    parser.add_argument(
        "--probe_type",
        default="linear_eoi_probe",
        help="Probe type to collect (default: linear_eoi_probe). Pass 'all' to collect every type.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_CONFIGS,
        metavar="MODEL_CFG",
        help='Model configs as "org/model|maxlen|k|temp". Defaults to DEFAULT_MODEL_CONFIGS.',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    # Build a dict of (org, model) -> gen_str for filtering
    model_filter: dict[tuple[str, str], str] = {}
    for cfg in args.models:
        org, model, gen_str = parse_model_config(cfg)
        model_filter[(org, model)] = gen_str

    collect_all_probe_types = args.probe_type == "all"

    # Collect all probe_metadata.json files, grouped by their 6-part key
    # Structure: results/{org}/{model}/{dataset}/{probe_type}/{gen_str}/label_{metric}/{timestamp}/
    groups: dict[tuple, list] = defaultdict(list)

    for meta_path in results_dir.rglob("probe_metadata.json"):
        parts = meta_path.parts
        try:
            root_idx = parts.index(results_dir.parts[-1])
            org        = parts[root_idx + 1]
            model      = parts[root_idx + 2]
            dataset    = parts[root_idx + 3]
            probe_type = parts[root_idx + 4]
            gen_str    = parts[root_idx + 5]
            label      = parts[root_idx + 6]
            timestamp  = parts[root_idx + 7]
        except (ValueError, IndexError):
            continue  # old path structure — silently skip

        # Skip cross-dataset inference outputs
        if "probe_from_" in str(meta_path):
            continue

        # Filter by model
        if (org, model) not in model_filter:
            continue

        # Filter by gen_str (must match the model's configured maxlen/k/temp)
        if gen_str != model_filter[(org, model)]:
            continue

        # Filter by probe type
        if not collect_all_probe_types and probe_type != args.probe_type:
            continue

        key = (org, model, dataset, probe_type, gen_str, label)
        groups[key].append((timestamp, meta_path))

    copied = 0
    for key, entries in groups.items():
        org, model, dataset, probe_type, gen_str, label = key

        # Keep only the latest timestamp (lexicographic sort works for YYYYMMDD_HHMMSS)
        latest_ts, src_path = sorted(entries)[-1]

        dest_path = (
            out_dir / org / model / dataset / probe_type / gen_str / label / latest_ts / "probe_metadata.json"
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        copied += 1

    print(f"Copied {copied} probe_metadata.json files to {out_dir}/")
    print(f"  probe_type : {args.probe_type}")
    for cfg in args.models:
        org, model, gen_str = parse_model_config(cfg)
        print(f"  {org}/{model}  ({gen_str})")


if __name__ == "__main__":
    main()
