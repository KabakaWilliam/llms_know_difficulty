#!/usr/bin/env python3
"""
Prepare probes for HuggingFace Hub upload.

This script:
1. Scans data/results for trained probes
2. Organizes them into a clean folder structure
3. Creates a manifest of all available probes
4. Generates the model card README

Usage:
    python scripts/prepare_hf_probes.py --output-dir probes_for_hf
    python scripts/prepare_hf_probes.py --output-dir probes_for_hf --probe-type linear_eoi_probe
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


def sanitize_name(name: str) -> str:
    """Convert a path-style name to a clean identifier."""
    return name.replace("/", "--").replace("_", "-")


def get_clean_probe_name(
    model_name: str,
    dataset: str,
    probe_type: str,
    label_type: str,
    k: int,
    temp: float
) -> str:
    """Generate a clean, descriptive probe folder name."""
    model_short = model_name.split("/")[-1]
    dataset_short = dataset.split("/")[-1].split("_")[0]  # First part of dataset name
    return f"{model_short}--{dataset_short}--{probe_type}--{label_type}--k{k}-t{temp}"


def parse_probe_path(probe_dir: Path) -> Optional[dict]:
    """
    Parse probe directory path to extract metadata.
    
    Expected structure:
    data/results/{org}/{model}/{dataset}/{probe_type}/maxlen_{}_k_{}_temp_{}/label_{}/timestamp/
    """
    parts = probe_dir.parts
    
    # Find the 'results' marker
    try:
        results_idx = parts.index("results")
    except ValueError:
        return None
    
    remaining = parts[results_idx + 1:]
    if len(remaining) < 6:
        return None
    
    # Parse the path components
    org = remaining[0]
    model = remaining[1]
    dataset = remaining[2]
    probe_type = remaining[3]
    
    # Parse config string like "maxlen_3000_k_5_temp_0.7"
    config_str = remaining[4]
    config = {}
    if config_str.startswith("maxlen_"):
        config_parts = config_str.split("_")
        for i, part in enumerate(config_parts):
            if part == "maxlen" and i + 1 < len(config_parts):
                config["max_len"] = int(config_parts[i + 1])
            elif part == "k" and i + 1 < len(config_parts):
                config["k"] = int(config_parts[i + 1])
            elif part == "temp" and i + 1 < len(config_parts):
                config["temperature"] = float(config_parts[i + 1])
    else:
        # Old format without config prefix
        config = {"max_len": None, "k": None, "temperature": None}
    
    # Parse label type
    label_str = remaining[5] if len(remaining) > 5 else "unknown"
    label_type = label_str.replace("label_", "") if label_str.startswith("label_") else label_str
    
    # Timestamp
    timestamp = remaining[6] if len(remaining) > 6 else None
    
    return {
        "org": org,
        "model": model,
        "model_name": f"{org}/{model}",
        "dataset": dataset,
        "probe_type": probe_type,
        "label_type": label_type,
        "max_len": config.get("max_len"),
        "k": config.get("k"),
        "temperature": config.get("temperature"),
        "timestamp": timestamp,
        "source_path": str(probe_dir),
    }


def find_all_probes(results_dir: Path, probe_type_filter: Optional[str] = None) -> list[dict]:
    """Find all trained probes in the results directory."""
    probes = []
    
    # Look for probe marker files
    for probe_file in results_dir.rglob("best_probe.joblib"):
        probe_dir = probe_file.parent
        
        # Parse path metadata
        path_meta = parse_probe_path(probe_dir)
        if path_meta is None:
            print(f"  Skipping (couldn't parse path): {probe_dir}")
            continue
        
        # Filter by probe type if specified
        if probe_type_filter and path_meta["probe_type"] != probe_type_filter:
            continue
        
        # Load the probe metadata JSON
        metadata_file = probe_dir / "probe_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                json_meta = json.load(f)
            path_meta.update(json_meta)
        
        # Check for additional files
        path_meta["has_platt_scaler"] = (probe_dir / "platt_scaler.joblib").exists()
        
        probes.append(path_meta)
    
    # Also look for TF-IDF probes (model.joblib)
    for probe_file in results_dir.rglob("model.joblib"):
        probe_dir = probe_file.parent
        
        # Only if this is a tfidf_probe
        if "tfidf_probe" not in str(probe_dir):
            continue
        
        # Filter by probe type if specified
        if probe_type_filter and probe_type_filter != "tfidf_probe":
            continue
        
        path_meta = parse_probe_path(probe_dir)
        if path_meta is None:
            continue
        
        path_meta["has_vectorizer"] = (probe_dir / "vectorizer.joblib").exists()
        
        metadata_file = probe_dir / "probe_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                json_meta = json.load(f)
            path_meta.update(json_meta)
        
        probes.append(path_meta)
    
    return probes


def select_best_probes(probes: list[dict]) -> list[dict]:
    """
    Select the best probe for each (model, dataset, probe_type, label_type) combination.
    Uses test_score or best_val_score to select.
    """
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for p in probes:
        key = (p["model_name"], p["dataset"], p["probe_type"], p["label_type"])
        grouped[key].append(p)
    
    best_probes = []
    for key, candidates in grouped.items():
        # Sort by test_score (or best_val_score as fallback)
        def get_score(p):
            return p.get("test_score") or p.get("best_val_score") or 0
        
        candidates.sort(key=get_score, reverse=True)
        best_probes.append(candidates[0])
    
    return best_probes


def copy_probe_to_output(probe: dict, output_dir: Path) -> str:
    """Copy a probe to the output directory with clean naming."""
    # Generate clean folder name
    k = probe.get("k") or 1
    temp = probe.get("temperature") or 0.0
    clean_name = get_clean_probe_name(
        model_name=probe["model_name"],
        dataset=probe["dataset"],
        probe_type=probe["probe_type"],
        label_type=probe["label_type"],
        k=k,
        temp=temp,
    )
    
    dest_dir = output_dir / clean_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(probe["source_path"])
    
    # Copy probe files
    files_to_copy = [
        "best_probe.joblib",
        "platt_scaler.joblib",
        "probe_metadata.json",
        "model.joblib",        # for tfidf
        "vectorizer.joblib",   # for tfidf
    ]
    
    copied_files = []
    for filename in files_to_copy:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, dest_dir / filename)
            copied_files.append(filename)
    
    # Create enhanced metadata
    enhanced_meta = {
        "probe_name": clean_name,
        "model_name": probe["model_name"],
        "dataset": probe["dataset"],
        "probe_type": probe["probe_type"],
        "label_type": probe["label_type"],
        "k": k,
        "temperature": temp,
        "max_len": probe.get("max_len"),
        "best_layer_idx": probe.get("best_layer_idx"),
        "best_position_value": probe.get("best_position_value"),
        "best_alpha": probe.get("best_alpha"),
        "d_model": probe.get("d_model"),
        "task_type": probe.get("task_type"),
        "test_score": probe.get("test_score"),
        "best_val_score": probe.get("best_val_score"),
        "files": copied_files,
        "exported_at": datetime.now().isoformat(),
    }
    
    with open(dest_dir / "config.json", "w") as f:
        json.dump(enhanced_meta, f, indent=2)
    
    return clean_name


def generate_model_card(probes: list[dict], output_dir: Path) -> None:
    """Generate the HuggingFace model card README.md."""
    
    # Group probes by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for p in probes:
        by_model[p["model_name"]].append(p)
    
    readme = f"""---
library_name: pika
tags:
  - probe
  - difficulty-prediction
  - routing
  - llm-routing
  - interpretability
license: mit
---

# PIKA Probes - Pre-trained Difficulty Prediction Probes

This repository contains pre-trained linear probes for predicting task difficulty from LLM activations.
These probes are part of the [PIKA (Probe-Informed K-Aware Routing)](https://github.com/KabakaWilliam/llms_know_difficulty) project.

## Overview

These probes are trained on internal LLM representations to predict:
- **Success rate**: Probability that the model will correctly solve a problem
- **Majority vote correctness**: Whether the most common answer is correct

The predictions can be used for:
- **Routing**: Direct easy/hard problems to appropriate models
- **Calibration**: Estimate uncertainty before generation
- **Analysis**: Understand model capabilities across problem types

## Installation

```bash
pip install pika
# or
pip install huggingface_hub
```

## Quick Start

### Load a probe directly

```python
from huggingface_hub import snapshot_download
from pika.probe import LinearEoiProbe

# Download probe
probe_path = snapshot_download(
    repo_id="KabakaWilliam/pika-probes",
    allow_patterns=["Qwen2.5-Math-7B-Instruct--MATH--linear-eoi-probe--success-rate--k5-t0.7/*"]
)

# Load and use
probe = LinearEoiProbe.load_from_checkpoint(probe_path)
predictions = probe.predict(texts=["Solve: 2x + 3 = 7"])
```

### Using PIKA's built-in downloader

```python
from pika.hub import download_probe

probe = download_probe(
    repo_id="KabakaWilliam/pika-probes",
    probe_name="Qwen2.5-Math-7B-Instruct--MATH--linear-eoi-probe--success-rate--k5-t0.7"
)
```

## Available Probes

| Model | Dataset | Probe Type | Label | k | Temp | Score |
|-------|---------|------------|-------|---|------|-------|
"""
    
    # Build table rows
    for model_name in sorted(by_model.keys()):
        for p in sorted(by_model[model_name], key=lambda x: (x["dataset"], x["probe_type"])):
            model_short = model_name.split("/")[-1]
            dataset_short = p["dataset"].split("_")[0]
            probe_type = p["probe_type"].replace("_", " ")
            label = p["label_type"].replace("_", " ")
            k = p.get("k") or "-"
            temp = p.get("temperature") or "-"
            score = p.get("test_score") or p.get("best_val_score") or "-"
            if isinstance(score, float):
                score = f"{score:.3f}"
            readme += f"| {model_short} | {dataset_short} | {probe_type} | {label} | {k} | {temp} | {score} |\n"

    readme += """
## Probe Structure

Each probe folder contains:
- `best_probe.joblib`: Trained sklearn LogisticRegression model
- `platt_scaler.joblib`: Optional Platt calibration scaler
- `config.json`: Full probe configuration and metadata

## Training Your Own Probes

```bash
python -m pika.main \\
    --probe linear_eoi_probe \\
    --dataset DigitalLearningGmbH_MATH-lighteval \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --max_len 3000 --k 50 --temperature 0.7
```

## Citation

If you use these probes in your research, please cite:

```bibtex
@misc{{pika2026,
  title={{PIKA: Probe-Informed K-Aware Routing for LLMs}},
  author={{Kabaka, William et al.}},
  year={{2026}},
  url={{https://github.com/KabakaWilliam/llms_know_difficulty}}
}}
```

## License

MIT License
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def generate_manifest(probes: list[dict], output_dir: Path) -> None:
    """Generate a manifest.json listing all probes."""
    manifest = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "num_probes": len(probes),
        "probes": [],
    }
    
    for p in probes:
        k = p.get("k") or 1
        temp = p.get("temperature") or 0.0
        clean_name = get_clean_probe_name(
            model_name=p["model_name"],
            dataset=p["dataset"],
            probe_type=p["probe_type"],
            label_type=p["label_type"],
            k=k,
            temp=temp,
        )
        manifest["probes"].append({
            "name": clean_name,
            "model": p["model_name"],
            "dataset": p["dataset"],
            "probe_type": p["probe_type"],
            "label_type": p["label_type"],
        })
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare probes for HuggingFace upload")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"),
                        help="Directory containing trained probes")
    parser.add_argument("--output-dir", type=Path, default=Path("probes_for_hf"),
                        help="Output directory for clean probe structure")
    parser.add_argument("--probe-type", type=str, default=None,
                        help="Filter by probe type (e.g., linear_eoi_probe)")
    parser.add_argument("--select-best", action="store_true",
                        help="Only keep the best probe per (model, dataset, probe_type, label) combo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying")
    
    args = parser.parse_args()
    
    print(f"Scanning for probes in: {args.results_dir}")
    probes = find_all_probes(args.results_dir, args.probe_type)
    print(f"Found {len(probes)} probes")
    
    if args.select_best:
        probes = select_best_probes(probes)
        print(f"Selected {len(probes)} best probes")
    
    if args.dry_run:
        print("\nDry run - would organize these probes:")
        for p in probes:
            k = p.get("k") or 1
            temp = p.get("temperature") or 0.0
            name = get_clean_probe_name(
                p["model_name"], p["dataset"], p["probe_type"], p["label_type"], k, temp
            )
            score = p.get("test_score") or p.get("best_val_score") or "?"
            print(f"  {name} (score: {score})")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy probes
    print(f"\nCopying probes to: {args.output_dir}")
    copied = []
    for p in probes:
        name = copy_probe_to_output(p, args.output_dir)
        copied.append(name)
        print(f"  ✓ {name}")
    
    # Generate manifest and model card
    print("\nGenerating manifest.json...")
    generate_manifest(probes, args.output_dir)
    
    print("Generating README.md (model card)...")
    generate_model_card(probes, args.output_dir)
    
    print(f"\n✅ Done! {len(copied)} probes prepared in {args.output_dir}")
    print(f"\nNext step: Run upload_to_hf.py to push to HuggingFace Hub")


if __name__ == "__main__":
    main()
