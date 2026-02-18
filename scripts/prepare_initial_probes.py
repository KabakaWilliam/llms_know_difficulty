#!/usr/bin/env python3
"""
Prepare specific probes for initial HuggingFace Hub upload.

This script prepares a curated set of probes for the first release:
- GPT-OSS-20B (high, medium, low) on MATH
- Qwen2.5-Math-7B-Instruct on MATH

Usage:
    python scripts/prepare_initial_probes.py
    python scripts/prepare_initial_probes.py --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

# Base directory for results
RESULTS_BASE = Path("data/results")

# Curated list of probes to upload (relative to RESULTS_BASE)
PROBES_TO_UPLOAD = [
    # GPT-OSS-20B variants on MATH
    "openai/gpt-oss-20b_high/DigitalLearningGmbH_MATH-lighteval/linear_eoi_probe/maxlen_131072_k_5_temp_1.0/label_majority_vote_is_correct/20260211_144858",
    "openai/gpt-oss-20b_low/DigitalLearningGmbH_MATH-lighteval/linear_eoi_probe/maxlen_131072_k_5_temp_1.0/label_majority_vote_is_correct/20260204_060511",
    "openai/gpt-oss-20b_medium/DigitalLearningGmbH_MATH-lighteval/linear_eoi_probe/maxlen_131072_k_5_temp_1.0/label_majority_vote_is_correct/20260205_162444",
    # Qwen2.5-Math-7B on MATH
    "Qwen/Qwen2.5-Math-7B-Instruct/DigitalLearningGmbH_MATH-lighteval/linear_eoi_probe/maxlen_3000_k_5_temp_0.7/label_majority_vote_is_correct/20260205_134741",
    # Qwen3-8B on AIME
    "Qwen/Qwen3-8B/gneubig_aime-1983-2024/linear_eoi_probe/maxlen_32768_k_5_temp_0.6/label_majority_vote_is_correct/20260217_115640"
]


def parse_probe_path(probe_path: str) -> dict:
    """Parse probe path to extract metadata."""
    parts = probe_path.split("/")
    
    # Expected: org/model/dataset/probe_type/config/label_type/timestamp
    org = parts[0]
    model = parts[1]
    dataset = parts[2]
    probe_type = parts[3]
    config_str = parts[4]  # e.g., "maxlen_131072_k_5_temp_1.0"
    label_str = parts[5]   # e.g., "label_majority_vote_is_correct"
    timestamp = parts[6]
    
    # Parse config
    config = {}
    config_parts = config_str.split("_")
    for i, part in enumerate(config_parts):
        if part == "maxlen" and i + 1 < len(config_parts):
            config["max_len"] = int(config_parts[i + 1])
        elif part == "k" and i + 1 < len(config_parts):
            config["k"] = int(config_parts[i + 1])
        elif part == "temp" and i + 1 < len(config_parts):
            config["temperature"] = float(config_parts[i + 1])
    
    label_type = label_str.replace("label_", "")
    
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
    }


def get_clean_probe_name(meta: dict) -> str:
    """Generate clean folder name for probe."""
    model_short = meta["model"].replace("_", "-")
    dataset_short = "MATH"  # Simplify dataset name
    probe_type = meta["probe_type"].replace("_", "-")
    label_short = "mv-correct" if "majority_vote" in meta["label_type"] else meta["label_type"].replace("_", "-")
    k = meta.get("k", 1)
    temp = meta.get("temperature", 0.0)
    
    return f"{model_short}--{dataset_short}--{probe_type}--{label_short}--k{k}-t{temp}"


def copy_probe(source_dir: Path, dest_dir: Path, clean_name: str, meta: dict) -> None:
    """Copy probe files to destination with clean naming."""
    probe_dest = dest_dir / clean_name
    probe_dest.mkdir(parents=True, exist_ok=True)
    
    # Files to copy (excluding large prediction files)
    files_to_copy = [
        "best_probe.joblib",
        "platt_scaler.joblib",
        "probe_metadata.json",
    ]
    
    copied = []
    for filename in files_to_copy:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, probe_dest / filename)
            copied.append(filename)
    
    # Load original metadata
    original_meta = {}
    meta_file = source_dir / "probe_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            original_meta = json.load(f)
    
    # Create enhanced config.json
    config = {
        "probe_name": clean_name,
        "model_name": meta["model_name"],
        "dataset": meta["dataset"],
        "dataset_display": "MATH (DigitalLearningGmbH)",
        "probe_type": meta["probe_type"],
        "label_type": meta["label_type"],
        "k": meta.get("k"),
        "temperature": meta.get("temperature"),
        "max_len": meta.get("max_len"),
        # Activation extraction settings
        "best_layer_idx": original_meta.get("best_layer_idx"),
        "best_position_value": original_meta.get("best_position_value"),
        "eoi_token_offset": original_meta.get("best_position_value"),
        "activation_position_description": f"Token at position {original_meta.get('best_position_value')} from end of templated input (e.g., -1 = last token, -3 = 3rd from end)",
        "best_alpha": original_meta.get("best_alpha"),
        "d_model": original_meta.get("d_model"),
        "task_type": original_meta.get("task_type"),
        "test_auc": original_meta.get("auc") or original_meta.get("test_score"),
        "val_auc": original_meta.get("best_val_score"),
        "ece": original_meta.get("ece"),
        "calibration_method": original_meta.get("calibration_method"),
        "files": copied,
        "exported_at": datetime.now().isoformat(),
    }
    
    with open(probe_dest / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return copied


def generate_readme(probes: list[dict], output_dir: Path) -> None:
    """Generate model card README.md."""
    
    readme = """---
library_name: pika
tags:
  - probe
  - difficulty-prediction
  - routing
  - llm-routing
  - interpretability
license: mit
---

# PIKA Probes — Pre-trained Difficulty Prediction Probes

This repository contains pre-trained linear probes for predicting task difficulty from LLM activations.
These probes are part of the [PIKA (Probe-Informed K-Aware Routing)](https://github.com/KabakaWilliam/llms_know_difficulty) project.

## Overview

These probes predict whether a model will correctly solve a problem based on its internal activations **before generation**.

### Use Cases
- **Routing**: Direct easy/hard problems to appropriate models
- **Calibration**: Estimate uncertainty before generation  
- **Analysis**: Understand model capabilities across problem types

## Installation

```bash
pip install pika
# or just
pip install huggingface_hub joblib
```

## Quick Start

```python
from huggingface_hub import snapshot_download
from pika.probe import LinearEoiProbe

# Download a probe
probe_path = snapshot_download(
    repo_id="CoffeeGitta/pika-probes",
    allow_patterns=["gpt-oss-20b-high--MATH--linear-eoi-probe--mv-correct--k5-t1.0/*"]
)

# Load and use
probe = LinearEoiProbe.load_from_checkpoint(probe_path)
predictions = probe.predict(texts=["Solve: Find all real x such that x^3 - 3x + 1 = 0"])
```

### Using PIKA's built-in downloader

```python
from pika.hub import download_probe

probe = download_probe(
    repo_id="CoffeeGitta/pika-probes",
    model_name="gpt-oss-20b_high",
    dataset="MATH"
)
```

## Available Probes

| Probe | Model | Layer | EOI Pos | Test AUC |
|-------|-------|-------|---------|----------|
"""
    
    # Add probe rows
    for p in probes:
        clean_name = get_clean_probe_name(p["meta"])
        model_short = p["meta"]["model"]
        auc = p.get("test_auc", "—")
        if isinstance(auc, float):
            auc = f"{auc:.3f}"
        layer = p.get("layer", "—")
        eoi_pos = p.get("eoi_position", "—")
        
        readme += f"| `{clean_name}` | {model_short} | {layer} | {eoi_pos} | {auc} |\n"
    
    readme += """
## Probe Structure

Each probe folder contains:
- `best_probe.joblib`: Trained sklearn LogisticRegression model
- `platt_scaler.joblib`: Platt calibration scaler (for probability calibration)
- `probe_metadata.json`: Full training metadata
- `config.json`: Clean configuration summary

### Activation Extraction

The probes are trained on hidden state activations extracted at a specific layer and token position:

- **`best_layer_idx`**: Which transformer layer to extract activations from
- **`eoi_token_offset`**: Token position relative to end of templated input
  - `-1` = last token of input
  - `-3` = 3rd token from end
  - This is the position **before** any generation tokens

Example: For `eoi_token_offset=-3`, extract the hidden state at `hidden_states[best_layer_idx][:, -3, :]` after tokenizing the input prompt.

## Model Details

### GPT-OSS-20B Variants
The GPT-OSS-20B models are different training checkpoints (low/medium/high training compute).
Probes trained on layer 15 activations at position -3 (3 tokens before end of input).

### Qwen2.5-Math-7B-Instruct
Probe trained on layer 17 activations at position -1 (last token of input).

## Training Your Own Probes

```bash
python -m pika.main \\
    --probe linear_eoi_probe \\
    --dataset DigitalLearningGmbH_MATH-lighteval \\
    --model Qwen/Qwen2.5-Math-7B-Instruct \\
    --max_len 3000 --k 5 --temperature 0.7
```

## Citation

If you use these probes in your research, please cite:

```bibtex
@misc{lugoloobi_llms_2026,
    title = {{LLMs} {Encode} {Their} {Failures}: {Predicting} {Success} from {Pre}-{Generation} {Activations}},
    shorttitle = {{LLMs} {Encode} {Their} {Failures}},
    url = {http://arxiv.org/abs/2602.09924},
    doi = {10.48550/arXiv.2602.09924},
    publisher = {arXiv},
    author = {Lugoloobi, William and Foster, Thomas and Bankes, William and Russell, Chris},
    month = feb,
    year = {2026},
    note = {arXiv:2602.09924 [cs]},
}
```

See also our earlier work: [LLMs Encode How Difficult Problems Are](https://arxiv.org/abs/2510.18147) (2025)

## License

MIT License
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def generate_manifest(probes: list[dict], output_dir: Path) -> None:
    """Generate manifest.json."""
    manifest = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "repository": "CoffeeGitta/pika-probes",
        "num_probes": len(probes),
        "probes": [],
    }
    
    for p in probes:
        clean_name = get_clean_probe_name(p["meta"])
        manifest["probes"].append({
            "name": clean_name,
            "model": p["meta"]["model_name"],
            "dataset": p["meta"]["dataset"],
            "probe_type": p["meta"]["probe_type"],
            "label_type": p["meta"]["label_type"],
            "test_auc": p.get("test_auc"),
        })
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare initial probes for HuggingFace upload")
    parser.add_argument("--output-dir", type=Path, default=Path("probes_for_hf"),
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIKA Probes - Initial HuggingFace Upload Preparation")
    print("=" * 60)
    print(f"\nTarget repository: CoffeeGitta/pika-probes")
    print(f"Output directory:  {args.output_dir}")
    print(f"\nProbes to prepare: {len(PROBES_TO_UPLOAD)}")
    
    prepared_probes = []
    
    for probe_path in PROBES_TO_UPLOAD:
        source_dir = RESULTS_BASE / probe_path
        meta = parse_probe_path(probe_path)
        clean_name = get_clean_probe_name(meta)
        
        print(f"\n  → {clean_name}")
        
        if not source_dir.exists():
            print(f"    ❌ Source not found: {source_dir}")
            continue
        
        # Load metrics from metadata
        meta_file = source_dir / "probe_metadata.json"
        test_auc = None
        if meta_file.exists():
            with open(meta_file) as f:
                orig_meta = json.load(f)
                test_auc = orig_meta.get("auc") or orig_meta.get("test_score")
                layer = orig_meta.get("best_layer_idx")
                eoi_position = orig_meta.get("best_position_value")
        else:
            layer = None
            eoi_position = None
        
        print(f"    Model: {meta['model_name']}")
        print(f"    Layer: {layer}, EOI pos: {eoi_position}")
        print(f"    Test AUC: {test_auc:.4f}" if test_auc else "    Test AUC: —")
        
        if not args.dry_run:
            copied = copy_probe(source_dir, args.output_dir, clean_name, meta)
            print(f"    ✓ Copied: {', '.join(copied)}")
        
        prepared_probes.append({
            "meta": meta, 
            "test_auc": test_auc, 
            "clean_name": clean_name,
            "layer": layer,
            "eoi_position": eoi_position,
        })
    
    if not args.dry_run and prepared_probes:
        print(f"\n{'=' * 60}")
        print("Generating manifest and README...")
        generate_manifest(prepared_probes, args.output_dir)
        generate_readme(prepared_probes, args.output_dir)
        print(f"✓ manifest.json")
        print(f"✓ README.md")
    
    print(f"\n{'=' * 60}")
    print(f"✅ Prepared {len(prepared_probes)} probes in {args.output_dir}")
    
    if not args.dry_run:
        print(f"""
Next steps:
  1. Review the prepared probes:
     ls -la {args.output_dir}/
     
  2. Login to HuggingFace:
     huggingface-cli login
     
  3. Upload:
     python scripts/upload_to_hf.py --repo-id CoffeeGitta/pika-probes --probes-dir {args.output_dir}
""")


if __name__ == "__main__":
    main()
