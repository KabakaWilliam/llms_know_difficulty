#!/usr/bin/env python3
"""
Train probes for all checkpoint steps and store results for comparison.

This script trains probes for each checkpoint step, allowing comparison of 
how probe performance and optimal layers/positions change during training.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

from .config import Config
from .model_wrapper.list import get_supported_model_class
from .dataset.load_dataset import prepare_all_datasets
from .get_directions import train_probes
from .utils import save_to_json_file

logging.basicConfig(level=logging.INFO)
torch.set_grad_enabled(False)


def find_checkpoint_steps(checkpoint_dir: str) -> List[Tuple[str, int]]:
    """
    Find all checkpoint steps in the given directory.
    
    Returns:
        List of (step_path, step_number) tuples
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    steps = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                steps.append((str(item), step_num))
            except (ValueError, IndexError):
                logging.warning(f"Skipping invalid step directory: {item.name}")
    
    # Sort by step number
    steps.sort(key=lambda x: x[1])
    return steps


def load_cached_data(cfg: Config, dataset_name: str, split: str) -> Tuple[List[str], List[float]]:
    """Load cached dataset split."""
    try:
        data = json.load(open(cfg.artifact_path() / f"datasplits/{dataset_name}_{split}.json", "r"))
        prompts = [x["formatted_prompt"] for x in data]
        scores = [x["difficulty"] for x in data]
        return (prompts, scores)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cached data not found for {dataset_name}_{split}")


def train_probes_for_checkpoint(
    model_path: str,
    step_num: int,
    cfg: Config,
    datasets: Dict[str, Tuple[List[str], List[float]]],
    output_dir: Path
) -> Dict:
    """
    Train probes for a single checkpoint step.
    
    Args:
        model_path: Path to the checkpoint
        step_num: Step number
        cfg: Configuration object
        datasets: Dictionary of dataset_name -> (train_data, test_data)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with training results and metadata
    """
    print(f"\n🔍 Training probes for checkpoint step {step_num}")
    print(f"Model path: {model_path}")
    
    try:
        # Load model
        model = get_supported_model_class(model_path)
        
        results = {
            "step": step_num,
            "model_path": model_path,
            "datasets": {}
        }
        
        # Train probes for each dataset
        for dataset_name, (train_data, test_data) in datasets.items():
            print(f"  📊 Training probes for {dataset_name}")
            
            dataset_output_dir = output_dir / f"step_{step_num}" / dataset_name
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # Train probes
            train_probes(
                model,
                train_data=train_data,
                test_data=test_data,
                artifact_dir=str(dataset_output_dir),
                batch_size=cfg.batch_size,
                seed=cfg.cv_seed,
                k_fold=cfg.use_k_fold,
                n_folds=cfg.n_folds
            )
            
            # Load and process results
            performance_path = dataset_output_dir / "probe_performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    performance_data = json.load(f)
                
                # Find best performing probe
                best_performance = -float('inf')
                best_pos_idx = 0
                best_layer = 0
                
                layer_performance = performance_data["layer_performance"]
                positions = performance_data["positions"]
                
                for pos_idx, pos in enumerate(positions):
                    for layer_idx in range(len(layer_performance["test"][pos_idx])):
                        test_perf = layer_performance["test"][pos_idx][layer_idx]
                        if not (test_perf != test_perf) and test_perf > best_performance:
                            best_performance = test_perf
                            best_pos_idx = pos_idx
                            best_layer = layer_idx
                
                # Store results
                results["datasets"][dataset_name] = {
                    "best_performance": best_performance,
                    "best_position": positions[best_pos_idx],
                    "best_layer": best_layer,
                    "best_position_index": best_pos_idx,
                    "full_performance": performance_data
                }
                
                print(f"    ✅ Best probe: Position {positions[best_pos_idx]}, Layer {best_layer}, Performance: {best_performance:.4f}")
            else:
                print(f"    ❌ Performance file not found: {performance_path}")
                results["datasets"][dataset_name] = {"error": "Performance file not found"}
        
        return results
        
    except Exception as e:
        print(f"    ❌ Error training probes for step {step_num}: {str(e)}")
        return {
            "step": step_num,
            "model_path": model_path,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Train probes for all checkpoint steps")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint steps (e.g., converted_hf_models/model_name/)')
    parser.add_argument('--config_file', type=str, required=False, default=None,
                       help='Load configuration from file')
    parser.add_argument('--baseline_model', type=str, required=False,
                       help='Path to baseline model for comparison')
    parser.add_argument('--use_k_fold', action="store_true",
                       help="Enable K-Fold cross-validation for probe training")
    parser.add_argument('--n_folds', type=int, default=5,
                       help="Number of folds for K-Fold cross-validation")
    parser.add_argument('--cv_seed', type=int, default=42,
                       help="Random seed for cross-validation")
    parser.add_argument('--output_dir', type=str, required=False,
                       help='Output directory for results (default: runs/checkpoint_comparison)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        cfg = Config.load(args.config_file)
        if hasattr(args, 'use_k_fold') and args.use_k_fold:
            cfg.use_k_fold = args.use_k_fold
        if hasattr(args, 'n_folds'):
            cfg.n_folds = args.n_folds
        if hasattr(args, 'cv_seed'):
            cfg.cv_seed = args.cv_seed
    else:
        # Create basic config
        checkpoint_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
        cfg = Config(
            model_path=args.checkpoint_dir,
            model_alias=f"checkpoint_comparison_{checkpoint_name}",
            use_k_fold=args.use_k_fold,
            n_folds=args.n_folds,
            cv_seed=args.cv_seed
        )
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs") / "checkpoint_comparison" / os.path.basename(args.checkpoint_dir.rstrip('/'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎯 Checkpoint Probe Training Pipeline")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Cross-validation: {cfg.use_k_fold} ({'with ' + str(cfg.n_folds) + ' folds' if cfg.use_k_fold else 'disabled'})")
    
    # Find checkpoint steps
    checkpoint_steps = find_checkpoint_steps(args.checkpoint_dir)
    print(f"Found {len(checkpoint_steps)} checkpoint steps: {[s[1] for s in checkpoint_steps]}")
    
    if not checkpoint_steps:
        print("❌ No valid checkpoint steps found!")
        return
    
    # Prepare datasets (use the first checkpoint to load the model for dataset preparation)
    print("\n📊 Preparing datasets...")
    first_checkpoint_path = checkpoint_steps[0][0]
    temp_model = get_supported_model_class(first_checkpoint_path)
    
    datasets = prepare_all_datasets(
        dataset_names=cfg.subset_datasets,
        n_train=cfg.n_train,
        n_test=cfg.n_test
    )
    
    # Convert datasets to the expected format
    dataset_splits = {}
    for dataset_name in cfg.subset_datasets:
        train_dataset = datasets[dataset_name]["train"]
        test_dataset = datasets[dataset_name]["test"]
        
        # Apply chat template
        train_instructions = train_dataset["formatted_prompt"].to_list()
        train_formatted = temp_model.apply_chat_template(train_instructions)
        train_ratings = train_dataset["difficulty"].to_list()
        
        test_instructions = test_dataset["formatted_prompt"].to_list()
        test_formatted = temp_model.apply_chat_template(test_instructions)
        test_ratings = test_dataset["difficulty"].to_list()
        
        dataset_splits[dataset_name] = (
            (train_formatted, train_ratings),
            (test_formatted, test_ratings)
        )
    
    print(f"✅ Prepared {len(dataset_splits)} datasets: {list(dataset_splits.keys())}")
    
    # Train probes for each checkpoint
    all_results = []
    
    for step_path, step_num in checkpoint_steps:
        result = train_probes_for_checkpoint(
            model_path=step_path,
            step_num=step_num,
            cfg=cfg,
            datasets=dataset_splits,
            output_dir=output_dir
        )
        all_results.append(result)
    
    # Train baseline model if provided
    if args.baseline_model:
        print(f"\n🔍 Training probes for baseline model")
        baseline_result = train_probes_for_checkpoint(
            model_path=args.baseline_model,
            step_num=-1,  # Use -1 to indicate baseline
            cfg=cfg,
            datasets=dataset_splits,
            output_dir=output_dir
        )
        baseline_result["is_baseline"] = True
        all_results.insert(0, baseline_result)  # Put baseline first
    
    # Save combined results
    summary_file = output_dir / "checkpoint_probe_summary.json"
    save_to_json_file(str(summary_file), all_results)
    
    print(f"\n✅ Training completed!")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"📁 Individual results in: {output_dir}")
    
    # Print quick summary
    print(f"\n📊 Quick Summary:")
    for result in all_results:
        step = result.get("step", "unknown")
        step_label = "Baseline" if result.get("is_baseline") else f"Step {step}"
        
        if "error" in result:
            print(f"  {step_label}: ❌ Error - {result['error']}")
        else:
            print(f"  {step_label}:")
            for dataset_name, dataset_result in result.get("datasets", {}).items():
                if "error" in dataset_result:
                    print(f"    {dataset_name}: ❌ Error")
                else:
                    perf = dataset_result.get("best_performance", 0)
                    layer = dataset_result.get("best_layer", "?")
                    pos = dataset_result.get("best_position", "?")
                    print(f"    {dataset_name}: {perf:.4f} (Layer {layer}, Pos {pos})")


if __name__ == "__main__":
    main()
