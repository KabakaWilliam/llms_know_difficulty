"""
Simple script to load a trained probe and make predictions on a specified dataset.

Usage:
    python predict_with_probe.py --probe_path /path/to/probe/checkpoint \
                                  --dataset dataset_name \
                                  --model model_name \
                                  [--output_dir /path/to/output]
"""

import argparse
import torch
from pathlib import Path
from pika.utils import DataIngestionWorkflow
from pika.metrics import compute_metrics
from pika.config import PROMPT_COLUMN_NAME, LABEL_COLUMN_NAME
from pika.probe.probe_factory import ProbeFactory
from pika.probe.linear_eoi_probe import LinearEoiProbe
from pika.probe.tfidf_probe import TfidfProbe
import json
import os

def detect_probe_type(probe_path: Path) -> str:
    """Detect probe type from checkpoint files."""
    probe_path = Path(probe_path)
    
    # Check for probe-specific files
    if (probe_path / "best_probe.joblib").exists():
        # Could be LinearEoiProbe or SklearnProbe
        # Try to check metadata for hint
        metadata_file = probe_path / "probe_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # LinearEoiProbe has best_layer_idx, SklearnProbe doesn't
                    if "best_layer_idx" in metadata:
                        return "linear_eoi_probe"
                    else:
                        return "sklearn_probe"
            except:
                pass
        return "linear_eoi_probe"  # Default to LinearEoiProbe if joblib exists
    elif (probe_path / "vectorizer.joblib").exists():
        return "tfidf_probe"
    
    return "unknown"

def load_probe(probe_path: Path, device: str = "cuda", batch_size: int = 16):
    """Load a trained probe from checkpoint."""
    probe_path = Path(probe_path)
    
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe checkpoint not found at {probe_path}")
    
    # Try to detect and load the probe type
    probe_type = detect_probe_type(probe_path)
    
    print(f"📦 Loading probe from: {probe_path}")
    print(f"   Detected probe type: {probe_type}")
    
    if probe_type == "unknown":
        raise ValueError(f"Cannot determine probe type from checkpoint at {probe_path}. No recognized probe files found.")
    
    try:
        if probe_type == "linear_eoi_probe":
            probe = LinearEoiProbe.load_from_checkpoint(probe_path, device=device)
        elif probe_type == "tfidf_probe":
            probe = TfidfProbe.load_from_checkpoint(probe_path)
        # elif probe_type == "sklearn_probe":
        #     probe = SklearnProbe.load_from_checkpoint(probe_path, device=device)
        else:
            raise ValueError(f"Unsupported probe type: {probe_type}")
    except Exception as e:
        import traceback
        print(f"\n❌ Error loading probe:")
        print(f"   Probe type: {probe_type}")
        print(f"   Path: {probe_path}")
        print(f"   Error: {e}")
        print(f"\n   Traceback:")
        traceback.print_exc()
        raise RuntimeError(f"Failed to load probe: {e}")
    
    probe.batch_size = batch_size
    
    print(f"✅ Probe loaded successfully!")
    return probe

def main():
    parser = argparse.ArgumentParser(
        description="Load a trained probe and make predictions on a dataset"
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        required=True,
        help="Path to the trained probe checkpoint directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to make predictions on"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model used for the dataset"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=3000,
        help="Maximum length of the response"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of rollouts per question"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the model"
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default=PROMPT_COLUMN_NAME,
        help="Name of prompt column"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=LABEL_COLUMN_NAME,
        help="Name of label column"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save predictions (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to make predictions on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for prediction (default: 16)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 Probe Prediction Script")
    print("=" * 60)
    print(f"Probe checkpoint: {args.probe_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"Prompt column: {args.prompt_column}")
    print(f"Label column: {args.label_column}")
    print("=" * 60)
    
    # Load the dataset
    print(f"\n📂 Loading dataset...")
    print(f"   Using prompt column: '{args.prompt_column}'")
    print(f"   Using label column: '{args.label_column}'")
    
    train_data, val_data, test_data = DataIngestionWorkflow.load_dataset(
        dataset_name=args.dataset,
        model_name=args.model,
        max_len=args.max_len,
        k=args.k,
        temperature=args.temperature,
        prompt_column=args.prompt_column,
        label_column=args.label_column
    )
    
    # Select which split to predict on
    split_data = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }[args.split]
    
    print(f"   ✓ Loaded {len(split_data[0])} samples from {args.split} split")
    
    # Load the probe
    print(f"\n🔧 Loading probe...")
    probe = load_probe(Path(args.probe_path), device=args.device, batch_size=args.batch_size)
    
    # Make predictions
    print(f"\n🎯 Making predictions on {args.split} data...")
    indices_tensor, predictions_tensor = probe.predict(split_data)
    print(f"   ✓ Made predictions for {len(indices_tensor)} samples")
    
    # Compute metrics if labels are available
    print(f"\n📊 Computing metrics...")
    try:
        all_labels = split_data[2]
        labels = all_labels[:len(indices_tensor)]
        metrics = compute_metrics(labels, predictions_tensor, full_metrics=True)
        
        # Extract main performance metric (AUC or Spearman)
        performance_metric = metrics.get('auc') or metrics.get('spearman')
        metric_name = 'AUC' if 'auc' in metrics else 'Spearman'
        
        # Calculate mean of predictions
        pred_mean = float(predictions_tensor.mean().item())
        
        print(f"   {metric_name}: {performance_metric:.4f}")
        print(f"   Mean prediction: {pred_mean:.4f}")
    except Exception as e:
        print(f"   ⚠️  Could not compute metrics: {e}")
        metrics = {}
    
    # Save predictions if output directory is specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving predictions to {output_dir}...")
        
        # Save predictions as torch tensors
        torch.save({
            'indices': indices_tensor,
            'predictions': predictions_tensor,
        }, output_dir / "predictions.pt")
        
        # Save predictions as JSON
        predictions_dict = {
            'indices': indices_tensor.tolist() if isinstance(indices_tensor, torch.Tensor) else indices_tensor,
            'predictions': predictions_tensor.tolist() if isinstance(predictions_tensor, torch.Tensor) else predictions_tensor,
            'metrics': metrics,
            'split': args.split,
            'dataset': args.dataset,
            'model': args.model,
            'mean_prediction': pred_mean,
        }
        
        with open(output_dir / "predictions.json", 'w') as f:
            json.dump(predictions_dict, f, indent=2)
        
        print(f"   ✓ Saved predictions.pt and predictions.json")
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
