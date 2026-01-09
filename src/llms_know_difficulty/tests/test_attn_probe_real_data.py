#!/usr/bin/env python
"""
Test SklearnProbe with real data from DigitalLearningGmbH_MATH-lighteval dataset.
Uses formatted_prompt as text and success_rate as labels.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path
import time

from llms_know_difficulty.probe.attn_probe import AttnProbe
from llms_know_difficulty.config import AttentionProbeConfig


def load_math_data() -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float]]:
    """
    Load data from DigitalLearningGmbH_MATH-lighteval parquet files.
    
    Splits the original train file into train/val, uses original test file as-is.
    
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """

    # Setup for Will B's setup:
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "20251229_170934"
    
    # Load train and test data
    train_path = data_dir / "test-Qwen-Qwen2-1.5B-Instruct_maxlen_3000_k_1_temp_0.0.parquet"
    test_path = data_dir / "test-Qwen-Qwen2-1.5B-Instruct_maxlen_3000_k_1_temp_0.0.parquet"
    
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"  Loaded {len(train_df)} samples")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_parquet(test_path)
    print(f"  Loaded {len(test_df)} samples")
    
    # Extract formatted_prompt and success_rate columns from train data
    train_texts_all = list(train_df["formatted_prompt"].values)
    train_labels_all = list(train_df["success_rate"].values)
        
    # Extract test data as-is
    test_texts = list(test_df["formatted_prompt"].values)
    test_labels = list(test_df["success_rate"].values)

    
    # Split train data into train (75%) and val (25%)
    train_size = int(0.75 * len(train_texts_all))
    
    train_texts = train_texts_all[:train_size]
    train_labels = train_labels_all[:train_size]

    val_texts = train_texts_all[train_size:]
    val_labels = train_labels_all[train_size:]
    
    print(f"\nData split:")
    print(f"  Train (from train file): {len(train_texts)} samples")
    print(f"  Val (from train file): {len(val_texts)} samples")
    print(f"  Test (from test file): {len(test_texts)} samples")
    
    # Show label statistics
    all_labels = train_labels + val_labels + test_labels
    print(f"\nSuccess rate statistics:")
    print(f"  Overall: mean={np.mean(all_labels):.4f}")
    print(f"  Train: mean={np.mean(train_labels):.4f}")
    print(f"  Val: mean={np.mean(val_labels):.4f}")
    print(f"  Test: mean={np.mean(test_labels):.4f}")
        
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def test_attn_probe_real_data():
    """Test AttnProbe with real MATH dataset."""
    
    test_start_time = time.time()
    
    print("=" * 80)
    print("Testing AttnProbe with Real MATH Data")
    print("=" * 80)
    
    # Create probe instance
    probe = AttnProbe(config=AttentionProbeConfig())
    
    # Setup with gpt2 for testing
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    # model_name = "Qwen/Qwen3-0.6B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. Setting up probe with model: {model_name}")
    print(f"   Device: {device}")
    setup_start = time.time()
    probe.setup(model_name=model_name, device=device)
    setup_time = time.time() - setup_start
    print("   ✓ Setup complete")
    
    # Load real data
    print("\n2. Loading MATH dataset...")
    load_start = time.time()
    try:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_math_data()
        load_time = time.time() - load_start
        print("   ✓ Data loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Train the probe
    print("\n3. Training probe with train/val/test split...")
    print("   This extracts activations and performs alpha grid search...")
    
    train_start = time.time()
    try:
        probe.train(
            train_data=(train_texts, train_labels),
            val_data=(val_texts, val_labels),
            #test_data=(test_texts, test_labels),
            # alpha_grid=None will use default from config
        )
        train_time = time.time() - train_start
        print("   ✓ Training complete")
    except Exception as e:
        print(f"   ✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check that best probe metadata was set
    print("\n4. Verifying best probe metadata was stored...")
    assert probe.best_probe is not None, "best_probe not set"
    assert probe.best_hyperparameters is not None, "best_hyperparameters not set"
    print(f"   Best layer: {probe.best_layer_id}")
    print("   ✓ Metadata verification complete")
    
    # Test the best probe on the test set
    print("\n5. Testing best probe on test set...")
    try:
        test_predictions = probe.predict(test_texts)
        test_labels_array = np.array(test_labels)
        
        # Use the same metric that was used during training
        from scipy.stats import spearmanr
        test_score, metric_name = spearmanr(test_predictions, test_labels_array)
        
        print(f"   Test set predictions shape: {test_predictions.shape}")
        print(f"   Test set {metric_name}: {test_score:.4f}")
        print("   ✓ Test set evaluation complete")
    except Exception as e:
        print(f"   ✗ Test set evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction on new prompts from the dataset
    print("\n6. Testing predictions on new prompts...")
    new_prompts = test_texts[:5]  # Use first 5 test prompts
    
    try:
        predictions = probe.predict(new_prompts)
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Prediction stats:")
        print(f"     Min: {predictions.min():.4f}")
        print(f"     Max: {predictions.max():.4f}")
        print(f"     Mean: {predictions.mean():.4f}")
        print("   ✓ Prediction complete")
    except Exception as e:
        print(f"   ✗ Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify predictions have correct shape
    assert predictions.shape[0] == len(new_prompts), f"Expected {len(new_prompts)} predictions, got {predictions.shape[0]}"
    print(f"\n7. Final Verification:")
    print(f"   Expected {len(new_prompts)} predictions on new prompts, got {predictions.shape[0]} ✓")
    
    # Calculate and display timing
    total_time = time.time() - test_start_time
    total_time_mins = total_time/60
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nTiming Breakdown:")
    print(f"  Setup: {setup_time:.2f}s")
    print(f"  Data loading: {load_time:.2f}s")
    print(f"  Training (extraction + grid search): {train_time:.2f}s")
    print(f"  Total test time: {total_time:.2f}s || {total_time_mins:.1f} mins")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_attn_probe_real_data()
    exit(0 if success else 1)