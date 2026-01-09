#!/usr/bin/env python
"""
End-to-end test for SklearnProbe with specified_eoi_position parameter.
Tests the full train -> predict pipeline.
"""

import torch
import numpy as np
from typing import List, Tuple

from ..probe.sklearn_probe import SklearnProbe
from ..probe.sk_train_utils import compute_metric


def create_synthetic_data(n_samples: int = 50) -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float]]:
    """Create simple synthetic prompts and labels for testing."""
    
    # Training data
    train_texts = [f"This is training sample {i}. Some context here." for i in range(n_samples)]
    train_labels = [float(i % 2) for i in range(n_samples)]  # Binary labels
    
    # Validation data
    val_texts = [f"This is validation sample {i}. Some context here." for i in range(n_samples // 2)]
    val_labels = [float(i % 2) for i in range(n_samples // 2)]
    
    # Test data
    test_texts = [f"This is test sample {i}. Some context here." for i in range(n_samples // 4)]
    test_labels = [float(i % 2) for i in range(n_samples // 4)]
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def test_sklearn_probe():
    """Test SklearnProbe with specified_eoi_position parameter."""
    
    print("=" * 80)
    print("Testing SklearnProbe End-to-End Pipeline")
    print("=" * 80)
    
    # Create probe instance
    probe = SklearnProbe(config={})
    
    # Setup with a small model for testing
    model_name = "gpt2"  # Small model for fast testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. Setting up probe with model: {model_name}")
    print(f"   Device: {device}")
    probe.setup(model_name=model_name, device=device)
    print("   ✓ Setup complete")
    
    # Create synthetic data
    print("\n2. Creating synthetic training data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = create_synthetic_data(n_samples=50)
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Val: {len(val_texts)} samples")
    print(f"   Test: {len(test_texts)} samples")
    print("   ✓ Data created")
    
    # Train the probe
    print("\n3. Training probe with train/val/test split...")
    print("   This extracts activations and performs alpha grid search...")
    
    try:
        probe.train(
            train_data=(train_texts, train_labels),
            val_data=(val_texts, val_labels),
            test_data=(test_texts, test_labels),
            alpha_grid=[0.01, 0.1, 1.0, 10.0],
        )
        print("   ✓ Training complete")
    except Exception as e:
        print(f"   ✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check that best probe metadata was set
    print("\n4. Verifying best probe metadata was stored...")
    assert probe.best_probe is not None, "best_probe not set"
    assert probe.best_pos_idx is not None, "best_pos_idx not set"
    assert probe.best_position_value is not None, "best_position_value not set"
    assert probe.best_layer_idx is not None, "best_layer_idx not set"
    assert probe.best_alpha is not None, "best_alpha not set"
    assert probe.best_val_score is not None, "best_val_score not set"
    print(f"   Best layer: {probe.best_layer_idx}")
    print(f"   Best position (idx): {probe.best_pos_idx}")
    print(f"   Best position (value): {probe.best_position_value}")
    print(f"   Best alpha: {probe.best_alpha}")
    print(f"   Best val score: {probe.best_val_score:.4f}")
    print("   ✓ Metadata verification complete")
    
    # Test the best probe on the test set
    print("\n5. Testing best probe on test set...")
    try:
        test_predictions = probe.predict(test_texts)
        test_labels_array = np.array(test_labels)
        
        # Use the same metric that was used during training
        test_score, metric_name = compute_metric(test_predictions, test_labels_array, probe.task_type)
        
        print(f"   Test set predictions shape: {test_predictions.shape}")
        print(f"   Test set {metric_name}: {test_score:.4f}")
        assert test_score > 0.0, "Test score should be positive"
        print("   ✓ Test set evaluation complete")
    except Exception as e:
        print(f"   ✗ Test set evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction on new prompts
    print("\n6. Testing predictions on new (unseen) prompts...")
    new_prompts = [
        "This is a new test prompt with some context.",
        "Another test prompt for prediction.",
        "And one more prompt to test.",
    ]
    
    try:
        predictions = probe.predict(new_prompts)
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predictions: {predictions}")
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
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_sklearn_probe()
    exit(0 if success else 1)
