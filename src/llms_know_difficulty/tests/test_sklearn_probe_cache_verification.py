#!/usr/bin/env python
"""
Test to verify that caching works correctly.
Runs activation extraction twice and verifies:
1. Second run loads from cache (not extracted again)
2. Cached activations match original activations
"""

import torch
import numpy as np
from pathlib import Path
import time

from ..probe.sklearn_probe import SklearnProbe
from ..probe import sk_activation_utils


def test_cache_verification():
    """Verify that caching works correctly."""
    
    print("=" * 80)
    print("Testing Activation Caching Verification")
    print("=" * 80)
    
    # Create probe instance
    probe = SklearnProbe(config={})
    
    # Setup with gpt2
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. Setting up probe with model: {model_name}")
    print(f"   Device: {device}")
    probe.setup(model_name=model_name, device=device)
    print("   ✓ Setup complete")
    
    # Create simple test data
    print("\n2. Creating test data...")
    test_texts = [
        "This is a test sentence for caching.",
        "Another test sentence here.",
        "And one more for good measure.",
    ]
    test_labels = [0.0, 1.0, 0.5]
    print(f"   Created {len(test_texts)} test samples")
    print("   ✓ Data created")
    
    # First extraction - should extract and cache
    print("\n3. First activation extraction (will extract and cache)...")
    extract_start = time.time()
    result_1 = sk_activation_utils.extract_or_load_activations(
        model=probe.model,
        tokenizer=probe.tokenizer,
        texts=test_texts,
        labels=test_labels,
        model_name=model_name,
        split="test",
        device=device,
        batch_size=16,
        max_length=512,
        eoi_tokens=None,
        layer_indices=None,
        cache_dir=str(Path(__file__).parent.parent / "data" / "activations" / "sklearn_probe"),
        use_cache=True,
    )
    extract_time = time.time() - extract_start
    
    activations_1 = result_1['activations']
    from_cache_1 = result_1['from_cache']
    print(f"   Shape: {activations_1.shape}")
    print(f"   From cache: {from_cache_1}")
    print(f"   Time: {extract_time:.2f}s")
    assert not from_cache_1, "First extraction should not be from cache"
    print("   ✓ First extraction complete")
    
    # Second extraction - should load from cache
    print("\n4. Second activation extraction (should load from cache)...")
    cache_start = time.time()
    result_2 = sk_activation_utils.extract_or_load_activations(
        model=probe.model,
        tokenizer=probe.tokenizer,
        texts=test_texts,
        labels=test_labels,
        model_name=model_name,
        split="test",
        device=device,
        batch_size=16,
        max_length=512,
        eoi_tokens=None,
        layer_indices=None,
        cache_dir=str(Path(__file__).parent.parent / "data" / "activations" / "sklearn_probe"),
        use_cache=True,
    )
    cache_time = time.time() - cache_start
    
    activations_2 = result_2['activations']
    from_cache_2 = result_2['from_cache']
    print(f"   Shape: {activations_2.shape}")
    print(f"   From cache: {from_cache_2}")
    print(f"   Time: {cache_time:.2f}s")
    assert from_cache_2, "Second extraction should be from cache"
    print("   ✓ Cache load complete")
    
    # Verify activations match
    print("\n5. Verifying cached activations match original...")
    
    # Convert to numpy for comparison
    if torch.is_tensor(activations_1):
        act_1_np = activations_1.numpy()
    else:
        act_1_np = np.array(activations_1)
    
    if torch.is_tensor(activations_2):
        act_2_np = activations_2.numpy()
    else:
        act_2_np = np.array(activations_2)
    
    # Check shapes match
    assert act_1_np.shape == act_2_np.shape, f"Shapes don't match: {act_1_np.shape} vs {act_2_np.shape}"
    print(f"   ✓ Shapes match: {act_1_np.shape}")
    
    # Check values match (allow for small floating point differences)
    max_diff = np.max(np.abs(act_1_np - act_2_np))
    print(f"   Max difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Activations differ too much: max diff = {max_diff}"
    print(f"   ✓ Activations match (max diff < 1e-5)")
    
    # Verify cache provides speedup
    speedup = extract_time / cache_time if cache_time > 0 else float('inf')
    print(f"\n6. Cache Performance:")
    print(f"   First extraction: {extract_time:.2f}s")
    print(f"   From cache: {cache_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    if cache_time > extract_time:
        print("   ⚠ Cache load is slower (might be I/O bound for small data)")
    else:
        print("   ✓ Cache provides speedup")
    
    print("\n" + "=" * 80)
    print("✓ Cache verification passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_cache_verification()
    exit(0 if success else 1)
