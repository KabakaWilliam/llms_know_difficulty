#!/usr/bin/env python3
"""
Quick test to verify metrics integration and file saving work correctly.
"""
import sys
import json
from pathlib import Path

# Test 1: Check main.py imports are correct
print("=" * 60)
print("TEST 1: Checking main.py imports...")
print("=" * 60)

try:
    with open("src/llms_know_difficulty/main.py", "r") as f:
        main_content = f.read()
    
    # Check for duplicate imports
    import_count = main_content.count("from metrics import compute_metrics")
    if import_count > 0:
        print(f"❌ FAIL: Found {import_count} duplicate import(s) of 'from metrics import compute_metrics'")
    else:
        print("✓ PASS: No duplicate imports")
    
    # Check for correct import
    if "from llms_know_difficulty.metrics import compute_metrics" in main_content:
        print("✓ PASS: Correct import found")
    else:
        print("❌ FAIL: Correct import not found")
        
    # Check for task_type parameter
    if "task_type=probe.task_type" in main_content:
        print("✓ PASS: task_type parameter in compute_metrics call")
    else:
        print("❌ FAIL: task_type parameter missing in compute_metrics call")
        
    # Check for test_score in metadata
    if "'test_score': probe.test_score" in main_content:
        print("✓ PASS: test_score included in metadata")
    else:
        print("❌ FAIL: test_score not in metadata")
        
    # Check for metadata.update(test_metrics)
    if "metadata.update(test_metrics)" in main_content:
        print("✓ PASS: Full test metrics merged into metadata")
    else:
        print("❌ FAIL: test_metrics not merged into metadata")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 2: Check utils.py save_probe_predictions
print("\n" + "=" * 60)
print("TEST 2: Checking save_probe_predictions in utils.py...")
print("=" * 60)

try:
    with open("src/llms_know_difficulty/utils.py", "r") as f:
        utils_content = f.read()
    
    # Check for jsonl writing
    if 'json.dump({"idx": int(row["idx"]), "pred": float(row["pred"])}' in utils_content:
        print("✓ PASS: probe_preds.jsonl writing implemented")
    else:
        print("❌ FAIL: probe_preds.jsonl writing not found")
        
    # Check for parquet saving
    if 'df.to_parquet(results_path / "probe_preds.parquet")' in utils_content:
        print("✓ PASS: Parquet saving implemented")
    else:
        print("❌ FAIL: Parquet saving not found")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 3: Verify sklearn_probe attributes exist
print("\n" + "=" * 60)
print("TEST 3: Checking sklearn_probe attributes...")
print("=" * 60)

try:
    with open("src/llms_know_difficulty/probe/sklearn_probe.py", "r") as f:
        probe_content = f.read()
    
    required_attrs = [
        'self.best_layer_idx',
        'self.best_pos_idx',
        'self.best_position_value',
        'self.best_alpha',
        'self.best_val_score',
        'self.test_score',
        'self.task_type',
        'self.model_name',
        'self.d_model'
    ]
    
    for attr in required_attrs:
        if attr in probe_content:
            print(f"✓ PASS: {attr} found in sklearn_probe.py")
        else:
            print(f"❌ FAIL: {attr} NOT found in sklearn_probe.py")
            
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 60)
print("Integration test complete!")
print("=" * 60)
