#!/usr/bin/env python3
"""
Quick validation script for cascade marker parsing.
Tests the new executor parser functions without running full inference.
"""

import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from router_code.execute_strategies import parse_cascade_marker, is_cascade_marker


def test_cascade_detection():
    """Test is_cascade_marker() function."""
    print("Testing cascade marker detection...")
    
    test_cases = [
        # (input, expected_result)
        ("mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0", True),
        ("mv_entropy_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0", True),
        ("mv_cascade_7B_k5_t0.7_72B_k1_t0.0", True),
        ("mv_1.5B_k8_t0.7", False),  # Simple MV
        ("mv_7B_k8_t0.7", False),   # Simple MV
        ("Qwen/Qwen2.5-Math-1.5B-Instruct", False),  # Plain model
        ("some_random_string", False),
    ]
    
    all_passed = True
    for marker, expected in test_cases:
        result = is_cascade_marker(marker)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} is_cascade_marker('{marker[:50]}...') = {result} (expected {expected})")
    
    return all_passed


def test_cascade_parsing():
    """Test parse_cascade_marker() function."""
    print("\nTesting cascade marker parsing...")
    
    test_cases = [
        {
            "marker": "mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0",
            "expected_tiers": 3,
            "expected_first": ("Qwen/Qwen2.5-Math-1.5B-Instruct", 4, 0.7),
        },
        {
            "marker": "mv_cascade_7B_k5_t0.7_72B_k1_t0.0",
            "expected_tiers": 2,
            "expected_first": ("Qwen/Qwen2.5-Math-7B-Instruct", 5, 0.7),
        },
        {
            "marker": "mv_entropy_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0",
            "expected_tiers": 3,
            "expected_first": ("Qwen/Qwen2.5-Math-1.5B-Instruct", 4, 0.7),
        },
        {
            "marker": "mv_1.5B_k8_t0.7",  # Simple MV, not cascade
            "expected_tiers": 0,
            "expected_first": None,
        },
    ]
    
    all_passed = True
    for test in test_cases:
        marker = test["marker"]
        tiers = parse_cascade_marker(marker)
        
        num_tiers_ok = len(tiers) == test["expected_tiers"]
        status_num = "✓" if num_tiers_ok else "✗"
        print(f"  {status_num} parse_cascade_marker('{marker[:50]}...') → {len(tiers)} tiers")
        
        if not num_tiers_ok:
            all_passed = False
            print(f"      Expected {test['expected_tiers']} tiers, got {len(tiers)}")
        
        if tiers and test["expected_first"]:
            first_ok = tiers[0] == test["expected_first"]
            status_first = "✓" if first_ok else "✗"
            print(f"      {status_first} First tier: {tiers[0]} (expected {test['expected_first']})")
            if not first_ok:
                all_passed = False
    
    return all_passed


def test_tier_extraction():
    """Test detailed tier extraction."""
    print("\nTesting detailed tier extraction...")
    
    marker = "mv_cascade_1.5B_k4_t0.7_7B_k6_t0.7_72B_k1_t0.0"
    tiers = parse_cascade_marker(marker)
    
    expected_tiers = [
        ("Qwen/Qwen2.5-Math-1.5B-Instruct", 4, 0.7),
        ("Qwen/Qwen2.5-Math-7B-Instruct", 6, 0.7),
        ("Qwen/Qwen2.5-Math-72B-Instruct", 1, 0.0),
    ]
    
    all_passed = True
    for i, (actual, expected) in enumerate(zip(tiers, expected_tiers)):
        tier_ok = actual == expected
        status = "✓" if tier_ok else "✗"
        model_name = expected[0].split("/")[-1]
        print(f"  {status} Tier {i+1} ({model_name}): k={actual[1]}, t={actual[2]}")
        if not tier_ok:
            all_passed = False
    
    return all_passed


def main():
    print("=" * 80)
    print("CASCADE EXECUTOR PARSER VALIDATION")
    print("=" * 80)
    
    test1 = test_cascade_detection()
    test2 = test_cascade_parsing()
    test3 = test_tier_extraction()
    
    print("\n" + "=" * 80)
    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
