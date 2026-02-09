#!/usr/bin/env python3
"""
Standalone cascade parser validation - tests just the parsing logic.
No external dependencies beyond basic Python.
"""

def parse_cascade_marker_standalone(marker: str) -> list:
    """
    Parse cascade marker format into list of (model_name, k, temperature) tuples.
    
    Standalone version for testing.
    """
    if not marker.startswith("mv_") or "cascade" not in marker:
        return []
    
    try:
        # Remove "mv_" prefix and "entropy_" prefix if present
        marker_content = marker.replace("mv_", "").replace("entropy_", "")
        
        # Skip "cascade_" prefix
        if not marker_content.startswith("cascade_"):
            return []
        marker_content = marker_content.replace("cascade_", "")
        
        # Parse by splitting on model size boundaries
        # Look for patterns: 1.5B, 7B, 72B followed by _k
        tiers = []
        i = 0
        
        while i < len(marker_content):
            # Find next model size marker (check 1.5B first since it's longer)
            if i + 4 <= len(marker_content) and marker_content[i:i+4] == "1.5B":
                # 1.5B found
                model_size = "1.5B"
                i += 4
            elif i + 3 <= len(marker_content) and marker_content[i:i+3] == "72B":
                # 72B found
                model_size = "72B"
                i += 3
            elif i + 2 <= len(marker_content) and marker_content[i:i+2] == "7B":
                # 7B found
                model_size = "7B"
                i += 2
            else:
                # Skip unknown character
                i += 1
                continue
            
            # Now extract _kK_tT part
            if i < len(marker_content) and marker_content[i] == "_":
                i += 1  # Skip the underscore
                
                # Parse k value: _kN
                if i < len(marker_content) and marker_content[i] == "k":
                    i += 1
                    k_str = ""
                    while i < len(marker_content) and marker_content[i] in "0123456789":
                        k_str += marker_content[i]
                        i += 1
                    
                    if k_str and i < len(marker_content) and marker_content[i] == "_":
                        k_val = int(k_str)
                        i += 1  # Skip the underscore after k
                        
                        # Parse t value: _tT
                        if i < len(marker_content) and marker_content[i] == "t":
                            i += 1
                            t_str = ""
                            # Collect temperature: could be 0, 0.0, 0.7, etc.
                            while i < len(marker_content) and marker_content[i] in "0123456789.":
                                t_str += marker_content[i]
                                i += 1
                            
                            if t_str:
                                t_val = float(t_str)
                                
                                # Convert model size to full model name
                                if "1.5" in model_size:
                                    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
                                elif "7B" in model_size:
                                    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
                                elif "72B" in model_size:
                                    model_name = "Qwen/Qwen2.5-Math-72B-Instruct"
                                else:
                                    continue  # Skip unknown model size
                                
                                tiers.append((model_name, k_val, t_val))
                                
                                # Skip the underscore before next model if exists
                                if i < len(marker_content) and marker_content[i] == "_":
                                    i += 1
        
        return tiers
    
    except Exception as e:
        print(f"    ⚠️ Error parsing cascade marker '{marker}': {e}")
        return []


def is_cascade_marker_standalone(route_str: str) -> bool:
    """Check if a routing string is a cascade marker."""
    return "mv_" in route_str and "cascade" in route_str


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
        result = is_cascade_marker_standalone(marker)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        display = marker[:50] + "..." if len(marker) > 50 else marker
        print(f"  {status} is_cascade_marker('{display}') = {result} (expected {expected})")
    
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
        tiers = parse_cascade_marker_standalone(marker)
        
        num_tiers_ok = len(tiers) == test["expected_tiers"]
        status_num = "✓" if num_tiers_ok else "✗"
        display = marker[:50] + "..." if len(marker) > 50 else marker
        print(f"  {status_num} parse_cascade_marker('{display}') → {len(tiers)} tiers")
        
        if not num_tiers_ok:
            all_passed = False
            print(f"      Expected {test['expected_tiers']} tiers, got {len(tiers)}")
        
        if tiers and test["expected_first"]:
            first_ok = tiers[0] == test["expected_first"]
            status_first = "✓" if first_ok else "✗"
            print(f"      {status_first} First tier: {tiers[0][:2]} (expected {test['expected_first'][:2]})")
            if not first_ok:
                all_passed = False
                print(f"          Full: {tiers[0]}")
                print(f"          Expected: {test['expected_first']}")
    
    return all_passed


def test_tier_extraction():
    """Test detailed tier extraction."""
    print("\nTesting detailed tier extraction...")
    
    marker = "mv_cascade_1.5B_k4_t0.7_7B_k6_t0.7_72B_k1_t0.0"
    tiers = parse_cascade_marker_standalone(marker)
    
    expected_tiers = [
        ("Qwen/Qwen2.5-Math-1.5B-Instruct", 4, 0.7),
        ("Qwen/Qwen2.5-Math-7B-Instruct", 6, 0.7),
        ("Qwen/Qwen2.5-Math-72B-Instruct", 1, 0.0),
    ]
    
    all_passed = True
    print(f"  Parsing: '{marker}'")
    for i, (actual, expected) in enumerate(zip(tiers, expected_tiers)):
        tier_ok = actual == expected
        status = "✓" if tier_ok else "✗"
        model_short = expected[0].split("-")[-1].replace("-Instruct", "")
        print(f"  {status} Tier {i+1} ({model_short}): k={actual[1]}, t={actual[2]}")
        if not tier_ok:
            all_passed = False
            print(f"      Expected: k={expected[1]}, t={expected[2]}")
    
    return all_passed


def main():
    print("=" * 80)
    print("CASCADE EXECUTOR PARSER VALIDATION (Standalone)")
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
    import sys
    sys.exit(main())
