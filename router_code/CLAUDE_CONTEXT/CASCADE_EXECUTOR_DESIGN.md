# Cascade Executor Design - Implementation Guide

## Overview

This document explains how the executor integrates cascade markers from adaptive k-sampling strategies while maintaining **full backward compatibility** with existing strategies.

## Architecture

### 1. Cascade Marker Format

The new adaptive k-sampling strategies return markers in a standardized format:

```
mv_[entropy_]cascade_MODEL_kK_tT_MODEL_kK_tT_..._MODEL_kK_tT
```

**Examples:**
```python
# Base adaptive k-sampling (always tries all tiers)
"mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0"

# Cost-aware escalation (skips 1.5B for hard problems)
"mv_cascade_7B_k5_t0.7_72B_k1_t0.0"        # k >= 5 threshold

# Entropy-aware escalation (signals special executor behavior)
"mv_entropy_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0"
```

Each tier is explicitly specified as `MODEL_kK_tT`:
- `MODEL`: One of {1.5B, 7B, 72B}
- `kK`: Sample count (e.g., k4 = 4 samples)
- `tT`: Temperature (e.g., t0.7 = 0.7)

### 2. Three Routing String Types (Backward Compatible)

The executor now handles three types of routing strings:

#### Type 1: Plain Model Names (unchanged)
```python
"Qwen/Qwen2.5-Math-1.5B-Instruct"
"Qwen/Qwen2.5-Math-7B-Instruct"
"Qwen/Qwen2.5-Math-72B-Instruct"
```
- **Behavior**: Single greedy inference (k=1, t=0.0)
- **Backward compatible**: ✅ Works as before

#### Type 2: Simple MV Markers (unchanged)
```python
"mv_1.5B_k8_t0.7"
"mv_7B_k8_t0.7"
"mv_72B_k8_t0.7"
```
- **Behavior**: Majority voting on single tier
- **Backward compatible**: ✅ Works as before
- **Note**: 72B is always overridden to greedy (k=1, t=0.0) for memory safety

#### Type 3: Cascade Markers (NEW)
```python
"mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0"
"mv_cascade_7B_k5_t0.7_72B_k1_t0.0"
"mv_entropy_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0"
```
- **Behavior**: Try first tier → if success_prob < target_conf, escalate to next tier
- **Executor integration**: Implemented via `parse_cascade_marker()` and early-stopping logic
- **Entropy handling**: `mv_entropy_*` prefix signals special exploration behavior (future work)

### 3. Executor Integration Points

#### Parser Functions (New, Non-Invasive)

```python
def parse_cascade_marker(marker: str) -> List[tuple]:
    """
    Convert: "mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0"
    To:      [(1.5B_model, 4, 0.7), (7B_model, 4, 0.7), (72B_model, 1, 0.0)]
    """
```

```python
def is_cascade_marker(route_str: str) -> bool:
    """Detect if string is a cascade marker vs plain model or simple MV."""
```

#### Modified Routing Logic (Minimal Changes)

In `execute_routing_strategies()`:

1. **Detect routing type** (cascade vs simple MV vs plain model)
2. **For cascade markers**: Parse into tier list, store in `cascade_tiers` column
3. **Set initial inference** to first tier's (model, k, temperature)
4. **Pass cascade info** to `run_routed_vllm_inference()` for escalation logic

**Code snippet:**
```python
# Mark cascade vs non-cascade routes
routing_df["is_cascade"] = routing_df["route_to"].apply(is_cascade_marker)

# For cascade routes, store tier information
for idx, row in routing_df.iterrows():
    if row["is_cascade"]:
        tiers = parse_cascade_marker(row["route_to"])
        if tiers:
            routing_df.at[idx, "cascade_tiers"] = tiers
            # Use first tier for initial inference
            _, k_val, t_val = tiers[0]
            routing_df.at[idx, "route_to"] = tiers[0][0]
            routing_df.at[idx, "sc_n"] = k_val
            routing_df.at[idx, "sc_temp"] = t_val
```

### 4. Current Implementation Status

✅ **Completed:**
- Cascade marker parsing function (`parse_cascade_marker()`)
- Type detection function (`is_cascade_marker()`)
- Routing dataframe augmentation with `cascade_tiers` column
- Full backward compatibility with existing strategies

⏳ **Next Phase (Executor Escalation Logic):**

Currently, cascades are handled as "first-tier-only". To enable actual escalation, `run_routed_vllm_inference()` needs:

1. **Accept `cascade_tiers` column** containing tier lists
2. **After first-tier inference**, compute confidence from majority vote
3. **If confidence < target_conf**, escalate to next tier:
   - Save first-tier results for potential pass@k computation
   - Run second-tier inference with same problem
   - Recompute majority vote on escalated tier results
4. **Repeat** until confidence threshold met or final tier reached

## Integration Timeline

### Phase 1 (Current): Parser Infrastructure
- ✅ Cascade marker detection
- ✅ Cascade tier parsing
- ✅ Backward compatibility validation
- ⏳ Testing with dry-run

### Phase 2 (Next): Executor Escalation Logic
- ⏳ Modify `run_routed_vllm_inference()` to accept cascade tiers
- ⏳ Implement tier-by-tier escalation with confidence checking
- ⏳ Results aggregation across tiers
- ⏳ Testing with first two strategies

### Phase 3 (Future): Entropy-Aware Exploration
- ⏳ Recognize `mv_entropy_*` prefix
- ⏳ Compute vote entropy after first tier
- ⏳ Implement retry logic before escalation
- ⏳ Testing with third strategy

## Key Design Decisions

### 1. Non-Invasive Architecture
- **Why**: Existing strategies use simple MV markers that are fully functional
- **How**: Cascade parsing is isolated in helper functions
- **Benefit**: Zero risk of breaking existing strategies, easy to test incrementally

### 2. Parser-First Approach
- **Why**: Enables testing cascade detection independently
- **How**: Parse once, store parsed tiers, execute later
- **Benefit**: Clear separation of concerns, easier debugging

### 3. First-Tier-First Inference
- **Why**: Minimizes changes to `run_routed_vllm_inference()`
- **How**: Cascade marker contains full tier info; executor extracts first tier for now
- **Benefit**: Can test parsing + storage before implementing escalation logic

### 4. Explicit Tier Encoding
- **Why**: Eliminates ambiguity about which tier is which
- **How**: Every tier in marker is fully specified: MODEL_kK_tT
- **Benefit**: Machine-parseable, no implicit fallback chains

## Testing Strategy

### Step 1: Parser Tests
```bash
# Test cascade marker parsing
python -c "
from router_code.execute_strategies import parse_cascade_marker, is_cascade_marker

# Test detection
assert is_cascade_marker('mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0')
assert not is_cascade_marker('mv_1.5B_k8_t0.7')

# Test parsing
tiers = parse_cascade_marker('mv_cascade_1.5B_k4_t0.7_7B_k4_t0.7_72B_k1_t0.0')
assert len(tiers) == 3
assert tiers[0][1] == 4  # k value
assert tiers[0][2] == 0.7  # temperature
"
```

### Step 2: Dry-Run Test
```bash
python router_code/execute_strategies.py \
  --dry-run \
  --probing-dataset gneubig_aime-1983-2024 \
  --datasets openai_gsm8k DigitalLearningGmbH_MATH-lighteval
```

This will:
- Parse all strategies including cascades
- Verify `cascade_tiers` column is populated correctly
- Show routing breakdown without running inference
- Validate format compatibility

### Step 3: Single Strategy Test
```bash
# Create a wrapper script that tests one strategy at a time
# Run first strategy (simple adaptive_k_sampling) on small dataset
```

### Step 4: Full Integration Test
- Run all three strategies on test dataset
- Verify results are saved correctly
- Validate metrics computation

## Troubleshooting

### Issue: `cascade_tiers` column is None for cascade routes
**Cause**: Parsing failed
**Solution**: 
1. Check marker format is exactly `mv_[entropy_]cascade_MODEL_kK_tT_...`
2. Add debug print in `parse_cascade_marker()` to see error
3. Verify model sizes are one of {1.5B, 7B, 72B}

### Issue: First-tier inference runs but escalation doesn't happen
**Expected**: This is correct for current phase
**Note**: Escalation logic will be added in Phase 2
**Current behavior**: Uses first tier only (e.g., 1.5B for `mv_cascade_1.5B_k4...`)

### Issue: Cascade strategies appear to route differently than simple MV
**Expected**: Yes, by design
**Reason**: 
- `mv_adaptive_k_sampling` uses adaptive k based on probe score
- `mv_cascade_1.5B_k8_t0.7` always uses k=8
- So routing k values should differ

## Future Enhancements

1. **Entropy-aware exploration** (Phase 3)
   - Implement retry logic before escalation
   - Compute vote entropy to detect ambiguity
   - Selectively increase k before escalating

2. **Cost tracking across tiers**
   - Track which tier made the final decision
   - Analyze cost distribution (what % escalate to 7B, 72B?)
   - Optimize tier thresholds based on cost/accuracy trade-off

3. **Confidence estimation**
   - Option 1: Use majority vote consensus strength
   - Option 2: Use model score directly (for non-MV tiers)
   - Option 3: Use entropy of vote distribution

4. **Flexible early-stopping**
   - User-configurable confidence thresholds per tier
   - Cost-aware escalation (don't escalate if cost > budget)
   - Time-based stopping (escalate if tier takes too long)

## Questions & Design Decisions

**Q: Why store cascade_tiers as column instead of re-parsing each time?**
A: Efficiency (parse once) + clarity (tiers available for logging/debugging) + testability

**Q: Why not implement full escalation logic immediately?**
A: Incremental approach reduces risk; can test parsing independently first

**Q: What about entropy-aware exploration?**
A: Will be added after basic cascade escalation works. Marked with `mv_entropy_*` prefix.

**Q: How does this interact with confidence thresholds?**
A: `target_confidence` will be used in Phase 2 for escalation decisions. Currently, Phase 1 just gets first-tier results.

---

**Document Version**: 1.0 (Phase 1 - Parser Infrastructure)  
**Last Updated**: 2025-01-04  
**Maintainer**: Router Code Team
