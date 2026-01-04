# Executor Integration - Quick Start Guide

## Current Status: ✅ Phase 1 Complete

The executor now **detects, parses, and stores cascade markers** while maintaining **100% backward compatibility** with existing strategies.

## Try It Out

### 1. Test the Parser (Quick Validation)
```bash
conda activate diff-direction
python router_code/test_cascade_parser_standalone.py
```

Expected output: `✅ ALL TESTS PASSED` (all 14 test cases)

### 2. Test with Dry Run (Full Data Processing)
```bash
python router_code/execute_strategies.py \
  --dry-run \
  --probing-dataset gneubig_aime-1983-2024 \
  --datasets openai_gsm8k DigitalLearningGmbH_MATH-lighteval
```

This will:
- ✅ Load probe data
- ✅ Apply all strategies (including new adaptive k-sampling ones)
- ✅ Parse cascade markers
- ✅ Display routing breakdown
- ⏳ Skip inference (dry run)

You'll see routes like:
```
→ Executing strategy: adaptive_k_sampling
  Routing breakdown:
    1.5B-Instruct: 45 (45.0%) - MV (k=1-4)
    7B-Instruct: 30 (30.0%) - MV (k=2-6)
    72B-Instruct: 25 (25.0%) - Greedy
```

### 3. Key Things to Verify
- ✅ No errors in routing logic
- ✅ Cascade markers are detected
- ✅ First-tier k and temperature are extracted correctly
- ✅ All existing strategies still work

## What's Working Now

### ✅ Implemented (Phase 1)
- Cascade marker detection via `is_cascade_marker()`
- Cascade parsing via `parse_cascade_marker()`
- Tier extraction (model, k, temperature)
- Storage in `cascade_tiers` dataframe column
- First-tier inference setup
- Full backward compatibility
- Comprehensive testing

### ⏳ Coming Next (Phase 2)
- Escalation logic in `run_routed_vllm_inference()`
- Confidence-based tier switching
- Multi-tier inference execution
- Results aggregation across tiers
- Cost tracking per tier

### 🔮 Future (Phase 3)
- Entropy-aware exploration
- Retry logic before escalation
- Adaptive tier thresholds

## Code Structure

```
router_code/
├── execute_strategies.py          ← Main file (added parser functions)
├── routing_strategies.py            ← Unchanged (generates cascade markers)
├── test_cascade_parser_standalone.py ← New (validates parser)
├── CASCADE_EXECUTOR_DESIGN.md       ← Full architecture doc
├── EXECUTOR_INTEGRATION_SUMMARY.md  ← Phase 1 status
└── EXECUTOR_CHANGES_DETAIL.md       ← Exact code changes
```

## Important Notes

1. **Cascade Markers Are Parsed But Not Yet Escalated**
   - First tier is used for initial inference
   - Subsequent tiers are stored but not yet executed
   - This happens in Phase 2

2. **Backward Compatibility Is Guaranteed**
   - Plain models work as before
   - Simple MV markers work as before
   - Zero impact on existing strategies

3. **Temperature Semantics Are Enforced**
   - k=1 always uses t=0.0 (greedy)
   - k>1 always uses t=0.7 (sampling)
   - 72B is always greedy

## Troubleshooting

### Parser tests fail
```bash
# Make sure you're in the right conda environment
conda activate diff-direction
python router_code/test_cascade_parser_standalone.py
```

### Cascade markers not detected in dry-run
- Check that routing strategies are generating markers correctly
- Look at the generated routes in the log output
- Verify marker format: `mv_[entropy_]cascade_MODEL_kK_tT_...`

### Cascade_tiers column is empty
- Parser may have failed (check for warning messages)
- Marker format might be invalid
- Check the routing_df output to see what markers were generated

## Next Steps

### Short Term (Ready to Run)
1. ✅ Test parser locally
2. ✅ Run dry-run to validate data flow
3. ✅ Verify cascade detection and parsing

### Medium Term (Requires Implementation)
1. Implement escalation logic in `run_routed_vllm_inference()`
2. Test with single adaptive k-sampling strategy
3. Validate cascade results match expectations

### Long Term (Post-Escalation)
1. Implement entropy-aware exploration
2. Add cost tracking and analysis
3. Optimize tier thresholds based on data

## Documentation

- **[CASCADE_EXECUTOR_DESIGN.md](CASCADE_EXECUTOR_DESIGN.md)**: Full design (80+ lines)
- **[EXECUTOR_INTEGRATION_SUMMARY.md](EXECUTOR_INTEGRATION_SUMMARY.md)**: Phase 1 overview
- **[EXECUTOR_CHANGES_DETAIL.md](EXECUTOR_CHANGES_DETAIL.md)**: Exact code changes
- **Inline comments**: In `execute_strategies.py`

## Contact & Questions

For questions about:
- **Parser logic**: See `parse_cascade_marker()` in execute_strategies.py
- **Design decisions**: See CASCADE_EXECUTOR_DESIGN.md
- **Code changes**: See EXECUTOR_CHANGES_DETAIL.md
- **Testing**: Run test_cascade_parser_standalone.py

---

**Status**: Phase 1 ✅ Complete and Tested  
**Next**: Phase 2 (Escalation Logic) when ready  
**Risk Level**: Minimal (non-invasive, fully backward compatible)  
**Ready for Production**: Yes (Phase 1 infrastructure)
