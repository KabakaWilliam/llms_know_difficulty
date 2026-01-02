# 🚀 Routing Strategies: Complete Workflow Implementation

## Overview

You now have a complete end-to-end workflow to:
1. **Execute** 13+ routing strategies
2. **Save** results with identifiable names
3. **Visualize** cost-accuracy Pareto frontiers
4. **Compare** strategies across datasets

All without modifying the existing cascade script structure!

---

## 📁 What Was Created

### Core Files (in `router_code/`)

```
router_code/
├── routing_strategies.py              [NEW] Strategy library (13+ strategies)
├── execute_strategies.py              [NEW] Execution script
├── ROUTING_STRATEGIES_README.md       [NEW] Full documentation
├── IMPLEMENTATION_SUMMARY.md          [NEW] Technical summary
├── QUICK_REFERENCE.md                 [NEW] Quick guide
└── notebooks/
    └── compare_all_routers.ipynb      [UPDATED] Added 3 visualization cells
```

---

## 🎯 The Workflow

### **Phase 1: Execute Strategies** (5-30 minutes)

```bash
cd router_code
python execute_strategies.py
```

**What happens:**
- Loads probe data for each dataset
- Applies all 13 routing strategies
- Runs vLLM inference
- Computes metrics (accuracy, pass@k, cost)
- Saves with identifiable names

**Results saved to:**
```
pika_cascade_trial/{probing_dataset}/
└── {dataset}_routed/
    ├── routing_cascade_conf0.9.parquet
    ├── answered_cascade_conf0.9.parquet
    ├── routing_cost_utility_conf0.9.parquet
    ├── answered_cost_utility_conf0.9.parquet
    └── ... (one pair per strategy)
```

### **Phase 2: Visualize Results** (1 minute)

Open `notebooks/compare_all_routers.ipynb` and run:

**Cell 8: Load All Strategy Results**
- Auto-discovers result files
- Extracts metrics
- Creates consolidated DataFrame

**Cell 9: Pareto Frontier Analysis**
- Cost vs Accuracy scatter plot for all strategies
- Strategy rankings by accuracy
- Strategy rankings by cost
- Efficiency rankings (accuracy/cost)

**Cell 10: Dataset-Specific Comparisons**
- Per-dataset cost vs accuracy plots
- Annotated with strategy names
- Per-dataset ranking tables

### **Phase 3: Analyze & Deploy** (your decision)

From the visualizations, determine:
- Which strategy dominates cost-accuracy tradeoff?
- How does performance vary by dataset?
- What's the best strategy for your constraints?

---

## 🎮 13 Available Strategies

### Baseline Strategies (3)
| Strategy | Description | Cost | Accuracy |
|----------|-------------|------|----------|
| `random` | Random selection | Medium | Low |
| `always_1.5B` | Always use smallest | Very Low | Low |
| `always_72B` | Always use largest | Very High | High |

### Confidence-Based Strategies (3)
| Strategy | Description | Best For |
|----------|-------------|----------|
| `cascade` | Try models sequentially | Quick, cheap routing |
| `bayesian_robust` | Both models must agree | Reliable routing |
| `72b_robust` | Agreement required | Conservative routing |

### Agreement-Aware Strategies (3)
| Strategy | Disagreement Threshold | Use When |
|----------|----------------------|----------|
| `disagreement_0.10` | ±0.10 | Very strict agreement |
| `disagreement_0.15` | ±0.15 | Default (balanced) |
| `disagreement_0.20` | ±0.20 | Relaxed agreement |

### Cost-Aware Strategies (3) ⭐
| Strategy | Logic | Best Use Case |
|----------|-------|---------------|
| `cost_utility` | Maximize accuracy/cost | **Best efficiency** |
| `adjusted_thresholds` | Lower thresholds for cheap models | Flexible tradeoff |
| `expected_cost` | Minimize failure cost | Avoid worst cases |

---

## 📊 Key Features

### Identifiable Naming
Results are easy to find and understand:
```
answered_cascade_conf0.9.parquet        ← Strategy name + confidence
answered_cost_utility_conf0.9.parquet   ← Clear, searchable
```

### Automatic Discovery
Notebook automatically:
1. Finds all result files
2. Parses strategy names and datasets
3. Loads metrics without manual config

### Extensibility
Add a new strategy in 2 steps:
```python
# 1. Define function in routing_strategies.py
def route_my_strategy(row, target_conf, cost_ratios=None):
    return model_name

# 2. Register it
ROUTING_STRATEGIES["my_strategy"] = route_my_strategy

# Done! It's automatically included in execute_strategies.py
```

---

## 📈 Expected Results Pattern

Based on your data:

```
┌─ Cheapest ─────────────────────────────────────── Most Expensive ─┐
│                                                                    │
│  random/always_1.5B  ←  cascade  ←  disagreement/bayesian  ←  always_72B
│  (Low accuracy)           (Variable)      (High accuracy)    (Highest acc)
│                                                                    │
│                ↘                                      ↙             │
│                 └─→ cost_utility (Pareto optimal) ←─┘             │
└────────────────────────────────────────────────────────────────────┘

Cost-utility should emerge as the best strategy in most cases,
offering the best accuracy-per-dollar on the Pareto frontier.
```

---

## 💻 Common Commands

### Execute with defaults
```bash
python execute_strategies.py
```

### Dry run (routing only, no inference)
```bash
python execute_strategies.py --dry-run
```

### Custom confidence threshold
```bash
python execute_strategies.py --target-conf 0.85
```

### Specific datasets
```bash
python execute_strategies.py --datasets openai_gsm8k opencompass_AIME2025
```

### Combined
```bash
python execute_strategies.py --dry-run --datasets openai_gsm8k --target-conf 0.80
```

---

## 📋 Metrics in Result Files

Each `answered_*.parquet` contains:

| Column | Type | Meaning |
|--------|------|---------|
| `accuracy` | float | Majority vote correctness (0-1) |
| `passk_score` | float | Pass@K metric (0-1) |
| `cost` | float | Total USD cost for that sample |
| `route_to` | str | Which model was selected |
| `majority_vote_is_correct` | bool | Correct/incorrect |
| `generated_solutions` | str | JSON of all generations |
| And more... | ... | Model outputs, token counts, etc. |

---

## 🔗 Integration with Existing Code

The new workflow **doesn't modify** existing files:
- ✅ PIKA_cascade.py works as before
- ✅ Can run alongside other routers
- ✅ Results saved separately with clear naming

You can:
1. Keep using existing cascade scripts
2. Run new strategies in parallel
3. Compare all results in the notebook

---

## 📚 Documentation

Three levels of documentation available:

| File | Audience | Length |
|------|----------|--------|
| `QUICK_REFERENCE.md` | Users in a hurry | 2 pages |
| `ROUTING_STRATEGIES_README.md` | Technical users | 5 pages |
| `IMPLEMENTATION_SUMMARY.md` | Developers | 5 pages |

---

## ✅ Success Checklist

You'll know everything works when:

- [ ] `execute_strategies.py` runs without errors
- [ ] Files appear in `pika_cascade_trial/` with strategy names
- [ ] Notebook Cell 8 prints "Loaded X result files"
- [ ] Pareto frontier plots display (Cell 9)
- [ ] Dataset comparisons show all strategies (Cell 10)
- [ ] Results make intuitive sense (cost-utility is efficient, etc.)

---

## 🚀 Next Steps

1. **Run the pipeline:**
   ```bash
   cd router_code
   python execute_strategies.py
   ```

2. **Visualize results:**
   - Open `notebooks/compare_all_routers.ipynb`
   - Run cells 8-10
   - Examine Pareto frontiers

3. **Analyze results:**
   - Which strategy dominates?
   - How do costs compare?
   - What's the efficiency ranking?

4. **Deploy best strategy:**
   ```python
   from router_code.routing_strategies import get_routing_strategy
   
   route = get_routing_strategy("cost_utility")
   model = route(row, target_conf=0.90)
   ```

---

## 🔧 Architecture

```
User Input (execute_strategies.py)
         ↓
Load Probe Data
         ↓
routing_strategies.py ← [13 strategies]
         ↓
Apply Each Strategy to Each Dataset
         ↓
Run vLLM Inference
         ↓
Compute Metrics (accuracy, cost, pass@k)
         ↓
Save with Identifiable Names
         ↓
compare_all_routers.ipynb (Cells 8-10)
         ↓
Pareto Frontier Visualization
         ↓
Deploy Best Strategy
```

---

## 📞 Troubleshooting

**Q: Results not appearing in notebook?**
- Check: `ls pika_cascade_trial/DigitalLearningGmbH_MATH-lighteval_probe/*/answered_*.parquet`
- Re-run Cell 8 to refresh

**Q: How do I add a new strategy?**
- Edit `routing_strategies.py`: define function + add to dict
- Re-run `execute_strategies.py` - automatic!

**Q: Can I compare just 2 strategies?**
```python
filtered = strategy_results_df[
    strategy_results_df['strategy'].isin(['cascade', 'cost_utility'])
]
```

**Q: Why is my strategy not in results?**
- Make sure it's in `ROUTING_STRATEGIES` dict
- Re-run `execute_strategies.py`
- Check output files were created

---

## 🎓 Educational Notes

The strategies demonstrate important concepts:

- **cascade**: Sequential decision-making
- **bayesian_robust**: Ensemble agreement
- **disagreement_***: Uncertainty quantification
- **cost_utility**: Utility optimization
- **adjusted_thresholds**: Cost-aware ML
- **expected_cost**: Decision theory

These patterns are useful beyond routing!

---

## 📄 File Reference

| What | Where |
|------|-------|
| Strategy implementations | `routing_strategies.py` |
| Execution script | `execute_strategies.py` |
| Result files | `pika_cascade_trial/` |
| Visualization | `notebooks/compare_all_routers.ipynb` (cells 8-10) |
| Quick guide | `QUICK_REFERENCE.md` |
| Full docs | `ROUTING_STRATEGIES_README.md` |
| Implementation details | `IMPLEMENTATION_SUMMARY.md` |

---

## 🏁 Summary

You have a complete, extensible system to:
- ✅ Execute multiple routing strategies
- ✅ Save results with clear, searchable names
- ✅ Automatically load and visualize in notebooks
- ✅ Compare cost-accuracy tradeoffs
- ✅ Deploy best strategies to production

**Total time to results:** ~30 minutes for full execution + visualization

**Maintenance burden:** Minimal - strategies are modular and independent

**Scalability:** Easy to add new strategies and datasets

Ready to run! 🚀
