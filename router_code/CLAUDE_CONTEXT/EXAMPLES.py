"""
Example: Using the Routing Strategies Workflow

This file shows practical examples of how to use the routing strategies
system for evaluation and deployment.
"""

# ============================================================================
# EXAMPLE 1: Run all strategies and visualize
# ============================================================================

"""
Simplest case - use defaults:

cd router_code
python execute_strategies.py
"""

# Then in notebook (compare_all_routers.ipynb cells 8-10):
# Results auto-load and visualize

# ============================================================================
# EXAMPLE 2: Run with custom parameters
# ============================================================================

from pathlib import Path
import sys

# Setup paths
repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))

from router_code.execute_strategies import execute_routing_strategies

# Run with custom confidence threshold
execute_routing_strategies(
    labelled_datasets=[
        "openai/gsm8k",
        "opencompass/AIME2025",
    ],
    target_confidence=0.85,  # Lower threshold
    probing_dataset="DigitalLearningGmbH_MATH-lighteval",
    dry_run=False,  # Do run inference
)

# ============================================================================
# EXAMPLE 3: Use specific strategy in production
# ============================================================================

import pandas as pd
from router_code.routing_strategies import get_routing_strategy, ROUTING_STRATEGIES

# Show all available strategies
print("Available strategies:")
for name in ROUTING_STRATEGIES.keys():
    print(f"  - {name}")

# Get a specific strategy
route_func = get_routing_strategy("cost_utility")

# Apply to a single question
row = pd.Series({
    "score_1.5B": 0.85,
    "score_7B": 0.92,
    "problem": "What is 2+2?"
})

target_confidence = 0.90
model = route_func(row, target_conf=target_confidence)
print(f"Selected model: {model}")  # Output: Qwen/Qwen2.5-Math-7B-Instruct

# ============================================================================
# EXAMPLE 4: Apply strategy to dataset
# ============================================================================

from router_code.routing_strategies import apply_routing_strategy

# Load data with probe scores
df = pd.read_parquet("path/to/probe_data.parquet")

# Apply a strategy
routes = apply_routing_strategy(
    df,
    strategy_name="disagreement_0.15",
    target_conf=0.90,
)

df["routed_model"] = routes

# Check distribution
print(df["routed_model"].value_counts())

# ============================================================================
# EXAMPLE 5: Compare multiple strategies
# ============================================================================

from router_code.routing_strategies import get_all_strategies

# Get all strategies
all_strategies = get_all_strategies()

# Apply each strategy to same data
results = {}
for strategy_name, strategy_func in all_strategies.items():
    routes = apply_routing_strategy(
        df,
        strategy_name=strategy_name,
        target_conf=0.90,
    )
    
    # Count routing decisions
    dist = pd.Series(routes).value_counts()
    results[strategy_name] = {
        "1.5B": dist.get("Qwen/Qwen2.5-Math-1.5B-Instruct", 0),
        "7B": dist.get("Qwen/Qwen2.5-Math-7B-Instruct", 0),
        "72B": dist.get("Qwen/Qwen2.5-Math-72B-Instruct", 0),
    }

# Compare
comparison_df = pd.DataFrame(results).T
print("\nStrategy Comparison:")
print(comparison_df)

# ============================================================================
# EXAMPLE 6: Load and analyze results from executed runs
# ============================================================================

import glob

# Find all results
results_files = glob.glob(
    "pika_cascade_trial/DigitalLearningGmbH_MATH-lighteval_probe/*/answered_*.parquet"
)

# Load and consolidate
all_results = []
for path in results_files:
    df = pd.read_parquet(path)
    
    # Extract metadata from path
    parts = path.split("/")
    dataset = parts[2].replace("_routed", "")
    strategy = parts[3].split("_")[1:-1]  # Extract strategy name
    strategy = "_".join(strategy)
    
    # Compute metrics
    accuracy = df["majority_vote_is_correct"].mean()
    cost = df["total_cost_usd"].sum()
    passk = df["passk_score"].mean()
    
    all_results.append({
        "dataset": dataset,
        "strategy": strategy,
        "accuracy": accuracy,
        "cost": cost,
        "passk": passk,
    })

results_df = pd.DataFrame(all_results)

# Find best strategy per dataset
print("\nBest Accuracy per Dataset:")
for dataset in results_df["dataset"].unique():
    best = results_df[results_df["dataset"] == dataset].nlargest(1, "accuracy")
    print(f"  {dataset}: {best['strategy'].values[0]} ({best['accuracy'].values[0]:.4f})")

# ============================================================================
# EXAMPLE 7: Create custom strategy
# ============================================================================

from router_code.routing_strategies import ROUTING_STRATEGIES

def route_high_confidence_or_72b(row, target_conf, cost_ratios=None):
    """
    Custom strategy: Only route to small models if very confident,
    otherwise always use 72B for safety.
    """
    # Aggressive threshold for small models
    small_model_threshold = target_conf + 0.05
    
    if row["score_1.5B"] >= small_model_threshold:
        return "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif row["score_7B"] >= small_model_threshold:
        return "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        # Always use 72B unless very confident with smaller models
        return "Qwen/Qwen2.5-Math-72B-Instruct"

# Register it
ROUTING_STRATEGIES["high_confidence_or_72b"] = route_high_confidence_or_72b

# Now it's automatically included in execute_strategies.py!
# Just run: python execute_strategies.py

# ============================================================================
# EXAMPLE 8: Deploy best strategy as a service
# ============================================================================

class RouterService:
    """Wraps a routing strategy for deployment."""
    
    def __init__(self, strategy_name: str, target_confidence: float = 0.90):
        from router_code.routing_strategies import get_routing_strategy
        
        self.route_func = get_routing_strategy(strategy_name)
        self.target_confidence = target_confidence
        self.strategy_name = strategy_name
    
    def route(self, question_scores):
        """
        Route a question based on model scores.
        
        Args:
            question_scores: dict with "score_1.5B" and "score_7B"
        
        Returns:
            model_name: str
        """
        row = pd.Series(question_scores)
        return self.route_func(row, self.target_conf=self.target_confidence)


# Usage
router = RouterService(strategy_name="cost_utility")
model = router.route({
    "score_1.5B": 0.85,
    "score_7B": 0.92,
})
print(f"Route to: {model}")

# ============================================================================
# EXAMPLE 9: Analyze disagreement between models
# ============================================================================

# Load probe data
probe_df = pd.read_parquet("path/to/probe_data.parquet")

# Add disagreement column
probe_df["disagreement"] = abs(probe_df["score_1.5B"] - probe_df["score_7B"])

# Analyze
print("\nDisagreement Statistics:")
print(probe_df["disagreement"].describe())

# See impact on routing
for threshold in [0.10, 0.15, 0.20, 0.25]:
    routes_strict = apply_routing_strategy(
        probe_df,
        strategy_name="disagreement_" + str(threshold).replace(".", "_"),
        target_conf=0.90,
    )
    pct_1_5b = (pd.Series(routes_strict) == "Qwen/Qwen2.5-Math-1.5B-Instruct").mean()
    print(f"Threshold ±{threshold}: {pct_1_5b*100:.1f}% routed to 1.5B")

# ============================================================================
# EXAMPLE 10: Compute cost savings
# ============================================================================

# Load results from different strategies
cascade_results = pd.read_parquet(
    "pika_cascade_trial/DigitalLearningGmbH_MATH-lighteval_probe/"
    "openai_gsm8k_routed/answered_cascade_conf0.9.parquet"
)

cost_utility_results = pd.read_parquet(
    "pika_cascade_trial/DigitalLearningGmbH_MATH-lighteval_probe/"
    "openai_gsm8k_routed/answered_cost_utility_conf0.9.parquet"
)

# Compare
print("\nCost Comparison:")
print(f"Cascade Cost:       ${cascade_results['total_cost_usd'].sum():.2f}")
print(f"Cost-Utility Cost:  ${cost_utility_results['total_cost_usd'].sum():.2f}")

savings = cascade_results["total_cost_usd"].sum() - cost_utility_results["total_cost_usd"].sum()
print(f"Savings:            ${savings:.2f}")

# Accuracy
cascade_acc = cascade_results["majority_vote_is_correct"].mean()
cost_util_acc = cost_utility_results["majority_vote_is_correct"].mean()

print(f"\nAccuracy Comparison:")
print(f"Cascade:            {cascade_acc:.4f}")
print(f"Cost-Utility:       {cost_util_acc:.4f}")
print(f"Difference:         {(cost_util_acc - cascade_acc)*100:+.2f}%")

# ============================================================================
# Summary of Key Functions
# ============================================================================

"""
Key imports:
- get_routing_strategy(name) -> Callable
- apply_routing_strategy(df, strategy_name, target_conf) -> List[str]
- get_all_strategies() -> Dict[str, Callable]
- execute_routing_strategies(...) -> Executes all strategies

All strategies have signature:
    strategy(row: pd.Series, target_conf: float, cost_ratios: Optional[Dict]) -> str

Result files:
    pika_cascade_trial/{probing_dataset}/{dataset}_routed/
    ├── routing_{strategy}_conf{threshold}.parquet
    └── answered_{strategy}_conf{threshold}.parquet
"""
