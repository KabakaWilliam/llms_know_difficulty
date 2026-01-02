"""
Routing Strategies for PIKA Router
===================================

This module contains all routing strategies that can be applied to questions
based on success probability predictions from multiple models.

Each strategy takes:
- row: pandas Series with columns like score_1.5B, score_7B
- target_conf: confidence threshold
- cost_ratios: dict mapping model names to cost multipliers

Returns: model name (str)
"""

import random
from typing import Dict, Optional, Callable, List


class RoutingStrategy:
    """Base class for routing strategies with metadata."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, row, target_conf, cost_ratios=None):
        return self.func(row, target_conf, cost_ratios)


# ============================================================================
# BASELINE STRATEGIES
# ============================================================================

def route_random(row, target_conf, cost_ratios=None):
    """
    Route randomly to one of the three models.
    Baseline strategy for comparison.
    """
    return random.choice([
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-Math-72B-Instruct",
    ])


def route_always_smallest(row, target_conf, cost_ratios=None):
    """
    Always use the smallest model.
    Baseline strategy - cheap but may have low accuracy.
    """
    return "Qwen/Qwen2.5-Math-1.5B-Instruct"


def route_always_largest(row, target_conf, cost_ratios=None):
    """
    Always use the largest model.
    Baseline strategy - most expensive but highest accuracy.
    """
    return "Qwen/Qwen2.5-Math-72B-Instruct"


# ============================================================================
# CONFIDENCE-BASED STRATEGIES
# ============================================================================

def route_cascade(row, target_conf, cost_ratios=None):
    """
    Sequential cascade: try each model in order until success prob >= threshold.
    
    Logic:
    - If 1.5B >= threshold: use 1.5B
    - Else if 7B >= threshold: use 7B
    - Else: use 72B
    """
    if row["score_1.5B"] >= target_conf:
        return "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif row["score_7B"] >= target_conf:
        return "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        return "Qwen/Qwen2.5-Math-72B-Instruct"


def route_bayesian_robust(row, target_conf, cost_ratios=None):
    """
    Route based on agreement between models.
    
    Logic:
    - If both models confident: use smallest (1.5B)
    - If only 7B confident: use 7B
    - Otherwise: use 72B
    
    Note: This requires both models to agree for 1.5B route.
    """
    if row["score_1.5B"] >= target_conf and row["score_7B"] >= target_conf:
        return "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif row["score_7B"] >= target_conf:
        return "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        return "Qwen/Qwen2.5-Math-72B-Instruct"


def route_72b_robust(row, target_conf, cost_ratios=None):
    """
    Alias for bayesian_robust - kept for backward compatibility.
    
    This is identical to route_bayesian_robust and is deprecated.
    Use route_bayesian_robust instead.
    """
    return route_bayesian_robust(row, target_conf, cost_ratios)


# ============================================================================
# AGREEMENT-AWARE STRATEGIES
# ============================================================================

def route_with_disagreement_threshold(row, target_conf, cost_ratios=None, disagreement_threshold=0.15):
    """
    Route considering disagreement between models.
    
    Logic:
    - If 1.5B confident AND models agree (|score_1.5B - score_7B| < threshold): use 1.5B
    - Elif 7B confident: use 7B
    - Else: use 72B
    """
    disagreement = abs(row["score_1.5B"] - row["score_7B"])
    if row["score_1.5B"] >= target_conf and disagreement < disagreement_threshold:
        return "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif row["score_7B"] >= target_conf:
        return "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        return "Qwen/Qwen2.5-Math-72B-Instruct"


def route_with_disagreement_0_10(row, target_conf, cost_ratios=None):
    """Disagreement-aware routing with threshold = 0.10"""
    return route_with_disagreement_threshold(row, target_conf, cost_ratios, disagreement_threshold=0.10)


def route_with_disagreement_0_15(row, target_conf, cost_ratios=None):
    """Disagreement-aware routing with threshold = 0.15"""
    return route_with_disagreement_threshold(row, target_conf, cost_ratios, disagreement_threshold=0.15)


def route_with_disagreement_0_20(row, target_conf, cost_ratios=None):
    """Disagreement-aware routing with threshold = 0.20"""
    return route_with_disagreement_threshold(row, target_conf, cost_ratios, disagreement_threshold=0.20)


# ============================================================================
# COST-AWARE STRATEGIES
# ============================================================================

def route_cost_utility(row, target_conf, cost_ratios=None):
    """
    Route to maximize success probability per cost unit (cost-efficiency).
    
    For each model, compute: efficiency = success_prob / cost_ratio
    Choose model with highest efficiency that meets target confidence.
    
    Logic:
    - For each model meeting threshold: calculate efficiency = score / cost
    - Choose model with highest efficiency
    - If no model meets threshold: use 72B
    """
    if cost_ratios is None:
        cost_ratios = {"Qwen/Qwen2.5-Math-1.5B-Instruct": 1.0,
                       "Qwen/Qwen2.5-Math-7B-Instruct": 2.0,
                       "Qwen/Qwen2.5-Math-72B-Instruct": 9.0}
    
    candidates = []
    
    if row["score_1.5B"] >= target_conf:
        efficiency = row["score_1.5B"] / cost_ratios["Qwen/Qwen2.5-Math-1.5B-Instruct"]
        candidates.append(("Qwen/Qwen2.5-Math-1.5B-Instruct", efficiency))
    
    if row["score_7B"] >= target_conf:
        efficiency = row["score_7B"] / cost_ratios["Qwen/Qwen2.5-Math-7B-Instruct"]
        candidates.append(("Qwen/Qwen2.5-Math-7B-Instruct", efficiency))
    
    if candidates:
        best_model, _ = max(candidates, key=lambda x: x[1])
        return best_model
    
    return "Qwen/Qwen2.5-Math-72B-Instruct"


def route_adjusted_thresholds(row, target_conf, cost_ratios=None, sensitivity=0.05):
    """
    Adjust confidence thresholds based on cost ratios.
    
    Cheaper models can be used with lower confidence thresholds.
    Adjustment formula: adjusted_threshold = target_conf - (cost_multiplier - 1) * sensitivity
    
    Logic:
    - Compute adjusted threshold for each model based on its cost
    - Use adjusted thresholds for routing decision
    """
    if cost_ratios is None:
        cost_ratios = {"Qwen/Qwen2.5-Math-1.5B-Instruct": 1.0,
                       "Qwen/Qwen2.5-Math-7B-Instruct": 2.0,
                       "Qwen/Qwen2.5-Math-72B-Instruct": 9.0}
    
    threshold_1_5B = target_conf - (cost_ratios["Qwen/Qwen2.5-Math-1.5B-Instruct"] - 1) * sensitivity
    threshold_7B = target_conf - (cost_ratios["Qwen/Qwen2.5-Math-7B-Instruct"] - 1) * sensitivity
    
    if row["score_1.5B"] >= threshold_1_5B:
        return "Qwen/Qwen2.5-Math-1.5B-Instruct"
    elif row["score_7B"] >= threshold_7B:
        return "Qwen/Qwen2.5-Math-7B-Instruct"
    else:
        return "Qwen/Qwen2.5-Math-72B-Instruct"


def route_expected_cost(row, target_conf, cost_ratios=None):
    """
    Route based on expected cost of failure.
    
    Expected cost = cost_ratio * (1 - success_prob)
    Choose model that minimizes expected cost while meeting target confidence.
    
    Logic:
    - For each model meeting threshold: calculate expected cost = cost * (1 - score)
    - Choose model with lowest expected cost
    - If no model meets threshold: use 72B
    """
    if cost_ratios is None:
        cost_ratios = {"Qwen/Qwen2.5-Math-1.5B-Instruct": 1.0,
                       "Qwen/Qwen2.5-Math-7B-Instruct": 2.0,
                       "Qwen/Qwen2.5-Math-72B-Instruct": 9.0}
    
    candidates = []
    
    if row["score_1.5B"] >= target_conf:
        expected_cost = cost_ratios["Qwen/Qwen2.5-Math-1.5B-Instruct"] * (1 - row["score_1.5B"])
        candidates.append(("Qwen/Qwen2.5-Math-1.5B-Instruct", expected_cost))
    
    if row["score_7B"] >= target_conf:
        expected_cost = cost_ratios["Qwen/Qwen2.5-Math-7B-Instruct"] * (1 - row["score_7B"])
        candidates.append(("Qwen/Qwen2.5-Math-7B-Instruct", expected_cost))
    
    if candidates:
        best_model, _ = min(candidates, key=lambda x: x[1])
        return best_model
    
    return "Qwen/Qwen2.5-Math-72B-Instruct"


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

ROUTING_STRATEGIES = {
    # Baseline strategies
    "random": route_random,
    "always_1.5B": route_always_smallest,
    # "always_72B": route_always_largest,
    
    # Cost-aware
    "cost_utility": route_cost_utility,
    "adjusted_thresholds": route_adjusted_thresholds,
    "expected_cost": route_expected_cost,

    # Confidence-based
    "cascade": route_cascade,
    "bayesian_robust": route_bayesian_robust,
    # "72b_robust": route_72b_robust,
    
    # Agreement-aware
    "disagreement_0.10": route_with_disagreement_0_10,
    "disagreement_0.15": route_with_disagreement_0_15,
    "disagreement_0.20": route_with_disagreement_0_20,
    
}


def get_routing_strategy(strategy_name: str) -> Callable:
    """
    Get a routing strategy by name.
    
    Args:
        strategy_name: name of the strategy (key in ROUTING_STRATEGIES)
    
    Returns:
        Callable routing function
    
    Raises:
        ValueError: if strategy_name not found
    """
    if strategy_name not in ROUTING_STRATEGIES:
        available = ", ".join(ROUTING_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    return ROUTING_STRATEGIES[strategy_name]


def get_all_strategies() -> Dict[str, Callable]:
    """Get all available routing strategies."""
    return ROUTING_STRATEGIES.copy()


def apply_routing_strategy(df, strategy_name: str, target_conf: float, 
                          cost_ratios: Optional[Dict] = None,
                          score_col_1_5B: str = "score_1.5B",
                          score_col_7B: str = "score_7B") -> List[str]:
    """
    Apply a routing strategy to a dataframe.
    
    Args:
        df: pandas DataFrame with score columns
        strategy_name: name of routing strategy
        target_conf: confidence threshold
        cost_ratios: optional cost ratios dict
        score_col_1_5B: column name for 1.5B scores
        score_col_7B: column name for 7B scores
    
    Returns:
        List of routed model names
    """
    strategy_func = get_routing_strategy(strategy_name)
    
    # Ensure we have the required columns
    df_copy = df.copy()
    if score_col_1_5B not in df_copy.columns or score_col_7B not in df_copy.columns:
        raise ValueError(f"DataFrame must have columns '{score_col_1_5B}' and '{score_col_7B}'")
    
    # Rename temporarily if needed
    if score_col_1_5B != "score_1.5B":
        df_copy["score_1.5B"] = df_copy[score_col_1_5B]
    if score_col_7B != "score_7B":
        df_copy["score_7B"] = df_copy[score_col_7B]
    
    # Apply strategy
    routes = df_copy.apply(
        lambda row: strategy_func(row, target_conf, cost_ratios),
        axis=1
    )
    
    return routes.tolist()
