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

def route_always_middle(row, target_conf, cost_ratios=None):
    """
    Always use the middle model.
    Baseline strategy - cheap but may have low accuracy.
    """
    return "Qwen/Qwen2.5-Math-7B-Instruct"


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
# MAJORITY VOTING (MV) STRATEGIES - PHASE 2
# ============================================================================

def route_mv_always_1_5b(row, target_conf, cost_ratios=None, mv_k=8, mv_temperature=0.7):
    """
    Always route to 1.5B with majority voting.
    
    Baseline MV strategy: cheap but may have low accuracy.
    
    Args:
        row: pandas Series with question data
        target_conf: unused for this strategy
        cost_ratios: unused for this strategy
        mv_k: number of samples for majority voting (default: 8)
        mv_temperature: temperature for MV sampling (default: 0.7)
    
    Returns:
        "mv_1.5B_k{mv_k}_t{mv_temperature}" marker indicating MV should be used
    """
    return "mv_1.5B_k8_t0.7"


def route_mv_always_7b(row, target_conf, cost_ratios=None, mv_k=8, mv_temperature=0.7):
    """
    Always route to 7B with majority voting.
    
    Baseline MV strategy: medium cost, better accuracy than 1.5B.
    
    Args:
        row: pandas Series with question data
        target_conf: unused for this strategy
        cost_ratios: unused for this strategy
        mv_k: number of samples for majority voting (default: 8)
        mv_temperature: temperature for MV sampling (default: 0.7)
    
    Returns:
        "mv_7B_k{mv_k}_t{mv_temperature}" marker indicating MV should be used
    """
    return "mv_7B_k8_t0.7"


def route_mv_always_72b(row, target_conf, cost_ratios=None, mv_k=8, mv_temperature=0.7):
    """
    Always route to 72B with majority voting.
    
    Baseline MV strategy: most expensive but highest accuracy.
    
    WARNING: 72B with MV (k=8) generates 8x KV cache overhead.
    This strategy may cause GPU memory errors and is NOT recommended for production use.
    Should be overridden to greedy 72B by safeguards in execute_strategies.py.
    
    Args:
        row: pandas Series with question data
        target_conf: unused for this strategy
        cost_ratios: unused for this strategy
        mv_k: number of samples for majority voting (default: 8)
        mv_temperature: temperature for MV sampling (default: 0.7)
    
    Returns:
        "mv_72B_k{mv_k}_t{mv_temperature}" marker indicating MV should be used
    """
    return "mv_72B_k8_t0.7"


def route_mv_adaptive_escalation(row, target_conf, cost_ratios=None, 
                                 disagreement_thresh_1_5b=0.15, 
                                 disagreement_thresh_7b=0.10,
                                 mv_k=8, mv_temperature=0.7):
    """
    Adaptive escalation routing with majority voting on uncertainty.
    
    Use disagreement between probe models as signal for uncertainty:
    - High disagreement (uncertain) → use MV on cheap 1.5B
    - Medium disagreement → escalate to MV on 7B
    - Low disagreement (confident about failure) → use 72B greedy
    
    Logic:
    - If disagreement > thresh_1_5b: route to 1.5B with MV
    - Elif disagreement > thresh_7b: route to 7B with MV
    - Else: route to 72B (greedy, no MV)
    
    Args:
        row: pandas Series with columns score_1.5B, score_7B
        target_conf: confidence threshold (used for fallback, not primary logic)
        cost_ratios: unused
        disagreement_thresh_1_5b: disagreement threshold for 1.5B escalation
        disagreement_thresh_7b: disagreement threshold for 7B escalation
        mv_k: number of samples for MV
        mv_temperature: temperature for MV sampling
    
    Returns:
        Model routing string (either MV marker or plain model name)
    """
    disagreement = abs(row["score_1.5B"] - row["score_7B"])
    
    if disagreement > disagreement_thresh_1_5b:
        return f"mv_1.5B_k{mv_k}_t{mv_temperature}"
    elif disagreement > disagreement_thresh_7b:
        return f"mv_7B_k{mv_k}_t{mv_temperature}"
    else:
        return "Qwen/Qwen2.5-Math-72B-Instruct"


def route_mv_adaptive_escalation_tight(row, target_conf, cost_ratios=None, mv_k=8, mv_temperature=0.7):
    """
    Adaptive escalation with tight thresholds (conservative MV use).
    
    Thresholds: 1.5B=0.15, 7B=0.10
    Use MV more sparingly, rely more on greedy 72B.
    """
    return route_mv_adaptive_escalation(row, target_conf, cost_ratios,
                                       disagreement_thresh_1_5b=0.15,
                                       disagreement_thresh_7b=0.10,
                                       mv_k=mv_k, mv_temperature=mv_temperature)


def route_mv_adaptive_escalation_loose(row, target_conf, cost_ratios=None, mv_k=8, mv_temperature=0.7):
    """
    Adaptive escalation with loose thresholds (aggressive MV use).
    
    Thresholds: 1.5B=0.20, 7B=0.15
    Use MV more liberally, rely less on greedy 72B.
    """
    return route_mv_adaptive_escalation(row, target_conf, cost_ratios,
                                       disagreement_thresh_1_5b=0.20,
                                       disagreement_thresh_7b=0.15,
                                       mv_k=mv_k, mv_temperature=mv_temperature)


# ============================================================================# ADAPTIVE K-SAMPLING STRATEGY (based on probe difficulty score)
# ============================================================================

def route_adaptive_k_sampling(row, target_conf, cost_ratios=None, 
                              high_confidence_threshold=0.85,
                              max_k=8):
    """
    Probe-aware adaptive k-sampling strategy with escalation.
    
    Strategy:
    1. One cheap forward pass on 1.5B → get probe score (ps)
    2. If ps >= 0.85: do 1 generation on 1.5B, return (high confidence easy problem)
    3. Else: 
       - Choose k = min(ks(ps), max_k) where ks maps probe score to sample count
       - Generate k traces on 1.5B, compute weighted majority vote + confidence
       - If confidence >= target_conf: return with 1.5B
       - Else: escalate to 7B (greedy or with MV)
    
    This exploits the probe as a difficulty estimator to adapt the number of samples
    needed per problem. Easy problems (high ps) need only 1 sample. Harder problems
    require multiple samples for consensus.
    
    Args:
        row: pandas Series with columns score_1.5B (probe score)
        target_conf: confidence threshold for early stopping
        cost_ratios: unused
        high_confidence_threshold: threshold for single-sample route (default: 0.85)
        max_k: maximum number of samples to generate (default: 8)
    
    Returns:
        Routing decision string:
        - "mv_1.5B_k1_t0.7" for very high confidence problems
        - "mv_1.5B_kN_t0.7" for adaptive k on 1.5B (N = min(ks(ps), max_k))
        - "Qwen/Qwen2.5-Math-7B-Instruct" for escalation
    """
    # Get probe score (difficulty estimate from 1.5B)
    probe_score = row.get("score_1.5B", 0.5)
    
    # High confidence → single sample
    if probe_score >= high_confidence_threshold:
        return "mv_1.5B_k1_t0.7"
    
    # Adaptive k mapping: as probe score decreases (harder), increase k
    # Linear mapping: k(ps) = max_k * (1 - ps)
    # Examples:
    #   ps=0.85 → k ≈ 1.2 (capped at 1)
    #   ps=0.70 → k ≈ 2.4 (capped at 2)
    #   ps=0.50 → k ≈ 4.0 (capped at 4)
    #   ps=0.30 → k ≈ 5.6 (capped at 6)
    
    adaptive_k = max(1, int(max_k * (1.0 - probe_score)))
    adaptive_k = min(adaptive_k, max_k)
    
    # Route with adaptive k on 1.5B
    return f"mv_1.5B_k{adaptive_k}_t0.7"


def route_adaptive_k_sampling_with_escalation(row, target_conf, cost_ratios=None,
                                               high_confidence_threshold=0.85,
                                               escalation_k_threshold=5,
                                               max_k=8):
    """
    Probe-aware adaptive k-sampling with 7B escalation.
    
    Extension of route_adaptive_k_sampling that escalates to 7B if needed:
    
    1. If ps >= 0.85: use 1.5B with k=1
    2. Else:
       - Compute adaptive k = min(ks(ps), max_k)
       - If adaptive k >= escalation_threshold: escalate to 7B (greedy or MV)
       - Else: use 1.5B with adaptive k
    
    This prevents expensive k=6-8 on cheap 1.5B, instead escalating to 7B
    which may be more cost-effective.
    
    Args:
        row: pandas Series with columns score_1.5B
        target_conf: confidence threshold
        cost_ratios: unused
        high_confidence_threshold: threshold for single-sample on 1.5B (default: 0.85)
        escalation_k_threshold: escalate to 7B if adaptive k >= this (default: 5)
        max_k: maximum k for 1.5B before escalation (default: 8)
    
    Returns:
        Routing decision
    """
    probe_score = row.get("score_1.5B", 0.5)
    
    # High confidence → single sample on cheap model
    if probe_score >= high_confidence_threshold:
        return "mv_1.5B_k1_t0.7"
    
    # Compute adaptive k
    adaptive_k = max(1, int(max_k * (1.0 - probe_score)))
    adaptive_k = min(adaptive_k, max_k)
    
    # Escalate to 7B if k is too large (expensive)
    if adaptive_k >= escalation_k_threshold:
        return "Qwen/Qwen2.5-Math-7B-Instruct"
    
    return f"mv_1.5B_k{adaptive_k}_t0.7"


def route_adaptive_k_sampling_with_entropy_escalation(row, target_conf, cost_ratios=None,
                                                       high_confidence_threshold=0.85,
                                                       entropy_threshold=0.5,
                                                       max_k=8):
    """
    Probe-aware adaptive k-sampling with entropy-based escalation.
    
    Uses entropy of the majority vote result (post-MV) to decide escalation:
    
    1. If ps >= 0.85: use 1.5B with k=1 (high confidence easy)
    2. Else:
       - Compute adaptive k = min(ks(ps), max_k)
       - Do MV on 1.5B with this k
       - After MV, compute entropy of the vote distribution
       - If entropy > entropy_threshold AND confidence < target_conf: escalate to 7B
       - Else: return 1.5B result
    
    This is more sophisticated than k-based escalation alone: it actually looks at
    how uncertain the majority vote is (high entropy = many tied votes = uncertain).
    
    High entropy signals:
    - The samples disagreed a lot (no clear consensus)
    - Need more powerful model or more samples
    - Good candidate for escalation
    
    The executor needs to support this by:
    1. Doing MV on 1.5B with adaptive k
    2. Computing entropy of vote distribution
    3. Checking if entropy > threshold AND confidence < target_conf
    4. If so, re-running on 7B; else returning 1.5B result
    
    Args:
        row: pandas Series with columns score_1.5B (probe score)
        target_conf: confidence threshold (passed to executor for entropy-based decision)
        cost_ratios: unused
        high_confidence_threshold: threshold for single-sample on 1.5B (default: 0.85)
        entropy_threshold: escalate if entropy > this value (default: 0.5)
        max_k: maximum k for 1.5B (default: 8)
    
    Returns:
        Special routing marker for entropy-aware escalation:
        - "mv_1.5B_k1_t0.7" for very high confidence
        - "mv_entropy_1.5B_k{k}_t0.7" for entropy-check escalation
        
        Executor should recognize "mv_entropy_*" and after computing majority vote:
        - Calculate entropy of vote results
        - If entropy > entropy_threshold AND confidence < target_conf: escalate to 7B
        - Otherwise: return 1.5B decision
    """
    probe_score = row.get("score_1.5B", 0.5)
    
    # High confidence → single sample, no entropy check needed
    if probe_score >= high_confidence_threshold:
        return "mv_1.5B_k1_t0.7"
    
    # Compute adaptive k
    adaptive_k = max(1, int(max_k * (1.0 - probe_score)))
    adaptive_k = min(adaptive_k, max_k)
    
    # Return special entropy-check marker for executor to handle
    # Format: mv_entropy_MODEL_kK_tT means "do MV, check entropy for escalation"
    return f"mv_entropy_1.5B_k{adaptive_k}_t0.7"


# ============================================================================# STRATEGY REGISTRY
# ============================================================================

ROUTING_STRATEGIES = {
    # Baseline strategies
    # "random": route_random,
    # "always_1.5B": route_always_smallest,
    # "always_7B": route_always_middle,
    # "always_72B": route_always_largest,
    
    # Confidence-based (Phase 0)
    "cascade": route_cascade,
    "bayesian_robust": route_bayesian_robust,

    # Cost-aware (Phase 0)
    "cost_utility": route_cost_utility,
    "adjusted_thresholds": route_adjusted_thresholds,
    "expected_cost": route_expected_cost,

    # Agreement-aware (Phase 0)
    "disagreement_0.10": route_with_disagreement_0_10,
    "disagreement_0.15": route_with_disagreement_0_15,
    "disagreement_0.20": route_with_disagreement_0_20,
    
    # Phase 2: Majority Voting Strategies
    # "mv_always_1.5B": route_mv_always_1_5b,
    # "mv_always_7B": route_mv_always_7b,
    # "mv_always_72B": route_mv_always_72b,
    "mv_adaptive_escalation_tight": route_mv_adaptive_escalation_tight,
    "mv_adaptive_escalation_loose": route_mv_adaptive_escalation_loose,
    
    # Adaptive k-sampling (Phase 3)
    "adaptive_k_sampling": route_adaptive_k_sampling,
    "adaptive_k_sampling_with_escalation": route_adaptive_k_sampling_with_escalation,
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
