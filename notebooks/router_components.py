"""
Router components for self-consistency evaluation.
Contains all class definitions and utility functions.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from collections import Counter
from sklearn.metrics import accuracy_score
from vllm import SamplingParams
from math_verify import parse, verify
from utils import parse_answers

# Token pricing (USD per 1M tokens)
INPUT_TOKEN_PRICE = 0.25   # $0.25 per 1M input tokens
OUTPUT_TOKEN_PRICE = 1.00  # $1.00 per 1M output tokens
AVG_PROMPT_TOKENS = 100    # Estimated average prompt length


class RoutingStrategy(Enum):
    """Different ways to decide which questions get SC"""
    RANDOM = "random"
    PROBE_THRESHOLD = "probe_threshold"
    PROBE_QUANTILE = "probe_quantile"
    ORACLE = "oracle"  # use ground truth (cheating, but useful for upper bound)
    ALL_GREEDY = "all_greedy"  # baseline: no SC at all
    ALL_SC = "all_sc"  # upper bound: SC on everything


@dataclass
class RouterConfig:
    """Configuration for routing strategy"""
    strategy: RoutingStrategy
    # For PROBE_THRESHOLD
    threshold: Optional[float] = None  # e.g., 0.5
    # For PROBE_QUANTILE
    quantile: Optional[float] = None  # e.g., 0.3 (bottom 30%)
    # For RANDOM
    fraction: Optional[float] = None  # e.g., 0.3 (30% of data)
    # Which column to use for probe-based routing
    score_column: str = "predicted_difficulty_sigmoid"
    # Random seed for reproducibility
    seed: int = 42
    
    def get_strategy_name(self) -> str:
        """Generate explicit strategy name with parameters for saving/display"""
        if self.strategy == RoutingStrategy.PROBE_THRESHOLD:
            # e.g., "probe_threshold_0.50"
            return f"probe_threshold_{self.threshold:.2f}"
        elif self.strategy == RoutingStrategy.PROBE_QUANTILE:
            # e.g., "probe_quantile_bottom_30pct"
            pct = int(self.quantile * 100)
            return f"probe_quantile_bottom_{pct}pct"
        elif self.strategy == RoutingStrategy.RANDOM:
            # e.g., "random_30pct"
            pct = int(self.fraction * 100)
            return f"random_{pct}pct"
        else:
            # For ALL_GREEDY, ALL_SC, ORACLE
            return self.strategy.value


@dataclass
class SolverResult:
    """Result from solving a question"""
    question_idx: int
    used_sc: bool  # did we route this to SC?
    is_correct: int
    final_answer: str
    num_samples_used: int  # 1 for greedy, k for SC
    responses: List[str]  # all generated responses
    token_lengths: List[int]  # token count for each response
    predicted_score: float  # probe score


@dataclass  
class EvaluationMetrics:
    """Metrics for a routing strategy"""

    strategy_name: str
    overall_accuracy: float
    baseline_accuracy: float  # greedy on everything
    accuracy_gain: float  # improvement over baseline
    
    # Routing stats
    fraction_routed_to_sc: float
    avg_samples_per_question: float #avg number of forward passes per question.
    total_tokens: int
    
    # Router quality (if we have ground truth)
    router_accuracy: Optional[float] = None  # did we route the right questions?
    router_precision: Optional[float] = None  # of questions we SC'd, how many benefited?
    router_recall: Optional[float] = None  # of questions that benefit from SC, how many did we catch?
    
    # SC effectiveness
    sc_subset_accuracy: Optional[float] = None  # accuracy on questions we SC'd
    greedy_subset_accuracy: Optional[float] = None  # accuracy on questions we didn't SC
    
    # Cost metrics
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    input_cost: Optional[float] = None  # in USD
    output_cost: Optional[float] = None  # in USD
    total_cost: Optional[float] = None  # in USD
    cost_per_question: Optional[float] = None
    num_correct: Optional[int] = None
    cost_per_correct: Optional[float] = None
    baseline_cost: Optional[float] = None
    cost_increase: Optional[float] = None
    roi: Optional[float] = None  # accuracy points per dollar spent
    baseline_tokens: Optional[int] = None  # total tokens from baseline
    token_multiplier: Optional[float] = None  # strategy_tokens / baseline_tokens
    
    def efficiency_score(self) -> float:
        """Accuracy gain per unit of compute (sample-based - legacy)"""
        return self.accuracy_gain / self.avg_samples_per_question
    
    def token_efficiency(self) -> float:
        """Accuracy gain per token multiplier (better metric)"""
        if self.token_multiplier and self.token_multiplier > 0:
            return self.accuracy_gain / self.token_multiplier
        return 0.0
    
    def cost_efficiency(self) -> float:
        """Accuracy gain per dollar spent (ROI in different form)"""
        if self.cost_increase and self.cost_increase > 0:
            return self.accuracy_gain / self.cost_increase
        elif self.cost_increase and self.cost_increase < 0 and self.accuracy_gain > 0:
            # Cheaper AND better = infinite efficiency
            return float('inf')
        return 0.0
        return self.accuracy_gain / self.avg_samples_per_question
    
    def __repr__(self):
        cost_section = ""
        if self.total_cost is not None:
            cost_section = f"""╠═══════════════════════════════════════════════════════════════╣
║  💰 COST METRICS:                                             ║
║    Total Cost:    ${self.total_cost:>8.5f} ({self.cost_increase/self.baseline_cost*100 if self.cost_increase and self.baseline_cost else 0:>+6.1f}% vs baseline)          ║
║    Per Question:  ${self.cost_per_question:>8.6f}                               ║
║    Per Correct:   ${self.cost_per_correct:>8.6f}                               ║
║    ROI:           {'∞ (FREE LUNCH!)' if self.roi and np.isinf(self.roi) else f'{self.roi*100:>6.1f} acc pts/$' if self.roi else '  0.0 acc pts/$'}                  ║
║    Input/Output:  {self.input_cost/self.total_cost*100 if self.total_cost else 0:>5.1f}% / {self.output_cost/self.total_cost*100 if self.total_cost else 0:>5.1f}%                           ║
"""
        
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║  Strategy: {self.strategy_name:<50} ║
╠═══════════════════════════════════════════════════════════════╣
║  🎯 ACCURACY:  {self.overall_accuracy:>6.2%} ({self.accuracy_gain:>+6.2%} vs baseline) ║
║  📊 Baseline:  {self.baseline_accuracy:>6.2%}                                  ║
╠═══════════════════════════════════════════════════════════════╣
║  🔀 Routing:   {self.fraction_routed_to_sc:>5.1%} → SC  |  {1-self.fraction_routed_to_sc:>5.1%} → Greedy     ║
║  ⚡ Avg Samples: {self.avg_samples_per_question:>5.2f}x  |  Tokens: {self.token_multiplier if self.token_multiplier else 0:>5.2f}x        ║
║  🏆 Efficiency (sample): {self.efficiency_score():>6.4f}                         ║
║  🎯 Efficiency (token):  {self.token_efficiency():>6.4f}                         ║
{cost_section}╠═══════════════════════════════════════════════════════════════╣
║  Router Quality:                                              ║
║    Accuracy:   {self.router_accuracy:>6.2%}                                 ║
║    Precision:  {self.router_precision:>6.2%}                                 ║
║    Recall:     {self.router_recall:>6.2%}                                 ║
╚═══════════════════════════════════════════════════════════════╝
"""


def cost_for_result(r: SolverResult, INPUT_TOKEN_PRICE=0.1, OUTPUT_TOKEN_PRICE=0.2) -> float:
    """Calculate cost for a result based on token usage"""
    return OUTPUT_TOKEN_PRICE * sum(r.token_lengths)


def run_greedy_baseline(llm, df: pd.DataFrame, greedy_temp: float, max_tokens: int) -> tuple:
    """Get baseline greedy performance on all questions"""
    prompts = df["formatted_prompt"].to_list()
    gts = df["answer"].to_list()
    
    params = SamplingParams(temperature=greedy_temp, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    
    responses = []
    token_lengths = []
    is_correct = []
    
    for output, gt in zip(outputs, gts):
        response = output.outputs[0].text
        token_length = len(output.outputs[0].token_ids)
        
        responses.append(response)
        token_lengths.append(token_length)
        
        # Verify answer
        parsed_answers = parse_answers([response])
        if parsed_answers and len(parsed_answers[0]) > 0:
            final_answer = parsed_answers[0][0]
        else:
            final_answer = ""
        
        correct = verify(parse(f"${gt}$"), final_answer)
        is_correct.append(int(correct))
    
    accuracy = np.mean(is_correct)
    print(f"Greedy Baseline Accuracy: {accuracy:.2%} ({sum(is_correct)}/{len(is_correct)})")
    
    return responses, token_lengths, is_correct


def create_routing_mask(df: pd.DataFrame, config: RouterConfig) -> np.ndarray:
    """
    Create boolean mask: True = route to SC, False = use greedy
    
    Returns:
        np.ndarray of bools, same length as df
    """
    n = len(df)
    
    if config.strategy == RoutingStrategy.ALL_GREEDY:
        return np.zeros(n, dtype=bool)
    
    elif config.strategy == RoutingStrategy.ALL_SC:
        return np.ones(n, dtype=bool)
    
    elif config.strategy == RoutingStrategy.RANDOM:
        assert config.fraction is not None, "fraction required for RANDOM strategy"
        np.random.seed(config.seed)
        mask = np.random.rand(n) < config.fraction
        return mask
    
    elif config.strategy == RoutingStrategy.PROBE_THRESHOLD:
        assert config.threshold is not None, "threshold required for PROBE_THRESHOLD"
        # Lower score = harder = should use SC
        scores = df[config.score_column].values
        mask = scores < config.threshold
        return mask
    
    elif config.strategy == RoutingStrategy.PROBE_QUANTILE:
        assert config.quantile is not None, "quantile required for PROBE_QUANTILE"
        # Route bottom-k% (hardest questions)
        scores = df[config.score_column].values
        threshold = np.quantile(scores, config.quantile)
        mask = scores <= threshold
        return mask
    
    elif config.strategy == RoutingStrategy.ORACLE:
        # Cheat: use ground truth to identify questions that benefit from SC
        # This requires us to have SC results already... we'll implement this last
        raise NotImplementedError("Oracle routing requires SC results first")
    
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


def print_routing_stats(mask: np.ndarray, config: RouterConfig, num_sc_samples: int):
    """Print stats about routing decisions"""
    n_sc = mask.sum()
    n_total = len(mask)
    frac_sc = n_sc / n_total
    
    print(f"\n{'='*50}")
    print(f"Routing Strategy: {config.get_strategy_name()}")
    print(f"{'='*50}")
    print(f"SC: {n_sc}/{n_total} ({frac_sc:.1%})")
    print(f"Greedy: {n_total - n_sc}/{n_total} ({1-frac_sc:.1%})")
    print(f"Avg samples per question: {1 + frac_sc * (num_sc_samples - 1):.2f}x")
    print(f"{'='*50}\n")


def run_self_consistency(llm, prompt: str, gt: str, sc_temp: float, max_tokens: int, num_samples: int) -> Dict:
    """
    Run self-consistency on a single question
    
    Returns:
        dict with keys: is_correct, final_answer, responses, token_lengths
    """
    params = SamplingParams(temperature=sc_temp, max_tokens=max_tokens)
    prompts = [prompt] * num_samples
    
    outputs = llm.generate(prompts, params)
    
    responses = []
    token_lengths = []
    
    for output in outputs:
        response = output.outputs[0].text
        token_length = len(output.outputs[0].token_ids)
        responses.append(response)
        token_lengths.append(token_length)
    
    # Parse all answers and do majority vote
    parsed_answers_list = parse_answers(responses)
    
    # Flatten: each parse can return multiple options
    all_answers = []
    for parsed in parsed_answers_list:
        all_answers.extend(parsed)
    
    # Majority vote
    if all_answers:
        final_answer = Counter(all_answers).most_common(1)[0][0]
    else:
        final_answer = ""
    
    # Verify
    is_correct = int(verify(parse(f"${gt}$"), final_answer))
    
    return {
        "is_correct": is_correct,
        "final_answer": final_answer,
        "responses": responses,
        "token_lengths": token_lengths,
    }


def evaluate_routing_strategy(
    llm, 
    df: pd.DataFrame, 
    config: RouterConfig,
    baseline_accuracy: float,
    sc_temp: float,
    max_tokens: int,
    num_sc_samples: int
) -> tuple[EvaluationMetrics, List[SolverResult]]:
    """
    Evaluate a routing strategy: run SC on selected questions, greedy on others
    
    Returns:
        Tuple of (EvaluationMetrics, List[SolverResult])
    """
    from tqdm import tqdm
    
    # Get routing mask
    routing_mask = create_routing_mask(df, config)
    print_routing_stats(routing_mask, config, num_sc_samples)
    
    # For questions not routed to SC, use cached greedy results
    results = []
    
    for idx in tqdm(range(len(df)), desc=f"Evaluating {config.get_strategy_name()}"):
        row = df.iloc[idx]
        
        if routing_mask[idx]:
            # Run SC
            sc_result = run_self_consistency(
                llm, 
                row["formatted_prompt"], 
                row["answer"],
                sc_temp=sc_temp,
                max_tokens=max_tokens,
                num_samples=num_sc_samples
            )
            
            results.append(SolverResult(
                question_idx=idx,
                used_sc=True,
                is_correct=sc_result["is_correct"],
                final_answer=sc_result["final_answer"],
                num_samples_used=num_sc_samples,
                responses=sc_result["responses"],
                token_lengths=sc_result["token_lengths"],
                predicted_score=row[config.score_column],
            ))
        else:
            # Use greedy (cached)
            parsed = parse_answers([row["baseline_response"]])
            final_answer = parsed[0][0] if parsed and len(parsed[0]) > 0 else ""
            
            results.append(SolverResult(
                question_idx=idx,
                used_sc=False,
                is_correct=row["baseline_is_correct"],
                final_answer=final_answer,
                num_samples_used=1,
                responses=[row["baseline_response"]],
                token_lengths=[row["baseline_token_length"]],
                predicted_score=row[config.score_column],
            ))
    
    # Compute metrics
    overall_accuracy = np.mean([r.is_correct for r in results])
    accuracy_gain = overall_accuracy - baseline_accuracy
    
    fraction_routed_to_sc = routing_mask.mean()
    avg_samples = np.mean([r.num_samples_used for r in results])
    total_tokens = sum([sum(r.token_lengths) for r in results])
    
    # Use explicit strategy name with parameters
    strategy_name = config.get_strategy_name()
    
    # SC subset stats
    sc_results = [r for r in results if r.used_sc]
    greedy_results = [r for r in results if not r.used_sc]
    
    sc_subset_accuracy = np.mean([r.is_correct for r in sc_results]) if sc_results else None
    greedy_subset_accuracy = np.mean([r.is_correct for r in greedy_results]) if greedy_results else None
    
    # Router quality: did we route questions that actually benefit from SC?
    # "Benefit" = SC improves over greedy baseline
    benefits_from_sc = []
    router_predicted_hard = []
    
    for idx, result in enumerate(results):
        baseline_correct = df.iloc[idx]["baseline_is_correct"]
        # If we SC'd this question, check if it improved
        if result.used_sc:
            # In practice, we need to compare against baseline for same question
            # For now, we'll compute this based on whether question was wrong in baseline
            benefits_from_sc.append(baseline_correct == 0)  # wrong in baseline = could benefit
        else:
            benefits_from_sc.append(baseline_correct == 0)
        
        router_predicted_hard.append(result.used_sc)
    
    # Router metrics
    router_accuracy = accuracy_score(benefits_from_sc, router_predicted_hard)
    
    # Precision: of questions we SC'd, how many actually benefited?
    sc_indices = [i for i, r in enumerate(results) if r.used_sc]
    if sc_indices:
        router_precision = np.mean([benefits_from_sc[i] for i in sc_indices])
    else:
        router_precision = 0.0
    
    # Recall: of questions that benefit, how many did we SC?
    benefit_indices = [i for i, b in enumerate(benefits_from_sc) if b]
    if benefit_indices:
        router_recall = np.mean([router_predicted_hard[i] for i in benefit_indices])
    else:
        router_recall = 0.0
        
    # Calculate cost metrics
    # Note: baseline_cost and baseline_tokens will be set externally after first run
    cost_metrics = calculate_cost_metrics(results, baseline_cost=None)
    
    # Calculate total tokens (input + output)
    strategy_total_tokens = cost_metrics['total_input_tokens'] + cost_metrics['total_output_tokens']
    
    metrics = EvaluationMetrics(
        strategy_name=strategy_name,
        overall_accuracy=overall_accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_gain=accuracy_gain,
        fraction_routed_to_sc=fraction_routed_to_sc,
        avg_samples_per_question=avg_samples,
        total_tokens=total_tokens,
        router_accuracy=router_accuracy,
        router_precision=router_precision,
        router_recall=router_recall,
        sc_subset_accuracy=sc_subset_accuracy,
        greedy_subset_accuracy=greedy_subset_accuracy,
        total_input_tokens=cost_metrics['total_input_tokens'],
        total_output_tokens=cost_metrics['total_output_tokens'],
        input_cost=cost_metrics['input_cost'],
        output_cost=cost_metrics['output_cost'],
        total_cost=cost_metrics['total_cost'],
        cost_per_question=cost_metrics['cost_per_question'],
        num_correct=cost_metrics['num_correct'],
        cost_per_correct=cost_metrics['cost_per_correct'],
        baseline_cost=cost_metrics['baseline_cost'],
        cost_increase=cost_metrics['cost_increase'],
        roi=None,  # Will be calculated after we have baseline
        baseline_tokens=None,  # Will be set externally
        token_multiplier=None,  # Will be calculated after we have baseline_tokens
    )
    
    return metrics, results


def calculate_cost_metrics(
    results: List[SolverResult],
    baseline_cost: Optional[float] = None,
    avg_prompt_tokens: int = AVG_PROMPT_TOKENS
) -> Dict[str, float]:
    """
    Calculate token usage and cost metrics for a set of results.
    
    Args:
        results: List of SolverResult objects
        baseline_cost: Cost of baseline strategy (for comparison)
        avg_prompt_tokens: Average prompt length in tokens
    
    Returns:
        Dict with cost metrics
    """
    # Input tokens: each sample requires a full prompt
    total_input_tokens = sum(r.num_samples_used * avg_prompt_tokens for r in results)
    
    # Output tokens: sum of all generated tokens
    total_output_tokens = sum(sum(r.token_lengths) for r in results)
    
    # Calculate costs (per 1M tokens)
    input_cost = (total_input_tokens / 1_000_000) * INPUT_TOKEN_PRICE
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE
    total_cost = input_cost + output_cost
    
    # Per-question and per-correct metrics
    num_questions = len(results)
    num_correct = sum(r.is_correct for r in results)
    cost_per_question = total_cost / num_questions if num_questions > 0 else 0
    cost_per_correct = total_cost / num_correct if num_correct > 0 else float('inf')
    
    # ROI calculation (accuracy points per dollar)
    cost_increase = total_cost - baseline_cost if baseline_cost is not None else None
    
    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'cost_per_question': cost_per_question,
        'num_correct': num_correct,
        'cost_per_correct': cost_per_correct,
        'baseline_cost': baseline_cost,
        'cost_increase': cost_increase,
    }


def get_strategy_color(strategy_name: str) -> str:
    """Assign distinct colors to each strategy type"""
    if 'quantile' in strategy_name.lower():
        # Different shades for different quantiles
        if '10pct' in strategy_name:
            return '#66BB6A'  # Light green for 10% quantile
        elif '20pct' in strategy_name:
            return '#4CAF50'  # Medium green for 20% quantile
        elif '30pct' in strategy_name:
            return '#2E7D32'  # Dark green for 30% quantile
        else:
            return '#1B5E20'  # Very dark green for other quantiles
    elif 'threshold' in strategy_name.lower():
        return '#1976D2'  # Blue for threshold-based
    elif 'random' in strategy_name.lower():
        return '#FF9800'  # Orange for random
    elif 'all_sc' in strategy_name.lower():
        return '#9E9E9E'  # Gray for all_sc
    elif 'all_greedy' in strategy_name.lower():
        return '#757575'  # Dark gray for all_greedy
    else:
        return '#BDBDBD'  # Light gray for others
