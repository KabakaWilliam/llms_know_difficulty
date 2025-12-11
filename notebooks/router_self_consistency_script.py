# %%

##For tmux TMUX_DS_K_5_T_0.6(diff_amc) this uses the 5 sampled predictions for deepseek. looks crap
##For tmux TMUX_DS_K_1_T_0.0(diff_3) this uses the 1 greedy sampled predictions for deepseek
##For tmux TMUX_QwM_K_1_T_0.0(diff_4) this uses the 1 greedy sampled predictions for qwenmath inst

import pandas as pd
import numpy as np
import os
import uuid
import base64
from vllm import LLM, SamplingParams
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import evaluate_responses, verify_answer, parse_answers
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_auc_score, 
    roc_curve
)
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Callable
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
sns.set_style("whitegrid")

# %%

# %% [markdown]
# ## Configuration

# %%
# Model and dataset config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ALIAS = MODEL_NAME.split("/")[-1]

DATASETS_TO_ROUTE = ["E2H-GSM8K", "GSM_HARD", "AIME_1983_2024", "AIME_2025"]
# Generation settings
GENERATION_SETTING_STR = "max_{MAX_TOKENS}_k_{K_SAMPLE}_temp_{TEMPERATURE}"
# GENERATION_SETTING_STR = "max_3000_k_1_temp_0"
GREEDY_TEMP = 0.0
PROBE_TEMP = 0.6
PROBE_K = 5
MAX_TOKENS =3000 #32768 #3000

SC_TEMP = 0.6
NUM_SC_SAMPLES = 5 #not yet run

# Initialize model
llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9)
print(f"Loaded model: {MODEL_NAME}")

for DATASET_NAME in DATASETS_TO_ROUTE:
    print("=========="*10)
    print("\n")
    print(f"🍨Performing routing analysis for {DATASET_NAME} 🍨\n")
    print("=========="*10)


    # DATASET_NAME = "E2H-GSM8K" #"AIME_1983_2024"  # or "E2H-GSM8K", "AIME_2025"
    PROBE_SOURCE = f"{MODEL_ALIAS}_" + GENERATION_SETTING_STR.format(MAX_TOKENS=MAX_TOKENS,
                                                                    K_SAMPLE=PROBE_K,
                                                                    TEMPERATURE=PROBE_TEMP)
    PROBE_PREDICTING_STR = f"predicting_MATH_learnability_{PROBE_SOURCE}"


    FULL_PROBE_PREDICTION_SOURCE = f"{DATASET_NAME}_predicted_by_{PROBE_PREDICTING_STR}"
    # Paths
    LABELLED_DATA_PATH = f"../runs/{MODEL_ALIAS}/datasplits/{FULL_PROBE_PREDICTION_SOURCE}.json"
    RESULTS_DIR = f"../predicting_learnability/MAJORITY_VOTE_DATA/{DATASET_NAME}/{PROBE_PREDICTING_STR}_probe"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # %%


    # %% [markdown]
    # ## Routing Strategy Definitions

    # %%
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
        avg_samples_per_question: float
        total_tokens: int
        
        # Router quality (if we have ground truth)
        router_accuracy: Optional[float] = None  # did we route the right questions?
        router_precision: Optional[float] = None  # of questions we SC'd, how many benefited?
        router_recall: Optional[float] = None  # of questions that benefit from SC, how many did we catch?
        
        # SC effectiveness
        sc_subset_accuracy: Optional[float] = None  # accuracy on questions we SC'd
        greedy_subset_accuracy: Optional[float] = None  # accuracy on questions we didn't SC
        
        def efficiency_score(self) -> float:
            """Accuracy gain per unit of compute (normalized)"""
            return self.accuracy_gain / self.avg_samples_per_question
        
        def __repr__(self):
            return f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  Strategy: {self.strategy_name:<50} ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  🎯 ACCURACY:  {self.overall_accuracy:>6.2%} ({self.accuracy_gain:>+6.2%} vs baseline) ║
    ║  📊 Baseline:  {self.baseline_accuracy:>6.2%}                                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  🔀 Routing:   {self.fraction_routed_to_sc:>5.1%} → SC  |  {1-self.fraction_routed_to_sc:>5.1%} → Greedy     ║
    ║  ⚡ Avg Samples: {self.avg_samples_per_question:>5.2f}x                               ║
    ║  🏆 Efficiency: {self.efficiency_score():>6.4f}                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Router Quality:                                              ║
    ║    Accuracy:   {self.router_accuracy:>6.2%}                                 ║
    ║    Precision:  {self.router_precision:>6.2%}                                 ║
    ║    Recall:     {self.router_recall:>6.2%}                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    # %% [markdown]
    # ## Load Data

    # %%
    df = pd.read_json(LABELLED_DATA_PATH)

    MATH_PROMPT_FORMATTING = " Please put your final answer inside \\boxed{}."
    df["formatted_prompt"] = df["question"].apply(lambda x: x + MATH_PROMPT_FORMATTING)
    df["id"] = [str(uuid.uuid1()) for _ in range(len(df))]
    df["b64id"] = df["question"].apply(lambda x: base64.b64encode(x.encode()))

    print(f"Loaded {len(df)} questions from {DATASET_NAME}")
    print(f"\nPredicted difficulty stats:")
    print(df["predicted_difficulty_sigmoid"].describe())

    # %%
    # # Initialize model
    # llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.9)
    # print(f"Loaded model: {MODEL_NAME}")

    # %% [markdown]
    # ## Baseline: Greedy Decoding on Everything

    # %%
    def run_greedy_baseline(llm, df):
        """Get baseline greedy performance on all questions"""
        prompts = df["formatted_prompt"].to_list()
        gts = df["answer"].to_list()
        
        params = SamplingParams(temperature=GREEDY_TEMP, max_tokens=MAX_TOKENS)
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

    # %%
    # Run baseline (cache this so we don't regenerate)
    baseline_responses, baseline_tokens, baseline_correct = run_greedy_baseline(llm, df)

    df["baseline_response"] = baseline_responses
    df["baseline_token_length"] = baseline_tokens
    df["baseline_is_correct"] = baseline_correct

    BASELINE_ACCURACY = np.mean(baseline_correct)

    # %% [markdown]
    # ## Router Implementation

    # %%
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

    def print_routing_stats(mask: np.ndarray, config: RouterConfig):
        """Print stats about routing decisions"""
        n_sc = mask.sum()
        n_total = len(mask)
        frac_sc = n_sc / n_total
        
        print(f"\n{'='*50}")
        print(f"Routing Strategy: {config.strategy.value}")
        print(f"{'='*50}")
        print(f"SC: {n_sc}/{n_total} ({frac_sc:.1%})")
        print(f"Greedy: {n_total - n_sc}/{n_total} ({1-frac_sc:.1%})")
        print(f"Avg samples per question: {1 + frac_sc * (NUM_SC_SAMPLES - 1):.2f}x")
        print(f"{'='*50}\n")

    # %% [markdown]
    # ## Self-Consistency Evaluation

    # %%
    def run_self_consistency(llm, prompt: str, gt: str, num_samples: int = NUM_SC_SAMPLES) -> Dict:
        """
        Run self-consistency on a single question
        
        Returns:
            dict with keys: is_correct, final_answer, responses, token_lengths
        """
        params = SamplingParams(temperature=SC_TEMP, max_tokens=MAX_TOKENS)
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

    # %%
    def evaluate_routing_strategy(llm, df: pd.DataFrame, config: RouterConfig) -> EvaluationMetrics:
        """
        Evaluate a routing strategy: run SC on selected questions, greedy on others
        
        Returns:
            EvaluationMetrics with all the stats
        """
        # Get routing mask
        routing_mask = create_routing_mask(df, config)
        print_routing_stats(routing_mask, config)
        
        # For questions not routed to SC, use cached greedy results
        results = []
        
        for idx in tqdm(range(len(df)), desc=f"Evaluating {config.strategy.value}"):
            row = df.iloc[idx]
            
            if routing_mask[idx]:
                # Run SC
                sc_result = run_self_consistency(
                    llm, 
                    row["formatted_prompt"], 
                    row["answer"],
                    num_samples=NUM_SC_SAMPLES
                )
                
                results.append(SolverResult(
                    question_idx=idx,
                    used_sc=True,
                    is_correct=sc_result["is_correct"],
                    final_answer=sc_result["final_answer"],
                    num_samples_used=NUM_SC_SAMPLES,
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
        accuracy_gain = overall_accuracy - BASELINE_ACCURACY
        
        fraction_routed_to_sc = routing_mask.mean()
        avg_samples = np.mean([r.num_samples_used for r in results])
        total_tokens = sum([sum(r.token_lengths) for r in results])
        
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
        
        metrics = EvaluationMetrics(
            strategy_name=config.strategy.value,
            overall_accuracy=overall_accuracy,
            baseline_accuracy=BASELINE_ACCURACY,
            accuracy_gain=accuracy_gain,
            fraction_routed_to_sc=fraction_routed_to_sc,
            avg_samples_per_question=avg_samples,
            total_tokens=total_tokens,
            router_accuracy=router_accuracy,
            router_precision=router_precision,
            router_recall=router_recall,
            sc_subset_accuracy=sc_subset_accuracy,
            greedy_subset_accuracy=greedy_subset_accuracy,
        )
        
        return metrics, results

    # %% [markdown]
    # ## Run Experiments

    # %%
    # Define experiments to run
    experiments = [
        # Baseline: no SC
        RouterConfig(strategy=RoutingStrategy.ALL_GREEDY),
        
        # Upper bound: SC on everything
        RouterConfig(strategy=RoutingStrategy.ALL_SC),
        
        # Random: 30% of questions
        RouterConfig(strategy=RoutingStrategy.RANDOM, fraction=0.3, seed=42),
        
        # Probe-based: threshold at 0.5
        RouterConfig(
            strategy=RoutingStrategy.PROBE_THRESHOLD,
            threshold=0.5,
            score_column="predicted_difficulty_sigmoid"
        ),
        
        # Probe-based: bottom 30% quantile
        RouterConfig(
            strategy=RoutingStrategy.PROBE_QUANTILE,
            quantile=0.3,
            score_column="predicted_difficulty"
        ),
        
        # Probe-based: bottom 10% quantile
        RouterConfig(
            strategy=RoutingStrategy.PROBE_QUANTILE,
            quantile=0.1,
            score_column="predicted_difficulty"
        ),
    ]

    print(f"Will run {len(experiments)} experiments")

    # %%
    # Run all experiments
    all_metrics = []
    all_results = {}

    for config in experiments:
        print(f"\n{'#'*60}")
        print(f"Running: {config.strategy.value}")
        print(f"{'#'*60}")
        
        metrics, results = evaluate_routing_strategy(llm, df, config)
        all_metrics.append(metrics)
        all_results[config.strategy.value] = results
        
        print(metrics)
        print("\n" * 2)  # Add spacing to separate from next batch of VLLM logs

    # %% [markdown]
    # ## Analysis & Visualization

    # %%
    # Create comparison dataframe
    comparison_df = pd.DataFrame([
        {
            "Strategy": m.strategy_name,
            "Accuracy": m.overall_accuracy,
            "Accuracy Gain": m.accuracy_gain,
            "Fraction SC": m.fraction_routed_to_sc,
            "Avg Samples": m.avg_samples_per_question,
            "Efficiency": m.efficiency_score(),
            "Router Acc": m.router_accuracy,
            "Router Prec": m.router_precision,
            "Router Recall": m.router_recall,
        }
        for m in all_metrics
    ])

    comparison_df = comparison_df.sort_values("Efficiency", ascending=False)

    # Print a VERY VISIBLE summary
    print("\n" * 3)
    print("="*80)
    print("="*80)
    print("🏁 FINAL RESULTS SUMMARY 🏁".center(80))
    print("="*80)
    print("="*80)
    print("\n")

    # Format the dataframe nicely
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\n")
    print("="*80)
    print("📊 KEY TAKEAWAYS".center(80))
    print("="*80)

    # Find best strategies
    best_efficiency = comparison_df.iloc[0]
    best_accuracy = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_probe = comparison_df[comparison_df['Strategy'].str.contains('probe', case=False)]

    if len(best_probe) > 0:
        best_probe = best_probe.iloc[0]
        print(f"\n🏆 BEST PROBE STRATEGY: {best_probe['Strategy']}")
        print(f"   → Accuracy: {best_probe['Accuracy']:.2%} ({best_probe['Accuracy Gain']:+.2%} gain)")
        print(f"   → Efficiency: {best_probe['Efficiency']:.4f}")
        print(f"   → Routes {best_probe['Fraction SC']:.1%} to SC")

    print(f"\n⚡ BEST EFFICIENCY: {best_efficiency['Strategy']}")
    print(f"   → Efficiency: {best_efficiency['Efficiency']:.4f}")
    print(f"   → Accuracy: {best_efficiency['Accuracy']:.2%}")

    print(f"\n🎯 BEST ACCURACY: {best_accuracy['Strategy']}")
    print(f"   → Accuracy: {best_accuracy['Accuracy']:.2%}")
    print(f"   → Cost: {best_accuracy['Avg Samples']:.2f}x samples")

    print("\n" + "="*80)
    print("\n" * 2)

    comparison_df

    # %%
    # Plot: Accuracy vs Compute
    fig, ax = plt.subplots(figsize=(10, 6))

    for m in all_metrics:
        ax.scatter(
            m.avg_samples_per_question,
            m.overall_accuracy,
            s=200,
            alpha=0.7,
            label=m.strategy_name
        )
        ax.annotate(
            m.strategy_name,
            (m.avg_samples_per_question, m.overall_accuracy),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    ax.axhline(BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Average Samples per Question', fontsize=12)
    ax.set_ylabel('Overall Accuracy', fontsize=12)
    ax.set_title(f'{DATASET_NAME}: Accuracy vs Compute Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/accuracy_vs_compute.png', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    # Define color scheme for all plots
    def get_strategy_color(strategy_name):
        """Assign distinct colors to each strategy type"""
        if 'quantile' in strategy_name.lower():
            # Different shades for different quantiles
            if '0.3' in strategy_name:
                return '#2E7D32'  # Dark green for 30% quantile
            elif '0.1' in strategy_name:
                return '#66BB6A'  # Light green for 10% quantile
            else:
                return '#4CAF50'  # Medium green for other quantiles
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
    
    # %%
    # Plot: Efficiency scores
    fig, ax = plt.subplots(figsize=(12, 7))

    strategies = [m.strategy_name for m in all_metrics]
    efficiencies = [m.efficiency_score() for m in all_metrics]

    colors = [get_strategy_color(s) for s in strategies]
    bars = ax.barh(strategies, efficiencies, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Efficiency Score (Accuracy Gain / Avg Samples)', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Routing Strategy Efficiency', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (strat, eff) in enumerate(zip(strategies, efficiencies)):
        ax.text(eff, i, f'  {eff:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#2E7D32', edgecolor='black', label='Probe Quantile 30%'),
        Patch(facecolor='#66BB6A', edgecolor='black', label='Probe Quantile 10%'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Probe Threshold'),
        Patch(facecolor='#FF9800', edgecolor='black', label='Random'),
        Patch(facecolor='#9E9E9E', edgecolor='black', label='All SC'),
        Patch(facecolor='#757575', edgecolor='black', label='All Greedy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    # Plot: Router quality metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    metrics_to_plot = [
        ('router_accuracy', 'Router Accuracy', 0),
        ('router_precision', 'Router Precision', 1),
        ('router_recall', 'Router Recall', 2),
    ]

    for metric_name, title, idx in metrics_to_plot:
        ax = axes[idx]
        strategies = [m.strategy_name for m in all_metrics if getattr(m, metric_name) is not None]
        values = [getattr(m, metric_name) for m in all_metrics if getattr(m, metric_name) is not None]
        
        # Use same color scheme
        colors = [get_strategy_color(s) for s in strategies]
        ax.barh(strategies, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (strat, val) in enumerate(zip(strategies, values)):
            ax.text(val, i, f'  {val:.3f}', va='center', fontsize=8)
    
    # Add shared legend below all subplots
    legend_elements = [
        Patch(facecolor='#2E7D32', edgecolor='black', label='Probe Quantile 30%'),
        Patch(facecolor='#66BB6A', edgecolor='black', label='Probe Quantile 10%'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Probe Threshold'),
        Patch(facecolor='#FF9800', edgecolor='black', label='Random'),
        Patch(facecolor='#9E9E9E', edgecolor='black', label='All SC'),
        Patch(facecolor='#757575', edgecolor='black', label='All Greedy'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
               ncol=6, frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/router_quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # %% [markdown]
    # ## Deep Dive: Probe-based Routing Analysis

    # %%
    # For best probe-based strategy, analyze which questions benefited
    probe_metrics = [m for m in all_metrics if 'probe' in m.strategy_name.lower()]
    best_probe_metric = max(probe_metrics, key=lambda m: m.efficiency_score())

    print(f"Best probe-based strategy: {best_probe_metric.strategy_name}")
    print(best_probe_metric)

    # %%
    # Analyze improvement distribution
    best_probe_results = all_results[best_probe_metric.strategy_name]

    improvements = []
    for idx, result in enumerate(best_probe_results):
        baseline_correct = df.iloc[idx]["baseline_is_correct"]
        improvement = result.is_correct - baseline_correct
        improvements.append(improvement)

    df["improvement"] = improvements
    df["used_sc"] = [r.used_sc for r in best_probe_results]

    print("Improvement distribution:")
    print(pd.Series(improvements).value_counts().sort_index())

    # %%
    # Plot: improvement by predicted difficulty
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    labels = {-1: 'Worse with SC', 0: 'No change', 1: 'Better with SC'}

    for improvement_val in [-1, 0, 1]:
        subset = df[df["improvement"] == improvement_val]
        if len(subset) > 0:
            ax.scatter(
                subset["predicted_difficulty_sigmoid"],
                subset["baseline_is_correct"],
                c=colors[improvement_val],
                label=labels[improvement_val],
                alpha=0.6,
                s=50
            )

    ax.set_xlabel('Predicted Difficulty (sigmoid)', fontsize=12)
    ax.set_ylabel('Baseline Correctness', fontsize=12)
    ax.set_title(f'SC Impact by Predicted Difficulty ({best_probe_metric.strategy_name})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/sc_impact_by_difficulty.png', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    # Confusion matrix for router
    # Ground truth: did question benefit from SC? (baseline wrong, SC right)
    actually_benefits = (df["baseline_is_correct"] == 0) & (df["improvement"] == 1)
    router_sent_to_sc = df["used_sc"]

    cm = confusion_matrix(actually_benefits, router_sent_to_sc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Benefit', 'Benefits'])
    disp.plot(cmap='Blues')
    plt.title(f'Router Confusion Matrix: {best_probe_metric.strategy_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/router_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # %% [markdown]
    # ## Save Results

    # %%
    # Save comparison table
    comparison_df.to_csv(f"{RESULTS_DIR}/routing_comparison.csv", index=False)

    # Save detailed results for each strategy
    for strategy_name, results in all_results.items():
        results_df = pd.DataFrame([
            {
                "question_idx": r.question_idx,
                "used_sc": r.used_sc,
                "is_correct": r.is_correct,
                "num_samples": r.num_samples_used,
                "predicted_score": r.predicted_score,
                "total_tokens": sum(r.token_lengths),
            }
            for r in results
        ])
        
        safe_name = strategy_name.replace("/", "_")
        results_df.to_parquet(f"{RESULTS_DIR}/{MODEL_ALIAS}_{safe_name}_results.parquet")

    print(f"\nResults saved to {RESULTS_DIR}/")

    # %%
    # FINAL SUMMARY - This is what you'll see at the bottom of tmux!
    print("\n" * 5)
    print("█" * 80)
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  🎉 EXPERIMENT COMPLETE! 🎉  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("█" * 80)
    print()

    # Quick reference table
    print("┌" + "─" * 78 + "┐")
    print("│" + " QUICK REFERENCE ".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    print(f"│  Dataset: {DATASET_NAME:<65}│")
    print(f"│  Model: {MODEL_ALIAS:<67}│")
    print(f"│  Baseline Accuracy: {BASELINE_ACCURACY:>6.2%}{' ' * 51}│")
    print(f"│  Total Experiments: {len(all_metrics):<57}│")
    print("└" + "─" * 78 + "┘")
    print()

    # Top 3 strategies
    print("🏆 TOP 3 STRATEGIES BY EFFICIENCY:")
    print("═" * 80)
    top3 = comparison_df.head(3)
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{i}. {row['Strategy']}")
        print(f"   Accuracy: {row['Accuracy']:.2%} | Efficiency: {row['Efficiency']:.4f} | SC: {row['Fraction SC']:.1%}")

    print("\n\n")
    print("🎯 TOP 3 STRATEGIES BY ACCURACY:")
    print("═" * 80)
    top3_acc = comparison_df.sort_values("Accuracy", ascending=False).head(3)
    for i, (idx, row) in enumerate(top3_acc.iterrows(), 1):
        print(f"\n{i}. {row['Strategy']}")
        print(f"   Accuracy: {row['Accuracy']:.2%} | Gain: {row['Accuracy Gain']:+.2%} | Cost: {row['Avg Samples']:.2f}x")

    print("\n\n")
    print("=" * 80)
    print(f"✅ Full results saved to: {RESULTS_DIR}/")
    print("=" * 80)
    print("\n" * 3)

    # %% [markdown]
    # ## Key Insights
    # 
    # **Questions to answer:**
    # 1. Does probe-based routing beat random routing?
    # 2. What's the efficiency gain of routing vs SC-on-everything?
    # 3. Which questions does the probe miss? (high predicted difficulty but actually easy)
    # 4. Is there a sweet spot for the routing threshold/quantile?
    # 
    # **Next steps:**
    # - Try different SC sample sizes (k=3, k=10, etc.)
    # - Test on different datasets (OOD generalization)
    # - Implement oracle routing (use actual SC benefit as ground truth)
    # - Try ensemble of probes or more sophisticated routing logic


