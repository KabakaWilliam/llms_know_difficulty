"""
Self-Consistency Router Evaluation Script

Evaluates different routing strategies for deciding when to use self-consistency
vs greedy decoding based on predicted difficulty from probes.
"""

import pandas as pd
import numpy as np
import os
import uuid
import base64
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import requests

# Import our router components
from router_components import (
    RoutingStrategy,
    RouterConfig,
    SolverResult,
    EvaluationMetrics,
    run_greedy_baseline,
    evaluate_routing_strategy,
    get_strategy_color
)

sns.set_style("whitegrid")


# ============================================================================
# CONFIGURATION
# ============================================================================

TOKEN_BUDGET_DICT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 32768,
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 3000
}

# Experiment Configuration
CONFIG = {
    # Device settings
    "device": 1,
    
    # Model settings
    "model_name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "memory_util": 0.6,

    #Probe settings
    "probe_source": "GSM8K",
    "probe_temp": 0.0,  # 0.6 is default; 0.0 temp was probe at greedy
    "probe_k": 1,  # Num samples for which probe label was calculated

    # Dataset settings
    "datasets": ["AIME_2025","E2H-GSM8K", "GSM_HARD", "AIME_1983_2024"],
    "dataset_sample_fraction": 1.0,  # Fraction of dataset to use (1.0 = all)
    
    # Generation settings
    "greedy_temp": 0.0,
    "sc_temp": 0.6,
    "num_sc_samples": 5,
    
    # Output settings
    "results_base_dir": "predicting_learnability",
    "router_directory_name": "ROUTER_SELF_CONCISTENCY_EXPERIMENTS",
    
    # Notification settings
    "send_notifications": True,
    "notification_url": "https://ntfy.sh/router_runs_lugoloobi",
}

# Derived settings
CONFIG["model_alias"] = CONFIG["model_name"].split("/")[-1]
CONFIG["max_tokens"] = TOKEN_BUDGET_DICT[CONFIG["model_name"]]


# ============================================================================
# SETUP
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = f"{CONFIG['device']}"

GENERATION_SETTING_STR = "max_{MAX_TOKENS}_k_{K_SAMPLE}_temp_{TEMPERATURE}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def send_notification(title: str, message: str):
    """Send notification to ntfy.sh"""
    if not CONFIG.get("send_notifications", True):
        return
    
    try:
        requests.post(
            CONFIG["notification_url"],
            data=message.encode(encoding='utf-8'),
            headers={"Title": title}
        )
    except:
        pass  # Don't fail if notification doesn't work


# ============================================================================
# MAIN PROCESSING
# ============================================================================

print(f"{'='*70}")
print(f"Self-Consistency Router Evaluation")
print(f"{'='*70}")
print(f"Model: {CONFIG['model_name']}")
print(f"Datasets: {', '.join(CONFIG['datasets'])}")
print(f"SC Samples: {CONFIG['num_sc_samples']} @ temp {CONFIG['sc_temp']}")
print(f"{'='*70}\n")

# Initialize model
print("Loading model...")
llm = LLM(model=CONFIG["model_name"], gpu_memory_utilization=CONFIG["memory_util"])
print(f"✓ Loaded model: {CONFIG['model_name']}\n")

# Process each dataset
for DATASET_NAME in CONFIG["datasets"]:
    print("="*100)
    print(f"\n🍨 Performing routing analysis for {DATASET_NAME} 🍨\n")
    print("="*100)
    
    # Setup paths
    MODEL_ALIAS = CONFIG["model_alias"]
    MAX_TOKENS = CONFIG["max_tokens"]
    PROBE_K = CONFIG["probe_k"]
    PROBE_TEMP = CONFIG["probe_temp"]
    GREEDY_TEMP = CONFIG["greedy_temp"]
    ROUTER_DIRECTORY_NAME = CONFIG["router_directory_name"]
    DATASET_SAMPLE_AMOUNT = CONFIG["dataset_sample_fraction"]
    SC_TEMP = CONFIG["sc_temp"]
    NUM_SC_SAMPLES = CONFIG["num_sc_samples"]
    PROBE_SOURCE = CONFIG["probe_source"]
    
    PROBE_FILE_SOURCE = f"{MODEL_ALIAS}_" + GENERATION_SETTING_STR.format(
        MAX_TOKENS=MAX_TOKENS,
        K_SAMPLE=PROBE_K,
        TEMPERATURE=PROBE_TEMP
    )
    PROBE_PREDICTING_STR = ""
    if PROBE_SOURCE == "MATH": #SHAKY FIX. JUST HAVE CONSISTENT FILE STRUCTURE
        PROBE_PREDICTING_STR = f"predicting_{PROBE_SOURCE}_learnability_{PROBE_FILE_SOURCE}"
    elif PROBE_SOURCE == "GSM8K":
        PROBE_PREDICTING_STR = f"predicting_{PROBE_SOURCE}_SR_{PROBE_FILE_SOURCE}"

    FULL_PROBE_PREDICTION_SOURCE = f"{DATASET_NAME}_predicted_by_{PROBE_PREDICTING_STR}"
    
    LABELLED_DATA_PATH = f"../runs/{MODEL_ALIAS}/datasplits/{FULL_PROBE_PREDICTION_SOURCE}.json"
    RESULTS_DIR = f"../{CONFIG['results_base_dir']}/{ROUTER_DIRECTORY_NAME}/{DATASET_NAME}/{PROBE_PREDICTING_STR}_probe"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Load data
    df = pd.read_json(LABELLED_DATA_PATH)

    LOCAL_DATASET_SAMPLE_AMOUNT= int(DATASET_SAMPLE_AMOUNT * len(df))
    df = df.sample(n=LOCAL_DATASET_SAMPLE_AMOUNT, random_state=42)
    MATH_PROMPT_FORMATTING = " Please put your final answer inside \\boxed{}."
    df["formatted_prompt"] = df["question"].apply(lambda x: x + MATH_PROMPT_FORMATTING)
    df["id"] = [str(uuid.uuid1()) for _ in range(len(df))]
    df["b64id"] = df["question"].apply(lambda x: base64.b64encode(x.encode()))
    
    print(f"Loaded {len(df)} questions from {DATASET_NAME}")
    print(f"\nPredicted difficulty stats:")
    print(df["predicted_difficulty_sigmoid"].describe())
    
    # Run baseline (cache this so we don't regenerate)
    print("\n" + "="*50)
    print("Running Greedy Baseline...")
    print("="*50)
    baseline_responses, baseline_tokens, baseline_correct = run_greedy_baseline(
        llm, df, GREEDY_TEMP, MAX_TOKENS
    )
    
    df["baseline_response"] = baseline_responses
    df["baseline_token_length"] = baseline_tokens
    df["baseline_is_correct"] = baseline_correct
    BASELINE_ACCURACY = np.mean(baseline_correct)
    
    # Define experiments to run
    experiments = [
        # Baseline: no SC
        RouterConfig(strategy=RoutingStrategy.ALL_GREEDY),
        
        # Upper bound: SC on everything
        RouterConfig(strategy=RoutingStrategy.ALL_SC),
        
        
        # Probe-based: threshold at 0.5
        RouterConfig(
            strategy=RoutingStrategy.PROBE_THRESHOLD,
            threshold=0.5,
            score_column="predicted_difficulty_sigmoid"
        ),
        
        # Random: 30% of questions
        RouterConfig(strategy=RoutingStrategy.RANDOM, fraction=0.3, seed=42),

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
    
    print(f"\nWill run {len(experiments)} experiments\n")
    
    # Run all experiments
    all_metrics = []
    all_results = {}
    baseline_cost = None
    baseline_tokens = None
    
    for config in experiments:
        strategy_name = config.get_strategy_name()
        print(f"\n{'#'*60}")
        print(f"Running: {strategy_name}")
        print(f"{'#'*60}")
        
        metrics, results = evaluate_routing_strategy(
            llm, df, config, BASELINE_ACCURACY, SC_TEMP, MAX_TOKENS, NUM_SC_SAMPLES
        )
        
        # Set baseline cost and tokens from first experiment (all_greedy)
        if baseline_cost is None and config.strategy == RoutingStrategy.ALL_GREEDY:
            baseline_cost = metrics.total_cost
            baseline_tokens = metrics.total_input_tokens + metrics.total_output_tokens
            print(f"\n💰 Baseline cost: ${baseline_cost:.5f}")
            print(f"📊 Baseline tokens: {baseline_tokens:,}")
        
        # Calculate ROI and cost increase now that we have baseline
        if baseline_cost is not None and baseline_tokens is not None:
            metrics.baseline_cost = baseline_cost
            metrics.cost_increase = metrics.total_cost - baseline_cost
            metrics.baseline_tokens = baseline_tokens
            
            strategy_tokens = metrics.total_input_tokens + metrics.total_output_tokens
            metrics.token_multiplier = strategy_tokens / baseline_tokens if baseline_tokens > 0 else 1.0
            
            # ROI calculation:
            # - If cost increases and accuracy improves: normal ROI
            # - If cost decreases and accuracy improves: infinite ROI (free lunch!)
            # - If accuracy doesn't improve: ROI = 0
            if metrics.cost_increase > 0 and metrics.accuracy_gain > 0:
                metrics.roi = metrics.accuracy_gain / metrics.cost_increase
            elif metrics.cost_increase <= 0 and metrics.accuracy_gain > 0:
                # Cheaper AND better = infinite ROI (or we could use negative cost as "profit")
                metrics.roi = float('inf')
            elif metrics.accuracy_gain < 0:
                # Worse accuracy = negative ROI (you're losing)
                metrics.roi = metrics.accuracy_gain / abs(metrics.cost_increase) if metrics.cost_increase != 0 else -999
            else:
                metrics.roi = 0.0
        
        all_metrics.append(metrics)
        all_results[strategy_name] = results
        
        print(metrics)
        print("\n" * 2)
    
    # Analysis & Visualization
    print("\n" + "="*80)
    print("ANALYSIS & VISUALIZATION")
    print("="*80 + "\n")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([
        {
            "Strategy": m.strategy_name,
            "Accuracy": m.overall_accuracy,
            "Accuracy Gain": m.accuracy_gain,
            "Fraction SC": m.fraction_routed_to_sc,
            "Avg Samples": m.avg_samples_per_question,
            "Token Multiplier": m.token_multiplier,
            "Efficiency (sample)": m.efficiency_score(),
            "Efficiency (token)": m.token_efficiency(),
            "Efficiency (cost)": m.cost_efficiency(),
            "Total Cost": m.total_cost,
            "Cost/Question": m.cost_per_question,
            "Cost/Correct": m.cost_per_correct,
            "ROI (pts/$)": m.roi * 100 if m.roi and not np.isinf(m.roi) else 0,  # accuracy points per dollar
            "Router Acc": m.router_accuracy,
            "Router Prec": m.router_precision,
            "Router Recall": m.router_recall,
            "Pass@1": m.avg_pass_at_1 if m.avg_pass_at_1 else 0,
            "Pass@5": m.pass_at_k.get(5, 0) if m.pass_at_k else 0,
            "Avg First Correct": m.avg_first_correct_sample if m.avg_first_correct_sample != float('inf') else 0,
            "Total Correct Samples": m.total_correct_samples if m.total_correct_samples else 0,
            "Best-Case Oracle Acc": m.oracle_best_case_accuracy if m.oracle_best_case_accuracy else 0,
            "Best-Case Oracle Gain": m.oracle_best_case_gain if m.oracle_best_case_gain else 0,
            "Sample Utilization": m.sample_utilization if m.sample_utilization else 0,
        }
        for m in all_metrics
    ])
    
    comparison_df = comparison_df.sort_values("Efficiency (token)", ascending=False)
    
    # Print summary
    print("\n" * 3)
    print("="*80)
    print("="*80)
    print("🏁 FINAL RESULTS SUMMARY 🏁".center(80))
    print("="*80)
    print("="*80)
    print("\n")
    
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
        print(f"   → Efficiency (token): {best_probe['Efficiency (token)']:.4f}")
        print(f"   → Efficiency (sample): {best_probe['Efficiency (sample)']:.4f}")
        print(f"   → Routes {best_probe['Fraction SC']:.1%} to SC")
    
    print(f"\n⚡ BEST EFFICIENCY (TOKEN-WEIGHTED): {best_efficiency['Strategy']}")
    print(f"   → Token Efficiency: {best_efficiency['Efficiency (token)']:.4f}")
    print(f"   → Sample Efficiency: {best_efficiency['Efficiency (sample)']:.4f}")
    print(f"   → Accuracy: {best_efficiency['Accuracy']:.2%}")
    
    print(f"\n🎯 BEST ACCURACY: {best_accuracy['Strategy']}")
    print(f"   → Accuracy: {best_accuracy['Accuracy']:.2%}")
    print(f"   → Cost: {best_accuracy['Avg Samples']:.2f}x samples")
    
    # Theoretical Best-Case Oracle Analysis
    print("\n" + "="*80)
    print("🔮 THEORETICAL BEST-CASE ORACLE ANALYSIS".center(80))
    print("="*80)
    print("\nIf we could always pick the best sample (theoretical oracle selector):\n")
    
    # Sort by oracle gain (potential improvement)
    oracle_sorted = comparison_df.sort_values("Best-Case Oracle Gain", ascending=False)
    
    print("Top 3 strategies with highest improvement potential:")
    for i, (idx, row) in enumerate(oracle_sorted.head(3).iterrows(), 1):
        print(f"\n{i}. {row['Strategy']}")
        print(f"   → Current Accuracy: {row['Accuracy']:.2%}")
        print(f"   → Best-Case Oracle Accuracy:  {row['Best-Case Oracle Acc']:.2%}")
        print(f"   → Potential Gain:   {row['Best-Case Oracle Gain']:+.2%}")
        print(f"   → Sample Utilization: {row['Sample Utilization']:.1%} (capturing {row['Sample Utilization']:.1%} of available correct samples)")
    
    # Find strategies with best sample utilization
    best_utilization = comparison_df.sort_values("Sample Utilization", ascending=False).iloc[0]
    worst_utilization_df = comparison_df[comparison_df['Sample Utilization'] > 0].sort_values("Sample Utilization", ascending=True)
    
    print(f"\n📊 Sample Utilization Insights:")
    print(f"   → Best utilization: {best_utilization['Strategy']} at {best_utilization['Sample Utilization']:.1%}")
    print(f"     (Majority voting captures {best_utilization['Sample Utilization']:.1%} of correct samples)")
    
    if len(worst_utilization_df) > 0:
        worst_utilization = worst_utilization_df.iloc[0]
        print(f"   → Worst utilization: {worst_utilization['Strategy']} at {worst_utilization['Sample Utilization']:.1%}")
        print(f"     (Leaving {worst_utilization['Best-Case Oracle Gain']:.1%} accuracy points on the table)")
    else:
        print(f"   → No strategies with multiple samples to analyze utilization")
    
    print("\n" + "="*80)
    print("\n" * 2)
    
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
    plt.close()
    
    # Plot: Actual vs Oracle (Theoretical Best) Accuracy
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for m in all_metrics:
        color = get_strategy_color(m.strategy_name)
        # Plot actual accuracy
        ax.scatter(
            m.avg_samples_per_question,
            m.overall_accuracy,
            s=200,
            alpha=0.8,
            color=color,
            edgecolor='black',
            linewidth=2,
            label=f'{m.strategy_name} (actual)',
            marker='o'
        )
        
        # Plot theoretical best-case accuracy (if we could pick best sample)
        if m.oracle_best_case_accuracy:
            ax.scatter(
                m.avg_samples_per_question,
                m.oracle_best_case_accuracy,
                s=200,
                alpha=0.5,
                color=color,
                edgecolor='black',
                linewidth=2,
                linestyle='--',
                marker='^'
            )
            
            # Draw line connecting actual to theoretical best-case
            ax.plot(
                [m.avg_samples_per_question, m.avg_samples_per_question],
                [m.overall_accuracy, m.oracle_best_case_accuracy],
                color=color,
                linestyle=':',
                linewidth=2,
                alpha=0.5
            )
            
            # Annotate with best-case gain
            mid_y = (m.overall_accuracy + m.oracle_best_case_accuracy) / 2
            ax.annotate(
                f'+{m.oracle_best_case_gain:.1%}',
                (m.avg_samples_per_question, mid_y),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=8,
                color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.7)
            )
    
    # Add legend elements for actual vs theoretical best-case
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
               label='Actual (majority vote)', markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, 
               label='Theoretical Best-Case', markeredgecolor='black', markeredgewidth=2, alpha=0.5),
    ]
    
    ax.axhline(BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Baseline')
    ax.set_xlabel('Average Samples per Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Actual vs Theoretical Best-Case Accuracy\n(Lines show potential improvement if we could always pick the best sample)', 
                 fontsize=14, fontweight='bold')
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/accuracy_vs_compute_with_theoretical_best_case.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Cost vs Accuracy Trade-off
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for m in all_metrics:
        color = get_strategy_color(m.strategy_name)
        ax.scatter(
            m.total_cost * 1000,  # convert to cents
            m.overall_accuracy * 100,  # percentage
            s=300,
            alpha=0.8,
            color=color,
            edgecolor='black',
            linewidth=2
        )
        ax.annotate(
            m.strategy_name,
            (m.total_cost * 1000, m.overall_accuracy * 100),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7)
        )
    
    # Draw Pareto frontier
    sorted_by_cost = sorted(all_metrics, key=lambda m: m.total_cost)
    pareto_points = []
    max_acc = 0
    for m in sorted_by_cost:
        if m.overall_accuracy > max_acc:
            pareto_points.append(m)
            max_acc = m.overall_accuracy
    
    if len(pareto_points) > 1:
        pareto_costs = [m.total_cost * 1000 for m in pareto_points]
        pareto_accs = [m.overall_accuracy * 100 for m in pareto_points]
        ax.plot(pareto_costs, pareto_accs, 'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')
    
    ax.set_xlabel('Total Cost (cents)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Cost-Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/cost_vs_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Efficiency scores
    fig, ax = plt.subplots(figsize=(12, 7))
    
    strategies = [m.strategy_name for m in all_metrics]
    efficiencies = [m.efficiency_score() for m in all_metrics]
    
    colors = [get_strategy_color(s) for s in strategies]
    bars = ax.barh(strategies, efficiencies, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Efficiency Score (Accuracy Gain / Avg Samples)', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Routing Strategy Efficiency', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (strat, eff) in enumerate(zip(strategies, efficiencies)):
        ax.text(eff, i, f'  {eff:.4f}', va='center', fontsize=9, fontweight='bold')
    
    legend_elements = [
        Patch(facecolor='#66BB6A', edgecolor='black', label='Probe Quantile Bottom 10%'),
        Patch(facecolor='#4CAF50', edgecolor='black', label='Probe Quantile Bottom 20%'),
        Patch(facecolor='#2E7D32', edgecolor='black', label='Probe Quantile Bottom 30%'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Probe Threshold 0.50'),
        Patch(facecolor='#FF9800', edgecolor='black', label='Random'),
        Patch(facecolor='#9E9E9E', edgecolor='black', label='All SC'),
        Patch(facecolor='#757575', edgecolor='black', label='All Greedy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Sample Utilization Efficiency (how well majority voting works)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    strategies_with_oracle = [m for m in all_metrics if m.sample_utilization is not None and m.sample_utilization > 0]
    if strategies_with_oracle:
        strategy_names = [m.strategy_name for m in strategies_with_oracle]
        utilization = [m.sample_utilization for m in strategies_with_oracle]
        oracle_gains = [m.oracle_best_case_gain for m in strategies_with_oracle]
        
        colors_list = [get_strategy_color(s) for s in strategy_names]
        
        bars = ax.barh(strategy_names, utilization, color=colors_list, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Sample Utilization (Actual / Theoretical Best-Case)', fontsize=12, fontweight='bold')
        ax.set_title(f'{DATASET_NAME}: Sample Utilization Efficiency\n(Higher = Majority voting effectively captures correct samples)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.05])
        ax.grid(axis='x', alpha=0.3)
        
        # Add text showing utilization percentage and potential gain
        for i, (name, util, gain) in enumerate(zip(strategy_names, utilization, oracle_gains)):
            ax.text(util, i, f'  {util:.1%} (unused: {gain:+.1%})', va='center', fontsize=9, fontweight='bold')
        
        # Add vertical line at 100%
        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect utilization')
        ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/sample_utilization_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Oracle Gain Potential
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if strategies_with_oracle:
        strategy_names = [m.strategy_name for m in strategies_with_oracle]
        oracle_gains = [m.oracle_best_case_gain * 100 for m in strategies_with_oracle]  # Convert to percentage points
        
        # Sort by oracle gain
        sorted_indices = np.argsort(oracle_gains)[::-1]  # Descending order
        strategy_names_sorted = [strategy_names[i] for i in sorted_indices]
        oracle_gains_sorted = [oracle_gains[i] for i in sorted_indices]
        colors_sorted = [get_strategy_color(s) for s in strategy_names_sorted]
        
        bars = ax.barh(strategy_names_sorted, oracle_gains_sorted, color=colors_sorted, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Best-Case Oracle Gain (Percentage Points)', fontsize=12, fontweight='bold')
        ax.set_title(f'{DATASET_NAME}: Theoretical Best-Case Improvement Potential\n(Accuracy gain if we could always pick the best sample)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (name, gain) in enumerate(zip(strategy_names_sorted, oracle_gains_sorted)):
            ax.text(gain, i, f'  +{gain:.1f} pts', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/theoretical_best_case_gain_potential.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
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
        
        colors = [get_strategy_color(s) for s in strategies]
        ax.barh(strategies, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (strat, val) in enumerate(zip(strategies, values)):
            ax.text(val, i, f'  {val:.3f}', va='center', fontsize=8)
    
    legend_elements = [
        Patch(facecolor='#66BB6A', edgecolor='black', label='Probe Quantile Bottom 10%'),
        Patch(facecolor='#4CAF50', edgecolor='black', label='Probe Quantile Bottom 20%'),
        Patch(facecolor='#2E7D32', edgecolor='black', label='Probe Quantile Bottom 30%'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Probe Threshold 0.50'),
        Patch(facecolor='#FF9800', edgecolor='black', label='Random'),
        Patch(facecolor='#9E9E9E', edgecolor='black', label='All SC'),
        Patch(facecolor='#757575', edgecolor='black', label='All Greedy'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
               ncol=7, frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/router_quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Actual vs Oracle Accuracy Comparison (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    strategies_list = [m.strategy_name for m in all_metrics]
    actual_acc = [m.overall_accuracy * 100 for m in all_metrics]
    oracle_acc = [m.oracle_best_case_accuracy * 100 if m.oracle_best_case_accuracy else m.overall_accuracy * 100 for m in all_metrics]
    
    x = np.arange(len(strategies_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, actual_acc, width, label='Actual (Majority Vote)', 
                   color='#42A5F5', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, oracle_acc, width, label='Theoretical Best-Case', 
                   color='#66BB6A', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add text labels on bars
    for i, (actual, oracle) in enumerate(zip(actual_acc, oracle_acc)):
        ax.text(i - width/2, actual + 0.5, f'{actual:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width/2, oracle + 0.5, f'{oracle:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Show gap
        gap = oracle - actual
        if gap > 0.5:  # Only show if gap is meaningful
            ax.annotate('', xy=(i + width/2, oracle - 0.3), xytext=(i - width/2, actual + 0.3),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=1.5, alpha=0.6))
            ax.text(i, (actual + oracle) / 2, f'+{gap:.1f}', ha='center', va='center',
                   fontsize=7, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.8))
    
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Actual vs Theoretical Best-Case Accuracy\n(Gap shows potential improvement if we could always pick the best sample)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies_list], fontsize=9, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(BASELINE_ACCURACY * 100, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Baseline')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/actual_vs_theoretical_best_case_accuracy_bars.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Cost per Correct Answer
    fig, ax = plt.subplots(figsize=(12, 7))
    
    strategies = [m.strategy_name for m in all_metrics]
    costs_per_correct = [m.cost_per_correct * 1000 if not np.isinf(m.cost_per_correct) else 0 for m in all_metrics]  # cents
    
    # Sort by cost per correct
    sorted_indices = np.argsort(costs_per_correct)
    strategies_sorted = [strategies[i] for i in sorted_indices]
    costs_sorted = [costs_per_correct[i] for i in sorted_indices]
    colors_sorted = [get_strategy_color(s) for s in strategies_sorted]
    
    bars = ax.barh(strategies_sorted, costs_sorted, color=colors_sorted, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Cost per Correct Answer (cents)', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Cost Efficiency\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (strat, cost) in enumerate(zip(strategies_sorted, costs_sorted)):
        ax.text(cost, i, f'  {cost:.3f}¢', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/cost_per_correct_answer.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: ROI Analysis (Accuracy Points per Dollar)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Only show strategies with positive accuracy gain
    roi_metrics = [m for m in all_metrics if m.accuracy_gain > 0 and m.roi and not np.isinf(m.roi)]
    
    if roi_metrics:
        roi_strategies = [m.strategy_name for m in roi_metrics]
        roi_values = [m.roi * 100 for m in roi_metrics]  # accuracy points per dollar
        
        # Sort by ROI
        sorted_indices = np.argsort(roi_values)
        roi_strategies_sorted = [roi_strategies[i] for i in sorted_indices]
        roi_values_sorted = [roi_values[i] for i in sorted_indices]
        colors_sorted = [get_strategy_color(s) for s in roi_strategies_sorted]
        
        bars = ax.barh(roi_strategies_sorted, roi_values_sorted, color=colors_sorted, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Accuracy Points Gained per Dollar Spent', fontsize=12, fontweight='bold')
        ax.set_title(f'{DATASET_NAME}: Return on Investment (ROI)\n(Higher is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (strat, roi_val) in enumerate(zip(roi_strategies_sorted, roi_values_sorted)):
            ax.text(roi_val, i, f'  {roi_val:.1f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/roi_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"\n🚀 Best ROI: {roi_strategies_sorted[-1]} with {roi_values_sorted[-1]:.1f} accuracy points per dollar")
    
    # Plot: Token Cost Breakdown (Input vs Output)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Input vs Output tokens
    ax1 = axes[0]
    strategies_list = [m.strategy_name for m in all_metrics]
    input_tokens = [m.total_input_tokens / 1000 for m in all_metrics]  # thousands
    output_tokens = [m.total_output_tokens / 1000 for m in all_metrics]
    
    x = np.arange(len(strategies_list))
    width = 0.35
    
    ax1.bar(x - width/2, input_tokens, width, label='Input Tokens', color='#42A5F5', alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, output_tokens, width, label='Output Tokens', color='#EF5350', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tokens (thousands)', fontsize=11, fontweight='bold')
    ax1.set_title('Token Usage Breakdown', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', '\n') for s in strategies_list], fontsize=8, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Input vs Output costs
    ax2 = axes[1]
    input_costs = [m.input_cost * 1000 for m in all_metrics]  # cents
    output_costs = [m.output_cost * 1000 for m in all_metrics]
    
    ax2.bar(x - width/2, input_costs, width, label='Input Cost', color='#42A5F5', alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, output_costs, width, label='Output Cost', color='#EF5350', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cost (cents)', fontsize=11, fontweight='bold')
    ax2.set_title('Cost Breakdown', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('_', '\n') for s in strategies_list], fontsize=8, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/token_cost_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Multi-Axis Pareto Frontiers
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Define the four efficiency dimensions
    pareto_dimensions = [
        ('avg_samples_per_question', 'Sample-Based Compute', 'Avg Samples per Question', axes[0, 0]),
        ('token_multiplier', 'Token-Based Compute', 'Token Multiplier', axes[0, 1]),
        ('total_cost', 'Cost-Based', 'Total Cost ($)', axes[1, 0]),
        ('cost_per_correct', 'Cost per Correct Answer', 'Cost per Correct ($)', axes[1, 1]),
    ]
    
    for metric_attr, title, xlabel, ax in pareto_dimensions:
        # Extract data
        x_vals = []
        y_vals = []
        names = []
        colors_list = []
        
        for m in all_metrics:
            x = getattr(m, metric_attr)
            if x is not None and not np.isinf(x):
                x_vals.append(x)
                y_vals.append(m.overall_accuracy * 100)
                names.append(m.strategy_name)
                colors_list.append(get_strategy_color(m.strategy_name))
        
        # Plot points
        for x, y, name, color in zip(x_vals, y_vals, names, colors_list):
            ax.scatter(x, y, s=200, alpha=0.8, color=color, edgecolor='black', linewidth=2)
            ax.annotate(
                name,
                (x, y),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.7)
            )
        
        # Find and draw Pareto frontier
        # Sort by x (cost/compute metric)
        sorted_indices = np.argsort(x_vals)
        pareto_x = []
        pareto_y = []
        max_acc = 0
        
        for idx in sorted_indices:
            if y_vals[idx] > max_acc:
                pareto_x.append(x_vals[idx])
                pareto_y.append(y_vals[idx])
                max_acc = y_vals[idx]
        
        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, 'r--', linewidth=2.5, alpha=0.6, label='Pareto Frontier', zorder=10)
            # Highlight Pareto optimal points
            ax.scatter(pareto_x, pareto_y, s=300, facecolors='none', edgecolors='red', linewidth=3, zorder=11)
        
        # Baseline reference
        baseline_metric = all_metrics[0]  # all_greedy
        baseline_x = getattr(baseline_metric, metric_attr)
        baseline_y = baseline_metric.overall_accuracy * 100
        ax.axhline(baseline_y, color='gray', linestyle=':', alpha=0.5, label='Baseline Accuracy')
        if metric_attr != 'cost_per_correct':  # cost_per_correct doesn't have a baseline line
            ax.axvline(baseline_x if baseline_x else 0, color='gray', linestyle=':', alpha=0.5, label='Baseline Cost')
        
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nPareto Efficiency', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/pareto_frontiers_multi_axis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Efficiency Comparison Across All Metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    
    strategies = [m.strategy_name for m in all_metrics]
    x = np.arange(len(strategies))
    width = 0.25
    
    sample_eff = [m.efficiency_score() for m in all_metrics]
    token_eff = [m.token_efficiency() for m in all_metrics]
    cost_eff = [m.cost_efficiency() for m in all_metrics]
    
    # Normalize to make them comparable on same scale
    def normalize(vals):
        max_val = max([abs(v) for v in vals if not np.isinf(v)] + [1])
        return [v/max_val if not np.isinf(v) else 0 for v in vals]
    
    sample_eff_norm = normalize(sample_eff)
    token_eff_norm = normalize(token_eff)
    cost_eff_norm = normalize(cost_eff)
    
    bars1 = ax.bar(x - width, sample_eff_norm, width, label='Sample Efficiency', color='#42A5F5', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, token_eff_norm, width, label='Token Efficiency', color='#66BB6A', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, cost_eff_norm, width, label='Cost Efficiency', color='#FFA726', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Efficiency Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Multi-Dimensional Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=9, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/efficiency_comparison_multi_metric.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print Pareto-optimal strategies for each dimension
    print("\n" + "="*80)
    print("🏆 PARETO-OPTIMAL STRATEGIES BY DIMENSION")
    print("="*80)
    
    for metric_attr, title, _, _ in pareto_dimensions:
        print(f"\n{title}:")
        
        # Find Pareto frontier
        data = [(m, getattr(m, metric_attr), m.overall_accuracy) for m in all_metrics 
                if getattr(m, metric_attr) is not None and not np.isinf(getattr(m, metric_attr))]
        data.sort(key=lambda x: x[1])  # sort by cost metric
        
        pareto_strategies = []
        max_acc = 0
        for m, cost, acc in data:
            if acc > max_acc:
                pareto_strategies.append((m.strategy_name, cost, acc))
                max_acc = acc
        
        for i, (name, cost, acc) in enumerate(pareto_strategies, 1):
            print(f"  {i}. {name:<35} Cost: {cost:>10.6f}  Acc: {acc:>6.2%}")
    
    print("\n" + "="*80)
    
    # Plot: Pass@k Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Pass@k curves for each strategy
    ax1 = axes[0]
    for m in all_metrics:
        if m.pass_at_k:
            k_values = sorted(m.pass_at_k.keys())
            pass_rates = [m.pass_at_k[k] for k in k_values]
            color = get_strategy_color(m.strategy_name)
            ax1.plot(k_values, pass_rates, marker='o', label=m.strategy_name, 
                    color=color, linewidth=2, markersize=6, alpha=0.8)
    
    ax1.set_xlabel('Number of Samples (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pass@k Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Pass@k: Success Rate with k Samples', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Right plot: Average position of first correct answer
    ax2 = axes[1]
    strategies_with_sc = [m for m in all_metrics if m.avg_first_correct_sample and m.avg_first_correct_sample != float('inf')]
    if strategies_with_sc:
        strategy_names = [m.strategy_name for m in strategies_with_sc]
        first_correct_pos = [m.avg_first_correct_sample for m in strategies_with_sc]
        colors_list = [get_strategy_color(s) for s in strategy_names]
        
        bars = ax2.barh(strategy_names, first_correct_pos, color=colors_list, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Average Position of First Correct Sample', fontsize=12, fontweight='bold')
        ax2.set_title('Efficiency: How Quickly We Find Correct Answers', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (name, pos) in enumerate(zip(strategy_names, first_correct_pos)):
            ax2.text(pos, i, f'  {pos:.2f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/pass_at_k_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot: Theoretical Best-Case Accuracy vs Total Samples (Pareto Frontier)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Collect data points
    data_points = []
    for m in all_metrics:
        if m.oracle_best_case_accuracy is not None and m.avg_samples_per_question:
            total_samples = m.avg_samples_per_question * len(df)
            oracle_acc = m.oracle_best_case_accuracy * 100  # Convert to percentage
            data_points.append((m, total_samples, oracle_acc))
    
    # Plot all strategies
    for m, total_samples, oracle_acc in data_points:
        color = get_strategy_color(m.strategy_name)
        ax.scatter(
            total_samples,
            oracle_acc,
            s=300,
            alpha=0.8,
            color=color,
            edgecolor='black',
            linewidth=2
        )
        ax.annotate(
            m.strategy_name,
            (total_samples, oracle_acc),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7)
        )
    
    # Find and draw Pareto frontier
    # Sort by total samples
    sorted_points = sorted(data_points, key=lambda x: x[1])
    pareto_points = []
    max_acc = 0
    
    for m, samples, acc in sorted_points:
        if acc > max_acc:
            pareto_points.append((samples, acc))
            max_acc = acc
    
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2.5, alpha=0.6, label='Pareto Frontier', zorder=10)
        # Highlight Pareto optimal points
        ax.scatter(pareto_x, pareto_y, s=400, facecolors='none', edgecolors='red', linewidth=3, zorder=11)
    
    ax.set_xlabel('Total Samples Generated', fontsize=12, fontweight='bold')
    ax.set_ylabel('Theoretical Best-Case Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{DATASET_NAME}: Theoretical Best-Case Accuracy vs Compute\n(Pareto frontier shows optimal accuracy-compute trade-offs)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/theoretical_best_case_accuracy_vs_samples_pareto.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Deep Dive: Probe-based Routing Analysis
    probe_metrics = [m for m in all_metrics if 'probe' in m.strategy_name.lower()]
    best_probe_metric = max(probe_metrics, key=lambda m: m.efficiency_score())
    
    print(f"\nBest probe-based strategy: {best_probe_metric.strategy_name}")
    print(best_probe_metric)
    
    # Analyze improvement distribution
    best_probe_results = all_results[best_probe_metric.strategy_name]
    
    improvements = []
    for idx, result in enumerate(best_probe_results):
        baseline_correct = df.iloc[idx]["baseline_is_correct"]
        improvement = result.is_correct - baseline_correct
        improvements.append(improvement)
    
    df["improvement"] = improvements
    df["used_sc"] = [r.used_sc for r in best_probe_results]
    
    print("\nImprovement distribution:")
    print(pd.Series(improvements).value_counts().sort_index())
    
    # Plot: improvement by predicted difficulty
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_map = {-1: 'red', 0: 'gray', 1: 'green'}
    labels_map = {-1: 'Worse with SC', 0: 'No change', 1: 'Better with SC'}
    
    for improvement_val in [-1, 0, 1]:
        subset = df[df["improvement"] == improvement_val]
        if len(subset) > 0:
            ax.scatter(
                subset["predicted_difficulty_sigmoid"],
                subset["baseline_is_correct"],
                c=colors_map[improvement_val],
                label=labels_map[improvement_val],
                alpha=0.6,
                s=50
            )
    
    ax.set_xlabel('Predicted Difficulty (sigmoid)', fontsize=12)
    ax.set_ylabel('Baseline Correctness', fontsize=12)
    ax.set_title(f'SC Impact by Predicted Difficulty ({best_probe_metric.strategy_name})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/sc_impact_by_difficulty.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Confusion matrix for router
    actually_benefits = (df["baseline_is_correct"] == 0) & (df["improvement"] == 1)
    router_sent_to_sc = df["used_sc"]
    
    # Check if we have both classes before creating confusion matrix
    unique_true = actually_benefits.unique()
    unique_pred = router_sent_to_sc.unique()
    
    if len(unique_true) > 1 or len(unique_pred) > 1:
        # Normal case: multiple classes exist
        cm = confusion_matrix(actually_benefits, router_sent_to_sc, labels=[False, True])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Benefit', 'Benefits'])
        disp.plot(cmap='Blues')
        plt.title(f'Router Confusion Matrix: {best_probe_metric.strategy_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/router_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        # Edge case: only one class (all samples identical)
        print(f"\n⚠️  Skipping confusion matrix - only one class present in data")
        print(f"   All samples: {'Benefits' if unique_true[0] else 'No Benefit'}")
        print(f"   Router decision: {'SC' if unique_pred[0] else 'Greedy'}" if len(unique_pred) > 0 else "   No routing decisions")
    
    # Save Results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    comparison_df.to_csv(f"{RESULTS_DIR}/routing_comparison.csv", index=False)
    
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
    
    # Final Summary
    print("\n" * 5)
    print("█" * 80)
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  🎉 EXPERIMENT COMPLETE! 🎉  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("█" * 80)
    print()
    
    print("┌" + "─" * 78 + "┐")
    print("│" + " QUICK REFERENCE ".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    print(f"│  Dataset: {DATASET_NAME:<65}│")
    print(f"│  Model: {MODEL_ALIAS:<67}│")
    print(f"│  Baseline Accuracy: {BASELINE_ACCURACY:>6.2%}{' ' * 51}│")
    print(f"│  Total Experiments: {len(all_metrics):<57}│")
    print("└" + "─" * 78 + "┘")
    print()
    
    print("🏆 TOP 3 STRATEGIES BY EFFICIENCY (TOKEN-WEIGHTED):")
    print("═" * 80)
    top3 = comparison_df.head(3)
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{i}. {row['Strategy']}")
        print(f"   Accuracy: {row['Accuracy']:.2%} | Token Eff: {row['Efficiency (token)']:.4f} | Sample Eff: {row['Efficiency (sample)']:.4f} | SC: {row['Fraction SC']:.1%}")
    
    print("\n\n")
    print("🎯 TOP 3 STRATEGIES BY ACCURACY:")
    print("═" * 80)
    top3_acc = comparison_df.sort_values("Accuracy", ascending=False).head(3)
    for i, (idx, row) in enumerate(top3_acc.iterrows(), 1):
        print(f"\n{i}. {row['Strategy']}")
        print(f"   Accuracy: {row['Accuracy']:.2%} | Gain: {row['Accuracy Gain']:+.2%} | Cost: {row['Avg Samples']:.2f}x")
    
    print("\n\n")
    print("🔮 THEORETICAL BEST-CASE ORACLE ANALYSIS SUMMARY:")
    print("═" * 80)
    top3_oracle = comparison_df.sort_values("Best-Case Oracle Gain", ascending=False).head(3)
    for i, (idx, row) in enumerate(top3_oracle.iterrows(), 1):
        print(f"\n{i}. {row['Strategy']}")
        print(f"   Current: {row['Accuracy']:.2%} | Best-Case: {row['Best-Case Oracle Acc']:.2%} | Potential: {row['Best-Case Oracle Gain']:+.2%}")
        print(f"   Utilization: {row['Sample Utilization']:.1%} (wasting {100 - row['Sample Utilization']*100:.1f}% of correct samples)")
    
    avg_utilization = comparison_df['Sample Utilization'].mean()
    avg_oracle_gain = comparison_df['Best-Case Oracle Gain'].mean()
    print(f"\n📊 Overall Statistics:")
    print(f"   → Average Sample Utilization: {avg_utilization:.1%}")
    print(f"   → Average Best-Case Oracle Gain: {avg_oracle_gain:+.2%}")
    print(f"   → Interpretation: On average, majority voting captures {avg_utilization:.1%} of available correct samples")
    
    print("\n\n")
    print("=" * 80)
    print(f"✅ Full results saved to: {RESULTS_DIR}/")
    print("=" * 80)
    print("\n" * 3)
    
    # Send notification after completing dataset
    try:
        best_strategy = comparison_df.iloc[0]
        notification_msg = (
            f"✅ {DATASET_NAME} routing analysis complete!\n"
            f"📊 Model: {MODEL_ALIAS}\n"
            f"🎯 Baseline Acc: {BASELINE_ACCURACY:.2%}\n"
            f"🏆 Best: {best_strategy['Strategy']}\n"
            f"   → Acc: {best_strategy['Accuracy']:.2%} ({best_strategy['Accuracy Gain']:+.2%})\n"
            f"   → Token Eff: {best_strategy['Efficiency (token)']:.4f}\n"
            f"📂 {RESULTS_DIR}"
        )
        send_notification(
            f"SC Router - {DATASET_NAME} Complete",
            notification_msg
        )
    except Exception as e:
        print(f"Note: Notification not sent - {e}")

print("\n" + "🎊" * 40)
print("ALL DATASETS PROCESSED SUCCESSFULLY!")
print("🎊" * 40 + "\n")

# Send final summary notification
try:
    final_msg = (
        f"🎉 All SC router experiments complete!\n"
        f"📊 Model: {CONFIG['model_alias']}\n"
        f"📈 Datasets processed: {len(CONFIG['datasets'])}\n"
        f"✓ {', '.join(CONFIG['datasets'])}"
    )
    send_notification(
        "SC Router - All Experiments Complete",
        final_msg
    )
except:
    pass
