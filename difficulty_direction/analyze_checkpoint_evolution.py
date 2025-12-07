#!/usr/bin/env python3
"""
Analyze checkpoint evolution by comparing probe performance and optimal layers/positions.

This script loads results from train_checkpoint_probes.py and provides comprehensive
analysis of how probe characteristics change during training.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

def load_checkpoint_results(results_file: str) -> List[Dict]:
    """Load checkpoint probe training results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_performance_trends(results: List[Dict]) -> pd.DataFrame:
    """
    Extract performance trends across checkpoints.
    
    Returns DataFrame with columns: step, dataset, performance, layer, position
    """
    rows = []
    
    for result in results:
        step = result.get("step", -1)
        is_baseline = result.get("is_baseline", False)
        step_label = "baseline" if is_baseline else step
        
        if "error" not in result:
            for dataset_name, dataset_result in result.get("datasets", {}).items():
                if "error" not in dataset_result:
                    rows.append({
                        "step": step,
                        "step_label": step_label,
                        "dataset": dataset_name,
                        "performance": dataset_result.get("best_performance", 0),
                        "layer": dataset_result.get("best_layer", -1),
                        "position": dataset_result.get("best_position", -1),
                        "is_baseline": is_baseline
                    })
    
    return pd.DataFrame(rows)


def plot_performance_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance trends across checkpoints."""
    plt.figure(figsize=(12, 8))
    
    datasets = df['dataset'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset].sort_values('step')
        
        # Separate baseline and checkpoint steps
        baseline_df = dataset_df[dataset_df['is_baseline']]
        checkpoint_df = dataset_df[~dataset_df['is_baseline']]
        
        # Plot baseline as a horizontal line
        if not baseline_df.empty:
            baseline_perf = baseline_df['performance'].iloc[0]
            plt.axhline(y=baseline_perf, color=colors[i], linestyle='--', alpha=0.7, 
                       label=f'{dataset} (baseline)')
        
        # Plot checkpoint progression
        if not checkpoint_df.empty:
            plt.plot(checkpoint_df['step'], checkpoint_df['performance'], 
                    marker='o', color=colors[i], linewidth=2, markersize=6,
                    label=f'{dataset} (checkpoints)')
    
    plt.xlabel('Training Step')
    plt.ylabel('Probe Performance')
    plt.title('Probe Performance Evolution Across Training Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_dir / "performance_trends.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Performance trends plot saved to: {plot_file}")


def plot_layer_position_evolution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot how optimal layers and positions change across checkpoints."""
    datasets = df['dataset'].unique()
    
    fig, axes = plt.subplots(2, len(datasets), figsize=(5*len(datasets), 10))
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset].sort_values('step')
        checkpoint_df = dataset_df[~dataset_df['is_baseline']]
        baseline_df = dataset_df[dataset_df['is_baseline']]
        
        if not checkpoint_df.empty:
            # Plot layer evolution
            axes[0, i].plot(checkpoint_df['step'], checkpoint_df['layer'], 
                           marker='o', linewidth=2, markersize=6, color='blue')
            if not baseline_df.empty:
                baseline_layer = baseline_df['layer'].iloc[0]
                axes[0, i].axhline(y=baseline_layer, color='red', linestyle='--', 
                                 alpha=0.7, label='Baseline')
            axes[0, i].set_title(f'{dataset} - Optimal Layer')
            axes[0, i].set_xlabel('Training Step')
            axes[0, i].set_ylabel('Layer Index')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # Plot position evolution
            axes[1, i].plot(checkpoint_df['step'], checkpoint_df['position'], 
                           marker='s', linewidth=2, markersize=6, color='green')
            if not baseline_df.empty:
                baseline_pos = baseline_df['position'].iloc[0]
                axes[1, i].axhline(y=baseline_pos, color='red', linestyle='--', 
                                 alpha=0.7, label='Baseline')
            axes[1, i].set_title(f'{dataset} - Optimal Position')
            axes[1, i].set_xlabel('Training Step')
            axes[1, i].set_ylabel('Position Index')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
    
    plt.tight_layout()
    plot_file = output_dir / "layer_position_evolution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Layer/position evolution plot saved to: {plot_file}")


def create_comparison_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create a detailed comparison table."""
    # Pivot table for better comparison
    comparison_dfs = []
    
    for metric in ['performance', 'layer', 'position']:
        pivot_df = df.pivot(index='step_label', columns='dataset', values=metric)
        pivot_df.columns = [f"{col}_{metric}" for col in pivot_df.columns]
        comparison_dfs.append(pivot_df)
    
    comparison_table = pd.concat(comparison_dfs, axis=1)
    
    # Reorder columns for better readability
    datasets = df['dataset'].unique()
    ordered_columns = []
    for dataset in datasets:
        ordered_columns.extend([f"{dataset}_performance", f"{dataset}_layer", f"{dataset}_position"])
    
    comparison_table = comparison_table[ordered_columns]
    
    # Save to CSV
    csv_file = output_dir / "checkpoint_comparison_table.csv"
    comparison_table.to_csv(csv_file)
    print(f"📄 Comparison table saved to: {csv_file}")
    
    return comparison_table


def analyze_performance_changes(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance changes and trends."""
    analysis = {}
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].sort_values('step')
        baseline_df = dataset_df[dataset_df['is_baseline']]
        checkpoint_df = dataset_df[~dataset_df['is_baseline']]
        
        if not baseline_df.empty and not checkpoint_df.empty:
            baseline_perf = baseline_df['performance'].iloc[0]
            final_perf = checkpoint_df['performance'].iloc[-1]
            
            # Calculate improvement/degradation
            improvement = final_perf - baseline_perf
            improvement_pct = (improvement / baseline_perf) * 100 if baseline_perf != 0 else 0
            
            # Check for consistent trends
            performances = checkpoint_df['performance'].values
            if len(performances) > 1:
                trend = "improving" if performances[-1] > performances[0] else "degrading"
                # Calculate correlation with step to see if trend is consistent
                steps = checkpoint_df['step'].values
                correlation = np.corrcoef(steps, performances)[0, 1] if len(steps) > 1 else 0
            else:
                trend = "single_point"
                correlation = 0
            
            analysis[dataset] = {
                "baseline_performance": baseline_perf,
                "final_performance": final_perf,
                "absolute_improvement": improvement,
                "percentage_improvement": improvement_pct,
                "trend": trend,
                "trend_correlation": correlation,
                "best_performance": checkpoint_df['performance'].max(),
                "best_step": checkpoint_df.loc[checkpoint_df['performance'].idxmax(), 'step'],
                "layer_stability": checkpoint_df['layer'].nunique() == 1,
                "position_stability": checkpoint_df['position'].nunique() == 1
            }
    
    return analysis


def generate_report(results: List[Dict], analysis: Dict[str, Any], output_dir: Path) -> None:
    """Generate a comprehensive text report."""
    report_file = output_dir / "analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("🎯 CHECKPOINT PROBE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary statistics
        f.write("📊 SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total checkpoints analyzed: {len([r for r in results if not r.get('is_baseline', False)])}\n")
        f.write(f"Datasets analyzed: {len(analysis)}\n")
        f.write(f"Baseline included: {'Yes' if any(r.get('is_baseline', False) for r in results) else 'No'}\n\n")
        
        # Per-dataset analysis
        for dataset, dataset_analysis in analysis.items():
            f.write(f"📈 DATASET: {dataset}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline performance: {dataset_analysis['baseline_performance']:.4f}\n")
            f.write(f"Final performance: {dataset_analysis['final_performance']:.4f}\n")
            f.write(f"Absolute change: {dataset_analysis['absolute_improvement']:+.4f}\n")
            f.write(f"Percentage change: {dataset_analysis['percentage_improvement']:+.2f}%\n")
            f.write(f"Best performance: {dataset_analysis['best_performance']:.4f} (Step {dataset_analysis['best_step']})\n")
            f.write(f"Overall trend: {dataset_analysis['trend']}\n")
            f.write(f"Trend consistency: {dataset_analysis['trend_correlation']:.3f}\n")
            f.write(f"Layer stability: {'Stable' if dataset_analysis['layer_stability'] else 'Changing'}\n")
            f.write(f"Position stability: {'Stable' if dataset_analysis['position_stability'] else 'Changing'}\n\n")
        
        # Key insights
        f.write("🔍 KEY INSIGHTS\n")
        f.write("-" * 15 + "\n")
        
        improving_datasets = [d for d, a in analysis.items() if a['percentage_improvement'] > 0]
        degrading_datasets = [d for d, a in analysis.items() if a['percentage_improvement'] < 0]
        
        if improving_datasets:
            f.write(f"✅ Improving datasets: {', '.join(improving_datasets)}\n")
        if degrading_datasets:
            f.write(f"❌ Degrading datasets: {', '.join(degrading_datasets)}\n")
        
        stable_layers = [d for d, a in analysis.items() if a['layer_stability']]
        stable_positions = [d for d, a in analysis.items() if a['position_stability']]
        
        if stable_layers:
            f.write(f"🔒 Stable optimal layers: {', '.join(stable_layers)}\n")
        if stable_positions:
            f.write(f"🔒 Stable optimal positions: {', '.join(stable_positions)}\n")
        
        # Recommendations
        f.write(f"\n💡 RECOMMENDATIONS\n")
        f.write("-" * 17 + "\n")
        
        best_overall = max(analysis.items(), key=lambda x: x[1]['best_performance'])
        f.write(f"🏆 Best performing setup: {best_overall[0]} at step {best_overall[1]['best_step']}\n")
        
        if any(a['percentage_improvement'] > 5 for a in analysis.values()):
            f.write("📈 Significant improvements detected - consider training longer\n")
        
        if any(not a['layer_stability'] or not a['position_stability'] for a in analysis.values()):
            f.write("⚠️  Optimal probe positions changing - model representations are evolving\n")
    
    print(f"📄 Detailed report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint probe evolution")
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to checkpoint_probe_summary.json from train_checkpoint_probes.py')
    parser.add_argument('--output_dir', type=str, required=False,
                       help='Output directory for analysis results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots (useful for headless environments)')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        return
    
    results = load_checkpoint_results(args.results_file)
    print(f"✅ Loaded results for {len(results)} checkpoints")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.results_file).parent / "analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Analysis output directory: {output_dir}")
    
    # Extract data
    df = extract_performance_trends(results)
    if df.empty:
        print("❌ No valid data found in results!")
        return
    
    print(f"📊 Extracted data for {len(df)} data points across {df['dataset'].nunique()} datasets")
    
    # Create comparison table
    comparison_table = create_comparison_table(df, output_dir)
    print("\n📋 Comparison Table (first few rows):")
    print(comparison_table.head())
    
    # Analyze performance changes
    analysis = analyze_performance_changes(df)
    
    # Generate plots (if not disabled)
    if not args.no_plots:
        try:
            plot_performance_trends(df, output_dir)
            plot_layer_position_evolution(df, output_dir)
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
            print("Continuing with text analysis...")
    
    # Generate report
    generate_report(results, analysis, output_dir)
    
    # Print summary to console
    print(f"\n🎯 ANALYSIS SUMMARY")
    print("=" * 30)
    for dataset, dataset_analysis in analysis.items():
        improvement = dataset_analysis['percentage_improvement']
        trend_icon = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        print(f"{trend_icon} {dataset}: {improvement:+.2f}% change from baseline")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
