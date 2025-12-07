#!/usr/bin/env python3
"""
Visualization script for checkpoint probe performance analysis.
Creates graphs showing performance trends across training steps with layer/position annotations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def load_checkpoint_results(summary_file: str) -> Dict:
    """Load the checkpoint probe summary results."""
    with open(summary_file, 'r') as f:
        return json.load(f)

def extract_model_name(results: List[Dict]) -> str:
    """Extract model name from the results."""
    for result in results:
        if "model_path" in result:
            model_path = result["model_path"]
            # Extract model name from path
            if "Llama" in model_path:
                if "3.2-3B" in model_path:
                    return "Llama-3.2-3B-Instruct"
                elif "3.1-8B" in model_path:
                    return "Llama-3.1-8B-Instruct"
                else:
                    return "Llama"
            elif "Qwen" in model_path:
                if "2.5-Math-1.5B" in model_path:
                    return "Qwen2.5-Math-1.5B"
                elif "2.5-Math-7B" in model_path:
                    return "Qwen2.5-Math-7B"
                else:
                    return "Qwen"
            else:
                # Try to extract from directory name
                path_parts = model_path.split('/')
                for part in path_parts:
                    if any(model in part for model in ["Llama", "Qwen", "GPT", "Claude"]):
                        # Clean up the name
                        clean_name = part.split('_')[0] if '_' in part else part
                        return clean_name
    
    # Fallback to extracting from file path
    return "Unknown Model"

def extract_performance_data(results: List[Dict]) -> Dict[str, Dict]:
    """Extract performance data organized by dataset."""
    datasets = {}
    
    for result in results:
        if "error" in result:
            continue
            
        step = result.get("step", "unknown")
        if result.get("is_baseline", False):
            step = "Baseline"
            
        for dataset_name, dataset_result in result.get("datasets", {}).items():
            if "error" in dataset_result:
                continue
                
            if dataset_name not in datasets:
                datasets[dataset_name] = {
                    "steps": [],
                    "performances": [],
                    "layers": [],
                    "positions": [],
                    "position_indices": []
                }
            
            datasets[dataset_name]["steps"].append(step)
            datasets[dataset_name]["performances"].append(dataset_result.get("best_performance", 0))
            datasets[dataset_name]["layers"].append(dataset_result.get("best_layer", 0))
            datasets[dataset_name]["positions"].append(dataset_result.get("best_position", 0))
            datasets[dataset_name]["position_indices"].append(dataset_result.get("best_position_index", 0))
    
    return datasets

def create_performance_graph(dataset_name: str, data: Dict, output_dir: Path, model_name: str, show_annotations: bool = True):
    """Create a performance graph for a single dataset."""
    steps = data["steps"]
    performances = data["performances"]
    layers = data["layers"]
    positions = data["positions"]
    
    # Convert steps to numeric for plotting (handle baseline)
    numeric_steps = []
    step_labels = []
    for step in steps:
        if step == "Baseline":
            numeric_steps.append(0)
            step_labels.append("Baseline")
        else:
            numeric_steps.append(int(step))
            step_labels.append(f"Step {step}")
    
    # Sort by step number
    sorted_data = sorted(zip(numeric_steps, performances, layers, positions, step_labels))
    numeric_steps, performances, layers, positions, step_labels = zip(*sorted_data)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot the performance line
    plt.plot(numeric_steps, performances, 'o-', linewidth=2.5, markersize=8, color='#2E86C1', alpha=0.8)
    
    # Add annotations for layer and position if requested
    if show_annotations:
        for i, (step, perf, layer, pos) in enumerate(zip(numeric_steps, performances, layers, positions)):
            # Offset annotation position slightly to avoid overlap
            offset_y = 0.002 if i % 2 == 0 else -0.008
            plt.annotate(f'L{layer}\nP{pos}', 
                        xy=(step, perf), 
                        xytext=(step, perf + offset_y),
                        ha='center', va='bottom' if offset_y > 0 else 'top',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.6))
    
    plt.title(f'{model_name}: Probe Performance Evolution - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Probe Performance (Accuracy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks and labels
    plt.xticks(numeric_steps, [f"{step}" if step != 0 else "Base" for step in numeric_steps], rotation=45)
    
    # Add performance values on points
    for step, perf in zip(numeric_steps, performances):
        plt.text(step, perf + 0.01, f'{perf:.3f}', ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f"{dataset_name}_performance_evolution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved performance graph: {output_file}")
    
    return output_file

def create_layer_position_heatmap(dataset_name: str, data: Dict, output_dir: Path, model_name: str):
    """Create a heatmap showing which layers and positions are optimal at each step."""
    steps = data["steps"]
    layers = data["layers"]
    positions = data["positions"]
    performances = data["performances"]
    
    # Convert steps to numeric
    numeric_steps = [0 if step == "Baseline" else int(step) for step in steps]
    
    # Sort by step
    sorted_data = sorted(zip(numeric_steps, layers, positions, performances, steps))
    numeric_steps, layers, positions, performances, step_labels = zip(*sorted_data)
    
    # Create subplot with 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Layer evolution
    ax1.plot(numeric_steps, layers, 'o-', linewidth=2, markersize=8, color='#E74C3C')
    ax1.set_title(f'{model_name}: Best Layer Evolution - {dataset_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Best Layer')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(numeric_steps)
    ax1.set_xticklabels([f"{step}" if step != 0 else "Base" for step in numeric_steps], rotation=45)
    
    # Add performance annotations
    for step, layer, perf in zip(numeric_steps, layers, performances):
        ax1.text(step, layer + 0.3, f'{perf:.3f}', ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Position evolution
    ax2.plot(numeric_steps, positions, 'o-', linewidth=2, markersize=8, color='#8E44AD')
    ax2.set_title(f'{model_name}: Best Position Evolution - {dataset_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Best Position')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(numeric_steps)
    ax2.set_xticklabels([f"{step}" if step != 0 else "Base" for step in numeric_steps], rotation=45)
    
    # Add performance annotations
    for step, pos, perf in zip(numeric_steps, positions, performances):
        ax2.text(step, pos + 0.1, f'{perf:.3f}', ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f"{dataset_name}_layer_position_evolution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved layer/position evolution: {output_file}")
    
    return output_file

def create_combined_comparison(datasets: Dict, output_dir: Path, model_name: str):
    """Create a combined comparison graph for all datasets."""
    plt.figure(figsize=(15, 10))
    
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD']
    
    for i, (dataset_name, data) in enumerate(datasets.items()):
        steps = data["steps"]
        performances = data["performances"]
        layers = data["layers"]
        positions = data["positions"]
        
        # Convert steps to numeric
        numeric_steps = [0 if step == "Baseline" else int(step) for step in steps]
        
        # Sort by step
        sorted_data = sorted(zip(numeric_steps, performances, layers, positions))
        numeric_steps, performances, layers, positions = zip(*sorted_data)
        
        color = colors[i % len(colors)]
        
        # Plot performance line
        plt.plot(numeric_steps, performances, 'o-', linewidth=2.5, markersize=8, 
                label=dataset_name, color=color, alpha=0.8)
        
        # Add annotations for the latest step
        latest_idx = -1
        step, perf, layer, pos = numeric_steps[latest_idx], performances[latest_idx], layers[latest_idx], positions[latest_idx]
        plt.annotate(f'{dataset_name}\nL{layer}, P{pos}\n{perf:.3f}', 
                    xy=(step, perf), 
                    xytext=(step + 0.5, perf),
                    ha='left', va='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color, alpha=0.6))
    
    plt.title(f'{model_name}: Probe Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Probe Performance (Accuracy)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis
    all_steps = set()
    for data in datasets.values():
        numeric_steps = [0 if step == "Baseline" else int(step) for step in data["steps"]]
        all_steps.update(numeric_steps)
    
    sorted_steps = sorted(all_steps)
    plt.xticks(sorted_steps, [f"{step}" if step != 0 else "Base" for step in sorted_steps], rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "combined_performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved combined comparison: {output_file}")
    
    return output_file

def create_performance_summary_table(datasets: Dict, output_dir: Path, model_name: str):
    """Create a summary table of performance metrics."""
    summary_data = []
    
    for dataset_name, data in datasets.items():
        steps = data["steps"]
        performances = data["performances"]
        layers = data["layers"]
        positions = data["positions"]
        
        # Convert steps to numeric for sorting
        numeric_steps = [0 if step == "Baseline" else int(step) for step in steps]
        sorted_data = sorted(zip(numeric_steps, performances, layers, positions, steps))
        
        max_perf = max(performances)
        min_perf = min(performances)
        final_perf = sorted_data[-1][1]  # Last step performance
        
        # Find best step
        best_idx = performances.index(max_perf)
        best_step = steps[best_idx]
        best_layer = layers[best_idx]
        best_pos = positions[best_idx]
        
        summary_data.append({
            'Dataset': dataset_name,
            'Best Performance': f"{max_perf:.4f}",
            'Best Step': best_step,
            'Best Layer': best_layer,
            'Best Position': best_pos,
            'Final Performance': f"{final_perf:.4f}",
            'Performance Range': f"{min_perf:.4f} - {max_perf:.4f}",
            'Performance Change': f"{final_perf - sorted_data[0][1]:+.4f}" if len(sorted_data) > 1 else "N/A"
        })
    
    # Create table
    fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = list(summary_data[0].keys())
    table_data = [[row[col] for col in headers] for row in summary_data]
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
    
    plt.title(f'{model_name}: Checkpoint Probe Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    # Save the table
    output_file = output_dir / "performance_summary_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved summary table: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Visualize checkpoint probe performance evolution")
    parser.add_argument('--summary_file', type=str, required=True,
                       help='Path to checkpoint_probe_summary.json file')
    parser.add_argument('--output_dir', type=str, required=False,
                       help='Output directory for graphs (default: same as summary file directory)')
    parser.add_argument('--no_annotations', action='store_true',
                       help='Disable layer/position annotations on main graphs')
    
    args = parser.parse_args()
    
    # Setup paths
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = summary_path.parent / "visualizations"
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Loading results from: {summary_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load and process data
    results = load_checkpoint_results(str(summary_path))
    model_name = extract_model_name(results)
    datasets = extract_performance_data(results)
    
    if not datasets:
        print("❌ No valid dataset results found in summary file")
        return
    
    print(f"🤖 Model: {model_name}")
    print(f"📈 Found {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Generate visualizations
    generated_files = []
    
    # Individual dataset graphs
    for dataset_name, data in datasets.items():
        print(f"\n📊 Creating graphs for {dataset_name}...")
        
        # Performance evolution graph
        perf_file = create_performance_graph(
            dataset_name, data, output_dir, model_name,
            show_annotations=not args.no_annotations
        )
        generated_files.append(perf_file)
        
        # Layer and position evolution
        layer_pos_file = create_layer_position_heatmap(dataset_name, data, output_dir, model_name)
        generated_files.append(layer_pos_file)
    
    # Combined comparison
    if len(datasets) > 1:
        print(f"\n📊 Creating combined comparison...")
        combined_file = create_combined_comparison(datasets, output_dir, model_name)
        generated_files.append(combined_file)
    
    # Summary table
    print(f"\n📊 Creating summary table...")
    table_file = create_performance_summary_table(datasets, output_dir, model_name)
    generated_files.append(table_file)
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Generated {len(generated_files)} files in: {output_dir}")
    for file in generated_files:
        print(f"   📊 {file.name}")

if __name__ == "__main__":
    main()
