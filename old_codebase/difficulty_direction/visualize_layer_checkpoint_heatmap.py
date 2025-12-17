#!/usr/bin/env python3
"""
Layer x Checkpoint Heatmap Visualization

This script creates two heatmaps from probe training results:
1. Best probe performance per layer/checkpoint (across all positions)
2. Performance at a specific position per layer/checkpoint

Usage:
    python visualize_layer_checkpoint_heatmap.py [--results_dir probe_checkpoint_results] [--position -1]
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional

def load_probe_results(results_dir: str) -> Dict:
    """Load all probe results from the results directory."""
    results = {}
    
    # First, try to load from the main summary file
    summary_file = Path(results_dir) / "checkpoint_probe_summary.json"
    if summary_file.exists():
        print(f"Loading from summary file: {summary_file}")
        with open(summary_file, 'r') as f:
            data = json.load(f)
            return data
    
    # Fallback: look for step directories with individual summary files
    for checkpoint_dir in Path(results_dir).glob("step_*"):
        if not checkpoint_dir.is_dir():
            continue
            
        step_num = checkpoint_dir.name.replace("step_", "")
        try:
            step_num = int(step_num)
        except ValueError:
            print(f"Warning: Could not parse step number from {checkpoint_dir.name}")
            continue
        
        # Load the summary file for this checkpoint
        step_summary_file = checkpoint_dir / "checkpoint_probe_summary.json"
        if step_summary_file.exists():
            with open(step_summary_file, 'r') as f:
                checkpoint_data = json.load(f)
                results[step_num] = checkpoint_data
    
    return results

def extract_best_performance_matrix(results: Dict, dataset_name: str = "E2H-AMC") -> Tuple[np.ndarray, List[int], List[int]]:
    """Extract matrix of best performance per layer/checkpoint across all positions."""
    if not results:
        return np.array([]), [], []
    
    # Handle both dictionary (old format) and list (new format) structures
    if isinstance(results, list):
        # New format: list of step dictionaries
        steps = []
        step_data_dict = {}
        
        for step_entry in results:
            if 'step' in step_entry and 'datasets' in step_entry:
                step_num = step_entry['step']
                if dataset_name in step_entry['datasets']:
                    steps.append(step_num)
                    step_data_dict[step_num] = step_entry['datasets'][dataset_name]
        
        steps = sorted(steps)
        
        # Get all layers from the performance data
        all_layers = set()
        for step_data in step_data_dict.values():
            if 'full_performance' in step_data and 'layer_performance' in step_data['full_performance']:
                layer_performance = step_data['full_performance']['layer_performance']
                if 'test' in layer_performance:
                    num_layers = len(layer_performance['test'][0]); all_layers.update(range(num_layers))
        
        layers = sorted(all_layers)
        
        
        # Create matrix
        matrix = np.full((len(layers), len(steps)), np.nan)
        
        for step_idx, step in enumerate(steps):
            if step in step_data_dict:
                step_data = step_data_dict[step]
                if 'full_performance' in step_data and 'layer_performance' in step_data['full_performance']:
                    layer_performance = step_data['full_performance']['layer_performance']
                    if 'test' in layer_performance:
                        test_performance = layer_performance['test']
                        for layer_idx, layer in enumerate(layers):
                            # Get best performance across all positions for this layer
                            layer_values = []
                            for pos_idx in range(len(test_performance)):
                                if layer < len(test_performance[pos_idx]):
                                    layer_values.append(test_performance[pos_idx][layer])
                            
                            if layer_values and not all(np.isnan(layer_values)):
                                best_perf = np.nanmax(layer_values)
                                matrix[layer_idx, step_idx] = best_perf
        
        return matrix, steps, layers
    
    else:
        # Old format: dictionary with step keys
        steps = sorted(results.keys())
        all_layers = set()
        
        for step_data in results.values():
            if 'layer_summaries' in step_data:
                all_layers.update(step_data['layer_summaries'].keys())
        
        layers = sorted([int(l) for l in all_layers])
        
        
        # Create matrix
        matrix = np.full((len(layers), len(steps)), np.nan)
        
        for step_idx, step in enumerate(steps):
            step_data = results[step]
            if 'layer_summaries' not in step_data:
                continue
                
            for layer_idx, layer in enumerate(layers):
                layer_str = str(layer)
                if layer_str in step_data['layer_summaries']:
                    layer_data = step_data['layer_summaries'][layer_str]
                    # Get the best performance across all positions for this layer
                    if 'best_performance' in layer_data:
                        matrix[layer_idx, step_idx] = layer_data['best_performance']
                    elif 'positions' in layer_data:
                        # Find best across all positions manually
                        best_perf = -1
                        for pos_data in layer_data['positions'].values():
                            if 'test_performance' in pos_data:
                                best_perf = max(best_perf, pos_data['test_performance'])
                        if best_perf > -1:
                            matrix[layer_idx, step_idx] = best_perf
        
        return matrix, steps, layers

def extract_position_performance_matrix(results: Dict, target_position: int = -1, dataset_name: str = "E2H-AMC") -> Tuple[np.ndarray, List[int], List[int]]:
    """Extract matrix of performance at specific position per layer/checkpoint."""
    if not results:
        return np.array([]), [], []
    
    # Handle both dictionary (old format) and list (new format) structures
    if isinstance(results, list):
        # New format: list of step dictionaries
        steps = []
        step_data_dict = {}
        
        for step_entry in results:
            if 'step' in step_entry and 'datasets' in step_entry:
                step_num = step_entry['step']
                if dataset_name in step_entry['datasets']:
                    steps.append(step_num)
                    step_data_dict[step_num] = step_entry['datasets'][dataset_name]
        
        steps = sorted(steps)
        
        # Get all layers from the performance data
        all_layers = set()
        for step_data in step_data_dict.values():
            if 'full_performance' in step_data and 'layer_performance' in step_data['full_performance']:
                layer_performance = step_data['full_performance']['layer_performance']
                if 'test' in layer_performance:
                    num_layers = len(layer_performance['test'][0]); all_layers.update(range(num_layers))
        
        layers = sorted(all_layers)
        
        
        # Create matrix
        matrix = np.full((len(layers), len(steps)), np.nan)
        
        for step_idx, step in enumerate(steps):
            if step in step_data_dict:
                step_data = step_data_dict[step]
                if 'full_performance' in step_data and 'layer_performance' in step_data['full_performance']:
                    layer_performance = step_data['full_performance']['layer_performance']
                    if 'test' in layer_performance:
                        test_performance = layer_performance['test']
                        for layer_idx, layer in enumerate(layers):
                            # Convert target_position to index
                            if target_position == -1:
                                position_idx = len(test_performance) - 1
                            elif target_position < 0:
                                position_idx = len(test_performance) + target_position
                            else:
                                position_idx = target_position
                            
                            if 0 <= position_idx < len(test_performance):
                                if layer < len(test_performance[position_idx]):
                                    value = test_performance[position_idx][layer]
                                    if not np.isnan(value):
                                        matrix[layer_idx, step_idx] = value
        
        return matrix, steps, layers
    
    else:
        # Old format: dictionary with step keys
        steps = sorted(results.keys())
        all_layers = set()
        
        for step_data in results.values():
            if 'layer_summaries' in step_data:
                all_layers.update(step_data['layer_summaries'].keys())
        
        layers = sorted([int(l) for l in all_layers])
        
        
        # Create matrix
        matrix = np.full((len(layers), len(steps)), np.nan)
        
        for step_idx, step in enumerate(steps):
            step_data = results[step]
            if 'layer_summaries' not in step_data:
                continue
                
            for layer_idx, layer in enumerate(layers):
                layer_str = str(layer)
                if layer_str in step_data['layer_summaries']:
                    layer_data = step_data['layer_summaries'][layer_str]
                    if 'positions' in layer_data:
                        pos_str = str(target_position)
                        if pos_str in layer_data['positions']:
                            pos_data = layer_data['positions'][pos_str]
                            if 'test_performance' in pos_data:
                                matrix[layer_idx, step_idx] = pos_data['test_performance']
        
        return matrix, steps, layers

def create_heatmap(matrix: np.ndarray, steps: List[int], layers: List[int], 
                   title: str, filename: str, task_name: str = "Difficulty") -> None:
    """Create and save a heatmap visualization."""
    if matrix.size == 0:
        print(f"Warning: No data available for {title}")
        return
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(matrix, 
                xticklabels=steps,
                yticklabels=[f"L{l}" for l in layers],
                annot=False,
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Spearman ρ'},
                vmin=0,
                vmax=1)
    
    plt.title(f"{task_name} — {title}")
    plt.xlabel("Checkpoint step (index)")
    plt.ylabel("Transformer layer (bottom→top)")
    
    # Invert y-axis so layer 0 is at bottom
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap: {filename}")

def detect_model_name(results_dir: str) -> str:
    """Try to detect the model name from the results directory structure."""
    results_path = Path(results_dir)
    
    # Check if we're in a model-specific subdirectory
    if "Llama" in str(results_path):
        return "Llama-3.2-3B"
    elif "Qwen" in str(results_path):
        return "Qwen2.5-Math-1.5B"
    
    # Check parent directories
    parent_dirs = str(results_path.resolve())
    if "Llama" in parent_dirs:
        return "Llama-3.2-3B"
    elif "Qwen" in parent_dirs:
        return "Qwen2.5-Math-1.5B"
    
    return "Model"

def main():
    parser = argparse.ArgumentParser(description="Generate layer x checkpoint heatmaps from probe results")
    parser.add_argument("--results_dir", type=str, default="runs/checkpoint_comparison",
                        help="Directory containing probe results (default: runs/checkpoint_comparison)")
    parser.add_argument("--position", type=int, default=-1,
                        help="Token position to analyze for position-specific heatmap (default: -1)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save visualizations (default: current directory)")
    parser.add_argument("--task_name", type=str, default="AMC (E2H-AMC)",
                        help="Task name for plot titles (default: AMC (E2H-AMC))")
    
    args = parser.parse_args()
    
    # Check if results directory exists
    results_path = Path(args.results_dir)
    if not results_path.is_absolute():
        # If relative path, resolve it from the script's parent directory
        script_dir = Path(__file__).parent.parent  # Go up from difficulty_direction to main dir
        results_path = script_dir / args.results_dir
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_path}' not found")
        return
    
    print(f"Loading probe results from: {results_path}")
    results = load_probe_results(str(results_path))
    
    if not results:
        print("Error: No probe results found")
        return
    
    print(f"Loaded results for {len(results)} checkpoints")
    
    # Detect model name
    model_name = detect_model_name(str(results_path))
    
    # Hardcode output directory to results_dir/visualizations
    output_dir = results_path / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Generate best performance heatmap
    print("Generating best performance heatmap...")
    best_matrix, steps, layers = extract_best_performance_matrix(results, dataset_name="E2H-AMC")
    print(f"DEBUG: Matrix shape: {best_matrix.shape}, Layers: {len(layers)}, Steps: {len(steps)}")
    print(f"DEBUG: Layer range: {layers[:5]}...{layers[-5:]}")
    print(f"DEBUG: Matrix stats - min: {np.nanmin(best_matrix):.4f}, max: {np.nanmax(best_matrix):.4f}, mean: {np.nanmean(best_matrix):.4f}")
    
    # Check specific layers
    print("DEBUG: Sample values for different layers:")
    for i in [0, 4, 10, 20, 27]:
        if i < len(layers):
            layer_data = best_matrix[i, :]
            valid_data = layer_data[~np.isnan(layer_data)]
            if len(valid_data) > 0:
                print(f"  Layer {layers[i]}: {len(valid_data)} valid values, range {valid_data.min():.4f}-{valid_data.max():.4f}")
            else:
                print(f"  Layer {layers[i]}: all NaN")
    if best_matrix.size > 0:
        best_filename = output_dir / f"layer_checkpoint_heatmap_best_{model_name.lower().replace('-', '_')}.png"
        create_heatmap(best_matrix, steps, layers, 
                      "Layer × Step Spearman ρ (Best Position per Layer)",
                      str(best_filename), args.task_name)
    
    # Generate position-specific heatmap
    print(f"Generating position {args.position} heatmap...")
    pos_matrix, steps, layers = extract_position_performance_matrix(results, args.position, dataset_name="E2H-AMC")
    if pos_matrix.size > 0:
        pos_filename = output_dir / f"layer_checkpoint_heatmap_pos{args.position}_{model_name.lower().replace('-', '_')}.png"
        create_heatmap(pos_matrix, steps, layers,
                      f"Layer × Step Spearman ρ (Position {args.position})",
                      str(pos_filename), args.task_name)
    
    # Print summary statistics
    if best_matrix.size > 0:
        valid_best = best_matrix[~np.isnan(best_matrix)]
        if len(valid_best) > 0:
            print(f"\nBest Performance Summary:")
            print(f"  Mean: {valid_best.mean():.3f}")
            print(f"  Max: {valid_best.max():.3f}")
            print(f"  Min: {valid_best.min():.3f}")
    
    if pos_matrix.size > 0:
        valid_pos = pos_matrix[~np.isnan(pos_matrix)]
        if len(valid_pos) > 0:
            print(f"\nPosition {args.position} Performance Summary:")
            print(f"  Mean: {valid_pos.mean():.3f}")
            print(f"  Max: {valid_pos.max():.3f}")
            print(f"  Min: {valid_pos.min():.3f}")

if __name__ == "__main__":
    main()
