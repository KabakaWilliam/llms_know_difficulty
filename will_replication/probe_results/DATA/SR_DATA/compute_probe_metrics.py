import json
import numpy as np
import os
import sys
from pathlib import Path
import torch
import numpy

# Add parent directory to path for imports
# We're in probe_results/DATA/SR_DATA, need to go up to will_replication, then up to llms_know_difficult
repo_root = Path.cwd().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

print(f"Added to path: {repo_root}")
print(f"Checking if thom_replication exists: {(repo_root / 'thom_replication').exists()}")

from thom_replication.utils.metrics import compute_metrics


def load_probe_results(filepath):
    """Load probe predictions and compute detailed metrics if needed."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if detailed metrics already exist
    if 'test_learnability_ys_mean' in data:
        # Metrics already computed, use them directly
        return data
    
    # Need to compute metrics
    test_preds = torch.tensor(data['test_predictions'])
    test_actual = torch.tensor(data['test_actual'])
    
    # Compute detailed metrics
    metrics = compute_metrics(test_preds, test_actual)
    
    # Add test_score from data
    metrics['test_score'] = data['test_score']
    
    return metrics

import glob

def process_probe_results_directory(base_dir):
    """
    Process all best_probe_predictions.json files in subdirectories of base_dir.
    Computes and adds detailed metrics if they don't exist.
    
    Args:
        base_dir: Path to directory containing probe result subdirectories (e.g., 'MATH', 'THOM_MATH')
    """
    # Find all best_probe_predictions.json files
    json_files = glob.glob(os.path.join(base_dir, '**/best_probe_predictions.json'), recursive=True)
    
    print(f"Found {len(json_files)} probe result files in {base_dir}")
    print("="*80)
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for json_file in json_files:
        rel_path = os.path.relpath(json_file, base_dir)
        print(f"\nProcessing: {rel_path}")
        
        try:
            # Load existing data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if detailed metrics already exist
            if 'detailed_test_metrics' in data:
                print(f"  ✓ Detailed metrics already exist, skipping")
                skipped_count += 1
                continue
            
            print(f"  → Computing metrics...")
            
            # Compute metrics
            test_preds = torch.tensor(data['test_predictions'])
            test_actual = torch.tensor(data['test_actual'])
            
            metrics = compute_metrics(test_preds, test_actual)
            
            # Create detailed_test_metrics dictionary with all metrics
            detailed_test_metrics = {}
            
            # Basic metrics
            detailed_test_metrics['mse'] = float(metrics['mse'])
            detailed_test_metrics['mae'] = float(metrics['mae'])
            detailed_test_metrics['spearman'] = float(metrics['spearman'])
            detailed_test_metrics['kendall_tau'] = float(metrics['kendall_tau'])
            detailed_test_metrics['acc_all'] = float(metrics['acc_all'])
            
            # Bin metrics
            for i in range(5):
                detailed_test_metrics[f'acc_bin_{i}'] = float(metrics[f'acc_bin_{i}'])
                detailed_test_metrics[f'count_bin_{i}'] = int(metrics[f'count_bin_{i}'])
                detailed_test_metrics[f'precision_bin_{i}'] = float(metrics[f'precision_bin_{i}'])
                detailed_test_metrics[f'recall_bin_{i}'] = float(metrics[f'recall_bin_{i}'])
                detailed_test_metrics[f'f1_bin_{i}'] = float(metrics[f'f1_bin_{i}'])
                detailed_test_metrics[f'num_predicted_bin_{i}'] = int(metrics[f'num_predicted_bin_{i}'])
            
            # Learnability metrics
            detailed_test_metrics['learnability_ys_mean'] = float(metrics['learnability_ys_mean'])
            detailed_test_metrics['learnability_selected_mean'] = float(metrics['learnability_selected_mean'])
            detailed_test_metrics['learnability_best_possible_mean'] = float(metrics['learnability_best_possible_mean'])
            
            # Add detailed_test_metrics to data
            data['detailed_test_metrics'] = detailed_test_metrics
            
            # Save updated data
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  ✓ Metrics computed and saved")
            print(f"    - Spearman: {detailed_test_metrics['spearman']:.4f}")
            print(f"    - Learnability (Ys): {detailed_test_metrics['learnability_ys_mean']:.4f}")
            print(f"    - Overall accuracy: {detailed_test_metrics['acc_all']:.4f}")
            updated_count += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files: {len(json_files)}")
    print(f"Updated: {updated_count}")
    print(f"Already had metrics: {skipped_count}")
    print(f"Errors: {error_count}")
    
    return updated_count, skipped_count, error_count


def generate_latex_table(base_dir, dataset_name):
    """
    Generate LaTeX table for all probe results in base_dir.
    """
    json_files = glob.glob(
        os.path.join(base_dir, '**/best_probe_predictions.json'),
        recursive=True
    )

    if not json_files:
        print(f"No probe results found in {base_dir}")
        return

    # ================= TABLE HEADER =================
    print(r'\begin{table*}[htbp]')
    print(r'\centering')
    print(fr'\caption{{Probe Performance on {dataset_name} Dataset}}')
    print(fr'\label{{tab:probe_results_{dataset_name.lower()}}}')
    print(r'\resizebox{\textwidth}{!}{%')
    print(r'\begin{tabular}{@{}lccccccccccc@{}}')
    print(r'\toprule')

    print(
        r'\multirow{3}{*}{\textbf{Model}} & '
        r'\multirow{3}{*}{\textbf{K}} & '
        r'\multirow{3}{*}{\textbf{Pass@K}} & '
        r'\multirow{3}{*}{\textbf{Spearman}} & '
        r'\multicolumn{5}{c}{\textbf{Bin Accuracy (\%)}} & '
        r'\multicolumn{3}{c}{\textbf{Learnability}} \\'
    )
    print(r'\cmidrule(lr){5-9} \cmidrule(lr){10-12}')
    print(
        r'& & & & '
        r'\textbf{B0} & \textbf{B1} & \textbf{B2} & \textbf{B3} & \textbf{B4} & '
        r'\textbf{Avg} & \textbf{Selected} & \textbf{Best} \\'
    )
    print(
        r'& & & & & & & & & '
        r'\textbf{(Ys)} & \textbf{(Ys)} & \textbf{Possible} \\'
    )
    print(r'\midrule')

    # ================= TABLE ROWS =================
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            dir_name = os.path.basename(os.path.dirname(json_file))

            # ---- Model name cleanup ----
            model = dir_name.split('_')[0].replace('-', ' ')
            model = model.replace('Instruct', 'I')

            # ---- Extract K ----
            if '_k_' in dir_name:
                k = dir_name.split('_k_')[1].split('_')[0]
            else:
                k = '?'

            # ---- Metrics ----
            pass_k = data.get('avg_benchmark_score', float('nan'))
            spearman = data.get('test_score', float('nan'))

            metrics = data.get('detailed_test_metrics', {})
            b = [metrics.get(f'acc_bin_{i}', 0.0) * 100 for i in range(5)]

            learn_avg = metrics.get('learnability_ys_mean', 0.0)
            learn_sel = metrics.get('learnability_selected_mean', 0.0)
            learn_best = metrics.get('learnability_best_possible_mean', 0.0)

            # ---- Print row ----
            print(
                f'{model} & {k} & {pass_k:.3f} & {spearman:.3f} & '
                f'{b[0]:.1f} & {b[1]:.1f} & {b[2]:.1f} & {b[3]:.1f} & {b[4]:.1f} & '
                f'{learn_avg:.3f} & {learn_sel:.3f} & {learn_best:.3f} \\\\'
            )

        except Exception as e:
            print(f'% Error processing {json_file}: {e}')

    # ================= TABLE FOOTER =================
    print(r'\midrule')
    print(
        r'\multicolumn{12}{l}{\footnotesize '
        r'B0--B4: Difficulty bins (0=hardest, 4=easiest). '
        r'Avg: Average dataset learnability, Selected: Learnability of probe predictions,} \\'
    )
    print(
        r'\multicolumn{12}{l}{\footnotesize '
        r'Best Possible: Optimal learnability achievable with perfect predictions.} \\'
    )
    print(r'\bottomrule')
    print(r'\end{tabular}%')
    print(r'}')
    print(r'\end{table*}')

# Process MATH directory
process_probe_results_directory('MATH')

print("\n\n" + "="*80)
print("LATEX TABLE FOR MATH DATASET")
print("="*80)
generate_latex_table('MATH', 'MATH')