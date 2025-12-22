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
            detailed_test_metrics['benchmark_mean_score'] = np.mean(test_actual)
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
    
    Args:
        base_dir: Path to directory containing probe result subdirectories
        dataset_name: Name to display in table (e.g., 'MATH', 'THOM_MATH')
    """
    json_files = glob.glob(os.path.join(base_dir, '**/best_probe_predictions.json'), recursive=True)
    
    if not json_files:
        print(f"No probe results found in {base_dir}")
        return
    
    print('\\begin{table*}[htbp]')
    print('\\centering')
    print(f'\\caption{{Probe Performance on {dataset_name} Dataset}}')
    print(f'\\label{{tab:probe_results_{dataset_name.lower()}}}')
    print('\\resizebox{\\textwidth}{!}{%')
    print('\\begin{tabular}{@{}llcccccccccccc@{}}')
    print('\\toprule')
    print('\\multirow{3}{*}{\\textbf{Model}} & \\multirow{3}{*}{\\textbf{Config}} & \\multirow{3}{*}{\\textbf{Temp}} & \\multirow{3}{*}{\\textbf{K}} & \\multirow{3}{*}{\\textbf{Spearman}} & \\multicolumn{5}{c}{\\textbf{Bin Accuracy (\\%)}} & \\multicolumn{3}{c}{\\textbf{Learnability}} \\\\')
    print('\\cmidrule(lr){6-10} \\cmidrule(lr){11-13}')
    print('& & & & & \\textbf{B0} & \\textbf{B1} & \\textbf{B2} & \\textbf{B3} & \\textbf{B4} & \\textbf{Avg} & \\textbf{Selected} & \\textbf{Best} \\\\')
    print('& & & & & & & & & & \\textbf{(Ys)} & \\textbf{(Ys)} & \\textbf{Possible} \\\\')
    print('\\midrule')
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract config from directory name
            dir_name = os.path.basename(os.path.dirname(json_file))
            
            # Parse model name and config
            if 'maxlen' in dir_name:
                # Format: Model_maxlen_X_k_Y_temp_Z
                parts = dir_name.split('_')
                model = parts[0].replace('-', ' ')
                k = parts[parts.index('k') + 1]
                temp = parts[parts.index('temp') + 1]
                config = f"maxlen={parts[parts.index('maxlen') + 1]}"
            elif 'samples' in dir_name:
                # Format: Model_samples_X_temp_Y
                parts = dir_name.split('_')
                model = parts[0].replace('-', ' ')
                k = parts[parts.index('samples') + 1]
                temp = parts[parts.index('temp') + 1]
                config = f"n={k}"
            else:
                model = dir_name
                k = "?"
                temp = "?"
                config = "-"
            
            if "Instruct" in model:
                model = model.replace("Instruct", "I")
            
            # Get metrics
            spearman = data['test_score']
            
            if 'detailed_test_metrics' in data:
                metrics = data['detailed_test_metrics']
                b0 = metrics['acc_bin_0'] * 100
                b1 = metrics['acc_bin_1'] * 100
                b2 = metrics['acc_bin_2'] * 100
                b3 = metrics['acc_bin_3'] * 100
                b4 = metrics['acc_bin_4'] * 100
                learn_avg = metrics['learnability_ys_mean']
                learn_sel = metrics['learnability_selected_mean']
                learn_best = metrics['learnability_best_possible_mean']
            else:
                # Compute metrics on the fly
                test_preds = torch.tensor(data['test_predictions'])
                test_actual = torch.tensor(data['test_actual'])
                metrics = compute_metrics(test_preds, test_actual)
                b0 = metrics['acc_bin_0'] * 100
                b1 = metrics['acc_bin_1'] * 100
                b2 = metrics['acc_bin_2'] * 100
                b3 = metrics['acc_bin_3'] * 100
                b4 = metrics['acc_bin_4'] * 100
                learn_avg = metrics['learnability_ys_mean']
                learn_sel = metrics['learnability_selected_mean']
                learn_best = metrics['learnability_best_possible_mean']
            
            print(f'{model} & {config} & {temp} & {k} & {spearman:.3f} & {b0:.1f} & {b1:.1f} & {b2:.1f} & {b3:.1f} & {b4:.1f} & {learn_avg:.3f} & {learn_sel:.3f} & {learn_best:.3f} \\\\')
            
        except Exception as e:
            print(f'% Error processing {json_file}: {e}')
    
    print('\\midrule')
    print('\\multicolumn{14}{l}{\\footnotesize B0-B4: Difficulty bins (0=hardest, 4=easiest). Avg: Average dataset learnability, Selected: Learnability of probe predictions,}\\\\')
    print('\\multicolumn{14}{l}{\\footnotesize Best Possible: Optimal learnability achievable with perfect predictions.}\\\\')
    print('\\bottomrule')
    print('\\end{tabular}%')
    print('}')
    print('\\end{table*}')


# Process MATH directory
process_probe_results_directory('MATH')

print("\n\n" + "="*80)
print("LATEX TABLE FOR MATH DATASET")
print("="*80)
generate_latex_table('MATH', 'MATH')