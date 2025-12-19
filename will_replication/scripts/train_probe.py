"""
Train linear probes on pre-extracted activations.

This script trains Ridge/Logistic regression probes on activations extracted
from post-instruction tokens, replicating the methodology from get_directions_store_preds.py.
"""

import argparse
import json
import os
import wandb
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from thom_replication.utils.metrics import compute_metrics


def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_task_type(y: np.ndarray, task_type: str = "auto") -> str:
    """Infer whether task is regression or classification."""
    if task_type in ("regression", "classification"):
        return task_type

    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return "classification"
    else:
        return "regression"


def make_cv(y: np.ndarray, n_splits: int, shuffle: bool, random_state: int):
    """Create cross-validator based on target type."""
    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )


def select_best_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    alpha_grid: List[float],
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, dict]:
    """
    Select best alpha via nested cross-validation.
    
    Returns:
        best_alpha, alpha_scores (dict mapping alpha -> mean CV score)
    """
    cv = make_cv(y_train, n_splits=n_folds, shuffle=True, random_state=random_state)
    alpha_scores = {}
    
    for alpha in alpha_grid:
        fold_scores = []
        
        for train_idx, val_idx in cv.split(x_train, y_train):
            x_train_fold = x_train[train_idx]
            x_val_fold = x_train[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            if task_type == "regression":
                model = Ridge(alpha=alpha, fit_intercept=False)
                model.fit(x_train_fold, y_train_fold)
                y_pred = model.predict(x_val_fold)
                
                corr_result = spearmanr(y_val_fold, y_pred)
                score = corr_result[0] if isinstance(corr_result, tuple) else corr_result.correlation
                
            else:  # classification
                if alpha == 0:
                    # No regularization
                    model = LogisticRegression(
                        penalty=None,
                        fit_intercept=False,
                        solver="lbfgs",
                        max_iter=1000,
                    )
                else:
                    model = LogisticRegression(
                        penalty="l2",
                        C=1.0 / alpha,
                        fit_intercept=False,
                        solver="lbfgs",
                        max_iter=1000,
                    )
                model.fit(x_train_fold, y_train_fold)
                y_proba = model.predict_proba(x_val_fold)[:, 1]
                score = roc_auc_score(y_val_fold, y_proba)
            
            fold_scores.append(score)
        
        alpha_scores[alpha] = float(np.mean(fold_scores))
    
    # Select alpha with best mean CV score
    best_alpha = max(alpha_scores.keys(), key=lambda a: alpha_scores[a])
    
    return best_alpha, alpha_scores


def train_single_layer_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    alpha: float = 1.0,
    alpha_grid: Optional[List[float]] = None,
    k_fold: bool = False,
    n_folds: int = 5,
    random_state: int = 42,
    aux_features_train: Optional[np.ndarray] = None,
    aux_features_test: Optional[np.ndarray] = None,
):
    """
    Train a single linear probe with optional alpha selection via nested CV.
    
    Args:
        aux_features_train: Optional auxiliary features to concatenate with activations [N, F]
        aux_features_test: Optional auxiliary features for test set [M, F]
    
    Returns:
        train_preds, test_preds, cv_preds, train_perf, test_perf, cv_perf, final_model, selected_alpha, alpha_scores
    """
    # Concatenate auxiliary features if provided
    if aux_features_train is not None:
        x_train = np.concatenate([x_train, aux_features_train], axis=1)
        x_test = np.concatenate([x_test, aux_features_test], axis=1)
    selected_alpha = alpha
    alpha_scores = None
    
    # Alpha selection via nested CV
    if alpha_grid is not None and len(alpha_grid) > 1:
        selected_alpha, alpha_scores = select_best_alpha(
            x_train, y_train, task_type, alpha_grid, n_folds, random_state
        )
    
    cv_preds = None
    cv_perf = None
    
    # Cross-validation (using selected_alpha)
    if k_fold:
        cv = make_cv(y_train, n_splits=n_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        cv_preds = np.zeros(len(y_train), dtype=float)
        
        for train_idx, val_idx in cv.split(x_train, y_train):
            x_train_fold = x_train[train_idx]
            x_val_fold = x_train[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            if task_type == "regression":
                model_fold = Ridge(alpha=selected_alpha, fit_intercept=False)
                model_fold.fit(x_train_fold, y_train_fold)
                y_pred_val = model_fold.predict(x_val_fold)
                cv_preds[val_idx] = y_pred_val
                
                corr_result = spearmanr(y_val_fold, y_pred_val)
                val_perf = corr_result[0] if isinstance(corr_result, tuple) else corr_result.correlation
                
            else:  # classification
                if selected_alpha == 0:
                    # No regularization
                    model_fold = LogisticRegression(
                        penalty=None,
                        fit_intercept=False,
                        solver="lbfgs",
                        max_iter=1000,
                    )
                else:
                    model_fold = LogisticRegression(
                        penalty="l2",
                        C=1.0 / selected_alpha,
                        fit_intercept=False,
                        solver="lbfgs",
                        max_iter=1000,
                    )
                model_fold.fit(x_train_fold, y_train_fold)
                y_proba_val = model_fold.predict_proba(x_val_fold)[:, 1]
                cv_preds[val_idx] = y_proba_val
                
                val_perf = roc_auc_score(y_val_fold, y_proba_val)
            
            cv_scores.append(val_perf)
        
        cv_perf = float(np.mean(cv_scores))
    
    # Train final model on full training set with selected alpha
    if task_type == "regression":
        final_model = Ridge(alpha=selected_alpha, fit_intercept=False)
        final_model.fit(x_train, y_train)
        
        train_preds = final_model.predict(x_train)
        test_preds = final_model.predict(x_test)
        
        train_corr = spearmanr(y_train, train_preds)
        train_perf = train_corr[0] if isinstance(train_corr, tuple) else train_corr.correlation
        
        test_corr = spearmanr(y_test, test_preds)
        test_perf = test_corr[0] if isinstance(test_corr, tuple) else test_corr.correlation
        
    else:  # classification
        if selected_alpha == 0:
            # No regularization
            final_model = LogisticRegression(
                penalty=None,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=1000,
            )
        else:
            final_model = LogisticRegression(
                penalty="l2",
                C=1.0 / selected_alpha,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=1000,
            )
        final_model.fit(x_train, y_train)
        
        train_preds = final_model.predict_proba(x_train)[:, 1]
        test_preds = final_model.predict_proba(x_test)[:, 1]
        
        train_perf = roc_auc_score(y_train, train_preds)
        test_perf = roc_auc_score(y_test, test_preds)
    
    return train_preds, test_preds, cv_preds, train_perf, test_perf, cv_perf, final_model, selected_alpha, alpha_scores


def train_probes(
    train_activations_path: str,
    test_activations_path: str,
    output_dir: str,
    task_type: str = "auto",
    alpha: float = 1.0,
    alpha_grid: Optional[List[float]] = None,
    k_fold: bool = False,
    n_folds: int = 5,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "diff_probe_training",
    wandb_name: Optional[str] = None,
):
    """Train linear probes on all layers and positions with optional alpha selection."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load activations and extract metadata
    print("Loading training activations...")
    train_data = torch.load(train_activations_path)
    train_activations = train_data['activations']  # [N, L, P, D]
    train_labels = train_data['labels'].numpy()  # [N]
    
    # Load auxiliary features if available (e.g., topic_num)
    aux_features_train = train_data.get('aux_features', None)  # [N, F] or None
    if aux_features_train is not None:
        if isinstance(aux_features_train, torch.Tensor):
            aux_features_train = aux_features_train.numpy()
        # Ensure 2D shape
        if aux_features_train.ndim == 1:
            aux_features_train = aux_features_train.reshape(-1, 1)
        print(f"Found auxiliary features with shape: {aux_features_train.shape}")
    
    # Extract metadata from filename
    train_filename = os.path.basename(train_activations_path)
    # Parse filename: Model_maxlen_X_k_Y_temp_Z_split.pt
    import re
    match = re.search(r'(.+?)_maxlen_(\d+)_k_(\d+(?:\.\d+)?)_temp_(\d+(?:\.\d+)?)_(train|test)\.pt', train_filename)
    if match:
        model_name_from_file = match.group(1)
        max_len = int(match.group(2))
        k_samples = float(match.group(3))
        temperature = float(match.group(4))
    else:
        # Fallback if pattern doesn't match
        model_name_from_file = train_data.get('model_name', 'unknown')
        max_len = None
        k_samples = None
        temperature = None
    
    print("Loading test activations...")
    test_data = torch.load(test_activations_path)
    test_activations = test_data['activations']  # [M, L, P, D]
    test_labels = test_data['labels'].numpy()  # [M]
    
    # Load auxiliary features for test set
    aux_features_test = test_data.get('aux_features', None)  # [M, F] or None
    if aux_features_test is not None:
        if isinstance(aux_features_test, torch.Tensor):
            aux_features_test = aux_features_test.numpy()
        if aux_features_test.ndim == 1:
            aux_features_test = aux_features_test.reshape(-1, 1)
        print(f"Found auxiliary features (test) with shape: {aux_features_test.shape}")
    
    # Initialize wandb if requested
    if use_wandb:
        config = {
            "train_activations": train_activations_path,
            "test_activations": test_activations_path,
            "alpha": alpha,
            "alpha_grid": alpha_grid,
            "k_fold": k_fold,
            "n_folds": n_folds,
            "seed": seed,
        }
        
        # Add parsed metadata from filename
        if max_len is not None:
            config["max_length"] = max_len
        if k_samples is not None:
            config["k_samples"] = k_samples
        if temperature is not None:
            config["temperature"] = temperature
        if model_name_from_file != 'unknown':
            config["model_name"] = model_name_from_file
        
        # Add auxiliary features info
        if aux_features_train is not None:
            config["aux_features_dim"] = aux_features_train.shape[1]
            config["using_aux_features"] = True
        else:
            config["using_aux_features"] = False
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config
        )
    
    n_train, n_layers, n_positions, d_model = train_activations.shape
    n_test = test_activations.shape[0]
    
    print(f"Train samples: {n_train}, Test samples: {n_test}")
    print(f"Number of layers: {n_layers}, Positions: {n_positions}, Hidden size: {d_model}")
    
    # Infer task type
    task_type = infer_task_type(train_labels, task_type)
    print(f"Task type: {task_type}")
    metric_name = "spearman" if task_type == "regression" else "roc_auc"
    
    # Train probe for each position and layer
    layer_performance = {
        "train": [],
        "test": [],
        "cv": [] if k_fold else None,
    }
    
    all_predictions = {}  # (pos_idx, layer_idx) -> {train_preds, test_preds, cv_preds}
    all_probe_weights = {}  # (pos_idx, layer_idx) -> probe weights
    all_selected_alphas = {}  # (pos_idx, layer_idx) -> selected alpha
    all_alpha_scores = {}  # (pos_idx, layer_idx) -> {alpha: score}
    
    positions = train_data.get('positions', list(range(-n_positions, 0)))
    
    # Print alpha selection info
    if alpha_grid is not None and len(alpha_grid) > 1:
        print(f"\nPerforming alpha grid search with values: {alpha_grid}")
        print(f"This will do nested CV for each (position, layer) combination\\n")
    
    for pos_idx in range(n_positions):
        pos = positions[pos_idx]
        print(f"\nTraining probes for position {pos}")
        
        pos_train_performance = []
        pos_test_performance = []
        pos_cv_performance = [] if k_fold else None
        
        for layer_idx in tqdm(range(n_layers), desc=f"Layer progress for position {pos}"):
            # Extract activations for this layer and position
            x_train = train_activations[:, layer_idx, pos_idx, :].numpy()  # [N, D]
            x_test = test_activations[:, layer_idx, pos_idx, :].numpy()  # [M, D]
            
            # Train probe with optional alpha selection and auxiliary features
            train_preds, test_preds, cv_preds, train_perf, test_perf, cv_perf, probe_model, selected_alpha, alpha_scores = train_single_layer_probe(
                    x_train, train_labels,
                    x_test, test_labels,
                    task_type=task_type,
                    alpha=alpha,
                    alpha_grid=alpha_grid,
                    k_fold=k_fold,
                    n_folds=n_folds,
                    random_state=seed,
                    aux_features_train=aux_features_train,
                    aux_features_test=aux_features_test,
                )
            
            # Store performance
            pos_train_performance.append(float(train_perf))
            pos_test_performance.append(float(test_perf))
            if k_fold:
                pos_cv_performance.append(float(cv_perf))
            
            # Log to wandb
            if use_wandb:
                # Per-layer metrics with step for easy plotting
                step = pos_idx * n_layers + layer_idx
                log_dict = {
                    f"pos_{pos}/train_{metric_name}": train_perf,
                    f"pos_{pos}/test_{metric_name}": test_perf,
                    f"pos_{pos}/selected_alpha": selected_alpha,
                    f"pos_{pos}/layer": layer_idx,
                    # Global metrics across all positions
                    f"layer_{layer_idx}/train_{metric_name}": train_perf,
                    f"layer_{layer_idx}/test_{metric_name}": test_perf,
                }
                if k_fold:
                    log_dict[f"pos_{pos}/cv_{metric_name}"] = cv_perf
                    log_dict[f"layer_{layer_idx}/cv_{metric_name}"] = cv_perf
                    # Generalization gap: train - cv
                    log_dict[f"pos_{pos}/train_cv_gap"] = train_perf - cv_perf
                    log_dict[f"pos_{pos}/cv_test_gap"] = cv_perf - test_perf
                
                # Log alpha scores if grid search was performed
                if alpha_scores is not None:
                    for alpha_val, score in alpha_scores.items():
                        log_dict[f"pos_{pos}/alpha_{alpha_val}_cv_score"] = score
                
                wandb.log(log_dict, step=step)
            
            # Store predictions
            all_predictions[(pos_idx, layer_idx)] = {
                "train_predictions": train_preds.tolist(),
                "test_predictions": test_preds.tolist(),
                "cv_predictions": cv_preds.tolist() if cv_preds is not None else None,
            }
            
            # Store probe weights (coefficients)
            if hasattr(probe_model, 'coef_'):
                weights = probe_model.coef_
                if weights.ndim == 2 and weights.shape[0] == 1:
                    weights = weights[0]  # Flatten from [1, D] to [D]
                all_probe_weights[(pos_idx, layer_idx)] = weights.tolist()
            
            # Store alpha selection results
            all_selected_alphas[(pos_idx, layer_idx)] = float(selected_alpha)
            if alpha_scores is not None:
                all_alpha_scores[(pos_idx, layer_idx)] = {str(k): v for k, v in alpha_scores.items()}
        
        layer_performance["train"].append(pos_train_performance)
        layer_performance["test"].append(pos_test_performance)
        if k_fold:
            layer_performance["cv"].append(pos_cv_performance)
        
        # Log per-position summary to wandb
        if use_wandb:
            pos_test_arr = np.array(pos_test_performance)
            best_layer_idx = int(np.nanargmax(pos_test_arr))
            summary_dict = {
                f"summary/pos_{pos}/best_layer": best_layer_idx,
                f"summary/pos_{pos}/best_test_{metric_name}": float(pos_test_arr[best_layer_idx]),
                f"summary/pos_{pos}/mean_test_{metric_name}": float(np.nanmean(pos_test_arr)),
                f"summary/pos_{pos}/std_test_{metric_name}": float(np.nanstd(pos_test_arr)),
            }
            if k_fold:
                pos_cv_arr = np.array(pos_cv_performance)
                best_cv_layer_idx = int(np.nanargmax(pos_cv_arr))
                summary_dict.update({
                    f"summary/pos_{pos}/best_cv_layer": best_cv_layer_idx,
                    f"summary/pos_{pos}/best_cv_{metric_name}": float(pos_cv_arr[best_cv_layer_idx]),
                    f"summary/pos_{pos}/mean_cv_{metric_name}": float(np.nanmean(pos_cv_arr)),
                })
            wandb.log(summary_dict)
    
    # Save performance metrics
    performance_data = {
        "layer_performance": layer_performance,
        "layer_indices": train_data['layer_indices'],
        "positions": positions,
        "n_layers": n_layers,
        "n_positions": n_positions,
        "d_model": d_model,
        "task_type": task_type,
        "metric": metric_name,
        "k_fold": k_fold,
        "n_folds": n_folds if k_fold else None,
        "alpha": alpha,
        "alpha_grid": alpha_grid,
        "selected_alphas": {f"pos_{positions[pi]}_layer_{li}": a 
                           for (pi, li), a in all_selected_alphas.items()},
    }
    
    # Save alpha scores if grid search was performed
    if alpha_grid is not None and len(alpha_grid) > 1:
        performance_data["alpha_cv_scores"] = {
            f"pos_{positions[pi]}_layer_{li}": scores 
            for (pi, li), scores in all_alpha_scores.items()
        }
    
    with open(f"{output_dir}/performance.json", 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"\nPerformance saved to {output_dir}/performance.json")
    
    # Find and report best layer for each position
    for pos_idx in range(n_positions):
        pos = positions[pos_idx]
        test_scores = np.array(layer_performance["test"][pos_idx])
        best_layer_by_test = int(np.nanargmax(test_scores))
        best_test = float(test_scores[best_layer_by_test])
        
        print(f"\n{'='*60}")
        print(f"Position {pos}: Best layer by test {metric_name}: Layer {best_layer_by_test}")
        print(f"Test {metric_name}: {best_test:.4f}")
        
        # Show selected alpha if grid search was performed
        if alpha_grid is not None and len(alpha_grid) > 1:
            selected_alpha_best = all_selected_alphas[(pos_idx, best_layer_by_test)]
            print(f"Selected alpha: {selected_alpha_best}")
        
        if k_fold:
            cv_scores = np.array(layer_performance["cv"][pos_idx])
            best_layer_by_cv = int(np.nanargmax(cv_scores))
            best_cv = float(cv_scores[best_layer_by_cv])
            test_at_cv_layer = float(test_scores[best_layer_by_cv])
            
            print(f"\nBest layer by CV {metric_name}: Layer {best_layer_by_cv}")
            print(f"CV {metric_name}: {best_cv:.4f}")
            print(f"Test {metric_name} at that layer: {test_at_cv_layer:.4f}")
    
    # Find overall best probe across all positions and layers
    if k_fold:
        best_cv_overall = -np.inf
        best_info = None  # (pos_idx, layer_idx)
        
        for pos_idx in range(n_positions):
            cv_scores = np.array(layer_performance["cv"][pos_idx])
            if cv_scores.size == 0:
                continue
            layer_idx = int(np.nanargmax(cv_scores))
            score_cv = float(cv_scores[layer_idx])
            if score_cv > best_cv_overall:
                best_cv_overall = score_cv
                best_info = (pos_idx, layer_idx)
        
        if best_info is not None:
            best_pos_idx, best_layer_idx = best_info
            best_pos = positions[best_pos_idx]
            test_scores_for_pos = np.array(layer_performance["test"][best_pos_idx])
            test_at_best_cv = float(test_scores_for_pos[best_layer_idx])
            
            print(f"\n{'='*60}")
            print("=== BEST PROBE OVERALL (selected by CV) ===")
            print(f"Position {best_pos}, Layer {best_layer_idx}")
            print(f"CV {metric_name}: {best_cv_overall:.4f}")
            print(f"Test {metric_name}: {test_at_best_cv:.4f}")
            if alpha_grid is not None and len(alpha_grid) > 1:
                best_alpha = all_selected_alphas[(best_pos_idx, best_layer_idx)]
                print(f"Selected alpha: {best_alpha}")
            
            # Compute detailed metrics using compute_metrics for test set
            test_preds_best = torch.tensor(all_predictions[(best_pos_idx, best_layer_idx)]["test_predictions"])
            test_labels_tensor = torch.tensor(test_labels)
            detailed_test_metrics = compute_metrics(test_labels_tensor, test_preds_best)
            
            # Compute detailed metrics for train set
            train_preds_best = torch.tensor(all_predictions[(best_pos_idx, best_layer_idx)]["train_predictions"])
            train_labels_tensor = torch.tensor(train_labels)
            detailed_train_metrics = compute_metrics(train_labels_tensor, train_preds_best)
            
            # Compute detailed metrics for CV set if available
            cv_preds_best = all_predictions[(best_pos_idx, best_layer_idx)]["cv_predictions"]
            if cv_preds_best is not None:
                cv_preds_tensor = torch.tensor(cv_preds_best)
                detailed_cv_metrics = compute_metrics(train_labels_tensor, cv_preds_tensor)
            else:
                detailed_cv_metrics = None
            
            # Print detailed test metrics
            print(f"\nDetailed Test Metrics:")
            print(f"  MSE: {detailed_test_metrics['mse']:.4f}")
            print(f"  MAE: {detailed_test_metrics['mae']:.4f}")
            print(f"  Spearman: {detailed_test_metrics['spearman']:.4f}")
            print(f"  Kendall Tau: {detailed_test_metrics['kendall_tau']:.4f}")
            print(f"  Binned Accuracy (5 bins): {detailed_test_metrics['acc_all']:.4f}")
            print(f"  Learnability (selected top 25%): {detailed_test_metrics['learnability_selected_mean']:.4f}")
            print(f"  Learnability (best possible): {detailed_test_metrics['learnability_best_possible_mean']:.4f}")
            
            # Print per-bin metrics
            print(f"\n  Per-Bin Metrics (Test Set):")
            for b in range(5):
                count = int(detailed_test_metrics[f'count_bin_{b}'])
                num_pred = int(detailed_test_metrics[f'num_predicted_bin_{b}'])
                acc = detailed_test_metrics[f'acc_bin_{b}']
                prec = detailed_test_metrics[f'precision_bin_{b}']
                rec = detailed_test_metrics[f'recall_bin_{b}']
                f1 = detailed_test_metrics[f'f1_bin_{b}']
                print(f"    Bin {b}: count={count:4d}, pred={num_pred:4d}, acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")
            
            print(f"{'='*60}")
            
            # Log best probe to wandb
            if use_wandb:
                train_scores_for_pos = np.array(layer_performance["train"][best_pos_idx])
                train_at_best_cv = float(train_scores_for_pos[best_layer_idx])
                
                best_probe_summary = {
                    "best_probe/position": best_pos,
                    "best_probe/layer": best_layer_idx,
                    f"best_probe/train_{metric_name}": train_at_best_cv,
                    f"best_probe/cv_{metric_name}": best_cv_overall,
                    f"best_probe/test_{metric_name}": test_at_best_cv,
                    "best_probe/train_cv_gap": train_at_best_cv - best_cv_overall,
                    "best_probe/cv_test_gap": best_cv_overall - test_at_best_cv,
                    "best_probe/selected_alpha": all_selected_alphas.get((best_pos_idx, best_layer_idx)),
                    "best_probe/using_aux_features": aux_features_train is not None,
                    "task_type": task_type,
                    "metric": metric_name,
                    # Add detailed test metrics
                    "best_probe/test_mse": detailed_test_metrics['mse'],
                    "best_probe/test_mae": detailed_test_metrics['mae'],
                    "best_probe/test_spearman": detailed_test_metrics['spearman'],
                    "best_probe/test_kendall_tau": detailed_test_metrics['kendall_tau'],
                    "best_probe/test_acc_all": detailed_test_metrics['acc_all'],
                    "best_probe/test_learnability_selected": detailed_test_metrics['learnability_selected_mean'],
                    "best_probe/test_learnability_best_possible": detailed_test_metrics['learnability_best_possible_mean'],
                    # Add detailed train metrics
                    "best_probe/train_mse": detailed_train_metrics['mse'],
                    "best_probe/train_mae": detailed_train_metrics['mae'],
                    "best_probe/train_spearman": detailed_train_metrics['spearman'],
                    "best_probe/train_kendall_tau": detailed_train_metrics['kendall_tau'],
                }
                
                # Add CV metrics if available
                if detailed_cv_metrics is not None:
                    best_probe_summary.update({
                        "best_probe/cv_mse": detailed_cv_metrics['mse'],
                        "best_probe/cv_mae": detailed_cv_metrics['mae'],
                        "best_probe/cv_spearman": detailed_cv_metrics['spearman'],
                        "best_probe/cv_kendall_tau": detailed_cv_metrics['kendall_tau'],
                    })
                
                # Add per-bin metrics for test set
                for b in range(5):
                    best_probe_summary[f"best_probe/test_acc_bin_{b}"] = detailed_test_metrics[f'acc_bin_{b}']
                    best_probe_summary[f"best_probe/test_precision_bin_{b}"] = detailed_test_metrics[f'precision_bin_{b}']
                    best_probe_summary[f"best_probe/test_recall_bin_{b}"] = detailed_test_metrics[f'recall_bin_{b}']
                    best_probe_summary[f"best_probe/test_f1_bin_{b}"] = detailed_test_metrics[f'f1_bin_{b}']
                    best_probe_summary[f"best_probe/test_count_bin_{b}"] = detailed_test_metrics[f'count_bin_{b}']
                    best_probe_summary[f"best_probe/test_num_predicted_bin_{b}"] = detailed_test_metrics[f'num_predicted_bin_{b}']
                
                # Add alpha scores if available
                if (best_pos_idx, best_layer_idx) in all_alpha_scores:
                    best_alpha_scores = all_alpha_scores[(best_pos_idx, best_layer_idx)]
                    for alpha_val, score in best_alpha_scores.items():
                        best_probe_summary[f"best_probe/alpha_{alpha_val}_cv_score"] = score
                
                wandb.log(best_probe_summary)
                
                # Create summary table for wandb
                wandb.run.summary["best_position"] = best_pos
                wandb.run.summary["best_layer"] = best_layer_idx
                wandb.run.summary[f"best_train_{metric_name}"] = train_at_best_cv
                wandb.run.summary[f"best_cv_{metric_name}"] = best_cv_overall
                wandb.run.summary[f"best_test_{metric_name}"] = test_at_best_cv
            
            # Save best probe predictions
            best_probe_data = {
                "best_position": best_pos,
                "best_position_idx": best_pos_idx,
                "best_layer": best_layer_idx,
                "cv_score": best_cv_overall,
                "test_score": test_at_best_cv,
                "selected_alpha": all_selected_alphas.get((best_pos_idx, best_layer_idx)),
                "alpha_cv_scores": all_alpha_scores.get((best_pos_idx, best_layer_idx)),
                "train_actual": train_labels.tolist(),
                "test_actual": test_labels.tolist(),
                "train_predictions": all_predictions[(best_pos_idx, best_layer_idx)]["train_predictions"],
                "test_predictions": all_predictions[(best_pos_idx, best_layer_idx)]["test_predictions"],
                "cv_predictions": all_predictions[(best_pos_idx, best_layer_idx)]["cv_predictions"],
                "probe_weights": all_probe_weights.get((best_pos_idx, best_layer_idx)),
                "task_type": task_type,
                "metric": metric_name,
                # Add detailed metrics
                "detailed_test_metrics": detailed_test_metrics,
                "detailed_train_metrics": detailed_train_metrics,
                "detailed_cv_metrics": detailed_cv_metrics,
            }
            
            with open(f"{output_dir}/best_probe_predictions.json", 'w') as f:
                json.dump(best_probe_data, f, indent=2)
            
            print(f"\nBest probe predictions saved to {output_dir}/best_probe_predictions.json")
    else:
        # Without CV, find best by test performance across all positions/layers
        best_test_overall = -np.inf
        best_info = None
        
        for pos_idx in range(n_positions):
            test_scores = np.array(layer_performance["test"][pos_idx])
            if test_scores.size == 0:
                continue
            layer_idx = int(np.nanargmax(test_scores))
            score_test = float(test_scores[layer_idx])
            if score_test > best_test_overall:
                best_test_overall = score_test
                best_info = (pos_idx, layer_idx)
        
        if best_info is not None:
            best_pos_idx, best_layer_idx = best_info
            best_pos = positions[best_pos_idx]
            
            print(f"\n{'='*60}")
            print("=== BEST PROBE OVERALL (selected by test) ===")
            print(f"Position {best_pos}, Layer {best_layer_idx}")
            print(f"Test {metric_name}: {best_test_overall:.4f}")
            print(f"{'='*60}")
    
    # Save all predictions with position and layer info
    all_preds_output = {
        "positions": positions,
        "predictions": {}
    }
    
    for (pos_idx, layer_idx), preds in all_predictions.items():
        key = f"pos_{positions[pos_idx]}_layer_{layer_idx}"
        all_preds_output["predictions"][key] = preds
    
    all_preds_output["train_labels"] = train_labels.tolist()
    all_preds_output["test_labels"] = test_labels.tolist()
    
    with open(f"{output_dir}/all_predictions.json", 'w') as f:
        json.dump(all_preds_output, f, indent=2)
    
    print(f"All predictions saved to {output_dir}/all_predictions.json")
    
    # Log overall statistics to wandb
    if use_wandb:
        # Aggregate statistics across all positions and layers
        all_test_scores = []
        all_train_scores = []
        all_cv_scores = [] if k_fold else None
        
        for pos_idx in range(n_positions):
            all_test_scores.extend(layer_performance["test"][pos_idx])
            all_train_scores.extend(layer_performance["train"][pos_idx])
            if k_fold:
                all_cv_scores.extend(layer_performance["cv"][pos_idx])
        
        overall_stats = {
            f"overall/mean_test_{metric_name}": float(np.nanmean(all_test_scores)),
            f"overall/std_test_{metric_name}": float(np.nanstd(all_test_scores)),
            f"overall/max_test_{metric_name}": float(np.nanmax(all_test_scores)),
            f"overall/min_test_{metric_name}": float(np.nanmin(all_test_scores)),
            f"overall/mean_train_{metric_name}": float(np.nanmean(all_train_scores)),
            f"overall/std_train_{metric_name}": float(np.nanstd(all_train_scores)),
        }
        
        if k_fold:
            overall_stats.update({
                f"overall/mean_cv_{metric_name}": float(np.nanmean(all_cv_scores)),
                f"overall/std_cv_{metric_name}": float(np.nanstd(all_cv_scores)),
                f"overall/mean_train_cv_gap": float(np.nanmean(np.array(all_train_scores) - np.array(all_cv_scores))),
                f"overall/mean_cv_test_gap": float(np.nanmean(np.array(all_cv_scores) - np.array(all_test_scores))),
            })
        
        wandb.log(overall_stats)
        
        # Update run summary with overall stats
        wandb.run.summary.update(overall_stats)
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on pre-extracted activations")
    
    # Data args
    parser.add_argument("--train_activations", type=str, required=True,
                        help="Path to training activations (.pt file)")
    parser.add_argument("--test_activations", type=str, required=True,
                        help="Path to test activations (.pt file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    
    # Training args
    parser.add_argument("--task_type", type=str, default="auto",
                        choices=["auto", "regression", "classification"],
                        help="Task type")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Regularization strength (used if --alpha_grid not specified)")
    parser.add_argument("--alpha_grid", type=str, default=None,
                        help="Comma-separated list of alpha values for grid search (e.g., '0.001,0.01,0.1,1,10,100')")
    parser.add_argument("--k_fold", action="store_true",
                        help="Enable k-fold cross-validation")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Wandb args
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="diff_probe_training",
                        help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name")
    
    args = parser.parse_args()
    
    # Parse alpha grid
    alpha_grid = None
    if args.alpha_grid is not None:
        alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",")]
        print(f"Using alpha grid: {alpha_grid}")
    
    train_probes(
        train_activations_path=args.train_activations,
        test_activations_path=args.test_activations,
        output_dir=args.output_dir,
        task_type=args.task_type,
        alpha=args.alpha,
        alpha_grid=alpha_grid,
        k_fold=args.k_fold,
        n_folds=args.n_folds,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


if __name__ == "__main__":
    main()
