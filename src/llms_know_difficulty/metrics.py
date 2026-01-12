import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from utils import infer_task_type

def bin(y, n_bins=10, min_val=0.0, max_val=1.0):
    # Use torch bucketize to bin y into n_bins between min_val and max_val
    bins = torch.linspace(min_val, max_val, n_bins + 1).to(y.device)  # [n_bins+1]
    # clamp y to [min_val, max_val]
    y = y.clamp(min=min_val, max=max_val)
    # add small epsilon to handle edge case where y == min_val
    return torch.bucketize(y + 1e-8, bins) - 1  # [B], bin indices

def compute_metrics(ys, preds, task_type: str = "auto", full_metrics: bool = True) -> dict:
    """Compute metrics between ys and preds.
    
    Args:
        ys: Ground truth values (torch.Tensor, np.ndarray, or list)
        preds: Predictions (torch.Tensor, np.ndarray, or list)
        task_type: "auto" to infer, or explicitly "regression"/"classification"
        full_metrics: If True, compute all metrics (binning, precision, recall, f1, learnability).
                     If False, compute only primary metrics (mse, mae, spearman/auc).
                     Use False for fast validation/training, True for final test evaluation.
    
    Returns:
        Dictionary of computed metrics.
    """
    # Convert to torch tensors if needed
    if not isinstance(ys, torch.Tensor):
        ys = torch.from_numpy(np.asarray(ys)).float()
    else:
        ys = ys.float()
    
    if not isinstance(preds, torch.Tensor):
        preds = torch.from_numpy(np.asarray(preds)).float()
    else:
        preds = preds.float()

    # Infer task type if not provided
    y_np = ys.view(-1).detach().cpu().numpy()
    p_np = preds.view(-1).detach().cpu().numpy()
    task_type = infer_task_type(y_np, task_type)

    mse = nn.MSELoss()(preds, ys).item()
    mae = (preds - ys).abs().mean().item()

    metrics = {
        "mse": mse,
        "mae": mae,
        "task_type": task_type,
    }

    # Compute primary metric based on task type
    if task_type == "regression":
        # Use Spearman correlation for regression
        spearman, _ = spearmanr(y_np, p_np)
        metrics["spearman"] = spearman
    else:
        # Use ROC-AUC for classification
        unique_labels = np.unique(y_np)
        if len(unique_labels) < 2:
            # Only one class present - AUC is undefined
            print(f"Warning: Cannot compute AUC - only one unique label value: {unique_labels}")
            metrics["auc"] = np.nan
        else:
            try:
                auc = roc_auc_score(y_np, p_np)
                metrics["auc"] = auc
            except (ValueError, RuntimeError) as e:
                # AUC cannot be computed due to other issues
                print(f"Warning: AUC computation failed - {str(e)}")
                print(f"  Unique labels: {unique_labels}")
                metrics["auc"] = np.nan

    # Return early if only primary metrics are needed (for validation/training)
    if not full_metrics:
        return metrics

    # Compute additional metrics for full evaluation (binning, precision, recall, f1, learnability)
    # binning + accuracies
    binned_ys = bin(ys, n_bins=5, min_val=0.0, max_val=1.0)
    binned_preds = bin(preds, n_bins=5, min_val=0.0, max_val=1.0)

    acc_all = (binned_ys == binned_preds).float().mean().item()
    metrics["acc_all"] = acc_all

    for b in range(5):
        mask = (binned_ys == b)
        correct = (binned_preds[mask] == b).float().sum()
        count = mask.sum()
        metrics[f"count_bin_{b}"] = count.item()
        metrics[f"acc_bin_{b}"] = (correct / count.clamp_min(1)).item()

    # do precision, recall, f1 and num predictions for each class
    for b in range(5):
        true_positives = ((binned_preds == b) & (binned_ys == b)).float().sum()
        predicted_positives = (binned_preds == b).float().sum()
        actual_positives = (binned_ys == b).float().sum()

        metrics[f"num_predicted_bin_{b}"] = predicted_positives.item()

        precision = (true_positives / predicted_positives.clamp_min(1)).item()
        recall = (true_positives / actual_positives.clamp_min(1)).item()
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        metrics[f"precision_bin_{b}"] = precision
        metrics[f"recall_bin_{b}"] = recall
        metrics[f"f1_bin_{b}"] = f1

    # learnability is defined as ys * (1-ys)
    clipped_ys = ys.clamp(min=0.0, max=1.0)
    clipped_preds = preds.clamp(min=0.0, max=1.0)
    learnability_ys = clipped_ys * (1.0 - clipped_ys)
    learnability_preds = clipped_preds * (1.0 - clipped_preds)
    # take the top 25% most learnable samples as estimated by the probe
    n_learnable = int(0.25 * ys.size(0))
    _, learnable_indices = torch.topk(learnability_preds, n_learnable)
    _, best_possible_learnable_indices = torch.topk(learnability_ys, n_learnable)
    learnability_selected = learnability_ys[learnable_indices]
    best_possible_learnability = learnability_ys[best_possible_learnable_indices]
    metrics["learnability_ys_mean"] = learnability_ys.mean().item()
    metrics["learnability_selected_mean"] = learnability_selected.mean().item()
    metrics["learnability_best_possible_mean"] = best_possible_learnability.mean().item()

    return metrics