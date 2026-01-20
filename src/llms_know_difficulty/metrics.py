import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from llms_know_difficulty.probe.probe_utils.linear_eoi_probe import linear_eoi_probe_train_utils


def bin(y, n_bins=10, min_val=0.0, max_val=1.0):
    """
    Bin values after sigmoid mapping (used for diagnostic bin metrics, not calibration).
    """
    bins = torch.linspace(min_val, max_val, n_bins + 1).to(y.device)
    y = torch.sigmoid(y)
    return torch.bucketize(y + 1e-8, bins) - 1


def compute_metrics(ys, preds, task_type: str = "auto", full_metrics: bool = True) -> dict:
    """Compute metrics between ys and preds.

    Args:
        ys: Ground truth values (torch.Tensor, np.ndarray, or list)
        preds: Predictions (torch.Tensor, np.ndarray, or list)
               For classification, these must be probabilities in [0,1].
        task_type: "auto" to infer, or explicitly "regression"/"classification"
        full_metrics: If True, compute all metrics (including calibration diagnostics).
                      If False, compute only primary metrics (mse, mae, spearman/auc).

    Returns:
        Dictionary of computed metrics.
    """
    # Convert to torch tensors
    if not isinstance(ys, torch.Tensor):
        ys = torch.from_numpy(np.asarray(ys)).float()
    else:
        ys = ys.float()

    if not isinstance(preds, torch.Tensor):
        preds = torch.from_numpy(np.asarray(preds)).float()
    else:
        preds = preds.float()

    # Flatten
    ys_flat = ys.view(-1)
    preds_flat = preds.view(-1)

    # Infer task type
    y_np = ys_flat.detach().cpu().numpy()
    p_np = preds_flat.detach().cpu().numpy()
    task_type = linear_eoi_probe_train_utils.infer_task_type(y_np, task_type)

    # Base metrics
    mse = nn.MSELoss()(preds_flat, ys_flat).item()
    mae = (preds_flat - ys_flat).abs().mean().item()

    metrics = {
        "mse": mse,
        "mae": mae,
        "task_type": task_type,
    }

    # Primary metric
    if task_type == "regression":
        spearman, _ = spearmanr(y_np, p_np)
        metrics["spearman"] = spearman
    else:
        unique_labels = np.unique(y_np)
        if len(unique_labels) < 2:
            print(f"Warning: Cannot compute AUC - only one unique label: {unique_labels}")
            metrics["auc"] = np.nan
        else:
            try:
                metrics["auc"] = roc_auc_score(y_np, p_np)
            except (ValueError, RuntimeError) as e:
                print(f"Warning: AUC computation failed - {str(e)}")
                metrics["auc"] = np.nan

    # Fast path for validation / training
    if not full_metrics:
        return metrics

    # ------------------------------------------------------------------
    # Calibration metric: Expected Calibration Error (ECE)
    # ------------------------------------------------------------------
    if task_type == "classification":
        n_bins = 10
        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=preds_flat.device)



        ece = torch.zeros(1, device=preds_flat.device)

        for i in range(n_bins):
            bin_mask = (preds_flat > bin_edges[i]) & (preds_flat <= bin_edges[i + 1])
            bin_size = bin_mask.float().sum()

            if bin_size > 0:
                bin_acc = ys_flat[bin_mask].mean()
                bin_conf = preds_flat[bin_mask].mean()
                ece += (bin_size / preds_flat.numel()) * torch.abs(bin_acc - bin_conf)

        metrics["ece"] = ece.item()

    # ------------------------------------------------------------------
    # Existing bin-based diagnostics (unchanged)
    # ------------------------------------------------------------------
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

    # Learnability diagnostics (unchanged)
    sigmoid_ys = torch.sigmoid(ys)
    sigmoid_preds = torch.sigmoid(preds)
    learnability_ys = sigmoid_ys * (1.0 - sigmoid_ys)
    learnability_preds = sigmoid_preds * (1.0 - sigmoid_preds)

    n_learnable = int(0.25 * ys.size(0))
    _, learnable_indices = torch.topk(learnability_preds, n_learnable)
    _, best_possible_learnable_indices = torch.topk(learnability_ys, n_learnable)

    metrics["learnability_ys_mean"] = learnability_ys.mean().item()
    metrics["learnability_selected_mean"] = learnability_ys[learnable_indices].mean().item()
    metrics["learnability_best_possible_mean"] = learnability_ys[best_possible_learnable_indices].mean().item()

    return metrics
