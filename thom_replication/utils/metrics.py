import torch
import torch.nn as nn


def bin(y, n_bins=10, min_val=0.0, max_val=1.0):
    # Use torch bucketize to bin y into n_bins between min_val and max_val
    bins = torch.linspace(min_val, max_val, n_bins + 1).to(y.device)  # [n_bins+1]
    # clamp y to [min_val, max_val]
    y = y.clamp(min=min_val, max=max_val)
    # add small epsilon to handle edge case where y == min_val
    return torch.bucketize(y + 1e-8, bins) - 1  # [B], bin indices


def compute_metrics(ys: torch.Tensor, preds: torch.Tensor) -> dict:
    """Compute various regression metrics between ys and preds."""

    mse = nn.MSELoss()(preds, ys).item()
    mae = (preds - ys).abs().mean().item()

    # Spearman's rank correlation (Pearson corr of ranks)
    y = ys.view(-1).float()
    p = preds.view(-1).float()

    # ranks via stable double-argsort (ties get arbitrary order; see note below)
    y_rank = torch.argsort(torch.argsort(y, stable=True), stable=True).float()
    p_rank = torch.argsort(torch.argsort(p, stable=True), stable=True).float()

    y_rank = y_rank - y_rank.mean()
    p_rank = p_rank - p_rank.mean()
    spearman = (y_rank * p_rank).sum() / (torch.sqrt((y_rank**2).sum()) * torch.sqrt((p_rank**2).sum()) + 1e-12)
    spearman = spearman.item()

    # binning + accuracies
    binned_ys = bin(ys, n_bins=5, min_val=0.0, max_val=1.0)
    binned_preds = bin(preds, n_bins=5, min_val=0.0, max_val=1.0)

    acc_all = (binned_ys == binned_preds).float().mean().item()

    metrics = {
        "mse": mse,
        "mae": mae,
        "spearman": spearman,
        "acc_all": acc_all,
    }

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

    # compute kendall's tau as well
    # since we have lots of ties spearmans may be misleading
    n = y.size(0)
    num_concordant = 0
    num_discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            concordant = ( (y[i] - y[j]) * (p[i] - p[j]) ) > 0
            discordant = ( (y[i] - y[j]) * (p[i] - p[j]) ) < 0
            if concordant:
                num_concordant += 1
            elif discordant:
                num_discordant += 1
    kendall_tau = (num_concordant - num_discordant) / (0.5 * n * (n - 1) + 1e-12)
    metrics["kendall_tau"] = kendall_tau

    return metrics