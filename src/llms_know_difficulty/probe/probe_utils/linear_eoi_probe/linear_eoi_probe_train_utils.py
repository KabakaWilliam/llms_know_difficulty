import numpy as np
import random
import torch

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.utils.multiclass import type_of_target
from llms_know_difficulty.utils import infer_task_type


def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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



def compute_metric(predictions: np.ndarray, labels: np.ndarray, task_type: str) -> tuple:
    """
    Compute metric based on task type.
    
    Args:
        predictions: Model predictions [N]
        labels: Ground truth labels [N]
        task_type: "regression" or "classification"
    
    Returns:
        (score, metric_name): Tuple of (float score, string metric name)
    """
    if task_type == "regression":
        score, _ = spearmanr(labels, predictions)
        metric_name = "spearman"
    else:  # classification
        score = roc_auc_score(labels, predictions)
        metric_name = "auc"
    
    return float(score), metric_name

def to_numpy(x):
    try:
        return x.numpy()
    except:
        return x.detach().to(torch.float16).cpu().numpy()