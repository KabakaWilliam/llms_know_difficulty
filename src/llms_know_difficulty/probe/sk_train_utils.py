import numpy as np
import random
import torch

from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold, StratifiedKFold



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


def train_probe(train_activations, test_activations, task_type, alpha, alpha_grid, k_fold, n_folds, seed):
    pass