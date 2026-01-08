from pathlib import Path
from typing import Union


ROOT_DATA_DIR = Path(r"../../data")
ROOT_ACTIVATION_DATA_DIR = Path(r"../../data/activations") # /model_name/dataset_name/probe_name/activations.pt

ATTN_PROBE_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 128,
    'num_epochs': 2,
    'weight_decay': 10.0
}

SKLEARN_PROBE_CONFIG = {
    "use_kfold": True,
    "alpha_grid": [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
}