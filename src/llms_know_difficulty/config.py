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

DEVICE = "cuda:0"

MODEL_HYPERPARAMETERS = {
    "Qwen/Qwen2.5-1.5B-Instruct":{
        'num_layers': 29 
    }
}


class AttentionProbeConfig:
    learning_rate: Union[float, list[float]] = 1e-3
    batch_size: Union[int, list[int]] = 128
    num_epochs: Union[int, list[int]] = 2
    weight_decay: Union[float, list[float]] = 10.0
    layer_indices: list[int] = list(range(29))
    cross_validate_hyperparameters: list[str] = []
    max_length: int = 512