from pathlib import Path
from typing import Union


# Get the directory where this config file is located
_CONFIG_DIR = Path(__file__).parent

# Define data directories relative to the config file location
# config.py is at: src/llms_know_difficulty/config.py
# So _CONFIG_DIR is: src/llms_know_difficulty/
# And data should be: src/llms_know_difficulty/data
ROOT_DATA_DIR = _CONFIG_DIR / "data"
ROOT_ACTIVATION_DATA_DIR = ROOT_DATA_DIR / "activations"

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