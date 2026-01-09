from pathlib import Path
from typing import Union


# Get the directory where this config file is located
_CONFIG_DIR = Path(__file__).parent

# Define data directories relative to the config file location
# config.py is at: src/llms_know_difficulty/config.py
# So _CONFIG_DIR is: src/llms_know_difficulty/
# Data is at project root: /VData/linna4335/llms_know_difficult/data
# So we go up 2 levels from config.py location
ROOT_DATA_DIR = _CONFIG_DIR.parent.parent / "data"
ROOT_ACTIVATION_DATA_DIR = ROOT_DATA_DIR / "activations"

ATTN_PROBE_CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 128,
    'num_epochs': 2,
    'weight_decay': 10.0
}

SKLEARN_PROBE_CONFIG = {
    "use_kfold": True,
    "alpha_grid": [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    "batch_size": 16,
    "max_length": 1024
}

PROMPTING_BASELINE = {
    "batch_size": 16,
    "generation_max_length": 500,
    "max_length": 1024,
    "generation_temperature": 1.0,

    "prompt_template": {
        "System": "You are a metacognitive scoring model. Your job is to estimate the probability that you will solve the problem correctly under the stated constraints. Do NOT solve the problem. Do NOT provide steps. Output a single number between 0.0 and 1.0.",

        "User": """Target solver: {SOLVER_MODEL_NAME}
Constraints: {k} attempts, max_tokens={T}, temperature={temp}

Problem:
{problem_text}

Output your probability estimate inside \\boxed{{}}:
\\boxed{{<float between 0.0 and 1.0>}}"""
    }
}
