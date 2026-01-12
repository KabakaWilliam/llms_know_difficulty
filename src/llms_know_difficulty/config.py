from pathlib import Path
from typing import Union
from pydantic import BaseModel


SEED = 42

# Get the directory where this config file is located
_CONFIG_DIR = Path(__file__).parent

# Define data directories relative to the config file location
# config.py is at: src/llms_know_difficulty/config.py
# So _CONFIG_DIR is: src/llms_know_difficulty/
# Data is at project root: /VData/linna4335/llms_know_difficult/data
# So we go up 2 levels from config.py location
ROOT_DATA_DIR = _CONFIG_DIR.parent.parent / "data"
ROOT_ACTIVATION_DATA_DIR = ROOT_DATA_DIR / "activations"

# How much of the train split to use for validation
VAL_TRAIN_SPLIT_RATIO = 0.2
PROMPT_COLUMN_NAME = "formatted_prompt"
LABEL_COLUMN_NAME = "success_rate"
IDX_COLUMN_NAME = "idx"

DEVICE = "cuda:0"
MODEL_HYPERPARAMETERS = {
    "Qwen/Qwen2.5-1.5B-Instruct":{
        'num_layers': 29 
    },
    "Qwen/Qwen2.5-Math-1.5B-Instruct":{
        'num_layers': 29 
    }
}


class AttentionProbeConfig(BaseModel):
    
    """
    Attention probe config, contains the hyperparameters which can be cross-validated over.

    Args:
    learning_rate: list[float]    List of learning rates to cross-validate during training (e.g. [1e-3, 5e-4]).
    batch_size: list[int]         List of batch sizes for mini-batch gradient descent to cross-validate.
    num_epochs: list[int]         List of number of epochs to try for training.
    weight_decay: list[float]     List of L2 regularization strengths to cross-validate.
    layers: list[int]             List of layer indices to cross-validate over during training.
    max_length: int               Maximum prompt length when tokenizing inputs during training and evaluation.
    cross_validated_hyperparameters: list[str]  
        List of parameter names to cross-validate. Should match field names above, e.g. 
        ['layers', 'batch_size', 'learning_rate', 'num_epochs', 'weight_decay', 'max_length']

    """

    learning_rate: list[float] = [1e-3]
    batch_size: list[int] = [128]
    num_epochs: list[int] = [4]
    weight_decay: list[float] = [10.0]
    layer: list[int] = [-1]
    max_length: list[int] = [512]
    test_mode: bool = False # TODO: Turn off for actual training.
    cv_metric: str = 'spearmanr'
    cross_validated_hyperparameters: list[str] = [
    'layer',
    'batch_size',
    'learning_rate',
    'num_epochs',
    'weight_decay',
    'max_length']

class LinearThenMaxProbeConfig(AttentionProbeConfig):
    """
    Inherits from the AttentionProbeConfig. Adds no new hyperparameters. 
    """

class LinearThenSoftmaxProbeConfig(AttentionProbeConfig):
    """
    Inherits from the AttentionProbeConfig. Adds no new hyperparameters. 
    """

class LinearThenRollingMaxProbeConfig(AttentionProbeConfig):
    """
    Inherits from the AttentionProbeConfig. Adds no new hyperparameters. 
    """
    window_size: list[int] = [10]
    cross_validated_hyperparameters: list[str] = [
        'layer',
        'batch_size',
        'learning_rate',
        'num_epochs',
        'weight_decay',
        'max_length',
        'window_size']


class LinearEOIProbeConfig(BaseModel):
    """
    LinearEOI probe config for ridge/logistic regression probes.
    
    Args:
        model_name: str          Name of the base model to use for activation extraction
        alpha_grid: list[float]  List of regularization strengths to grid search over
        batch_size: int          Batch size for activation extraction
        max_length: int          Maximum prompt length when tokenizing inputs
    """
    model_name: str = "gpt2"
    alpha_grid: list[float] = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    batch_size: int = 16
    max_length: int = 1024

class TfidfProbeConfig(BaseModel):
    """
    LinearEOI probe config for ridge/logistic regression probes.
    
    Args:
        model_name: str          Name of the base model to use for activation extraction
        alpha_grid: list[float]  List of regularization strengths to grid search over
        batch_size: int          Batch size for activation extraction
        max_length: int          Maximum prompt length when tokenizing inputs
    """
    alpha_grid: list[float] = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    fit_intercept: bool = True


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
