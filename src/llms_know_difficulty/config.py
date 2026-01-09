from pathlib import Path
from typing import Union
from pydantic import BaseModel

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


class AttentionProbeConfig(BaseModel):
    
    """
    The cross_validate_hyperparameters are the hyperparameters that will be cross-validated on,
    all variables that are relevant to training should be listed here, the variables should appear
    in the config and be a list of values. If more than one values is provided the probe will be cross
    validated on that value.

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
    num_epochs: list[int] = [2]
    weight_decay: list[float] = [10.0]
    layer: list[int] = list(range(2)) # 29
    max_length: list[int] = [512]
    test_mode: bool = True # TODO: Turn off for actual training.
    cv_metric: str = 'spearmanr'
    cross_validated_hyperparameters: list[str] = [
    'layer',
    'batch_size',
    'learning_rate',
    'num_epochs',
    'weight_decay',
    'max_length']