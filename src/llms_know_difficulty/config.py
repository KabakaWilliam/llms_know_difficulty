from pathlib import Path
from typing import Union
from pydantic import BaseModel

# Get the directory where this config file is located
_CONFIG_DIR = Path(__file__).parent

# Define data directories relative to the config file location
# config.py is at: src/llms_know_difficulty/config.py
# So _CONFIG_DIR is: src/llms_know_difficulty/
# Data is at project root: /VData/linna4335/llms_know_difficult/data
# So we go up 2 levels from config.py location
ROOT_DATA_DIR = _CONFIG_DIR.parent.parent / "data"
ROOT_ACTIVATION_DATA_DIR = ROOT_DATA_DIR / "activations"

DEVICE = "cuda:0"
MODEL_HYPERPARAMETERS = {
    "Qwen/Qwen2.5-1.5B-Instruct":{
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


class SklearnProbeConfig(BaseModel):
    """
    Sklearn probe config for ridge/logistic regression probes.
    
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


class PromptingBaselineConfig(BaseModel):
    """
    Prompting baseline config for scoring models.
    
    Args:
        batch_size: int                  Batch size for inference
        generation_max_length: int       Maximum tokens for generation
        max_length: int                  Maximum prompt length
        generation_temperature: float    Temperature for generation sampling
        prompt_template: dict            Dictionary with "System" and "User" prompt templates
    """
    batch_size: int = 16
    generation_max_length: int = 500
    max_length: int = 1024
    generation_temperature: float = 1.0
    prompt_template: dict = {
        "System": "You are a metacognitive scoring model. Your job is to estimate the probability that you will solve the problem correctly under the stated constraints. Do NOT solve the problem. Do NOT provide steps. Output a single number between 0.0 and 1.0.",
        "User": """Target solver: {SOLVER_MODEL_NAME}
Constraints: {k} attempts, max_tokens={T}, temperature={temp}

Problem:
{problem_text}

Output your probability estimate inside \\boxed{{}}:
\\boxed{{<float between 0.0 and 1.0>}}"""
    }


SKLEARN_PROBE_CONFIG = SklearnProbeConfig()


PROMPTING_BASELINE = PromptingBaselineConfig()
