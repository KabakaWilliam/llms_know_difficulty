from abc import ABC, abstractmethod
from typing import List

class Probe(ABC):
    def __init__(self, config):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the probe."""
        pass

    @abstractmethod
    def init_model(self, config: dict):
        """Load a probe from checkpoint"""
        pass

    @abstractmethod
    def setup(self, model_name: str):
        """
        Any pre-training steps, extracting activations,
        loading a model to extract activations on the fly
        """
        pass

    @abstractmethod
    def fit(self, prompts: List[str], targets: List[float]) -> None:
        """
        Can do whatever you like: cross-validation, backprop, etc.
        """
        pass

    @abstractmethod
    def predict(self, prompts: List[str]):
        """
        Returns logits or probits
        """
        pass
