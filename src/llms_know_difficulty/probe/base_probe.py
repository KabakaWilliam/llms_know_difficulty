from abc import ABC, abstractmethod
from typing import List

class Probe(ABC):
    
    def __init__(self, config, **kwargs):
        self.config = config
        self.setup()

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
    def train(self, train_data: tuple[list[str], list[float]], val_data: tuple[list[str], list[float]]) -> None:
        """
        Train the probe on the training data, evaluate on the validation data, and repeat, returning the best probe.
        """
        pass

    @abstractmethod
    def fit(self, prompts: List[str], targets: List[float], **kwargs) -> None:
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
    def get_probe_metadata(self) -> dict:
        """
        Return a dictionary of probe metadata for saving.
        Each probe type implements this with its own attributes.
        
        Returns:
            Dictionary containing probe-specific metadata
        """
        raise NotImplementedError(f"get_probe_metadata() not implemented for {self.__class__.__name__}")