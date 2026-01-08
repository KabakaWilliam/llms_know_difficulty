from typing import List
from base_probe import Probe

class SklearnProbe(Probe):
    def __init__(self, config):
        super().__init__(config)

    def name(self) -> str:
        return "sklearn_probe"

    def init_model(self, config: dict):
        pass

    def setup(self, model_name: str):
        pass

    def fit(self, prompts: List[str], targets: List[float]) -> None:
        return super().fit(prompts, targets)
    
    def predict(self, prompts: List[str]):
        return super().predict(prompts)