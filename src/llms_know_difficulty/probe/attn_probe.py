from base_probe import Probe

class AttnProbe(Probe):
    def __init__(self, config):
        super().__init__(config)

    def name(self) -> str:
        return "attn_probe"

    def init_model(self, config: dict):
        pass

    def setup(self, model_name: str):
        pass