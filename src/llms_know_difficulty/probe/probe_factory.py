
from llms_know_difficulty.config import *
from probe.base_probe import Probe
from probe.attn_probe import AttnProbe
from probe.sklearn_probe import SklearnProbe
from llms_know_difficulty.config import SKLEARN_PROBE_CONFIG, AttentionProbeConfig, DEVICE

class ProbeFactory:

    """
    Class which handles probe initialization, interfaces between the probe constructors and the configs
    loaded from config.py
    """

    @staticmethod
    def create_probe(probe_name: str, **kwargs) -> Probe:
        """
        Create a probe with a given name, load the config from the config.py file
        and do any other probe specific init steps that are needed
        """

        if probe_name == "attn_probe":
            probe_setup_args = {
                'model_name': kwargs.get('model'),
                'device': DEVICE,
                # other huggingface loading args go here ...
            }
            probe = AttnProbe(AttentionProbeConfig())
            return probe.setup(**probe_setup_args)
            
        elif probe_name == "sklearn_probe":

            probe_setup_args = {
                'model_name': kwargs.get('model_name'),
            }
            probe = SklearnProbe(SKLEARN_PROBE_CONFIG())
            return probe.setup(**probe_setup_args)
        

            
        else:
            raise NotImplementedError(f"Probe {probe_name} not implemented")