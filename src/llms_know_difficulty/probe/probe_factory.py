
from llms_know_difficulty.config import *
from probe.base_probe import Probe
from probe.attn_probe import AttnProbe

class ProbeFactory:

    """
    Class which handles probe initialization, interfaces between the probe constructors and the configs
    loaded from config.py
    """

    @staticmethod
    def create_probe(self, probe_name: str, **kwargs) -> Probe:
        """
        Create a probe with a given name, load the config from the config.py file
        and do any other probe specific init steps that are needed
        """

        if probe_name == "attn_probe":

            probe_setup_args = {
                'model_name': kwargs.get('model_name'),
                # other huggingface loading args go here ...
            }

            return AttnProbe(ATTN_PROBE_CONFIG), probe_setup_args

            
        else:
            raise NotImplementedError(f"Probe {probe_name} not implemented")