
from llms_know_difficulty.config import *
from llms_know_difficulty.probe.base_probe import Probe
from llms_know_difficulty.probe.torch_probe import (
    TorchLayerProbe,
    TorchProbe,
    AttnLite,
    LinearThenMax,
    LinearThenSoftmax,
    LinearThenRollingMax,
) 
from llms_know_difficulty.probe.linear_eoi_probe import LinearEoiProbe

from llms_know_difficulty.probe.tfidf_probe import TfidfProbe
from llms_know_difficulty.config import LinearEOIProbeConfig, AttentionProbeConfig, TfidfProbeConfig, DEVICE

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
                'ProbeClass': AttnLite,
            }
            probe = TorchProbe(AttentionProbeConfig())
            return probe.setup(**probe_setup_args)
            
        elif probe_name == "linear_eoi_probe":

            probe_setup_args = {
                'model_name': kwargs.get('model'),
                'device': DEVICE,
            }
            probe = LinearEoiProbe(LinearEOIProbeConfig())
            print("Lets set up the probe ⚙️ ...")
            return probe.setup(**probe_setup_args)
        
        elif probe_name == "tfidf_probe":

            probe = TfidfProbe(TfidfProbeConfig())
            print("Lets set up the probe ⚙️ ...")
            return probe.setup()

        elif probe_name == "linear_then_max_probe":

            probe_setup_args = {
                'model_name': kwargs.get('model'),
                'device': DEVICE,
                'ProbeClass': LinearThenMax
            }

            probe = TorchProbe(LinearThenMaxProbeConfig())
            print("Lets set up the probe ⚙️ ...")
            return probe.setup(**probe_setup_args)

        elif probe_name == "linear_then_softmax_probe":

            probe_setup_args = {
                'model_name': kwargs.get('model'),
                'device': DEVICE,
                'ProbeClass': LinearThenSoftmax
            }

            probe = TorchProbe(LinearThenSoftmaxProbeConfig())
            print("Lets set up the probe ⚙️ ...")
            return probe.setup(**probe_setup_args)


        elif probe_name == "linear_then_rolling_max_probe":
            probe_setup_args = {
                'model_name': kwargs.get('model'),
                'device': DEVICE,
                'ProbeClass': LinearThenRollingMax
            }

            probe = TorchProbe(LinearThenRollingMaxProbeConfig())
            print("Lets set up the probe ⚙️ ...")
            return probe.setup(**probe_setup_args)

        elif probe_name == "layer_attn_probe":
            probe_setup_args = {
                'model_name': kwargs.get('model'),
                'device': DEVICE,
                'ProbeClass': AttnLite
            }
            probe = TorchLayerProbe(LayerAttnProbeConfig())
            print("Lets set up the probe ⚙️ ...")
            return probe.setup(**probe_setup_args)

        else:
            raise NotImplementedError(f"Probe {probe_name} not implemented")