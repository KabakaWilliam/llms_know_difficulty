from .deepseek_qwen import Qwen2Model
from .qwen2_5 import Qwen2_5Model
from .qwen3 import Qwen3Model
from .llama import LlamaModel
from .gemma_3 import Gemma3Model
from .openhands import OpenHandsModel
from .swe_agent import SWE_Agent
from .gpt_neox import GPTNeoXModel
from .olmo import OlmoModel

def get_supported_model_class(model_name, device_override=None):
    print(f"model_name in supported class: {model_name}")
    if "swe" in model_name.lower():
        return SWE_Agent(model_name)
    if "openhands" in model_name.lower():
        return OpenHandsModel(model_name)
    if "gemma-3" in model_name.lower():
        return Gemma3Model(model_name)
    elif "olmo" in model_name.lower():
        return OlmoModel(model_name)
    elif "distill-qwen" in model_name.lower():
        return Qwen2Model(model_name)
    elif "qwen3" in model_name.lower():
        return Qwen3Model(model_name)
    elif "qwen2.5" in model_name.lower() or "qwen2" in model_name.lower():
        return Qwen2_5Model(model_name)
    elif "neox" in model_name.lower() or "pythia" in model_name.lower() or "chessgpt" in model_name.lower():
        return GPTNeoXModel(model_name)
    elif "llama" in model_name.lower():
        return LlamaModel(model_name)
    else:
        raise Exception("No supported model found.")

