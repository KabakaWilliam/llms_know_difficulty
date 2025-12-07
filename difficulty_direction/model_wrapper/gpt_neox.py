from operator import attrgetter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import ModelBase
from nnsight import LanguageModel


SYSTEM_PROMPT = """You are a helpful assistant."""


class GPTNeoXModel(ModelBase):
    """Model wrapper for GPT-NeoX based models (e.g., Pythia, GPT-NeoX-20B, etc.)"""
    
    def __init__(self, model_name):
        super().__init__(model_name, torch_dtype=torch.float16)

    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        tokenizer.padding_side = "left"

        # GPT-NeoX models typically don't have a pad token set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer
    
    def _load_model(self, model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        # Load the raw model to ensure that the configuration is available
        raw_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # Create the LanguageModel wrapper
        wrapped_model = LanguageModel(raw_model, tokenizer=tokenizer, dispatch=True, **kwargs)
        
        # Ensure config is accessible on the wrapped model
        if not hasattr(wrapped_model, 'config') or wrapped_model.config is None:
            wrapped_model.config = raw_model.config
            
        return wrapped_model 

    def _get_system_message(self):
        # GPT-NeoX models (especially base models) may not use system prompts
        # Setting system_role to False to use simpler formatting
        self.system_role = False
        return SYSTEM_PROMPT
    
    def _get_model_block_modules(self):
        # GPT-NeoX uses gpt_neox.layers instead of model.layers
        return attrgetter("gpt_neox.layers")(self.model)
    
    def _get_attn_modules(self):
        if self.model_block_modules is None:
            return []
        # GPT-NeoX uses 'attention' instead of 'self_attn'
        return [attrgetter("attention")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        if self.model_block_modules is None:
            return []
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]
