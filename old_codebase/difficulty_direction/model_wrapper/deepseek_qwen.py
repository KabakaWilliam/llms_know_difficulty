from operator import attrgetter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import ModelBase
from nnsight import LanguageModel



SYSTEM_PROMPT = """You are a helpful assistant."""
#Added think token as a post instruction token. 
CHAT_TEMPLATE = """\
{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>' }}\n\
{% elif message['role'] == 'user' %}\n{{ '<|im_start|>user\n' + message['content'] + '<|im_end|> '}}\n\
{% elif message['role'] == 'assistant' %}\n{{ '<|im_start|>assistant\n'  + message['content'] + '<|im_end|>' }}\n\
{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|im_start|>assistant<think>' }}\n{% endif %}\n{% endfor %}\
"""

class Qwen2Model(ModelBase):
    def __init__(self, model_name):
        super().__init__(model_name, torch_dtype=torch.float16)

    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, clean_up_tokenization_spaces=True, **kwargs)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.chat_template = CHAT_TEMPLATE
        return tokenizer
    
    def _load_model(self, model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        # First load the raw model to ensure config is available
        raw_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map=self.device)
        
        # Create the LanguageModel wrapper
        wrapped_model = LanguageModel(raw_model, tokenizer=tokenizer, dispatch=True, **kwargs)
        
        # Ensure config is accessible on the wrapped model
        if not hasattr(wrapped_model, 'config') or wrapped_model.config is None:
            wrapped_model.config = raw_model.config
            
        return wrapped_model 

    
    def _get_system_message(self):
        self.system_role = True
        return SYSTEM_PROMPT
    
    # def _get_refusal_toks(self):
    #     return QWEN_REFUSAL_TOKS
    
    def _get_model_block_modules(self):
        return attrgetter("model.layers")(self.model)
    
    def _get_attn_modules(self):
        if self.model_block_modules is None:
            return []
        return [attrgetter("self_attn")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        if self.model_block_modules is None:
            return []
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]