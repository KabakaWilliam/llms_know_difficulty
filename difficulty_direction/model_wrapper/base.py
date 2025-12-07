import os, warnings
from tqdm import tqdm
from operator import attrgetter
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torchtyping import TensorType
from transformers import AutoTokenizer, BatchEncoding
import nnsight
from nnsight import LanguageModel
from ..utils import ceildiv, chunks, orthogonal_rejection

# Turn off annoying warning messages
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if torch.cuda.is_available():
    device = "auto"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class ModelBase(ABC):
    def __init__(self, model_name: str, device_override: Optional[str] = None, **model_kwargs):
        self.model_name = model_name

        self.system_role = False
        self.tokenizer = self._load_tokenizer(model_name)
        self.system_message = self._get_system_message()
        self.eoi_toks = self._get_eoi_toks()
        # self.refusal_toks = self._get_refusal_toks()

        # Allow device override
        self.device = device_override if device_override else device
        self.model = self._load_model(model_name, self.tokenizer, **model_kwargs)
        
        # Debug: Check if model and config loaded properly
        if self.model is None:
            raise ValueError(f"Failed to load model: {model_name}")
        if not hasattr(self.model, 'config') or self.model.config is None:
            raise ValueError(f"Model config is None for model: {model_name}")
        
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        self.model_block_modules = self._get_model_block_modules()
        self.attn_modules = self._get_attn_modules()
        self.mlp_modules = self._get_mlp_modules()
        self.lm_head_module = attrgetter("lm_head")(self.model)
        self.intervene_direction = None
        self.actAdd_layer = None

    def _load_model(self, model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        # Check if model is > 70B parameters and should be quantized
        should_quantize = self._should_quantize_model(model_name)
        
        # if should_quantize:
        #     print(f"Loading large model ({model_name}) with 4-bit quantization")
        #     try:
        #         # Add quantization config for large models
        #         quantization_kwargs = {
        #             'load_in_4bit': True,
        #             'bnb_4bit_compute_dtype': torch.float16,
        #             'bnb_4bit_use_double_quant': True,
        #             'bnb_4bit_quant_type': 'nf4'
        #         }
        #         kwargs.update(quantization_kwargs)
        #     except Exception as e:
        #         print(f"Warning: Could not apply quantization for {model_name}: {e}")
        #         print("Loading model without quantization...")
        
        return LanguageModel(
            model_name, tokenizer=tokenizer, device_map=device, 
            dispatch=True, trust_remote_code=True, **kwargs)
    
    def _should_quantize_model(self, model_name: str) -> bool:
        """Determine if a model should be quantized based on its size (>70B parameters)"""
        model_name_lower = model_name.lower()
        
        # Check for explicit size indicators in model name
        size_indicators = [
            '70b', '72b', '80b', '90b', '100b', '110b', '120b', '130b', '140b', '150b',
            '70B', '72B', '80B', '90B', '100B', '110B', '120B', '130B', '140B', '150B'
        ]
        
        for size in size_indicators:
            if size in model_name:
                return True
        
        # Additional check for known large model families
        large_model_patterns = [
            'llama-2-70b', 'llama-3-70b', 'llama-3.1-70b', 'llama-3.2-70b',
            'code-llama-70b', 'qwen-72b', 'qwen2-72b', 'mixtral-8x22b'
        ]
        
        for pattern in large_model_patterns:
            if pattern in model_name_lower:
                return True
        
        return False
    

    
    @abstractmethod
    def _load_tokenizer(self, model_name, **kwargs) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_system_message(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass
    
    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass
    
    def _get_eoi_toks(self) -> str:
        '''Get post instruction tokens'''
        return self.tokenizer.encode(self.apply_chat_template(["{instruction}"])[0].split("{instruction}")[-1], add_special_tokens=False)

    def set_dtype(self, *vars):
        if len(vars) == 1:
            return vars[0].to(self.model.dtype)
        else:
            return (var.to(self.model.dtype) for var in vars)

    def set_intervene_direction(self, direction: TensorType["hidden_size"]):
        '''Set the default direction for intervention'''
        self.intervene_direction = self.set_dtype(direction)

    def set_actAdd_intervene_layer(self, layer: int):
        '''Set the default model layer for activation addition'''
        self.actAdd_layer = layer

    def tokenize(self, prompts: Union[List[str], BatchEncoding]):
        if isinstance(prompts, BatchEncoding):
            return prompts
        else:
            return self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    
    def get_token_ids(self, input: str) -> TensorType[-1]:
        if hasattr(self.tokenizer, "add_prefix_space") and self.tokenizer.add_prefix_space is True:
            return torch.tensor(self.tokenizer(input.lstrip(), add_special_tokens=False).input_ids)
        return torch.tensor(self.tokenizer(input, add_special_tokens=False).input_ids)
    
    def _has_chat_template(self) -> bool:
        """Check if the tokenizer has a chat template available"""
        return hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
    
    def _fallback_chat_format(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        """Fallback chat formatting for models without chat templates
        
        Provides multiple fallback strategies for different model types:
        1. Standard conversational format
        2. Simple prompt format for base models
        3. Instruction-following format
        """
        formatted_text = ""
        
        # Detect model type for better formatting
        model_name_lower = self.model_name.lower()
        is_base_model = any(keyword in model_name_lower for keyword in ['base', 'foundation', 'pretrained'])
        is_instruct_model = any(keyword in model_name_lower for keyword in ['instruct', 'chat', 'assistant'])
        
        has_system = any(msg["role"] == "system" for msg in messages)
        
        # Strategy 1: For single-turn base models, use simple format
        if (is_base_model or not is_instruct_model) and not has_system and len(messages) == 1 and messages[0]["role"] == "user":
            formatted_text = messages[0]["content"]
            if add_generation_prompt:
                formatted_text += "\n\nAnswer:"
            return formatted_text
        
        # Strategy 2: Standard conversational format for instruct/chat models
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_text += f"System: {content}\n\n"
            elif role == "user":
                if is_instruct_model:
                    formatted_text += f"### Instruction:\n{content}\n\n"
                else:
                    formatted_text += f"User: {content}\n"
            elif role == "assistant":
                if is_instruct_model:
                    formatted_text += f"### Response:\n{content}\n\n"
                else:
                    formatted_text += f"Assistant: {content}\n"
        
        # Add generation prompt
        if add_generation_prompt:
            if not messages or messages[-1]["role"] != "assistant":
                if is_instruct_model:
                    formatted_text += "### Response:\n"
                else:
                    formatted_text += "Assistant: "
        
        return formatted_text
    
    def apply_chat_template(self, instructions: Union[str, List[str]], outputs: Optional[List[str]] = None, use_system_prompt: Optional[bool] = False) -> List[str]:
        """Apply chat template to instructions with robust fallback for models without chat templates.
        
        This method handles the common error:
        "ValueError: Cannot use chat template functions because tokenizer.chat_template is not set"
        
        Fallback strategies:
        - Base models: Simple format with direct prompt + "Answer:"
        - Instruct models: Structured format with "### Instruction:" and "### Response:"
        - Chat models: Conversational format with "User:" and "Assistant:" 
        
        Args:
            instructions: Single instruction or list of instructions
            outputs: Optional list of expected outputs to append
            use_system_prompt: Whether to include system message
            
        Returns:
            List of formatted prompts ready for the model
        """
        if isinstance(instructions, str):
            instructions = [instructions]
        
        prompts = []
        for i in range(len(instructions)):
            messages = []
            inputs = instructions[i]

            if self.system_message is not None and use_system_prompt:
                if self.system_role:
                    messages.append({"role": "system", "content": self.system_message})
                else:
                    inputs = self.system_message + " " + instructions[i]

            messages.append({"role": "user", "content": inputs})
            
            # Try to use chat template, fallback to manual formatting if not available
            try:
                if self._has_chat_template():
                    inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    # No chat template available, use fallback
                    print(f"Info: Model '{self.model_name}' has no chat template, using fallback formatting")
                    inputs = self._fallback_chat_format(messages, add_generation_prompt=True)
            except (ValueError, AttributeError) as e:
                # Fallback if chat template fails for any reason
                print(f"Warning: Chat template failed for model '{self.model_name}' ({e}), using fallback formatting")
                inputs = self._fallback_chat_format(messages, add_generation_prompt=True)
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Error: Unexpected error with chat template for model '{self.model_name}' ({e}), using fallback formatting")
                inputs = self._fallback_chat_format(messages, add_generation_prompt=True)

            if inputs[-1] not in ["\n", " "]:
                inputs += " "

            if outputs is not None:
                inputs += outputs[i]
            prompts.append(inputs)

        return prompts
    
    def get_activations(
        self, layers: Union[List[int], int], prompts: Union[str, List[str], BatchEncoding],
        positions: Optional[List[int]] = [-1]
    ) -> List[TensorType["n_prompt", "n_pos", "hidden_size"]]:
        """Get output activations of prompts given a specific layer(s) and token position(s)"""
        if isinstance(layers, int):
            layers = [layers]

        all_acts = []
        with self.model.trace(prompts) as tracer:
            for layer in layers:
                if positions is None:
                    acts = self.model_block_modules[layer].input
                else:
                    acts = self.model_block_modules[layer].input[:, positions, :]

                acts = acts.detach().to("cpu").to(torch.float64).unsqueeze(0).save()
                all_acts.append(acts)

            self.model_block_modules[layer].output.stop() # Early stopping
        return torch.vstack(all_acts)
    
    def _prepare_act_add_inputs(
        self, prompts: Union[str, List[str], BatchEncoding],
        steering_vec: TensorType["hidden_size"], layer: int, 
        coeffs: Union[float, List[float], TensorType[-1]],
    ):
        inputs = self.tokenize(prompts)
        coeffs = torch.tensor(coeffs)

        if coeffs.dim() != 0:
            coeffs = coeffs[:, None, None]
        if steering_vec is None:
            steering_vec = self.intervene_direction
        if layer is None:
            layer = self.actAdd_layer

        steering_vec, coeffs = self.set_dtype(steering_vec, coeffs)
        return inputs, steering_vec, layer, coeffs
    
    def activation_addition(
        self, prompts: Union[str, List[str], BatchEncoding], 
        steering_vec: Optional[TensorType["hidden_size"]] = None, 
        layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        
        inputs, steering_vec, layer, coeffs = self._prepare_act_add_inputs(prompts, steering_vec, layer, coeffs)

        with self.model.trace(inputs) as tracer:
            self.model_block_modules[layer].input += (steering_vec * coeffs)
            logits = self.lm_head_module.output.detach().to("cpu").save()
        return logits
    
    def _prepare_ablation_inputs(
        self, prompts: Union[str, List[str], BatchEncoding],
        direction: TensorType["hidden_size"]
    ):
        if direction is None:
            direction = self.intervene_direction

        inputs = self.tokenize(prompts)
        unit_direction = direction / (direction.norm(dim=-1) + 1e-8)
        unit_direction = self.set_dtype(unit_direction)
        return inputs, unit_direction
    
    def directional_ablation(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        
        inputs, unit_direction = self._prepare_ablation_inputs(prompts, direction)

        with self.model.trace(inputs) as tracer:
            for layer in range(self.n_layers):
                acts = nnsight.apply(orthogonal_rejection, *(self.model_block_modules[layer].input, unit_direction))
                self.model_block_modules[layer].input = acts

                act_post_attn = nnsight.apply(orthogonal_rejection, *(self.attn_modules[layer].output[0], unit_direction))
                self.attn_modules[layer].output = (act_post_attn,) + self.attn_modules[layer].output[1:]

                act_post_mlp = nnsight.apply(orthogonal_rejection, *(self.mlp_modules[layer].output, unit_direction))
                self.mlp_modules[layer].output = act_post_mlp

            logits = self.lm_head_module.output.detach().to("cpu").save()
        return logits
    
    def get_logits(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        intervention_method: Optional[int] = None, 
        steering_layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        '''Get output logits of all token positions'''

        if intervention_method == "actadd":
            logits = self.activation_addition(prompts, direction, steering_layer, coeffs=coeffs)
        elif intervention_method == "ablation":
            logits = self.directional_ablation(prompts, direction)
        else:
            logits = self.model.trace(prompts, trace=False).logits.detach().to("cpu")
        logits = logits.to(torch.float64)
        return logits
    
    def get_last_position_logits(
        self, instructions: List[str], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        intervention_method: Optional[int] = None, 
        steering_layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0, 
        batch_size: Optional[int] = 16
    ) -> TensorType["n_instructions", "vocab_size"]:
        '''Get the logits of the last token position'''
        total = ceildiv(len(instructions), batch_size)
        if total > 5:
            pbar = tqdm(chunks(instructions, batch_size), total=total, desc="Getting last position logits")
        else:
            pbar = chunks(instructions, batch_size)

        last_pos_logits = []
        for instruction_batch in pbar:
            prompts = self.apply_chat_template(instructions=instruction_batch)
            logits = self.get_logits(prompts, direction, intervention_method, steering_layer, coeffs)[:, -1, :]
            
            if last_pos_logits is None:
                last_pos_logits = logits
            else:
                last_pos_logits.append(logits)

        return torch.vstack(last_pos_logits)
    
    def _generate_act_add(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0,
        max_new_tokens: Optional[int] = 10,
        do_sample: Optional[bool] = False, **kwargs
    ) -> TensorType["n_prompt", "seq_len"]:
        """Text generation with activation addition"""
        inputs, direction, layer, coeffs = self._prepare_act_add_inputs(prompts, direction, layer, coeffs)

        with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            self.model_block_modules[layer].input += (direction * coeffs)

            for _ in range(max_new_tokens - 1):
                acts = self.model_block_modules[layer].next().input[:, -1, :]
                if isinstance(coeffs, torch.Tensor):
                    coeff_value = coeffs.squeeze(-1) if coeffs.dim() > 0 else coeffs
                else:
                    coeff_value = coeffs
                self.model_block_modules[layer].input[:, -1, :] = acts + (direction * coeff_value)

            outputs = self.model.generator.output.detach().to("cpu").save()
        # Try different ways to get the tensor value
        if hasattr(outputs, 'value'):
            return outputs.value
        else:
            return outputs
    
    def _generate_abalation(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        max_new_tokens: Optional[int] = 10,
        do_sample: Optional[bool] = False, **kwargs
    )-> TensorType["n_prompt", "seq_len"]:
        """Text generation with directional ablation"""
        inputs, unit_direction = self._prepare_ablation_inputs(prompts, direction)

        with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            for layer in range(self.n_layers):
                acts = nnsight.apply(orthogonal_rejection, *(self.model_block_modules[layer].input, unit_direction))
                self.model_block_modules[layer].input = acts

                act_post_attn = nnsight.apply(orthogonal_rejection, *(self.attn_modules[layer].output[0], unit_direction))
                self.attn_modules[layer].output = (act_post_attn,) + self.attn_modules[layer].output[1:]

                act_post_mlp = nnsight.apply(orthogonal_rejection, *(self.mlp_modules[layer].output, unit_direction))
                self.mlp_modules[layer].output = act_post_mlp
            
                for _ in range(max_new_tokens - 1):
                    act_pre = nnsight.apply(orthogonal_rejection, *(self.model_block_modules[layer].next().input[:, -1, :], unit_direction))
                    self.model_block_modules[layer].input[:, -1, :] = act_pre

                    act_post_attn = nnsight.apply(orthogonal_rejection, *(self.attn_modules[layer].next().output[0][:, -1, :], unit_direction))
                    self.attn_modules[layer].output[0][:, -1, :] = act_post_attn

                    act_post_mlp = nnsight.apply(orthogonal_rejection, *(self.mlp_modules[layer].next().output[:, -1, :], unit_direction))
                    self.mlp_modules[layer].output[:, -1, :] = act_post_mlp

            outputs = self.model.generator.output.detach().to("cpu").save()
        # Try different ways to get the tensor value
        if hasattr(outputs, 'value'):
            return outputs.value
        else:
            return outputs
    
    def generate(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        intervention_method: Optional[str] = None, 
        steering_layer: Optional[int] = None, 
        coeffs: Union[float, List[float], TensorType[-1]] = 1.0,
        max_new_tokens: int = 10,
        do_sample: bool = False,
        **kwargs
    ) -> TensorType["n_prompt", "seq_len"]:
        if intervention_method =="actadd":
            return self._generate_act_add(prompts, direction, steering_layer, coeffs, max_new_tokens, do_sample, **kwargs)
        elif intervention_method == "ablation":
            return self._generate_abalation(prompts, direction, max_new_tokens, do_sample, **kwargs)
        
        inputs = self.tokenize(prompts)
        with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            output_proxy = self.model.generator.output.detach().to("cpu").save()
        
        # Access the saved value after the context
        return output_proxy.value if hasattr(output_proxy, 'value') else output_proxy
    
    def generate_completions(
        self, instructions: List[str], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        intervention_method: Optional[str] = None, 
        steering_layer: Optional[int] = None, 
        coeffs: Union[float, List[float], TensorType[-1]] = 1.0, 
        batch_size: int = 16,
        max_new_tokens: int = 10, 
        do_sample: bool = False,
        return_prompt: bool = False, 
        **generation_kwargs
    ) -> List[str]:
        '''Run text generation in batch with given intervention method and decode the outputs to strings'''
        completions = []
        total = ceildiv(len(instructions), batch_size)

        for instruction_batch in tqdm(chunks(instructions, batch_size), total=total, desc="Generating completions"):
            formatted_prompts = self.apply_chat_template(instruction_batch)
            inputs = self.tokenize(formatted_prompts)
            with torch.no_grad():
                outputs = self.generate(
                    inputs, direction, intervention_method, steering_layer, coeffs, 
                    max_new_tokens, do_sample, **generation_kwargs
                )
            
            if return_prompt:
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                seq_len = inputs.input_ids.shape[1]
                decoded_outputs = self.tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)

            completions.extend(decoded_outputs)
            torch.cuda.empty_cache()

        return completions
    