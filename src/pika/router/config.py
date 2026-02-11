"""
Configuration classes for the ProbeRouter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Generation-string configs (used to locate training data files)
# ──────────────────────────────────────────────────────────────────────────────
GEN_STR_CONFIGS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-20b_low":                     {"maxlen": 131072, "k": 5, "temp": 1.0},
    "openai/gpt-oss-20b_medium":                  {"maxlen": 131072, "k": 5, "temp": 1.0},
    "openai/gpt-oss-20b_high":                    {"maxlen": 131072, "k": 5, "temp": 1.0},
    "Qwen/Qwen2.5-1.5B-Instruct":                 {"maxlen": 3000,   "k": 5, "temp": 0.7},
    "Qwen/Qwen2.5-Math-1.5B-Instruct":            {"maxlen": 3000,   "k": 5, "temp": 0.7},
    "Qwen/Qwen2.5-Math-7B-Instruct":              {"maxlen": 3000,   "k": 5, "temp": 0.7},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct":           {"maxlen": 4096,   "k": 1, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-3B-Instruct":             {"maxlen": 4096,   "k": 5, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-7B-Instruct":             {"maxlen": 4096,   "k": 5, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-14B-Instruct":            {"maxlen": 4096,   "k": 1, "temp": 0.2},
    "Qwen/Qwen2.5-Coder-32B-Instruct":            {"maxlen": 4096,   "k": 1, "temp": 0.2},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":    {"maxlen": 32768,  "k": 5, "temp": 0.6},
}


def get_gen_str(model_name: str) -> str:
    """Return the generation string for a model (used to locate data files)."""
    cfg = GEN_STR_CONFIGS.get(model_name, GEN_STR_CONFIGS["openai/gpt-oss-20b_low"])
    return f"maxlen_{cfg['maxlen']}_k_{cfg['k']}_temp_{cfg['temp']}"


# ──────────────────────────────────────────────────────────────────────────────
# Fallback model costs (used when training data not available)
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_MODEL_COSTS: dict[str, float] = {
    # Qwen models
    "Qwen/Qwen2.5-1.5B-Instruct": 0.001,
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 0.001,
    "Qwen/Qwen2.5-Math-7B-Instruct": 0.005,
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": 0.001,
    "Qwen/Qwen2.5-Coder-3B-Instruct": 0.002,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 0.005,
    "Qwen/Qwen2.5-Coder-14B-Instruct": 0.010,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 0.020,
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.008,
    # OpenAI / GPT-OSS variants
    "openai/gpt-oss-20b_low": 0.010,
    "openai/gpt-oss-20b_medium": 0.025,
    "openai/gpt-oss-20b_high": 0.050,
}

# Default generation parameters per model
DEFAULT_GEN_PARAMS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-20b_low": {"max_tokens": 131072, "n": 5, "temperature": 1.0},
    "openai/gpt-oss-20b_medium": {"max_tokens": 131072, "n": 5, "temperature": 1.0},
    "openai/gpt-oss-20b_high": {"max_tokens": 131072, "n": 5, "temperature": 1.0},
    "Qwen/Qwen2.5-1.5B-Instruct": {"max_tokens": 3000, "n": 5, "temperature": 0.7},
    "Qwen/Qwen2.5-Math-1.5B-Instruct": {"max_tokens": 3000, "n": 5, "temperature": 0.7},
    "Qwen/Qwen2.5-Math-7B-Instruct": {"max_tokens": 3000, "n": 5, "temperature": 0.7},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {"max_tokens": 4096, "n": 1, "temperature": 0.2},
    "Qwen/Qwen2.5-Coder-3B-Instruct": {"max_tokens": 4096, "n": 5, "temperature": 0.2},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"max_tokens": 4096, "n": 5, "temperature": 0.2},
    "Qwen/Qwen2.5-Coder-14B-Instruct": {"max_tokens": 4096, "n": 1, "temperature": 0.2},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"max_tokens": 4096, "n": 1, "temperature": 0.2},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"max_tokens": 32768, "n": 5, "temperature": 0.6},
}


@dataclass
class ModelConfig:
    """Configuration for a single model in the router pool.
    
    Attributes:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-Math-7B-Instruct")
        probe_path: Path to trained probe checkpoint (optional, auto-discovered if None)
        cost: Estimated cost per query in USD (optional, uses defaults if None)
        generation_params: Override generation parameters (max_tokens, temperature, etc.)
    """
    model_id: str
    probe_path: Path | str | None = None
    cost: float | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.probe_path is not None:
            self.probe_path = Path(self.probe_path)
    
    def get_cost(self) -> float:
        """Return model cost, falling back to defaults if not set."""
        if self.cost is not None:
            return self.cost
        return FALLBACK_MODEL_COSTS.get(self.model_id, 0.01)  # Default to $0.01
    
    def get_generation_params(self) -> dict[str, Any]:
        """Return generation params, merging with defaults."""
        defaults = DEFAULT_GEN_PARAMS.get(self.model_id, {
            "max_tokens": 4096,
            "n": 1,
            "temperature": 0.7,
        })
        return {**defaults, **self.generation_params}


@dataclass
class RouterConfig:
    """Global configuration for the ProbeRouter.
    
    Attributes:
        lambda_val: Cost-accuracy tradeoff parameter.
            0.0 = maximize accuracy (ignore cost)
            0.5 = balanced
            1.0 = strongly penalize cost
        probe_model_type: Type of probe to use ("linear_eoi_probe", "tfidf_probe")
        probe_metric: Metric the probe was trained on ("majority_vote_is_correct", "pass_at_k")
        device: Device for inference ("cuda", "cpu", "auto")
        batch_size: Batch size for probe predictions
        normalize_costs: Whether to min-max normalize costs across models
        sequential_loading: Load probes one at a time and unload after prediction.
            Reduces peak memory usage but is slower. Recommended when using
            multiple large models that don't fit in GPU memory simultaneously.
        vllm_gpu_memory_utilization: GPU memory fraction for vLLM (0.0-1.0)
        vllm_tensor_parallel_size: Number of GPUs for tensor parallelism
    """
    lambda_val: float = 0.5
    probe_model_type: str = "linear_eoi_probe"
    probe_metric: str = "majority_vote_is_correct"
    device: str = "auto"
    batch_size: int = 16
    normalize_costs: bool = True
    sequential_loading: bool = False
    vllm_gpu_memory_utilization: float = 0.7
    vllm_tensor_parallel_size: int = 1
    
    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not 0.0 <= self.lambda_val <= 1.0:
            raise ValueError(f"lambda_val must be in [0, 1], got {self.lambda_val}")
