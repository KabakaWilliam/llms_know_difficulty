"""
PIKA Router — Probe-based model routing for optimal cost-accuracy tradeoffs.

Usage:
    from pika.router import ProbeRouter
    
    router = ProbeRouter(
        models=["Qwen/Qwen2.5-Math-7B-Instruct", "openai/gpt-oss-20b_low"],
        probe_dir="./probes",
    )
    response = router.generate("What is 2+2?")
    
    # Or just get routing decision:
    model_name = router.route("What is 2+2?")
"""

from pika.router.probe_router import ProbeRouter
from pika.router.config import (
    ModelConfig, 
    RouterConfig, 
    get_gen_str, 
    GEN_STR_CONFIGS,
    FALLBACK_MODEL_COSTS,
)

__all__ = [
    "ProbeRouter", 
    "ModelConfig", 
    "RouterConfig", 
    "get_gen_str",
    "GEN_STR_CONFIGS",
    "FALLBACK_MODEL_COSTS",
]
