"""PIKA — Probe-Informed Knowledge-Aware routing and difficulty prediction.

Quick Start:
    from pika.router import ProbeRouter
    
    router = ProbeRouter(
        models=["Qwen/Qwen2.5-Math-7B-Instruct", "openai/gpt-oss-20b_low"],
        probe_dir="./probes",
        dataset="DigitalLearningGmbH_MATH-lighteval",
    )
    
    # Route to optimal model and generate
    response = router.generate("What is 2+2?")
    
    # Or just get routing decision
    best_model = router.route("What is 2+2?")
"""

from pika.router import ProbeRouter, ModelConfig, RouterConfig

__all__ = ["ProbeRouter", "ModelConfig", "RouterConfig"]
