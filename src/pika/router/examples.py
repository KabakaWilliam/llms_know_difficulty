"""
Example usage of the ProbeRouter.

This script demonstrates how to use the PIKA router for
probe-based LLM selection and generation.

Run from project root:
    python -m pika.router.examples
"""

from pathlib import Path
from pika.router import ProbeRouter, ModelConfig, RouterConfig


def _get_data_dir() -> Path:
    """Get the data directory (works from any cwd)."""
    # Try relative to this file first
    here = Path(__file__).parent.resolve()
    # examples.py is at src/pika/router/examples.py
    # data is at project_root/data
    # Go up 3 levels: router -> pika -> src -> project_root
    project_root = here.parent.parent.parent
    data_dir = project_root / "data"
    if data_dir.exists():
        return data_dir
    # Fallback to cwd-relative
    if Path("./data").exists():
        return Path("./data").resolve()
    if Path("../data").exists():
        return Path("../data").resolve()
    raise FileNotFoundError("Could not find data directory. Run from project root.")


# =============================================================================
# Example 1: Basic usage with model IDs
# =============================================================================
def basic_example():
    """Simple router with default settings."""
    
    router = ProbeRouter(
        models=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            # "openai/gpt-oss-20b_low",
        ],
        probe_dir=_get_data_dir(),
        dataset="DigitalLearningGmbH_MATH-lighteval",
        lambda_val=0.5,  # Balanced cost-accuracy tradeoff
    )
    
    # Route to best model and generate
    response = router.generate("What is the derivative of x^2 + 3x?")
    
    print(f"Selected model: {response.model_id}")
    print(f"Response: {response.text}")
    print(f"Estimated cost: ${response.cost:.4f}")
    print(f"Probe scores: {response.probe_scores}")


# =============================================================================
# Example 2: Routing only (no generation)
# =============================================================================
def routing_only_example():
    """Use the router just for model selection without generating."""
    
    router = ProbeRouter(
        models=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            # "openai/gpt-oss-20b_low",
            # "openai/gpt-oss-20b_high",
        ],
        probe_dir=_get_data_dir(),
        dataset="DigitalLearningGmbH_MATH-lighteval",
    )
    
    # Just get the routing decision
    best_model = router.route("Solve: x^2 - 5x + 6 = 0")
    print(f"Best model for this problem: {best_model}")
    
    # Get detailed scores
    best_model, probes, utilities = router.route(
        "Prove that sqrt(2) is irrational",
        return_scores=True,
    )
    print(f"\nPredicted success rates:")
    for model, score in probes.items():
        print(f"  {model}: {score:.1%}")
    
    print(f"\nUtility scores (with λ={router.config.lambda_val}):")
    for model, util in utilities.items():
        print(f"  {model}: {util:.4f}")


# =============================================================================
# Example 3: Custom costs
# =============================================================================
def custom_costs_example():
    """Configure custom costs per model."""
    
    router = ProbeRouter(
        models=[
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            # "openai/gpt-oss-20b_high",
        ],
        probe_dir=_get_data_dir(),
        dataset="DigitalLearningGmbH_MATH-lighteval",
        model_costs={
            "Qwen/Qwen2.5-Math-7B-Instruct": 0.002,  # $0.002 per query
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.10,         # $0.10 per query
        },
        lambda_val=0.7,  # Strongly prefer cheaper models
    )
    
    # Or update costs dynamically
    router.update_costs({
        "openai/gpt-oss-20b_high": 0.05,  # Price dropped!
    })
    
    best_model = router.route("What is 2+2?")
    print(f"Selected: {best_model}")


# =============================================================================
# Example 4: Advanced configuration
# =============================================================================
def advanced_example():
    """Use ModelConfig and RouterConfig for fine-grained control."""
    
    # Configure individual models
    models = [
        ModelConfig(
            model_id="Qwen/Qwen2.5-Math-7B-Instruct",
            probe_path="../data/results/Qwen/Qwen2.5-Math-7B-Instruct/...",
            cost=0.005,
            generation_params={"max_tokens": 2048, "temperature": 0.3},
        ),
        ModelConfig(
            model_id="openai/gpt-oss-20b_high",
            cost=0.10,
            generation_params={"max_tokens": 8192, "temperature": 0.7},
        ),
    ]
    
    # Configure router behavior
    config = RouterConfig(
        lambda_val=0.4,
        probe_model_type="linear_eoi_probe",
        probe_metric="majority_vote_is_correct",
        device="cuda",
        batch_size=32,
        normalize_costs=True,
        vllm_gpu_memory_utilization=0.85,
    )
    
    router = ProbeRouter(
        models=models,
        dataset="DigitalLearningGmbH_MATH-lighteval",
        config=config,
    )
    
    # Adjust lambda on the fly
    router.set_lambda(0.8)  # More cost-sensitive
    
    response = router.generate("Factor: x^4 - 16")
    print(response)


# =============================================================================
# Example 5: Batch routing
# =============================================================================
def batch_routing_example():
    """Route multiple prompts efficiently."""
    
    router = ProbeRouter(
        models=[
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "openai/gpt-oss-20b_low",
            "openai/gpt-oss-20b_medium",
            "openai/gpt-oss-20b_high",
        ],
        probe_dir=_get_data_dir(),
        dataset="DigitalLearningGmbH_MATH-lighteval",
    )
    
    prompts = [
        # Easy arithmetic
        "What is 2 + 2?",
        "Calculate 15 × 7",
        "What is 144 ÷ 12?",
        # Basic algebra
        "Solve for x: 3x + 5 = 14",
        "Solve the equation: 2x - 7 = 15",
        "Find x if 5x = 45",
        # Intermediate algebra
        "Factor: x^2 - 5x + 6",
        "Solve the quadratic: x^2 - 4x - 5 = 0",
        "Simplify: (x^2 - 9) / (x - 3)",
        # Geometry
        "What is the area of a circle with radius 5?",
        "Find the hypotenuse of a right triangle with legs 3 and 4",
        "Calculate the volume of a sphere with radius 3",
        # Harder problems
        "Prove that sqrt(2) is irrational",
        "Find all prime numbers less than 100",
        "Prove the Pythagorean theorem",
        # Competition-level
        "Prove that for any integer n, n^3 - n is divisible by 6",
        "Find the sum of all positive divisors of 360",
        "How many ways can you arrange the letters in MISSISSIPPI?",
        # Advanced
        "Evaluate the integral of x^2 * e^x dx",
        "Find the derivative of ln(x^2 + 1)",
    ]
    
    
    # Get routing decisions for all prompts
    selected_models = router.batch_route(prompts, lambda_val=0.5)
    
    for prompt, model in zip(prompts, selected_models):
        print(f"'{prompt[:30]}...' -> {model}")


# =============================================================================
# Example 6: Lambda sweep (for analysis)
# =============================================================================
def lambda_sweep_example():
    """Explore cost-accuracy tradeoff across lambda values."""
    
    router = ProbeRouter(
        models=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            # "openai/gpt-oss-20b_low",
            # "openai/gpt-oss-20b_high",
        ],
        probe_dir=_get_data_dir(),
        dataset="DigitalLearningGmbH_MATH-lighteval",
    )
    
    prompt = "Prove that the sum of angles in a triangle is 180 degrees"
    
    print(f"Prompt: {prompt[:50]}...")
    print(f"\nLambda sweep:")
    print("-" * 50)
    
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        best_model, probes, utils = router.route(
            prompt, lambda_val=lam, return_scores=True
        )
        print(f"λ={lam:.2f}: {best_model.split('/')[-1]:30s} (utility={utils[best_model]:.3f})")


# =============================================================================
# Example 7: Sequential loading (memory-efficient) with routing summary
# =============================================================================
def sequential_loading_example(lambda_val: float = 0.5):
    """
    Use sequential loading for memory-constrained environments.
    
    When using multiple large models, loading all probes at once may exceed
    GPU memory. Sequential loading loads one probe at a time, makes predictions,
    then unloads before loading the next.
    
    This example routes a batch of math problems and shows a summary of
    which models would be selected.
    
    Args:
        lambda_val: Cost-accuracy tradeoff (0.0 = accuracy only, 1.0 = cost only)
    """
    from collections import Counter
    from pika.router import RouterConfig
    
    config = RouterConfig(
        sequential_loading=True,  # Key setting for memory efficiency
        device="cuda",
        lambda_val=lambda_val,
    )
    
    router = ProbeRouter(
        models=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "Qwen/Qwen2.5-Math-7B-Instruct",
        ],
        probe_dir=_get_data_dir(),
        dataset="DigitalLearningGmbH_MATH-lighteval",
        config=config,
    )
    
    # Diverse math problems of varying difficulty
    prompts = [
        # Easy arithmetic
        "What is 2 + 2?",
        "Calculate 15 × 7",
        "What is 144 ÷ 12?",
        # Basic algebra
        "Solve for x: 3x + 5 = 14",
        "Solve the equation: 2x - 7 = 15",
        "Find x if 5x = 45",
        # Intermediate algebra
        "Factor: x^2 - 5x + 6",
        "Solve the quadratic: x^2 - 4x - 5 = 0",
        "Simplify: (x^2 - 9) / (x - 3)",
        # Geometry
        "What is the area of a circle with radius 5?",
        "Find the hypotenuse of a right triangle with legs 3 and 4",
        "Calculate the volume of a sphere with radius 3",
        # Harder problems
        "Prove that sqrt(2) is irrational",
        "Find all prime numbers less than 100",
        "Prove the Pythagorean theorem",
        # Competition-level
        "Prove that for any integer n, n^3 - n is divisible by 6",
        "Find the sum of all positive divisors of 360",
        "How many ways can you arrange the letters in MISSISSIPPI?",
        # Advanced
        "Evaluate the integral of x^2 * e^x dx",
        "Find the derivative of ln(x^2 + 1)",
    ]
    
    print(f"\n📊 Routing {len(prompts)} math problems...")
    print("-" * 60)
    
    # Track routing decisions
    routing_decisions = []
    
    for i, prompt in enumerate(prompts):
        best_model = router.route(prompt, lambda_val)
        routing_decisions.append(best_model)
        
        # Print progress every 5 prompts
        if (i + 1) % 5 == 0:
            print(f"  Routed {i + 1}/{len(prompts)} prompts...")
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("ROUTING SUMMARY")
    print("=" * 60)
    
    # Count decisions
    counts = Counter(routing_decisions)
    total = len(routing_decisions)
    
    # Sort by count (most selected first)
    sorted_models = sorted(counts.items(), key=lambda x: -x[1])
    
    print(f"\n{'Model':<45} {'Count':>6} {'%':>8}")
    print("-" * 60)
    for model, count in sorted_models:
        short_name = model.split('/')[-1]
        pct = 100 * count / total
        bar = "█" * int(pct / 5)  # Simple bar chart
        print(f"{short_name:<45} {count:>6} {pct:>7.1f}% {bar}")
    
    print("-" * 60)
    print(f"{'Total':<45} {total:>6}")
    
    # Show some example routing decisions
    print("\n📝 Sample routing decisions:")
    print("-" * 60)
    for prompt, model in list(zip(prompts, routing_decisions))[:10]:
        short_name = model.split('/')[-1]
        print(f"  '{prompt[:40]:40s}' → {short_name}")
    print("  ...")


# =============================================================================
# Example 8: Compare lambda values
# =============================================================================
def compare_lambda_values():
    """
    Compare how different lambda values affect routing decisions.
    
    - lambda=0.0: Pure accuracy mode (ignore cost, pick best model)
    - lambda=0.5: Balanced (trade off accuracy vs cost)
    - lambda=1.0: Pure cost mode (always pick cheapest)
    
    This helps understand the cost-accuracy tradeoff in practice.
    """
    from collections import Counter
    from pika.router import RouterConfig
    
    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    ]
    
    prompts = [
        "What is 2 + 2?",
        "Solve for x: 3x + 5 = 14",
        "Factor: x^2 - 5x + 6",
        "Find the hypotenuse of a right triangle with legs 3 and 4",
        "Prove that sqrt(2) is irrational",
        "Prove that for any integer n, n^3 - n is divisible by 6",
        "Evaluate the integral of x^2 * e^x dx",
        "Find the derivative of ln(x^2 + 1)",
    ]
    
    print("\n" + "=" * 70)
    print("LAMBDA VALUE COMPARISON")
    print("=" * 70)
    print("λ=0: maximize accuracy | λ=0.5: balanced | λ=1: minimize cost")
    print("=" * 70)
    
    for lambda_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        config = RouterConfig(
            sequential_loading=True,
            device="cuda",
            lambda_val=lambda_val,
        )
        
        router = ProbeRouter(
            models=models,
            probe_dir=_get_data_dir(),
            dataset="DigitalLearningGmbH_MATH-lighteval",
            config=config,
        )
        
        # Route all prompts
        routing_decisions = []
        for prompt in prompts:
            result = router.route(prompt)  # returns model_id string
            routing_decisions.append(result.split('/')[-1])
        
        # Count distribution
        counts = Counter(routing_decisions)
        
        print(f"\nλ = {lambda_val:.2f}:")
        for model in models:
            short_name = model.split('/')[-1]
            count = counts.get(short_name, 0)
            pct = 100 * count / len(prompts)
            bar = "█" * int(pct / 10)
            print(f"  {short_name:<35} {count:>2} ({pct:>5.1f}%) {bar}")


if __name__ == "__main__":
    # Run examples (comment out ones that need vLLM)
    print("=" * 60)
    print("PIKA Router Examples")
    print("=" * 60)
    
    # These examples work without vLLM (routing only)
    # Use sequential_loading_example() if you have limited GPU memory
    # compare_lambda_values()
    # sequential_loading_example(lambda_val=0.0)  # accuracy-focused
    # routing_only_example()
    batch_routing_example()
    # lambda_sweep_example()
    
    # These examples require vLLM for generation
    # basic_example()
    
    print("\nSee the example functions above for usage patterns.")
