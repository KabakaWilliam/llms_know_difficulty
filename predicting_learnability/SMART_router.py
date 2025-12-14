"""
SMART Router - Cascading Model Selection Based on Predicted Success Rates

This is a practical routing system that doesn't waste compute on models that won't help.
Instead of throwing every hard question at expensive models, we intelligently cascade:
- Start with the cheapest model
- Only escalate if we think the next model can actually solve it
- Abstain if none of them can handle it (better to say "IDK" than waste tokens)

"""

import pandas as pd
import numpy as np
import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from vllm import LLM, SamplingParams
from collections import Counter

from router_components import (
    SolverResult,
    EvaluationMetrics,
    calculate_cost_metrics,
    calculate_pass_at_k,
    get_strategy_color
)
from utils import parse_answers
from math_verify import parse, verify

os.environ["CUDA_VISIBLE_DEVICES"] = f"2"

# ============================================================================
# NOTIFICATION CONFIGURATION
# ============================================================================

NOTIFICATION_CONFIG = {
    "enabled": True,
    "url": "https://ntfy.sh/router_runs_lugoloobi"
}

def send_notification(title: str, message: str):
    """Send notification to ntfy.sh"""
    if not NOTIFICATION_CONFIG.get("enabled", True):
        return
    
    try:
        requests.post(
            NOTIFICATION_CONFIG["url"],
            data=message.encode(encoding='utf-8'),
            headers={"Title": title}
        )
    except:
        pass  # Don't fail if notification doesn't work


# ============================================================================
# SMART ROUTER CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model in the cascade"""
    name: str                    # Model identifier (e.g., "Qwen2.5-Math-1.5B")
    display_name: str           # Human-readable name
    threshold: float            # Success rate threshold - use this model if success_rate >= threshold
    cost_multiplier: float      # Relative cost compared to base model
    max_tokens: int            # Max tokens for generation
    
    def __repr__(self):
        return f"{self.display_name} (threshold={self.threshold:.2f}, cost={self.cost_multiplier}x)"


@dataclass
class CascadeConfig:
    """Configuration for the full model cascade"""
    models: List[ModelConfig]    # Ordered list: cheapest to most expensive
    abstain_threshold: float     # If predicted_success < this, abstain with "IDK" (too hard)
    greedy_temp: float = 0.0
    sc_temp: float = 0.6
    num_sc_samples: int = 1      # Can enable SC for specific models
    probe_column: str = "predicted_difficulty_sigmoid"  # Misnomer: actually contains success rates!
    enable_abstain: bool = True  # If False, always route to a model (never abstain)
    
    def __post_init__(self):
        """Validate configuration"""
        assert len(self.models) > 0, "Need at least one model"
        assert 0 <= self.abstain_threshold <= 1, "Abstain threshold must be in [0, 1]"
        
        # Ensure models are sorted by cost (ascending)
        costs = [m.cost_multiplier for m in self.models]
        assert costs == sorted(costs), "Models should be ordered cheapest to most expensive"
    
    def get_model_for_question(self, predicted_success: float) -> Optional[ModelConfig]:
        """
        Determine which model should handle this question.
        
        Logic (for SUCCESS RATE scores, where high=easy, low=hard):
        1. If enable_abstain=True and predicted_success < abstain_threshold -> None (abstain)
        2. Otherwise, pick the CHEAPEST model where predicted_success >= model.threshold
        3. If no model threshold is met, use most expensive model (or abstain if enabled)
        
        Args:
            predicted_success: Probe's prediction of success rate (0.0=will fail, 1.0=will succeed)
        
        Returns:
            ModelConfig to use, or None if we should abstain
        """
        # Check if we should abstain entirely (success rate too low = too hard)
        if self.enable_abstain and predicted_success < self.abstain_threshold:
            return None
        
        # Find cheapest model that can handle this (success_rate >= threshold)
        # Models are ordered cheapest to most expensive
        for model in self.models:
            if predicted_success >= model.threshold:
                return model
        
        # If we get here, no model threshold is met
        # Use the most expensive model (last one) as a fallback
        return self.models[-1]


# ============================================================================
# SMART ROUTER IMPLEMENTATION
# ============================================================================

class SmartRouter:
    """
    Intelligent model cascade router.
    
    The whole point here is to not be dumb about compute:
    - Don't route easy questions to expensive models
    - Don't route impossible questions to any model
    - Only escalate when it actually makes sense
    """
    
    def __init__(self, config: CascadeConfig):
        self.config = config
        self.llm_cache: Dict[str, LLM] = {}
        
    def load_model(self, model_name: str, gpu_memory_util: float = 0.4) -> LLM:
        """Lazy load models as needed"""
        if model_name not in self.llm_cache:
            print(f"🔄 Loading model: {model_name}")
            self.llm_cache[model_name] = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_util
            )
            print(f"✅ Model loaded: {model_name}")
        return self.llm_cache[model_name]
    
    def solve_question(
        self,
        question_idx: int,
        prompt: str,
        ground_truth: str,
        predicted_success: float,
        model_config: Optional[ModelConfig] = None,
        use_sc: bool = False
    ) -> SolverResult:
        """
        Solve a single question with the assigned model.
        
        Args:
            question_idx: Question index
            prompt: Formatted prompt
            ground_truth: Answer for verification
            predicted_success: Probe prediction of success rate (0=fail, 1=succeed)
            model_config: Model to use (if None, use routing logic)
            use_sc: Whether to use self-consistency
        """
        # Determine which model to use
        if model_config is None:
            model_config = self.config.get_model_for_question(predicted_success)
        
        # Check if we should abstain
        if model_config is None:
            return SolverResult(
                question_idx=question_idx,
                used_sc=False,
                is_correct=0,
                final_answer="IDK",
                num_samples_used=0,
                responses=["IDK"],
                token_lengths=[0],
                predicted_score=predicted_success,
                individual_correct=[False]
            )
        
        # Load model
        llm = self.load_model(model_config.name)
        
        # Generate
        num_samples = self.config.num_sc_samples if use_sc else 1
        temp = self.config.sc_temp if use_sc else self.config.greedy_temp
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=model_config.max_tokens
        )
        
        prompts = [prompt] * num_samples
        outputs = llm.generate(prompts, params)
        
        responses = []
        token_lengths = []
        individual_correct = []
        
        for output in outputs:
            response = output.outputs[0].text
            token_length = len(output.outputs[0].token_ids)
            responses.append(response)
            token_lengths.append(token_length)
            
            # Check individual correctness
            parsed = parse_answers([response])
            if parsed and len(parsed[0]) > 0:
                answer = parsed[0][0]
                correct = verify(parse(f"${ground_truth}$"), answer)
                individual_correct.append(correct)
            else:
                individual_correct.append(False)
        
        # Final answer (majority vote if SC, otherwise just the response)
        if num_samples > 1:
            parsed_answers_list = parse_answers(responses)
            all_answers = []
            for parsed in parsed_answers_list:
                all_answers.extend(parsed)
            
            if all_answers:
                final_answer = Counter(all_answers).most_common(1)[0][0]
            else:
                final_answer = ""
        else:
            parsed = parse_answers([responses[0]])
            final_answer = parsed[0][0] if parsed and len(parsed[0]) > 0 else ""
        
        # Verify
        is_correct = int(verify(parse(f"${ground_truth}$"), final_answer))
        
        return SolverResult(
            question_idx=question_idx,
            used_sc=use_sc,
            is_correct=is_correct,
            final_answer=final_answer,
            num_samples_used=num_samples,
            responses=responses,
            token_lengths=token_lengths,
            predicted_score=predicted_success,
            individual_correct=individual_correct
        )
    
    def evaluate(
        self,
        df: pd.DataFrame,
        baseline_accuracy: float,
        baseline_cost: Optional[float] = None,
        baseline_tokens: Optional[int] = None
    ) -> Tuple[EvaluationMetrics, List[SolverResult]]:
        """
        Evaluate the SMART routing strategy on a dataset.
        
        Returns:
            Tuple of (metrics, results)
        """
        from tqdm import tqdm
        
        results = []
        routing_stats = {model.name: 0 for model in self.config.models}
        routing_stats["ABSTAIN"] = 0
        
        print("\n" + "="*80)
        print("🧠 SMART ROUTER EVALUATION")
        print("="*80)
        print(f"Models in cascade: {len(self.config.models)}")
        for model in self.config.models:
            print(f"  → {model}")
        print(f"Abstain threshold: {self.config.abstain_threshold:.2f}")
        print("="*80 + "\n")
        
        for idx in tqdm(range(len(df)), desc="Routing questions"):
            row = df.iloc[idx]
            predicted_success = row[self.config.probe_column]
            
            # Route to appropriate model
            model_config = self.config.get_model_for_question(predicted_success)
            
            # Track routing decision
            if model_config is None:
                routing_stats["ABSTAIN"] += 1
            else:
                routing_stats[model_config.name] += 1
            
            # Solve
            result = self.solve_question(
                question_idx=idx,
                prompt=row["formatted_prompt"],
                ground_truth=row["answer"],
                predicted_success=predicted_success,
                model_config=model_config,
                use_sc=False  # Can be made configurable per model
            )
            
            results.append(result)
        
        # Print routing stats
        print("\n" + "="*80)
        print("📊 ROUTING STATISTICS")
        print("="*80)
        total_questions = len(df)
        for key, count in routing_stats.items():
            pct = count / total_questions * 100
            print(f"  {key:40s}: {count:5d} ({pct:5.1f}%)")
        print("="*80 + "\n")
        
        # Calculate metrics
        overall_accuracy = np.mean([r.is_correct for r in results])
        accuracy_gain = overall_accuracy - baseline_accuracy
        
        # Calculate average samples and cost
        total_samples = sum(r.num_samples_used for r in results)
        avg_samples = total_samples / len(results)
        
        # Cost calculation (weighted by model usage)
        cost_metrics = calculate_cost_metrics(results, baseline_cost=baseline_cost)
        
        # Token metrics
        total_input_tokens = cost_metrics['total_input_tokens']
        total_output_tokens = cost_metrics['total_output_tokens']
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate token multiplier
        token_multiplier = None
        if baseline_tokens and baseline_tokens > 0:
            token_multiplier = total_tokens / baseline_tokens
        
        # Router quality metrics (simplified for SMART routing)
        # "Benefits from routing" = questions where we didn't use base model
        base_model_name = self.config.models[0].name
        routed_to_better = sum(1 for r in results if r.num_samples_used > 0 and 
                               routing_stats.get(base_model_name, 0) < len(results))
        router_precision = None  # Not directly applicable
        router_recall = None     # Not directly applicable
        router_accuracy = None   # Not directly applicable
        
        # Pass@k metrics
        pass_k_metrics = calculate_pass_at_k(results, max_k=self.config.num_sc_samples)
        
        # ROI calculation
        roi = None
        if cost_metrics['cost_increase'] and cost_metrics['cost_increase'] > 0:
            roi = accuracy_gain / cost_metrics['cost_increase']
        elif cost_metrics['cost_increase'] and cost_metrics['cost_increase'] < 0 and accuracy_gain > 0:
            roi = float('inf')  # Better AND cheaper!
        
        metrics = EvaluationMetrics(
            strategy_name="smart_cascade",
            overall_accuracy=overall_accuracy,
            baseline_accuracy=baseline_accuracy,
            accuracy_gain=accuracy_gain,
            fraction_routed_to_sc=0.0,  # Not using SC by default
            avg_samples_per_question=avg_samples,
            total_tokens=total_tokens,
            router_accuracy=router_accuracy,
            router_precision=router_precision,
            router_recall=router_recall,
            sc_subset_accuracy=None,
            greedy_subset_accuracy=None,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            input_cost=cost_metrics['input_cost'],
            output_cost=cost_metrics['output_cost'],
            total_cost=cost_metrics['total_cost'],
            cost_per_question=cost_metrics['cost_per_question'],
            num_correct=cost_metrics['num_correct'],
            cost_per_correct=cost_metrics['cost_per_correct'],
            baseline_cost=baseline_cost,
            cost_increase=cost_metrics['cost_increase'],
            roi=roi,
            baseline_tokens=baseline_tokens,
            token_multiplier=token_multiplier,
            pass_at_k=pass_k_metrics['pass_at_k'],
            avg_pass_at_1=pass_k_metrics['avg_pass_at_1'],
            avg_first_correct_sample=pass_k_metrics['avg_first_correct_sample'],
            total_correct_samples=pass_k_metrics['total_correct_samples'],
            oracle_best_case_accuracy=pass_k_metrics['oracle_best_case_accuracy'],
            oracle_best_case_gain=pass_k_metrics['oracle_best_case_accuracy'] - overall_accuracy,
            sample_utilization=overall_accuracy / pass_k_metrics['oracle_best_case_accuracy'] 
                              if pass_k_metrics['oracle_best_case_accuracy'] > 0 else 0.0,
        )
        
        return metrics, results


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

def get_example_cascade_tiny() -> CascadeConfig:
    """
    Example: Small model cascade for math problems.
    Use this as a template for your own configurations.
    
    IMPORTANT: Thresholds are for SUCCESS RATE scores (0=fail, 1=succeed)
    Note: probe_column is named 'predicted_difficulty_sigmoid' but actually contains success rates!
    
    - Small model handles success_rate >= 0.7 (easy questions, high chance of success)
    - Big model handles success_rate >= 0.4 (medium-hard questions)
    - If success_rate < 0.2, abstain (too hard, likely to fail)
    """
    return CascadeConfig(
        models=[
            ModelConfig(
                name="Qwen/Qwen2.5-Math-1.5B-Instruct",
                display_name="Qwen-Math-1.5B",
                threshold=0.7,  # Use for easy questions (success_rate >= 70%)
                cost_multiplier=1.0,  # Base cost
                max_tokens=3000
            ),
            ModelConfig(
                name="Qwen/Qwen2.5-Math-7B-Instruct",
                display_name="Qwen-Math-7B",
                threshold=0.0,  # Use for harder questions (success_rate >= 10%)
                cost_multiplier=4.0,  # ~4x more expensive
                max_tokens=3000
            ),
            # Add more models here as needed
        ],
        abstain_threshold=0.2,  # If success_rate < 20%, say "IDK" (too hard)
        greedy_temp=0.0,
        probe_column="predicted_difficulty_sigmoid",  # Misnomer: actually success rates!
        enable_abstain=False  # Disabled by default - route everything
    )


def get_example_cascade_with_reasoning() -> CascadeConfig:
    """
    Example: Cascade with reasoning models for harder questions.
    
    Uses success rate thresholds (0=fail, 1=succeed):
    - Small model: success_rate >= 0.8 (very easy, high confidence)
    - Medium model: success_rate >= 0.5 (medium difficulty)
    - Reasoning model: success_rate >= 0.2 (hard problems, use expensive reasoning)
    - Abstain: success_rate < 0.15 (impossible, too low success chance)
    """
    return CascadeConfig(
        models=[
            ModelConfig(
                name="Qwen/Qwen2.5-Math-1.5B-Instruct",
                display_name="Qwen-Math-1.5B",
                threshold=0.8,  # Very easy questions only (>80% success)
                cost_multiplier=1.0,
                max_tokens=3000
            ),
            ModelConfig(
                name="Qwen/Qwen2.5-Math-7B-Instruct",
                display_name="Qwen-Math-7B",
                threshold=0.5,  # Medium difficulty (50-80% success)
                cost_multiplier=4.0,
                max_tokens=3000
            ),
            ModelConfig(
                name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                display_name="DeepSeek-R1-1.5B",
                threshold=0.2,  # Hard problems (20-50% success, use reasoning)
                cost_multiplier=8.0,  # More expensive due to longer outputs
                max_tokens=32768
            ),
        ],
        abstain_threshold=0.15,  # Only abstain if success_rate < 15%
        greedy_temp=0.0,
        probe_column="predicted_difficulty_sigmoid",  # Misnomer: actually success rates!
        enable_abstain=False  # Disabled by default - route everything
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the SMART router.
    
    This is just a template - modify for your specific use case.
    """
    
    import os
    import uuid
    import base64
    
    # Configuration
    DATASET_NAME = "AIME_2025"  # OPTIONS:["AIME_2025","E2H-GSM8K", "GSM_HARD", "AIME_1983_2024"]
    MODEL_ALIAS = "Qwen2.5-Math-1.5B-Instruct"
    PROBE_SOURCE = "MATH" #GSM8K is other option

    PROBE_TEMP = 0.0
    PROBE_MAX_TOKENS = 3000
    PROBE_K = 1
    PROBE_SETTING_STR = f"max_{PROBE_MAX_TOKENS}_k_{PROBE_K}_temp_{PROBE_TEMP}"
    
    # Setup paths
    LABELLED_DATA_PATH = f"../runs/{MODEL_ALIAS}/datasplits/{DATASET_NAME}_predicted_by_predicting_{PROBE_SOURCE}_learnability_{MODEL_ALIAS}_{PROBE_SETTING_STR}.json"
    RESULTS_DIR = f"../predicting_learnability/SMART_ROUTER_EXPERIMENTS/{DATASET_NAME}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_json(LABELLED_DATA_PATH)
    # df = df.sample(n=100, random_state=42)
    df.rename(columns={
    'predicted_difficulty_sigmoid': 'predicted_success_rate_sigmoid',
    'predicted_difficulty': 'predicted_success_rate',})


    print(f"\n📏 LEN OF OUR DATASET: {len(df)} 📏\n")
    
    # Format prompts
    MATH_PROMPT_FORMATTING = " Please put your final answer inside \\boxed{}."
    df["formatted_prompt"] = df["question"].apply(lambda x: x + MATH_PROMPT_FORMATTING)
    df["id"] = [str(uuid.uuid1()) for _ in range(len(df))]
    
    print(f"✅ Loaded {len(df)} questions")
    print(f"\nPredicted success rate distribution:")
    print(df["predicted_difficulty_sigmoid"].describe())
    
    # Create router with your cascade configuration
    cascade_config = get_example_cascade_tiny()  # or get_example_cascade_with_reasoning()
    router = SmartRouter(cascade_config)
    
    # Run baseline (you might want to load this from cache)
    print("\n🏃 Running baseline...")
    from router_components import run_greedy_baseline
    
    llm = router.load_model(cascade_config.models[0].name)
    
    baseline_responses, baseline_tokens, baseline_correct = run_greedy_baseline(
        llm, df, cascade_config.greedy_temp, cascade_config.models[0].max_tokens
    )
    baseline_accuracy = np.mean(baseline_correct)
    baseline_total_tokens = sum(baseline_tokens) + len(baseline_tokens) * 100  # Approx input tokens
    
    # Calculate baseline cost
    from router_components import INPUT_TOKEN_PRICE, OUTPUT_TOKEN_PRICE, AVG_PROMPT_TOKENS
    baseline_input_tokens = len(baseline_tokens) * AVG_PROMPT_TOKENS
    baseline_output_tokens = sum(baseline_tokens)
    baseline_cost = (baseline_input_tokens / 1_000_000) * INPUT_TOKEN_PRICE + \
                    (baseline_output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE
    
    print(f"✅ Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"💰 Baseline cost: ${baseline_cost:.5f}")
    print(f"📊 Baseline tokens: {baseline_total_tokens:,}")
    
    # Evaluate SMART router
    metrics, results = router.evaluate(
        df=df,
        baseline_accuracy=baseline_accuracy,
        baseline_cost=baseline_cost,
        baseline_tokens=baseline_total_tokens
    )
    
    # Print results
    print("\n" + "="*80)
    print("🎉 SMART ROUTER RESULTS")
    print("="*80)
    print(metrics)
    
    # Save results
    results_df = pd.DataFrame([
        {
            "question_idx": r.question_idx,
            "predicted_success": r.predicted_score,
            "is_correct": r.is_correct,
            "final_answer": r.final_answer,
            "num_samples": r.num_samples_used,
            "total_tokens": sum(r.token_lengths),
        }
        for r in results
    ])
    
    results_df.to_csv(f"{RESULTS_DIR}/smart_router_results.csv", index=False)
    print(f"\n💾 Results saved to {RESULTS_DIR}/")
    
    # Send notification
    try:
        notification_msg = (
            f"✅ SMART Router evaluation complete!\n"
            f"📊 Dataset: {DATASET_NAME}\n"
            f"🎯 Baseline Acc: {baseline_accuracy:.2%}\n"
            f"🤖 Router Acc: {metrics.overall_accuracy:.2%} ({metrics.accuracy_gain:+.2%})\n"
            f"⚡ Token Multiplier: {metrics.token_multiplier:.2f}x\n"
            f"💰 Total Cost: ${metrics.total_cost:.5f}\n"
            f"📂 {RESULTS_DIR}"
        )
        send_notification(
            f"SMART Router - {DATASET_NAME} Complete",
            notification_msg
        )
    except Exception as e:
        print(f"Note: Notification not sent - {e}")
    
    print("\n" + "="*80)
    print("✨ Done! Hope this run went well ✨")
    print("="*80)
