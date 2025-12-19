"""
PIKA Router - Cascading Model Selection Based on Predicted Success Rates

HAKI-Route: Heuristic Allocation vis k-Inference Router (observation->predict p_success, armament -> increase k, conqueror->escalate to bigger model)
PIKA-Route: Probe-Informed K Allocation Router

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
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from vllm import LLM, SamplingParams
from collections import Counter
from datetime import datetime
from router_components import (
    SolverResult,
    EvaluationMetrics,
    calculate_cost_metrics,
    calculate_pass_at_k,
    get_strategy_color
)
from utils import parse_answers
from math_verify import parse, verify

# os.environ["CUDA_VISIBLE_DEVICES"] = f"1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
        print("📵 Notifications disabled in config")
        return
    
    try:
        response = requests.post(
            NOTIFICATION_CONFIG["url"],
            data=message.encode(encoding='utf-8'),
            headers={"Title": title},
            timeout=10
        )
        response.raise_for_status()
        print(f"✅ Notification sent: {title}")
    except Exception as e:
        print(f"⚠️  Failed to send notification: {e}")
        print(f"   Title: {title}")
        print(f"   URL: {NOTIFICATION_CONFIG['url']}")


def check_gpu_memory():
    """Check available GPU memory and warn if low"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_mem = float(result.stdout.strip().split('\n')[0])  # First GPU
            free_mem_gb = free_mem / 1024
            print(f"💾 GPU Memory Available: {free_mem_gb:.2f} GB")
            return free_mem_gb
    except:
        pass
    return None


def force_gpu_cleanup():
    """Aggressively clean up GPU memory"""
    import gc
    import torch
    
    print("🧹 Force cleaning GPU memory...")
    gc.collect()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    import time
    time.sleep(2)
    
    free_gb = check_gpu_memory()
    if free_gb and free_gb < 30:
        print(f"⚠️  WARNING: Only {free_gb:.2f} GB free - may cause OOM!")
        print("   Consider running: nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9")
    else:
        print(f"✅ GPU cleanup complete - {free_gb:.2f} GB available")



# ============================================================================
# PIKA ROUTER CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model in the cascade"""
    name: str                    # Model identifier (e.g., "Qwen2.5-Math-1.5B")
    display_name: str           # Human-readable name
    threshold: float            # Success rate threshold - use this model if success_rate >= threshold
    cost_multiplier: float      # Relative cost compared to base model
    max_tokens: int            # Max tokens for generation
    temperature: float = 0.0    # Temperature for this model (0.6 for DeepSeek, 0.0 for others)
    
    def __repr__(self):
        return f"{self.display_name} (threshold={self.threshold:.2f}, cost={self.cost_multiplier}x, temp={self.temperature})"


@dataclass
class CascadeConfig:
    """Configuration for the full model cascade"""
    models: List[ModelConfig]    # Ordered list: cheapest to most expensive
    abstain_threshold: float     # If predicted_success < this, abstain with "IDK" (too hard)
    greedy_temp: float = 0.0
    sc_temp: float = 0.6
    num_sc_samples: int = 1      # Can enable SC for specific models if >1.
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
# PIKA ROUTER IMPLEMENTATION
# ============================================================================

class PIKARouter:
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
        
    def load_model(self, model_name: str, gpu_memory_util: float = 0.45) -> LLM:
        """Lazy load models as needed"""
        if model_name not in self.llm_cache:
            print(f"🔄 Loading model: {model_name}")
            self.llm_cache[model_name] = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_util
            )
            print(f"✅ Model loaded: {model_name}")
        return self.llm_cache[model_name]
    
    def unload_model(self, model_name: str):
        """Unload a model to free GPU memory"""
        if model_name in self.llm_cache:
            print(f"🗑️  Unloading model: {model_name}")
            llm = self.llm_cache[model_name]
            
            # Explicitly destroy vLLM engine
            try:
                if hasattr(llm, 'llm_engine'):
                    del llm.llm_engine
                if hasattr(llm, '_run_engine'):
                    llm._run_engine = None
            except:
                pass
            
            del self.llm_cache[model_name]
            del llm
            
            # Aggressive garbage collection
            import gc
            import torch
            gc.collect()
            gc.collect()  # Call twice for cycles
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            print(f"✅ Model unloaded and GPU memory cleared")
    
    def unload_all_models(self):
        """Unload all cached models to completely free GPU memory"""
        if not self.llm_cache:
            return
        
        print(f"🧹 Unloading all {len(self.llm_cache)} cached models...")
        model_names = list(self.llm_cache.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        
        # Extra cleanup
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ All models unloaded, GPU memory completely freed")
    
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
                individual_correct=[False],
                cost_multiplier=0.0  # No cost for abstention
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
            individual_correct=individual_correct,
            cost_multiplier=model_config.cost_multiplier
        )
    
    def evaluate(
        self,
        df: pd.DataFrame,
        baseline_accuracy: float,
        baseline_cost: Optional[float] = None,
        baseline_tokens: Optional[List[int]] = None
    ) -> Tuple[EvaluationMetrics, List[SolverResult], Dict[str, int], Dict[str, float]]:
        """
        Evaluate the PIKA routing strategy on a dataset.
        Uses batched inference for significant speedup.
        
        Returns:
            Tuple of (metrics, results, routing_stats, model_hypothetical_costs)
        """
        from tqdm import tqdm
        
        results: List[SolverResult] = [None] * len(df)  # type: ignore  # Preallocate to maintain order
        routing_stats = {model.name: 0 for model in self.config.models}
        routing_stats["ABSTAIN"] = 0
        
        print("\n" + "="*80)
        print("🧠 PIKA ROUTER EVALUATION (BATCHED)")
        print("="*80)
        print(f"Models in cascade: {len(self.config.models)}")
        for model in self.config.models:
            print(f"  → {model}")
        print(f"Abstain threshold: {self.config.abstain_threshold:.2f}")
        print("="*80 + "\n")
        
        # PHASE 1: Route all questions and group by model
        print("📍 Phase 1: Routing questions to models...")
        batches_by_model = {model.name: [] for model in self.config.models}
        batches_by_model["ABSTAIN"] = []
        
        for idx in tqdm(range(len(df)), desc="Routing"):
            row = df.iloc[idx]
            predicted_success = row[self.config.probe_column]
            
            # Route to appropriate model
            model_config = self.config.get_model_for_question(predicted_success)
            
            # Track routing decision and group
            if model_config is None:
                routing_stats["ABSTAIN"] += 1
                batches_by_model["ABSTAIN"].append({
                    'idx': idx,
                    'prompt': row["formatted_prompt"],
                    'ground_truth': row["answer"],
                    'predicted_success': predicted_success
                })
            else:
                routing_stats[model_config.name] += 1
                batches_by_model[model_config.name].append({
                    'idx': idx,
                    'prompt': row["formatted_prompt"],
                    'ground_truth': row["answer"],
                    'predicted_success': predicted_success,
                    'model_config': model_config
                })
        
        # PHASE 2: Process each model's batch
        print("\n🚀 Phase 2: Processing batches...")
        
        # Handle abstentions first (no inference needed)
        if batches_by_model["ABSTAIN"]:
            print(f"⏭️  Abstaining on {len(batches_by_model['ABSTAIN'])} questions...")
            for item in batches_by_model["ABSTAIN"]:
                results[item['idx']] = SolverResult(
                    question_idx=item['idx'],
                    used_sc=False,
                    is_correct=0,
                    final_answer="IDK",
                    num_samples_used=0,
                    responses=["IDK"],
                    token_lengths=[0],
                    predicted_score=item['predicted_success'],
                    individual_correct=[False],
                    cost_multiplier=0.0  # No cost for abstention
                )
        
        # Process each model's batch
        for model_config in self.config.models:
            batch = batches_by_model[model_config.name]
            if not batch:
                continue
            
            print(f"⚡ Processing {len(batch)} questions with {model_config.display_name}...")
            
            # Load model once (with reduced memory usage for batching)
            llm = self.load_model(model_config.name, gpu_memory_util=0.45)
            
            # Prepare batch prompts
            num_samples = self.config.num_sc_samples  # Can make this per-model
            
            # Use model-specific temperature if set, otherwise fall back to config defaults
            if model_config.temperature > 0:
                temp = model_config.temperature
            else:
                temp = self.config.sc_temp if num_samples > 1 else self.config.greedy_temp
            
            params = SamplingParams(
                temperature=temp,
                max_tokens=model_config.max_tokens
            )
            
            # Create prompt list (with SC duplicates if needed)
            all_prompts = []
            prompt_to_idx = []  # Track which question each prompt belongs to
            for item in batch:
                for _ in range(num_samples):
                    all_prompts.append(item['prompt'])
                    prompt_to_idx.append(item['idx'])
            
            # BATCHED GENERATION
            batch_outputs = llm.generate(all_prompts, params)
            
            # Process outputs and group by question
            question_outputs = {}
            for prompt_idx, output in enumerate(batch_outputs):
                question_idx = prompt_to_idx[prompt_idx]
                if question_idx not in question_outputs:
                    question_outputs[question_idx] = []
                question_outputs[question_idx].append(output)
            
            # Create results for each question
            for item in tqdm(batch, desc=f"  Verifying {model_config.display_name}", leave=False):
                idx = item['idx']
                outputs = question_outputs[idx]
                
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
                        correct = verify(parse(f"${item['ground_truth']}$"), answer)
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
                is_correct = int(verify(parse(f"${item['ground_truth']}$"), final_answer))
                
                results[idx] = SolverResult(
                    question_idx=idx,
                    used_sc=(num_samples > 1),
                    is_correct=is_correct,
                    final_answer=final_answer,
                    num_samples_used=num_samples,
                    responses=responses,
                    token_lengths=token_lengths,
                    predicted_score=item['predicted_success'],
                    individual_correct=individual_correct,
                    cost_multiplier=model_config.cost_multiplier
                )
            
            # Unload model after processing this batch to free memory
            self.unload_model(model_config.name)
        
        # Print routing stats with per-model metrics if M > 2
        print("\n" + "="*80)
        print("📊 ROUTING STATISTICS")
        print("="*80)
        total_questions = len(df)
        for key, count in routing_stats.items():
            pct = count / total_questions * 100
            print(f"  {key:40s}: {count:5d} ({pct:5.1f}%)")
        print("="*80 + "\n")
        
        # Per-model performance breakdown (if M > 2) and calculate hypothetical costs
        model_hypothetical_costs = {}  # Store for notification later
        
        if len(self.config.models) > 2:
            print("\n" + "="*80)
            print("📈 PER-MODEL PERFORMANCE BREAKDOWN")
            print("="*80)
            
            from router_components import INPUT_TOKEN_PRICE, OUTPUT_TOKEN_PRICE, AVG_PROMPT_TOKENS
            
            for model_config in self.config.models:
                # Get results for this model
                model_results = [r for r in results if r.num_samples_used > 0 and 
                                routing_stats.get(model_config.name, 0) > 0]
                
                # Calculate hypothetical cost if ALL questions ran on this model
                # Use average output tokens from baseline as approximation
                if baseline_tokens and isinstance(baseline_tokens, list):
                    avg_baseline_output = sum(baseline_tokens) / len(baseline_tokens)
                else:
                    avg_baseline_output = 500
                    
                hypothetical_input = total_questions * AVG_PROMPT_TOKENS
                hypothetical_output = total_questions * avg_baseline_output
                hypothetical_cost = (hypothetical_input / 1_000_000) * INPUT_TOKEN_PRICE + \
                                   (hypothetical_output / 1_000_000) * OUTPUT_TOKEN_PRICE
                hypothetical_cost = hypothetical_cost * model_config.cost_multiplier
                model_hypothetical_costs[model_config.name] = hypothetical_cost
                
                if not model_results:
                    continue
                
                # Calculate metrics for this model using individual result costs
                model_correct = sum(r.is_correct for r in model_results)
                model_total = len(model_results)
                model_accuracy = model_correct / model_total if model_total > 0 else 0
                
                # Sum costs from individual results
                model_cost = sum(r.total_cost for r in model_results)
                
                print(f"\n{model_config.display_name}:")
                print(f"  Questions handled: {model_total} ({model_total/total_questions*100:.1f}%)")
                print(f"  Accuracy: {model_accuracy:.2%} ({model_correct}/{model_total})")
                print(f"  Accuracy vs baseline: {model_accuracy - baseline_accuracy:+.2%}")
                print(f"  Total cost: ${model_cost:.5f}")
                print(f"  Cost per question: ${model_cost/model_total:.5f}")
                print(f"  Cost multiplier: {model_config.cost_multiplier:.2f}x (configured)")
                print(f"  Temperature: {model_config.temperature if model_config.temperature > 0 else self.config.greedy_temp}")
            
            print("\n" + "="*80 + "\n")
        
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
        if baseline_tokens and len(baseline_tokens) > 0:
            baseline_total = sum(baseline_tokens) + len(baseline_tokens) * 100  # Approx input tokens
            token_multiplier = total_tokens / baseline_total
        
        # Router quality metrics (simplified for PIKA routing)
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
            strategy_name="PIKA_cascade",
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
        
        return metrics, results, routing_stats, model_hypothetical_costs


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================
def get_math_cascade_tiny() -> CascadeConfig:
    """
    """
    return CascadeConfig(
        models=[
            ModelConfig(
                name="Qwen/Qwen2.5-1.5B-Instruct",
                display_name="Qwen-Inst-1.5B",
                threshold=0.9,  # Use for easy questions (success_rate >= 90%)
                cost_multiplier=1.0,  # Base cost
                max_tokens=3000
            ),
            ModelConfig(
                name="Qwen/Qwen2.5-Math-1.5B-Instruct",
                display_name="Qwen-Math-1.5B",
                threshold=0.4,  # Use for easy questions (success_rate >= 40%)
                cost_multiplier=1.5,  # Base cost
                max_tokens=3000
            ),
            ModelConfig(
                name="Qwen/Qwen2.5-Math-7B-Instruct",
                display_name="Qwen-Math-7B",
                threshold=0.0,  # Use for harder questions (success_rate >= 00%)
                cost_multiplier=3.0,  # ~4x more expensive
                max_tokens=3000
            ),
            # Add more models here as needed
        ],
        abstain_threshold=0.2,  # If success_rate < 20%, say "IDK" (too hard)
        greedy_temp=0.0,
        probe_column="predicted_difficulty_sigmoid",  # Misnomer: actually success rates!
        enable_abstain=False  # Disabled by default - route everything
    )

def get_math_cascade_reasoning() -> CascadeConfig:
    """
    """
    return CascadeConfig(
        models=[
            ModelConfig(
                name="Qwen/Qwen2.5-Math-1.5B-Instruct",
                display_name="Qwen-Math-1.5B",
                threshold=0.9,  # Use for easy questions (success_rate >= 90%)
                cost_multiplier=1.0,  # Base cost
                max_tokens=3000
            ),
            ModelConfig(
                name="Qwen/Qwen2.5-Math-7B-Instruct",
                display_name="Qwen-Math-7B",
                threshold=0.4,  # Use for easy questions (success_rate >= 40%)
                cost_multiplier=1.5,  # Base cost
                max_tokens=3000
            ),
            ModelConfig(
                name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                display_name="DS-Qwen-8B",
                threshold=0.0,  # Use for harder questions (success_rate >= 00%)
                cost_multiplier=3.0,  # ~4x more expensive
                max_tokens=32768
            ),
            # Add more models here as needed
        ],
        abstain_threshold=0.2,  # If success_rate < 20%, say "IDK" (too hard)
        greedy_temp=0.0,
        probe_column="predicted_difficulty_sigmoid",  # Misnomer: actually success rates!
        enable_abstain=False  # Disabled by default - route everything
    )


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
                max_tokens=3000,
                temperature=0.0  # Greedy for Qwen
            ),
            ModelConfig(
                name="Qwen/Qwen2.5-Math-7B-Instruct",
                display_name="Qwen-Math-7B",
                threshold=0.3,  # Medium difficulty (30-80% success)
                cost_multiplier=4.0,
                max_tokens=3000,
                temperature=0.0  # Greedy for Qwen
            ),
            ModelConfig(
                name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                display_name="DeepSeek-R1-1.5B",
                threshold=0.0,  # Hard problems (00-30% success, use reasoning)
                cost_multiplier=8.0,  # More expensive due to longer outputs
                max_tokens=32768,
                temperature=0.6  # DeepSeek-specific temperature
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
    PIKA Router evaluation across multiple datasets and probe sources.
    """
    
    import os
    import uuid
    import base64
    
    # ========================================================================
    # EXPERIMENT CONFIGURATION
    # ========================================================================
    
    EXPERIMENT_CONFIG = {
        # Datasets to evaluate (use single-item list for one dataset)
        "datasets": ["AIME_2025", "E2H-GSM8K", "GSM_HARD", "AIME_1983_2024"],
        # "datasets": ["AIME_2025"],  # Single dataset example
        
        # Probe sources to try (use single-item list for one probe)
        "probe_sources": ["MATH", "GSM8K"],
        # "probe_sources": ["GSM8K"],  # Single probe example
        
        # Model configuration
        "model_alias": "Qwen2.5-Math-1.5B-Instruct",
        
        # Probe settings
        "probe_temp": 0.0,
        "probe_max_tokens": 3000,
        "probe_k": 1,
        
        # Self-consistency settings
        "use_sc": True,  # Enable self-consistency sampling
        "sc_num_samples": 10,  # Number of samples for SC (if enabled)
        "sc_temp": 0.6,  # Temperature for SC sampling
        
        "results_dir_path": "../predicting_learnability/PIKA_ROUTER_EXPERIMENTS_TINY_MATH_REVAMPED"
    }
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    # Check initial GPU memory
    print("\n" + "="*80)
    print("📊 INITIAL GPU MEMORY CHECK")
    print("="*80)
    check_gpu_memory()
    print("="*80 + "\n")
    
    print("=" * 80)
    print("🚀 PIKA ROUTER MULTI-DATASET EVALUATION")
    print("=" * 80)
    print(f"📊 Datasets: {', '.join(EXPERIMENT_CONFIG['datasets'])}")
    print(f"🔍 Probe Sources: {', '.join(EXPERIMENT_CONFIG['probe_sources'])}")
    print(f"🤖 Model: {EXPERIMENT_CONFIG['model_alias']}")
    print("=" * 80 + "\n")
    
    # Track overall results
    all_evaluations = []
    
    # Cache for baseline results (keyed by dataset name)
    # This avoids re-running baselines for each probe since they don't depend on probe
    baseline_cache = {}
    
    for DATASET_NAME in EXPERIMENT_CONFIG["datasets"]:
        
        for PROBE_SOURCE in EXPERIMENT_CONFIG["probe_sources"]:
            
            # Check GPU memory before starting
            print("\n" + "👀" * 40)
            check_gpu_memory()
            print("👀" * 40 + "\n")
            
            print("\n" + "█" * 80)
            print(f"📦 Dataset: {DATASET_NAME} | 🔍 Probe: {PROBE_SOURCE}")
            print("█" * 80 + "\n")
            
            # Configuration for this run
            MODEL_ALIAS = EXPERIMENT_CONFIG["model_alias"]
            PROBE_TEMP = EXPERIMENT_CONFIG["probe_temp"]
            PROBE_MAX_TOKENS = EXPERIMENT_CONFIG["probe_max_tokens"]
            PROBE_K = EXPERIMENT_CONFIG["probe_k"]
            PROBE_SETTING_STR = f"max_{PROBE_MAX_TOKENS}_k_{PROBE_K}_temp_{PROBE_TEMP}"
            RESULT_DIR_PATH = EXPERIMENT_CONFIG["results_dir_path"]
            
            # Setup paths (PLEASE FIX THIS SOON: LEARNABILITY VS SR)
            if "MATH" in PROBE_SOURCE:
                DONKEY_PATH_PATH_STR = "learnability"
            elif "GSM8K" in PROBE_SOURCE:
                DONKEY_PATH_PATH_STR = "SR" \

            LABELLED_DATA_PATH = f"../runs/{MODEL_ALIAS}/datasplits/{DATASET_NAME}_predicted_by_predicting_{PROBE_SOURCE}_{DONKEY_PATH_PATH_STR}_{MODEL_ALIAS}_{PROBE_SETTING_STR}.json"
            RESULTS_DIR = f"{RESULT_DIR_PATH}/{DATASET_NAME}/{PROBE_SOURCE}_probe"
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            # Check if data file exists
            if not os.path.exists(LABELLED_DATA_PATH):
                print(f"⚠️  Data file not found: {LABELLED_DATA_PATH}")
                print(f"⏭️  Skipping {DATASET_NAME} with {PROBE_SOURCE} probe\n")
                continue
            
            try:
                # Load data
                print("📊 Loading dataset...")
                df = pd.read_json(LABELLED_DATA_PATH)
                # df = df.sample(n=15, random_state=42)  # Uncomment to test on subset
                # DONKEY PATCH: Please fix this. We should be calling these success rate probes
                # df.rename(columns={
                #     'predicted_difficulty_sigmoid': 'predicted_success_rate_sigmoid',
                #     'predicted_difficulty': 'predicted_success_rate',
                # }, inplace=True)
                
                print(f"\n📏 Dataset size: {len(df)} questions 📏\n")
                
                # Format prompts
                MATH_PROMPT_FORMATTING = " Please put your final answer inside \\boxed{}."
                df["formatted_prompt"] = df["question"].apply(lambda x: x + MATH_PROMPT_FORMATTING)
                df["id"] = [str(uuid.uuid1()) for _ in range(len(df))]
                
                print(f"✅ Loaded {len(df)} questions")
                print(f"\nPredicted success rate distribution:")
                # print(df["predicted_success_rate_sigmoid"].describe())
                print(df["predicted_difficulty_sigmoid"].describe())
                
                # Create router with the chosen cascade configuration
                # cascade_config = get_example_cascade_tiny()  # or get_example_cascade_with_reasoning()
                # cascade_config = get_example_cascade_with_reasoning()  
                # cascade_config = get_math_cascade_tiny()
                cascade_config = get_math_cascade_reasoning()
                
                # Override SC settings from EXPERIMENT_CONFIG
                if EXPERIMENT_CONFIG.get("use_sc", False):
                    cascade_config.num_sc_samples = EXPERIMENT_CONFIG["sc_num_samples"]
                    cascade_config.sc_temp = EXPERIMENT_CONFIG["sc_temp"]
                    print(f"🔀 Self-Consistency ENABLED: {cascade_config.num_sc_samples} samples @ temp={cascade_config.sc_temp}")
                else:
                    cascade_config.num_sc_samples = 1
                    print("⚡ Self-Consistency DISABLED: Using greedy decoding")
                
                router = PIKARouter(cascade_config)
                
                # Check if we have cached baseline results for this dataset
                cache_key = f"{DATASET_NAME}_{len(df)}_{cascade_config.models[0].name}"
                
                if cache_key in baseline_cache:
                    print("\n♻️  Using cached baseline results...")
                    cached = baseline_cache[cache_key]
                    baseline_accuracy = cached['baseline_accuracy']
                    baseline_cost = cached['baseline_cost']
                    baseline_tokens = cached['baseline_tokens']
                    baseline_total_tokens = cached['baseline_total_tokens']
                    middle_accuracy = cached.get('middle_accuracy')
                    middle_cost = cached.get('middle_cost')
                    upper_accuracy = cached.get('upper_accuracy')
                    upper_cost = cached.get('upper_cost')
                    
                    print(f"✅ Baseline accuracy: {baseline_accuracy:.2%}")
                    print(f"💰 Baseline cost: ${baseline_cost:.5f}")
                    if middle_accuracy is not None:
                        print(f"✅ Middle model accuracy: {middle_accuracy:.2%}")
                        print(f"💰 Middle model cost: ${middle_cost:.5f}")
                    if upper_accuracy is not None:
                        print(f"✅ Upper bound accuracy: {upper_accuracy:.2%}")
                        print(f"💰 Upper bound cost: ${upper_cost:.5f}")
                else:
                    print("\n🏃 Running baseline (will be cached for other probes)...")
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
                
                # Unload baseline model to free GPU memory before cascade evaluation
                router.unload_model(cascade_config.models[0].name)
                print("🧹 Baseline model unloaded, GPU memory freed\n")
                
                # Run middle model baseline (for 3-model cascades)
                middle_accuracy = None
                middle_cost = None
                if len(cascade_config.models) == 3:
                    print("\n🏃 Running middle model baseline...")
                    middle_model = cascade_config.models[1]  # Middle model
                    llm = router.load_model(middle_model.name)
                    
                    middle_responses, middle_tokens, middle_correct = run_greedy_baseline(
                        llm, df, cascade_config.greedy_temp, middle_model.max_tokens
                    )
                    middle_accuracy = np.mean(middle_correct)
                    
                    # Calculate middle model cost
                    middle_input_tokens = len(middle_tokens) * AVG_PROMPT_TOKENS
                    middle_output_tokens = sum(middle_tokens)
                    middle_cost = (middle_input_tokens / 1_000_000) * INPUT_TOKEN_PRICE + \
                                    (middle_output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE
                    middle_cost = middle_cost * middle_model.cost_multiplier
                    
                    print(f"✅ Middle model accuracy: {middle_accuracy:.2%} ({middle_model.display_name})")
                    print(f"💰 Middle model cost: ${middle_cost:.5f}")
                    
                    # Unload middle model
                    router.unload_model(middle_model.name)
                    print("🧹 Middle model unloaded\n")
                
                # Run upper bound (most expensive model in cascade)
                upper_accuracy = None
                upper_cost = None
                if len(cascade_config.models) > 1:
                    print("\n🏃 Running upper bound (most expensive model)...")
                    upper_model = cascade_config.models[-1]  # Last model (most expensive)
                    llm = router.load_model(upper_model.name)
                    
                    upper_responses, upper_tokens, upper_correct = run_greedy_baseline(
                        llm, df, cascade_config.greedy_temp, upper_model.max_tokens
                    )
                    upper_accuracy = np.mean(upper_correct)
                    
                    # Calculate upper bound cost
                    upper_input_tokens = len(upper_tokens) * AVG_PROMPT_TOKENS
                    upper_output_tokens = sum(upper_tokens)
                    upper_cost = (upper_input_tokens / 1_000_000) * INPUT_TOKEN_PRICE + \
                                    (upper_output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE
                    upper_cost = upper_cost * upper_model.cost_multiplier
                    
                    print(f"✅ Upper bound accuracy: {upper_accuracy:.2%} ({upper_model.display_name})")
                    print(f"💰 Upper bound cost: ${upper_cost:.5f}")
                    
                    # Unload upper bound model
                    router.unload_model(upper_model.name)
                    print("🧹 Upper bound model unloaded\n")
                else:
                    upper_accuracy = baseline_accuracy  # Only one model, so baseline = upper bound
                    upper_cost = baseline_cost
                
                # Cache baseline results for reuse with other probes
                baseline_cache[cache_key] = {
                    'baseline_accuracy': baseline_accuracy,
                    'baseline_cost': baseline_cost,
                    'baseline_tokens': baseline_tokens,
                    'baseline_total_tokens': baseline_total_tokens,
                    'middle_accuracy': middle_accuracy,
                    'middle_cost': middle_cost,
                    'upper_accuracy': upper_accuracy,
                    'upper_cost': upper_cost,
                }
                print("💾 Baseline results cached for other probes\n")                # Evaluate PIKA router
                metrics, results, routing_stats, model_hypothetical_costs = router.evaluate(
                    df=df,
                    baseline_accuracy=baseline_accuracy,
                    baseline_cost=baseline_cost,
                    baseline_tokens=baseline_tokens
                )
                
                # Print results
                print("\n" + "="*80)
                print("🎉 PIKA ROUTER RESULTS")
                print("="*80)
                print(metrics)
                
                # Print model costs for comparison
                print("\n" + "="*80)
                print("💰 MODEL COST COMPARISON")
                print("="*80)
                print(f"  Baseline ({cascade_config.models[0].display_name}): ${baseline_cost:.5f}")
                if middle_cost is not None:
                    print(f"  Middle ({cascade_config.models[1].display_name}):   ${middle_cost:.5f}")
                if upper_cost is not None:
                    print(f"  Upper ({cascade_config.models[-1].display_name}):    ${upper_cost:.5f}")
                print(f"  Router:                            ${metrics.total_cost:.5f}")
                print("="*80 + "\n")
                # Save results
                results_df = pd.DataFrame([
                    {
                        "question_idx": r.question_idx,
                        "predicted_success": r.predicted_score,
                        "is_correct": r.is_correct,
                        "final_answer": r.final_answer,
                        "num_samples": r.num_samples_used,
                        "used_sc": r.used_sc,
                        "total_tokens": sum(r.token_lengths),
                    }
                    for r in results
                ])
                
                # Add SC suffix to filename if enabled
                sc_suffix = f"_sc{cascade_config.num_sc_samples}" if cascade_config.num_sc_samples > 1 else ""
                results_df.to_csv(f"{RESULTS_DIR}/PIKA_router_results{sc_suffix}.csv", index=False)
                print(f"\n💾 Results saved to {RESULTS_DIR}/")
                
                # Track this evaluation
                all_evaluations.append({
                    "dataset": DATASET_NAME,
                    "probe_source": PROBE_SOURCE,
                    "num_models": len(cascade_config.models),
                    "baseline_acc": baseline_accuracy,
                    "baseline_cost": baseline_cost,
                    "middle_acc": middle_accuracy if middle_accuracy is not None else None,
                    "middle_cost": middle_cost if middle_cost is not None else None,
                    "upper_acc": upper_accuracy if upper_accuracy is not None else baseline_accuracy,
                    "upper_cost": upper_cost if upper_cost is not None else baseline_cost,
                    "router_acc": metrics.overall_accuracy,
                    "router_cost": metrics.total_cost,
                    "accuracy_gain": metrics.accuracy_gain,
                    "token_multiplier": metrics.token_multiplier,
                    "total_cost": metrics.total_cost,
                    "used_sc": cascade_config.num_sc_samples > 1,
                    "sc_num_samples": cascade_config.num_sc_samples if cascade_config.num_sc_samples > 1 else None,
                    "results_dir": RESULTS_DIR
                })
                
                # Send notification
                try:
                    # Calculate routing distribution from routing_stats (already tracked during evaluation)
                    total_q = len(results)
                    routing_parts = []
                    for model in cascade_config.models:
                        count = routing_stats[model.name]
                        if count > 0:
                            pct = count / total_q * 100
                            routing_parts.append(f"{model.display_name}: {count} ({pct:.0f}%)")
                    if routing_stats["ABSTAIN"] > 0:
                        count = routing_stats["ABSTAIN"]
                        pct = count / total_q * 100
                        routing_parts.append(f"ABSTAIN: {count} ({pct:.0f}%)")
                    
                    routing_str = " | ".join(routing_parts)
                    
                    # Build model list
                    models_str = " → ".join([m.display_name for m in cascade_config.models])
                    
                    # Cost comparison
                    cost_diff = metrics.total_cost - baseline_cost
                    cost_pct = (cost_diff / baseline_cost * 100) if baseline_cost > 0 else 0
                    cost_sign = "💰" if cost_diff < 0 else "💸"
                    
                    # Enhanced cost comparison for M>2
                    if len(cascade_config.models) > 2:
                        cost_vs_parts = []
                        # Use ACTUAL costs from baseline runs, not hypothetical
                        actual_costs = {
                            cascade_config.models[0].name: baseline_cost,
                            cascade_config.models[1].name: middle_cost if middle_cost is not None else baseline_cost,
                            cascade_config.models[-1].name: upper_cost if upper_cost is not None else baseline_cost
                        }
                        
                        for i, model in enumerate(cascade_config.models):
                            actual_cost = actual_costs.get(model.name, baseline_cost)
                            pct_diff = ((metrics.total_cost - actual_cost) / actual_cost * 100) if actual_cost > 0 else 0
                            label = "baseline" if i == 0 else ("upper" if i == len(cascade_config.models)-1 else model.display_name)
                            cost_vs_parts.append(f"{label}:{pct_diff:+.1f}%")
                        cost_diff_str = f"${cost_diff:+.5f} ({', '.join(cost_vs_parts)})"
                    else:
                        cost_diff_str = f"${cost_diff:+.5f} ({cost_pct:+.1f}%)"
                    
                    # Upper bound comparison
                    upper_bound_acc = upper_accuracy if 'upper_accuracy' in locals() else baseline_accuracy
                    router_vs_upper = metrics.overall_accuracy - upper_bound_acc
                    upper_model_name = cascade_config.models[-1].display_name if len(cascade_config.models) > 1 else "N/A"
                    
                    # Build accuracy comparison message
                    if len(cascade_config.models) == 3 and middle_accuracy is not None:
                        # For 3-model cascades, show all three baselines
                        router_vs_middle = metrics.overall_accuracy - middle_accuracy
                        middle_model_name = cascade_config.models[1].display_name
                        acc_msg = (
                            f"🎯 Baseline: {baseline_accuracy:.2%} ({cascade_config.models[0].display_name}) - ${baseline_cost:.5f}\n"
                            f"🎯 Middle: {middle_accuracy:.2%} ({middle_model_name}) - ${middle_cost:.5f}\n"
                            f"🏆 Upper: {upper_bound_acc:.2%} ({upper_model_name}) - ${upper_cost:.5f}\n"
                            f"🤖 Router: {metrics.overall_accuracy:.2%} ({metrics.accuracy_gain:+.2%} vs baseline, {router_vs_middle:+.2%} vs middle, {router_vs_upper:+.2%} vs upper)"
                        )
                    else:
                        # For 2-model cascades, use original format
                        acc_msg = (
                            f"🎯 Baseline: {baseline_accuracy:.2%} ({cascade_config.models[0].display_name}) - ${baseline_cost:.5f}\n"
                            f"🏆 Upper: {upper_bound_acc:.2%} ({upper_model_name}) - ${upper_cost:.5f}\n"
                            f"🤖 Router: {metrics.overall_accuracy:.2%} ({metrics.accuracy_gain:+.2%} vs baseline, {router_vs_upper:+.2%} vs upper)"
                        )
                    
                    notification_msg = (
                        f"✅ PIKA Router evaluation complete!\n"
                        f"📊 Dataset: {DATASET_NAME} ({len(df)} questions)\n"
                        f"🔍 Probe: {PROBE_SOURCE}\n"
                        f"{acc_msg}\n"
                        f"⚡ Token Multiplier: {metrics.token_multiplier:.2f}x\n"
                        f"{cost_sign} Router Cost: ${metrics.total_cost:.5f}\n"
                        f"💵 Cost Diff: {cost_diff_str}\n"
                        f"🌡️ Temp: {cascade_config.greedy_temp} (greedy) / {cascade_config.sc_temp} (SC)\n"
                        f"🔀 SC Samples: {cascade_config.num_sc_samples}\n"
                        f"🤖 Models: {models_str}\n"
                        f"📈 Routing: {routing_str}\n"
                        f"📂 {RESULTS_DIR}"
                    )
                    send_notification(
                        f"PIKA Router - {DATASET_NAME} ({PROBE_SOURCE})",
                        notification_msg
                    )
                except Exception as e:
                    print(f"Note: Notification not sent - {e}")
                
                print("\n" + "="*80)
                print(f"✨ Completed: {DATASET_NAME} with {PROBE_SOURCE} probe ✨")
                print("="*80 + "\n")
                
                # Clean up all models before next iteration
                router.unload_all_models()
                del router  # Delete router object
                
                # Force aggressive GPU cleanup
                force_gpu_cleanup()
                
            except Exception as e:
                print(f"\n❌ Error processing {DATASET_NAME} with {PROBE_SOURCE} probe:")
                print(f"   {str(e)}")
                print("   Continuing to next dataset/probe combination...\n")
                notification_error_msg = (
                    f"\n❌ Error processing {DATASET_NAME} with {PROBE_SOURCE} probe:"
                    f"   {str(e)}"
                    "   Continuing to next dataset/probe combination...\n"
                    )

                send_notification(
                        f"PIKA Router - {DATASET_NAME} ({PROBE_SOURCE})",
                        notification_error_msg
                    )
                
                # Clean up any loaded models before continuing
                try:
                    if 'router' in locals():
                        router.unload_all_models()
                        del router
                    import gc
                    import torch
                    import time
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(3)  # Give GPU time to clean up after error
                except:
                    pass  # Best effort cleanup
                
                continue
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "🎊" * 40)
    print("ALL EVALUATIONS COMPLETE!")
    print("🎊" * 40 + "\n")
    
    if all_evaluations:
        summary_df = pd.DataFrame(all_evaluations)
        print("📊 SUMMARY OF ALL RUNS:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"{RESULT_DIR_PATH}/evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
        
        # Send final notification
        try:
            best_run = summary_df.loc[summary_df['accuracy_gain'].idxmax()]
            final_msg = (
                f"🎉 All PIKA Router evaluations complete!\n"
                f"📊 Total runs: {len(all_evaluations)}\n"
                f"🏆 Best: {best_run['dataset']} ({best_run['probe_source']})\n"
                f"   → Acc gain: {best_run['accuracy_gain']:+.2%}\n"
                f"   → Token mult: {best_run['token_multiplier']:.2f}x\n"
                f"📂 Results: {RESULT_DIR_PATH}/"
            )
            send_notification(
                "PIKA Router - All Evaluations Complete",
                final_msg
            )
        except:
            pass
    else:
        print("⚠️  No evaluations completed successfully.")
    
    print("\n✨ All done! ✨\n")
