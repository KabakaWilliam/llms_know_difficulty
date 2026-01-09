import re
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
from vllm import LLM, SamplingParams
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os

from .base_probe import Probe
from ..config import PROMPTING_BASELINE, ROOT_DATA_DIR
from ..utils import create_results_path


def parse_boxed_answer(text: str) -> Optional[float]:
    """Extract the boxed answer from model output.
    
    Looks for patterns like \\boxed{0.75} or \\boxed{{0.75}}
    """
    # Try to find \boxed{...} pattern
    match = re.search(r'\\boxed\{\{?(.*?)\}?\}', text)
    if match:
        try:
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None


class PromptingBaseline(Probe):
    def __init__(self, config):
        super().__init__(config)
        self._has_setup_run = False
        self.model_name = None
        self.vllm_model = None
        self.device = None
        self.batch_size = PROMPTING_BASELINE.get("batch_size", 16)
        self.generation_max_length = PROMPTING_BASELINE.get("generation_max_length", 128)
        self.generation_temperature = PROMPTING_BASELINE.get("generation_temperature", 0.7)
        self.prompt_template = PROMPTING_BASELINE.get("prompt_template")
        
        # Metadata for evaluation
        self.val_predictions = None
        self.val_labels = None
        self.val_generations = None  # Store generated text for debugging
        self.test_generations = None  # Store generated text for debugging
        self.best_val_score = None
        self.task_type = None
        self.metric_name = None
        self.k_value = None  # Number of generations (from constraints)

    def name(self) -> str:
        """The name of the probe."""
        return "prompting_baseline"

    def init_model(self, config: dict):
        """Load a probe from checkpoint."""
        pass

    def setup(self, model_name: str, device: str = "cuda") -> None:
        """Load VLLM model for inference."""
        self.model_name = model_name
        self.device = device
        
        # Initialize VLLM with GPU
        self.vllm_model = LLM(
            model=model_name,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
            dtype="bfloat16"
        )
        self._has_setup_run = True
        print(f"✓ VLLM model loaded: {model_name}")

    def _format_prompt(self, problem_text: str, solver_model: str, k: int, max_tokens: int, temperature: float) -> str:
        """Format the problem using the prompt template."""
        system = self.prompt_template["System"]
        user = self.prompt_template["User"]
        
        # Format user prompt with placeholders
        user_formatted = user.format(
            SOLVER_MODEL_NAME=solver_model,
            k=k,
            T=max_tokens,
            temp=temperature,
            problem_text=problem_text
        )
        
        # Build chat format: system + user
        formatted = f"{system}\n\n{user_formatted}"
        return formatted

    def train(self, train_data: Tuple[List[str], List[float]], val_data: Tuple[List[str], List[float]], 
              solver_model: str = "gpt2", k: int = 8, 
              max_tokens: int = 3000, temperature: float = 0.7) -> None:
        """
        Train (or rather, evaluate) the prompting baseline on val data.
        
        Args:
            train_data: (prompts, labels) - not used for prompting baseline
            val_data: (prompts, labels) - used for evaluation
            solver_model: Name of the target solver model
            k: Number of generation attempts (for label in constraints)
            max_tokens: Max tokens for generation (for label in constraints)
            temperature: Temperature for generation (for label in constraints)
        """
        val_prompts, val_labels = val_data
        
        # Determine task type based on k
        # k=1 means binary success/failure, k>1 means continuous success_rate
        self.k_value = k
        self.task_type = "classification" if k == 1 else "regression"
        
        print(f"\n{'='*80}")
        print(f"Training Prompting Baseline")
        print(f"  Solver Model: {solver_model}")
        print(f"  k (attempts): {k}")
        print(f"  Task Type: {self.task_type}")
        print(f"  Num validation samples: {len(val_prompts)}")
        print(f"{'='*80}")
        
        # Generate prompts and collect predictions
        self.val_predictions = []
        self.val_generations = []
        
        print(f"\nGenerating predictions on validation set...")
        for i, (problem, label) in enumerate(zip(val_prompts, val_labels)):
            # Format the prompt
            formatted_prompt = self._format_prompt(
                problem_text=problem,
                solver_model=solver_model,
                k=k,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Generate using VLLM
            sampling_params = SamplingParams(
                temperature=self.generation_temperature,
                max_tokens=self.generation_max_length,
                top_p=1.0
            )
            
            outputs = self.vllm_model.generate(
                [formatted_prompt],
                sampling_params,
                use_tqdm=False
            )
            
            generated_text = outputs[0].outputs[0].text
            self.val_generations.append(generated_text)  # Store for debugging
            
            # Parse the boxed answer
            pred = parse_boxed_answer(generated_text)
            if pred is None:
                # Fallback: try to extract any number
                numbers = re.findall(r'0\.\d+|\d+', generated_text)
                if numbers:
                    try:
                        pred = float(numbers[0])
                        pred = min(max(pred, 0.0), 1.0)  # Clamp to [0, 1]
                    except:
                        pred = 0.5
                else:
                    pred = 0.5
            
            # Clamp to [0, 1]
            pred = min(max(pred, 0.0), 1.0)
            self.val_predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(val_prompts)} samples processed")
        
        self.val_predictions = np.array(self.val_predictions)
        self.val_labels = np.array(val_labels)
        
        # Compute metric
        self.best_val_score, self.metric_name = self._compute_metric(
            self.val_predictions, self.val_labels, self.task_type
        )
        
        print(f"\n{'='*80}")
        print(f"Validation {self.metric_name}: {self.best_val_score:.4f}")
        print(f"{'='*80}\n")

    def fit(self, prompts: List[str], targets: List[float], **kwargs) -> None:
        """No-op for prompting baseline (no parameters to fit)."""
        pass

    def predict(self, prompts: List[str], solver_model: str = "gpt2",
                k: int = 8, max_tokens: int = 3000, temperature: float = 0.7) -> np.ndarray:
        """
        Generate predictions for new prompts using the prompting baseline.
        
        Args:
            prompts: List of problem texts
            solver_model: Name of the target solver model
            k: Number of generation attempts
            max_tokens: Max tokens for generation
            temperature: Temperature for generation
            
        Returns:
            Array of predicted success probabilities
        """
        predictions = []
        self.test_generations = []
        
        print(f"\nGenerating predictions on {len(prompts)} samples...")
        for i, problem in enumerate(prompts):
            # Format the prompt
            formatted_prompt = self._format_prompt(
                problem_text=problem,
                solver_model=solver_model,
                k=k,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Generate using VLLM
            sampling_params = SamplingParams(
                temperature=self.generation_temperature,
                max_tokens=self.generation_max_length,
                top_p=1.0
            )
            
            outputs = self.vllm_model.generate(
                [formatted_prompt],
                sampling_params,
                use_tqdm=False
            )
            
            generated_text = outputs[0].outputs[0].text
            self.test_generations.append(generated_text)  # Store for debugging
            
            # Parse the boxed answer
            pred = parse_boxed_answer(generated_text)
            if pred is None:
                # Fallback: try to extract any number
                numbers = re.findall(r'0\.\d+|\d+', generated_text)
                if numbers:
                    try:
                        pred = float(numbers[0])
                        pred = min(max(pred, 0.0), 1.0)
                    except:
                        pred = 0.5
                else:
                    pred = 0.5
            
            # Clamp to [0, 1]
            pred = min(max(pred, 0.0), 1.0)
            predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(prompts)} samples processed")
        
        return np.array(predictions)

    @staticmethod
    def _compute_metric(predictions: np.ndarray, labels: np.ndarray, task_type: str) -> Tuple[float, str]:
        """
        Compute metric based on task type.
        
        Args:
            predictions: Model predictions [N]
            labels: Ground truth labels [N]
            task_type: "regression" or "classification"
        
        Returns:
            (score, metric_name): Tuple of (float score, string metric name)
        """
        if task_type == "regression":
            score, _ = spearmanr(labels, predictions)
            metric_name = "Spearman"
        else:  # classification
            score = roc_auc_score(labels, predictions)
            metric_name = "ROC-AUC"
        
        return float(score), metric_name

    def save_results(self, model_name: str, dataset_name: str, gen_str: str = None) -> Path:
        """
        Save prompting baseline results (predictions, generations, metadata) to disk.
        Similar to sklearn_probe results structure.
        
        Args:
            model_name: Name of the model used for inference
            dataset_name: Name of the dataset
            gen_str: Generation string (e.g., "k_8_temp_0.7") for result organization
        
        Returns:
            Path to results directory
        """
        # Create results path
        results_path = create_results_path(
            dataset_name=dataset_name,
            model_name=model_name,
            probe_name=self.name(),
            gen_str=gen_str
        )
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save validation predictions and generations
        val_data = {
            "predictions": self.val_predictions.tolist() if self.val_predictions is not None else [],
            "labels": self.val_labels.tolist() if self.val_labels is not None else [],
            "generations": self.val_generations if self.val_generations is not None else [],
            "metric": self.metric_name,
            "score": float(self.best_val_score) if self.best_val_score is not None else None
        }
        
        with open(results_path / "val_results.json", "w") as f:
            json.dump(val_data, f, indent=2)
        
        # Save test predictions and generations if available
        if self.test_generations is not None:
            test_data = {
                "generations": self.test_generations
            }
            with open(results_path / "test_generations.json", "w") as f:
                json.dump(test_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "probe_name": self.name(),
            "model_name": self.model_name,
            "dataset_name": dataset_name,
            "task_type": self.task_type,
            "k_value": self.k_value,
            "metric": self.metric_name,
            "val_score": float(self.best_val_score) if self.best_val_score is not None else None,
            "num_val_samples": len(self.val_predictions) if self.val_predictions is not None else 0,
            "num_test_samples": len(self.test_generations) if self.test_generations is not None else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
        print(f"  - val_results.json: {len(self.val_predictions)} predictions + generations")
        if self.test_generations is not None:
            print(f"  - test_generations.json: {len(self.test_generations)} generations")
        print(f"  - metadata.json: Probe metadata")
        
        return results_path
