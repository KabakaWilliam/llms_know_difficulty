"""
ProbeRouter — Main router class for probe-based model selection.

Routes incoming prompts to the optimal model based on:
1. Probe-predicted success probability per model
2. Model cost
3. User-defined cost-accuracy tradeoff (lambda)

Usage:
    router = ProbeRouter(
        models=["Qwen/Qwen2.5-Math-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
        probe_dir="./probes",
    )
    response = router.generate("Solve: x^2 - 5x + 6 = 0")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pika.router.config import ModelConfig, RouterConfig, get_gen_str, FALLBACK_MODEL_COSTS


@dataclass
class RouterResponse:
    """Response from the router's generate() method.
    
    Attributes:
        text: Generated response text
        model_id: ID of the model that generated the response
        probe_scores: Dict of {model_id: probe_predicted_success_prob}
        utility_scores: Dict of {model_id: utility_score} (probe_pred - lambda * cost)
        cost: Estimated cost of this generation
    """
    text: str
    model_id: str
    probe_scores: dict[str, float]
    utility_scores: dict[str, float]
    cost: float


class ProbeRouter:
    """
    Probe-based router for selecting the optimal LLM based on predicted success.
    
    The router uses trained probes to predict which model is most likely to
    succeed on a given prompt, while considering cost-accuracy tradeoffs.
    
    Parameters
    ----------
    models : list[str] | list[ModelConfig]
        List of model IDs (HuggingFace format) or ModelConfig objects.
        Example: ["Qwen/Qwen2.5-Math-7B-Instruct", "openai/gpt-oss-20b_low"]
    
    probe_dir : str | Path, optional
        Directory containing trained probe checkpoints. Expected structure:
        probe_dir/{model_name}/{dataset}/{probe_type}/...
        If None, probes must be specified via ModelConfig.probe_path
    
    dataset : str, optional
        Dataset name for probe discovery (e.g., "DigitalLearningGmbH_MATH-lighteval")
    
    lambda_val : float, default=0.5
        Cost-accuracy tradeoff parameter:
        - 0.0: Maximize accuracy, ignore cost
        - 0.5: Balanced tradeoff
        - 1.0: Strongly penalize cost
    
    model_costs : dict[str, float], optional
        Override default costs per model. Keys are model IDs, values are USD per query.
    
    config : RouterConfig, optional
        Advanced configuration options. Overrides lambda_val if provided.
    
    Examples
    --------
    Basic usage:
    >>> router = ProbeRouter(
    ...     models=["Qwen/Qwen2.5-Math-7B-Instruct"],
    ...     probe_dir="./probes",
    ... )
    >>> response = router.generate("What is 2+2?")
    >>> print(response.text)
    
    With custom costs:
    >>> router = ProbeRouter(
    ...     models=["model_a", "model_b"],
    ...     model_costs={"model_a": 0.001, "model_b": 0.01},
    ... )
    
    Routing only (no generation):
    >>> best_model = router.route("What is the derivative of x^2?")
    >>> print(best_model)  # "Qwen/Qwen2.5-Math-7B-Instruct"
    """
    
    def __init__(
        self,
        models: list[str] | list[ModelConfig],
        probe_dir: str | Path | None = None,
        dataset: str | None = None,
        lambda_val: float = 0.5,
        model_costs: dict[str, float] | None = None,
        config: RouterConfig | None = None,
    ):
        # Initialize config
        self.config = config or RouterConfig(lambda_val=lambda_val)
        self.probe_dir = Path(probe_dir) if probe_dir else None
        self.dataset = dataset
        
        # Parse model configs
        self.model_configs: dict[str, ModelConfig] = {}
        for m in models:
            if isinstance(m, str):
                cost = model_costs.get(m) if model_costs else None
                self.model_configs[m] = ModelConfig(model_id=m, cost=cost)
            else:
                self.model_configs[m.model_id] = m
        
        # Apply cost overrides
        if model_costs:
            for model_id, cost in model_costs.items():
                if model_id in self.model_configs:
                    self.model_configs[model_id].cost = cost
        
        # Compute costs from training data (like utility_router.ipynb)
        # Falls back to FALLBACK_MODEL_COSTS if training data unavailable
        self._raw_costs = self._compute_costs_from_training_data(model_costs)
        if self.config.normalize_costs:
            lo, hi = min(self._raw_costs.values()), max(self._raw_costs.values())
            span = hi - lo if hi != lo else 1.0
            self._norm_costs = {m: (c - lo) / span for m, c in self._raw_costs.items()}
        else:
            self._norm_costs = self._raw_costs.copy()
        
        # Lazy-loaded components
        self._probes: dict[str, Any] = {}  # model_id -> probe (or None if failed)
        self._probe_paths: dict[str, Path | None] = {}  # model_id -> probe checkpoint path
        self._llm_engines: dict[str, Any] = {}  # model_id -> vLLM engine
        self._initialized = False
    
    @property
    def model_ids(self) -> list[str]:
        """List of model IDs in the router pool."""
        return list(self.model_configs.keys())
    
    def initialize(self, load_llms: bool = True) -> "ProbeRouter":
        """
        Explicitly initialize probes and optionally LLM engines.
        
        Called automatically on first generate() or route() call.
        
        Parameters
        ----------
        load_llms : bool
            If True, also load vLLM engines for generation.
            Set False if only using route() without generate().
        """
        if self._initialized:
            return self
        
        print(f"🔧 Initializing ProbeRouter with {len(self.model_configs)} models...")
        
        if self.config.sequential_loading:
            # Sequential mode: only discover probe paths, load on-demand
            print("  (sequential_loading=True: probes will load on-demand)")
            for model_id in self.model_configs:
                self._probe_paths[model_id] = self._discover_probe_path_for_model(model_id)
                if self._probe_paths[model_id]:
                    print(f"  ✓ Found probe for {model_id}")
                else:
                    print(f"  ⚠ No probe found for {model_id}")
        else:
            # Standard mode: load all probes upfront
            for model_id in self.model_configs:
                self._load_probe(model_id)
        
        # Load LLM engines if needed
        if load_llms:
            for model_id in self.model_configs:
                self._load_llm_engine(model_id)
        
        self._initialized = True
        loaded_count = len([p for p in self._probes.values() if p is not None])
        found_count = len([p for p in self._probe_paths.values() if p is not None])
        if self.config.sequential_loading:
            print(f"✅ Router ready ({found_count} probe paths discovered)")
        else:
            print(f"✅ Router ready with {loaded_count} probes loaded")
        return self
    
    def _discover_probe_path_for_model(self, model_id: str) -> Path | None:
        """Get the probe path for a model (from config or auto-discovery)."""
        cfg = self.model_configs[model_id]
        
        if cfg.probe_path:
            path = Path(cfg.probe_path)
            return path if path.exists() else None
        elif self.probe_dir and self.dataset:
            return self._discover_probe_path(model_id)
        return None
    
    def _compute_costs_from_training_data(
        self, 
        cost_overrides: dict[str, float] | None = None
    ) -> dict[str, float]:
        """
        Compute per-model costs from training data (total_output_cost_usd).
        
        This matches the approach in utility_router.ipynb:compute_tier_costs().
        Falls back to FALLBACK_MODEL_COSTS if training data is unavailable.
        
        Parameters
        ----------
        cost_overrides : dict[str, float], optional
            Override costs for specific models (takes highest priority)
        
        Returns
        -------
        dict[str, float]
            Raw costs per model (before normalization)
        """
        import pandas as pd
        
        raw_costs = {}
        
        for model_id, cfg in self.model_configs.items():
            # Priority 1: Explicit cost override
            if cost_overrides and model_id in cost_overrides:
                raw_costs[model_id] = cost_overrides[model_id]
                continue
            
            # Priority 2: Cost set in ModelConfig
            if cfg.cost is not None:
                raw_costs[model_id] = cfg.cost
                continue
            
            # Priority 3: Load from training data
            if self.probe_dir and self.dataset:
                # Find data directory (probe_dir is in data/results, we need data/)
                data_dir = self.probe_dir.parent
                gen_str = get_gen_str(model_id)
                train_path = data_dir / model_id / self.dataset / f"train_{gen_str}.parquet"
                
                if train_path.exists():
                    try:
                        train_df = pd.read_parquet(train_path)
                        if "total_output_cost_usd" in train_df.columns:
                            raw_costs[model_id] = train_df["total_output_cost_usd"].mean()
                            continue
                    except Exception as e:
                        warnings.warn(f"Failed to load training data for {model_id}: {e}")
            
            # Priority 4: Fallback to default costs
            raw_costs[model_id] = FALLBACK_MODEL_COSTS.get(model_id, 0.01)
        
        return raw_costs
    
    def _load_probe(self, model_id: str) -> None:
        """Load probe checkpoint for a model."""
        from pika.probe.linear_eoi_probe import LinearEoiProbe
        
        cfg = self.model_configs[model_id]
        
        # Determine probe path
        if cfg.probe_path:
            probe_path = cfg.probe_path
        elif self.probe_dir and self.dataset:
            # Auto-discover probe path
            probe_path = self._discover_probe_path(model_id)
        else:
            raise ValueError(
                f"No probe path for {model_id}. Provide probe_path in ModelConfig, "
                "or set probe_dir and dataset for auto-discovery."
            )
        
        if probe_path is None or not Path(probe_path).exists():
            warnings.warn(f"⚠ Probe not found for {model_id}, using uniform predictions")
            self._probes[model_id] = None
            return
        
        print(f"  📦 Loading probe for {model_id}...")
        try:
            probe = LinearEoiProbe.load_from_checkpoint(
                probe_path, device=self.config.device
            )
            probe.batch_size = self.config.batch_size
            self._probes[model_id] = probe
        except Exception as e:
            warnings.warn(f"⚠ Failed to load probe for {model_id}: {e}")
            self._probes[model_id] = None
    
    def _discover_probe_path(self, model_id: str) -> Path | None:
        """
        Find the latest probe checkpoint for a model.
        
        Searches in order of preference:
        1. Exact match: {probe_dir}/results/{model}/{dataset}/{probe_type}/{gen_str}/label_{metric}/{timestamp}
        2. Any gen_str with matching metric
        3. Any gen_str with any metric
        """
        if not self.probe_dir or not self.dataset:
            return None
        
        # Resolve to absolute path
        probe_dir = Path(self.probe_dir).resolve()
        
        # Build expected path structure
        base_dir = (
            probe_dir
            / "results"
            / model_id
            / self.dataset
            / self.config.probe_model_type
        )
        
        if not base_dir.exists():
            # Try without /results prefix (maybe user passed results dir directly)
            alt_base = probe_dir / model_id / self.dataset / self.config.probe_model_type
            if alt_base.exists():
                base_dir = alt_base
            else:
                return None
        
        # Strategy 1: Try exact gen_str match
        gen_str = self._get_gen_str(model_id)
        metric_dir = base_dir / gen_str / f"label_{self.config.probe_metric}"
        
        if metric_dir.exists():
            result = self._find_latest_probe_in_dir(metric_dir)
            if result:
                return result
        
        # Strategy 2: Try any gen_str with matching metric
        for gen_dir in base_dir.iterdir():
            if not gen_dir.is_dir():
                continue
            metric_dir = gen_dir / f"label_{self.config.probe_metric}"
            if metric_dir.exists():
                result = self._find_latest_probe_in_dir(metric_dir)
                if result:
                    print(f"    (using fallback gen_str: {gen_dir.name})")
                    return result
        
        # Strategy 3: Try any gen_str with any metric
        for gen_dir in base_dir.iterdir():
            if not gen_dir.is_dir():
                continue
            for metric_dir in gen_dir.iterdir():
                if metric_dir.is_dir() and metric_dir.name.startswith("label_"):
                    result = self._find_latest_probe_in_dir(metric_dir)
                    if result:
                        print(f"    (using fallback: {gen_dir.name}/{metric_dir.name})")
                        return result
        
        # Strategy 4: Check for old-style layout (timestamp directly under base_dir)
        result = self._find_latest_probe_in_dir(base_dir)
        if result:
            print(f"    (using legacy probe layout)")
            return result
        
        return None
    
    def _find_latest_probe_in_dir(self, directory: Path) -> Path | None:
        """Find the latest timestamp directory containing probe files."""
        if not directory.exists():
            return None
        
        # Look for timestamp directories (format: YYYYMMDD_HHMMSS or similar)
        candidates = []
        for d in directory.iterdir():
            if d.is_dir():
                # Check if it contains probe files
                if (d / "best_probe.joblib").exists() or (d / "probe_metadata.json").exists():
                    candidates.append(d)
        
        if not candidates:
            return None
        
        # Sort by name (timestamp format sorts correctly)
        return sorted(candidates)[-1]
    
    def _get_gen_str(self, model_id: str) -> str:
        """Get generation string for probe discovery."""
        cfg = self.model_configs[model_id]
        params = cfg.get_generation_params()
        return f"maxlen_{params['max_tokens']}_k_{params['n']}_temp_{params['temperature']}"
    
    def _load_llm_engine(self, model_id: str) -> None:
        """Load vLLM engine for a model."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for generation. Install with: pip install vllm"
            )
        
        # Skip if already loaded
        if model_id in self._llm_engines:
            return
        
        cfg = self.model_configs[model_id]
        
        # Extract base model name (remove thinking mode suffix)
        base_model_name = model_id
        for suffix in ['_low', '_medium', '_high', '_reasoning']:
            if base_model_name.endswith(suffix):
                base_model_name = base_model_name[:-len(suffix)]
                break
        
        print(f"  🚀 Loading vLLM engine for {model_id}...")
        try:
            engine = LLM(
                model=base_model_name,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                trust_remote_code=True,
            )
            self._llm_engines[model_id] = engine
        except Exception as e:
            warnings.warn(f"⚠ Failed to load vLLM for {model_id}: {e}")
            self._llm_engines[model_id] = None
    
    def get_probe_scores(self, prompt: str) -> dict[str, float]:
        """
        Get probe-predicted success probabilities for all models.
        
        Parameters
        ----------
        prompt : str
            The input prompt to evaluate.
        
        Returns
        -------
        dict[str, float]
            Mapping of model_id -> predicted success probability [0, 1]
        """
        if not self._initialized:
            self.initialize(load_llms=False)
        
        if self.config.sequential_loading:
            return self._get_probe_scores_sequential(prompt)
        else:
            return self._get_probe_scores_parallel(prompt)
    
    def _get_probe_scores_parallel(self, prompt: str) -> dict[str, float]:
        """Get scores using pre-loaded probes (standard mode)."""
        scores = {}
        for model_id, probe in self._probes.items():
            if probe is None:
                scores[model_id] = 0.5
            else:
                indices, preds = probe.predict(([0], [prompt]))
                scores[model_id] = float(preds[0])
        return scores
    
    def _get_probe_scores_sequential(self, prompt: str) -> dict[str, float]:
        """Get scores by loading probes one at a time (memory-efficient mode)."""
        import gc
        from pika.probe.linear_eoi_probe import LinearEoiProbe
        
        scores = {}
        for model_id in self.model_configs:
            probe_path = self._probe_paths.get(model_id)
            
            if probe_path is None:
                scores[model_id] = 0.5
                continue
            
            probe = None
            try:
                # Load probe
                probe = LinearEoiProbe.load_from_checkpoint(
                    probe_path, device=self.config.device
                )
                probe.batch_size = self.config.batch_size
                
                # Get prediction
                indices, preds = probe.predict(([0], [prompt]))
                scores[model_id] = float(preds[0])
                
            except Exception as e:
                warnings.warn(f"⚠ Failed to load probe for {model_id}: {e}")
                scores[model_id] = 0.5
            finally:
                # Always unload to free memory, even on error
                if probe is not None:
                    self._unload_probe(probe)
                    del probe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return scores
    
    def _unload_probe(self, probe) -> None:
        """Unload a probe and free GPU memory."""
        import gc
        
        if probe is None:
            return
        
        # Move model to CPU first (helps release GPU memory)
        if hasattr(probe, 'model') and probe.model is not None:
            try:
                probe.model.cpu()
            except:
                pass
            probe.model = None
            del probe.model
        
        # Clear tokenizer
        if hasattr(probe, 'tokenizer'):
            probe.tokenizer = None
            del probe.tokenizer
        
        # Clear any other large attributes
        if hasattr(probe, 'best_probe'):
            probe.best_probe = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def compute_utilities(
        self,
        probe_scores: dict[str, float],
        lambda_val: float | None = None,
    ) -> dict[str, float]:
        """
        Compute utility scores for each model.
        
        Utility = probe_pred - lambda * normalized_cost
        
        Parameters
        ----------
        probe_scores : dict[str, float]
            Predicted success probabilities per model.
        lambda_val : float, optional
            Override the router's lambda_val for this computation.
        
        Returns
        -------
        dict[str, float]
            Mapping of model_id -> utility score
        """
        lam = lambda_val if lambda_val is not None else self.config.lambda_val
        
        utilities = {}
        for model_id, pred in probe_scores.items():
            cost = self._norm_costs.get(model_id, 0.5)
            utilities[model_id] = pred - lam * cost
        
        return utilities
    
    def route(
        self,
        prompt: str,
        lambda_val: float | None = None,
        return_scores: bool = False,
    ) -> str | tuple[str, dict[str, float], dict[str, float]]:
        """
        Select the best model for a given prompt.
        
        Parameters
        ----------
        prompt : str
            The input prompt to route.
        lambda_val : float, optional
            Override the router's lambda_val for this decision.
        return_scores : bool
            If True, also return probe and utility scores.
        
        Returns
        -------
        str
            The model_id of the selected model.
        tuple[str, dict, dict]
            If return_scores=True: (model_id, probe_scores, utility_scores)
        """
        probe_scores = self.get_probe_scores(prompt)
        utilities = self.compute_utilities(probe_scores, lambda_val)
        
        best_model = max(utilities.keys(), key=lambda m: utilities[m])
        
        if return_scores:
            return best_model, probe_scores, utilities
        return best_model
    
    def generate(
        self,
        prompt: str,
        lambda_val: float | None = None,
        **generation_kwargs,
    ) -> RouterResponse:
        """
        Route the prompt to the best model and generate a response.
        
        Parameters
        ----------
        prompt : str
            The input prompt.
        lambda_val : float, optional
            Override the router's lambda_val for this request.
        **generation_kwargs
            Override generation parameters (max_tokens, temperature, etc.)
        
        Returns
        -------
        RouterResponse
            Contains the generated text, selected model, and scoring details.
        """
        if not self._initialized:
            self.initialize(load_llms=True)
        
        # Route to best model
        result = self.route(prompt, lambda_val, return_scores=True)
        # Unpack tuple result when return_scores=True
        best_model, probe_scores, utilities = result  # type: ignore
        
        # Generate with the selected model
        engine = self._llm_engines.get(best_model)
        if engine is None:
            raise RuntimeError(
                f"LLM engine not loaded for {best_model}. "
                "Ensure vLLM is installed and model is accessible."
            )
        
        # Prepare generation params
        from vllm import SamplingParams
        
        cfg = self.model_configs[best_model]
        gen_params = cfg.get_generation_params()
        gen_params.update(generation_kwargs)
        
        sampling_params = SamplingParams(
            max_tokens=gen_params.get("max_tokens", 4096),
            temperature=gen_params.get("temperature", 0.7),
            n=1,  # Always generate 1 response for routing
        )
        
        # Generate
        outputs = engine.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        return RouterResponse(
            text=response_text,
            model_id=best_model,
            probe_scores=probe_scores,  # type: ignore[arg-type]
            utility_scores=utilities,  # type: ignore[arg-type]
            cost=self._raw_costs[best_model],
        )
    
    def batch_route(
        self,
        prompts: list[str],
        lambda_val: float | None = None,
    ) -> list[str]:
        """
        Route multiple prompts to their optimal models.
        
        Parameters
        ----------
        prompts : list[str]
            List of prompts to route.
        lambda_val : float, optional
            Override the router's lambda_val.
        
        Returns
        -------
        list[str]
            List of model_ids, one per prompt.
        """
        return [str(self.route(p, lambda_val)) for p in prompts]
    
    def set_lambda(self, lambda_val: float) -> "ProbeRouter":
        """
        Update the cost-accuracy tradeoff parameter.
        
        Parameters
        ----------
        lambda_val : float
            New lambda value in [0, 1].
        
        Returns
        -------
        ProbeRouter
            Self, for method chaining.
        """
        if not 0.0 <= lambda_val <= 1.0:
            raise ValueError(f"lambda_val must be in [0, 1], got {lambda_val}")
        self.config.lambda_val = lambda_val
        return self
    
    def update_costs(self, costs: dict[str, float]) -> "ProbeRouter":
        """
        Update model costs and recompute normalization.
        
        Parameters
        ----------
        costs : dict[str, float]
            Mapping of model_id -> cost in USD.
        
        Returns
        -------
        ProbeRouter
            Self, for method chaining.
        """
        for model_id, cost in costs.items():
            if model_id in self.model_configs:
                self.model_configs[model_id].cost = cost
                self._raw_costs[model_id] = cost
        
        # Recompute normalized costs
        if self.config.normalize_costs:
            lo, hi = min(self._raw_costs.values()), max(self._raw_costs.values())
            span = hi - lo if hi != lo else 1.0
            self._norm_costs = {m: (c - lo) / span for m, c in self._raw_costs.items()}
        
        return self
    
    def __repr__(self) -> str:
        models = ", ".join(self.model_ids[:3])
        if len(self.model_ids) > 3:
            models += f", ... ({len(self.model_ids)} total)"
        return f"ProbeRouter(models=[{models}], λ={self.config.lambda_val})"
