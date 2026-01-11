import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import math
import einops
import numpy as np
import os
import json
from pathlib import Path
import joblib

from .base_probe import Probe
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from itertools import product

from typing import List, Tuple, Optional
from llms_know_difficulty.config import ROOT_ACTIVATION_DATA_DIR
from llms_know_difficulty.probe.probe_utils.sklearn_probe import sk_activation_utils, sk_train_utils
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm
from llms_know_difficulty.metrics import compute_metrics

ROOT_ACTIVATION_DATA_DIR = os.path.join(ROOT_ACTIVATION_DATA_DIR,"sklearn_probe")


class SklearnProbe(Probe):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._has_setup_run = False
        self.model_name = config.get("probing_model_name", "gpt2")
        self.tokenizer = None
        self.model = None
        self.device = None
        self.d_model = None
        self.batch_size = getattr(self.config, "batch_size", 16)
        self.max_length = getattr(self.config, "max_length", 512)
        self.alpha_grid = getattr(self.config, "alpha_grid", [1.0])
        
        # Data indices storage
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Activation storage
        self.train_activations = None
        self.train_labels = None
        self.val_activations = None
        self.val_labels = None
        self.test_activations = None
        self.test_labels = None
        self.layer_indices = None
        self.positions: List[int] = [-1]  # default to last token if none here

        # Probe storage
        self.weights = None
        self.probing_layer = None
        self.probing_position = None
        
        # Best probe metadata
        self.best_probe = None
        self.best_pos_idx = None
        self.best_position_value = None  # Actual position value (e.g., -1, -2) for use in predict()
        self.best_layer_idx = None
        self.best_alpha = None
        self.best_val_score:float|None = None
        self.test_score = None
        self.task_type: Optional[str] = None
        self.metric_name: Optional[str] = None

    def name(self) -> str:
        """The name of the probe."""
        return "eoi_probe"

    def init_model(self, config: dict):
        """
        Load a prior training checkpoint.
        """
        raise NotImplementedError("Initializing a model from a checkpoint is not implemented for the eoi probe.")

    def setup(self, model_name: str, device: str) -> "SklearnProbe":
        """
        Any pre-training loading steps, run before .fit or .predict.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.d_model = self.model.config.hidden_size
        self.device = device
        self._has_setup_run = True

        return self

    def _extract_and_load_all_activations(self, train_texts: List[str], train_labels: List[float], val_texts: List[str], val_labels: List[float], test_texts: Optional[List[str]] = None, test_labels: Optional[List[float]] = None) -> None:
        """
        Extract and cache activations for train, val, and optionally test splits, with logging.
        Stores results in instance variables for use in fit/predict methods.
        
        Args:
            train_texts: List of training prompts
            train_labels: List of training labels
            val_texts: List of validation prompts
            val_labels: List of validation labels
            test_texts: Optional list of test prompts
            test_labels: Optional list of test labels
        """
        n_test = len(test_texts) if test_texts is not None else 0
        print(f"\nProcessing activations for {len(train_texts)} train, {len(val_texts)} val" + 
              (f", and {n_test} test samples..." if test_texts is not None else " samples..."))
        activation_data = {}
        
        splits = [("train", train_texts, train_labels), ("val", val_texts, val_labels)]
        if test_texts is not None and test_labels is not None:
            splits.append(("test", test_texts, test_labels))
        
        for split_name, texts, labels in splits:
            print(f"  {split_name.capitalize()} split...", end=" ")
            
            activation_data[split_name] = sk_activation_utils.extract_or_load_activations(
                model=self.model,
                tokenizer=self.tokenizer,
                texts=texts,
                labels=labels,
                model_name=self.model_name,
                split=split_name,  # Pass split type explicitly (train/val/test)
                device=self.device,
                batch_size=self.batch_size,
                max_length=self.max_length,
                eoi_tokens=None,
                layer_indices=None,
                cache_dir=ROOT_ACTIVATION_DATA_DIR,
                use_cache=True,
            )
            
            # Log activation details
            shape = activation_data[split_name]['activations'].shape
            from_cache = activation_data[split_name]['from_cache']
            cache_status = "cached" if from_cache else "extracted"
            print(f"Shape [N={shape[0]}, L={shape[1]}, P={shape[2]}, D={shape[3]}] ({cache_status})")
        
        # Store for use in fit/predict methods
        self.train_activations = activation_data['train']['activations']
        self.train_labels = activation_data['train']['labels']
        self.val_activations = activation_data['val']['activations']
        self.val_labels = activation_data['val']['labels']
        if "test" in activation_data:
            self.test_activations = activation_data['test']['activations']
            self.test_labels = activation_data['test']['labels']
        self.layer_indices = activation_data['train']['layer_indices']
        self.positions = activation_data['train']['positions']

    def _select_best_alpha_on_validation(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, alpha_grid: List[float]) -> Tuple[float, dict]:
        """
        Select best alpha by training probes for each alpha and evaluating on validation set.
        
        Args:
            x_train: Training activations [N, D]
            y_train: Training labels [N]
            x_val: Validation activations [N_val, D]
            y_val: Validation labels [N_val]
            alpha_grid: List of alpha values to try
        
        Returns:
            best_alpha, {alpha: score for all alphas}
        """
        alpha_scores = {}
        
        for alpha in alpha_grid:
            # Train probe with this alpha
            probe = self.fit(x_train, y_train, alpha, self.task_type)
            
            # Evaluate on validation set
            val_score = self._evaluate_probe(probe, x_val, y_val, full_metrics=False)
            alpha_scores[alpha] = val_score
        
        # Select alpha with best validation score
        best_alpha = max(alpha_scores.keys(), key=lambda a: alpha_scores[a])
        
        return best_alpha, alpha_scores
    
    def _evaluate_probe(self, probe, x: np.ndarray, y: np.ndarray, full_metrics: bool = False) -> float:
        """
        Evaluate a probe on data and return metric score.
        
        Args:
            probe: Trained sklearn model (Ridge or LogisticRegression)
            x: Activations [N, D]
            y: Labels [N]
            full_metrics: If True, compute all metrics. If False, compute only primary metric.
                         Use False for fast validation/training, True for final test evaluation.
        
        Returns:
            score (float): Spearman correlation for regression, ROC-AUC for classification
        """
        assert self.task_type is not None, "task_type must be set before calling _evaluate_probe"
        
        if self.task_type == "regression":
            y_pred = probe.predict(x)
        else:  # classification
            y_pred = probe.predict_proba(x)[:, 1]
        
        # Compute metrics using the centralized metrics module
        metrics = compute_metrics(y, y_pred, task_type=self.task_type, full_metrics=full_metrics)
        
        # Extract the appropriate metric based on task type
        score = metrics["spearman"] if self.task_type == "regression" else metrics["auc"]
        
        return float(score)
    
    def _train_and_select_probes(self, alpha_grid: Optional[List[float]] = None) -> None:
        """
        Train candidate probes for all (position, layer) combinations, select best by validation score.
        
        Args:
            alpha_grid: List of alpha values to search over. If None, uses config default.
        """
        n_positions = len(self.positions)
        n_layers = len(self.layer_indices)
        
        if alpha_grid is None:
            alpha_grid = self.alpha_grid
        
        print(f"\n{'='*80}")
        print(f"Training candidate probes: {n_positions} positions × {n_layers} layers")
        if len(alpha_grid) > 1:
            print(f"Alpha grid search over {len(alpha_grid)} values: {alpha_grid}")
        print(f"{'='*80}")
        
        candidates = {}  # (pos_idx, layer_idx) -> {probe, alpha, val_score, alpha_scores}
        best_val_score = -np.inf
        best_key = None
        
        for pos_idx in range(n_positions):
            pos = self.positions[pos_idx]
            print(f"\nPosition {pos}:")
            
            for layer_idx in tqdm(self.layer_indices, desc=f"  Layer progress"):
                # Extract activations for this (pos, layer) from training and validation splits
                try:
                    x_train = self.train_activations[:, layer_idx, pos_idx, :].numpy()
                    x_val = self.val_activations[:, layer_idx, pos_idx, :].numpy()
                except:
                    x_train = self.train_activations[:, layer_idx, pos_idx, :].detach().cpu().numpy()
                    x_val = self.val_activations[:, layer_idx, pos_idx, :].detach().cpu().numpy()
                
                # Extract test activations if available
                if self.test_activations is not None:
                    try:
                        x_test = self.test_activations[:, layer_idx, pos_idx, :].numpy()
                    except:
                        x_test = self.test_activations[:, layer_idx, pos_idx, :].detach().cpu().numpy()
                else:
                    x_test = None
                
                # Convert labels to numpy if needed
                y_train = np.array(self.train_labels) if not isinstance(self.train_labels, np.ndarray) else self.train_labels
                y_val = np.array(self.val_labels) if not isinstance(self.val_labels, np.ndarray) else self.val_labels
                y_test = None
                if self.test_labels is not None:
                    y_test = np.array(self.test_labels) if not isinstance(self.test_labels, np.ndarray) else self.test_labels
                
                # Select best alpha using validation set
                if len(alpha_grid) > 1:
                    best_alpha, alpha_scores = self._select_best_alpha_on_validation(x_train, y_train, x_val, y_val, alpha_grid)
                else:
                    best_alpha = alpha_grid[0]
                    alpha_scores = {best_alpha: None}
                
                # Train final probe with selected alpha
                probe = self.fit(x_train, y_train, best_alpha, self.task_type)
                
                # Evaluate on validation set
                val_score = self._evaluate_probe(probe, x_val, y_val, full_metrics=False)
                
                # Store candidate
                candidates[(pos_idx, layer_idx)] = {
                    'probe': probe,
                    'alpha': best_alpha,
                    'val_score': val_score,
                    'alpha_scores': alpha_scores,
                    'x_test': x_test,
                    'test_activated': False  # Will be set after selection
                }
                
                # Track best so far
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_key = (pos_idx, layer_idx)
        
        # Select best probe by validation score
        self.best_pos_idx, self.best_layer_idx = best_key
        self.best_probe = candidates[best_key]['probe']
        self.best_alpha = candidates[best_key]['alpha']
        self.best_val_score = candidates[best_key]['val_score']
        
        # Store the actual position value (e.g., -1, -2, etc.) for use in predict()
        self.best_position_value = self.positions[self.best_pos_idx]
        
        # Evaluate on test set if available
        if self.test_activations is not None and self.test_labels is not None:
            x_test_best = candidates[best_key]['x_test']
            y_test = np.array(self.test_labels) if not isinstance(self.test_labels, np.ndarray) else self.test_labels
            # Use full_metrics=True for final test evaluation
            self.test_score = self._evaluate_probe(self.best_probe, x_test_best, y_test, full_metrics=True)
        
        # Print summary
        best_pos = self.positions[self.best_pos_idx]
        print(f"\n{'='*80}")
        print(f"Best Probe Selected:")
        print(f"  Position: {best_pos}")
        print(f"  Layer: {self.best_layer_idx}")
        print(f"  Alpha: {self.best_alpha}")
        print(f"  Validation {self.metric_name}: {self.best_val_score:.4f}")
        if self.test_score is not None:
            print(f"  Test {self.metric_name}: {self.test_score:.4f}")
        print(f"{'='*80}")

    def train(self, train_data: Tuple[List[int], List[str], List[float]], val_data: Tuple[List[int], List[str], List[float]], test_data: Optional[Tuple[List[int], List[str], List[float]]] = None, alpha_grid: Optional[List[float]] = None) -> "SklearnProbe":
        """
        Train probes on train data, select best using validation data, optionally evaluate on test data.
        
        Extracts activations for train and val splits (test is optional).
        Trains a candidate probe at each (position, layer) combination with optional alpha grid search.
        Selects the single best probe based on validation set performance.
        If test data is provided, reports final performance on held-out test set.
        
        Args:
            train_data: Tuple of (indices, prompts, targets) for training
            val_data: Tuple of (indices, prompts, targets) for validation
            test_data: Optional tuple of (indices, prompts, targets) for testing
            alpha_grid: Optional list of alpha values to search. If None, uses [1.0]
        
        Returns:
            self (for method chaining)
        """
        if not self._has_setup_run:
            raise RuntimeError("Must call setup() before train()")
        
        # Convert data tuples to lists
        train_idxs, train_texts, train_labels = train_data
        val_idxs, val_texts, val_labels = val_data
        test_idxs = None
        test_texts = None
        test_labels = None
        if test_data is not None:
            test_idxs, test_texts, test_labels = test_data
        
        # Store indices for reference
        self.train_indices = train_idxs
        self.val_indices = val_idxs
        self.test_indices = test_idxs
        
        print(f"\n{'='*80}")
        print(f"SklearnProbe Training: Extracting activations")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}")
        if test_texts is not None:
            print(f"Test samples: {len(test_texts)}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Extract and load activations for train, val, and optionally test splits
        self._extract_and_load_all_activations(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
        
        # Infer task type from labels
        self.task_type = sk_train_utils.infer_task_type(train_labels)
        self.metric_name = "spearman" if self.task_type == "regression" else "auc"
        print(f"Task type: {self.task_type}")
        print(f"Metric: {self.metric_name}")
        
        # Assert task_type is set for type checking
        assert self.task_type is not None, "task_type should be set by infer_task_type"
        
        # Train and select probes
        self._train_and_select_probes(alpha_grid)

        return self

    def fit(self, prompt_activations: np.ndarray, targets: np.ndarray, alpha: float, task_type: str) -> Ridge | LogisticRegression:
        """
        Train a single probe model with specified alpha and task type.
        
        This is the primary factory method for instantiating and training models.
        It eliminates code duplication by centralizing the task_type logic.
        
        Args:
            prompt_activations: Activation features [N, D]
            targets: Target labels [N]
            alpha: Regularization parameter (used for Ridge alpha or LogisticRegression C=1/alpha)
            task_type: "regression" or "classification"
        
        Returns:
            Trained probe model (Ridge or LogisticRegression)
        """
        if task_type == "regression":
            probe_model = Ridge(
                alpha=alpha,
                fit_intercept=False
            )
        else:  # classification
            if alpha == 0:
                # No regularization
                probe_model = LogisticRegression(
                    penalty=None,
                    fit_intercept=False,
                    solver="lbfgs",
                    max_iter=1000
                )
            else:
                # L2 regularization with C = 1/alpha
                probe_model = LogisticRegression(
                    penalty="l2",
                    C=1.0 / alpha,
                    fit_intercept=False,
                    solver="lbfgs",
                    max_iter=1000
                )
        
        probe_model.fit(prompt_activations, targets)
        return probe_model

    def predict(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on new data using the best trained probe.
        
        Args:
            data: Either:
                - Tuple of (indices, prompts, targets) - extracts activations from prompts
                - List of prompt strings - extracts activations first
                - Numpy array of pre-extracted activations
        
        Returns:
            Tuple of (indices_tensor, predictions_tensor):
                - indices_tensor: torch.Tensor of dtype int32 with shape [N]
                - predictions_tensor: torch.Tensor of dtype float32 with shape [N]
        """
        if self.best_probe is None:
            raise RuntimeError("Must call train() before predict()")
        
        assert self.task_type is not None, "task_type must be set before calling predict"
        
        # Handle tuple input (indices, prompts, targets) and extract indices
        if isinstance(data, tuple) and len(data) == 3:
            indices = list(data[0])  # Extract indices from (indices, prompts, targets)
            prompts = list(data[1])  # Extract prompts
        else:
            # For non-tuple inputs, generate sequential indices
            prompts = data
            if isinstance(prompts, list):
                indices = list(range(len(prompts)))
            else:  # numpy array
                indices = list(range(len(prompts)))
        
        # If prompts are strings, extract activations
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], str):
            # Extract activations for the best position across all layers
            with torch.no_grad():
                texts = prompts
                activations_data = sk_activation_utils.extract_activations_from_texts(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    texts=texts,
                    labels=None,  # We don't have labels for prediction
                    device=self.device,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    eoi_tokens=None,
                    layer_indices=None,  # Extract all layers
                    specified_eoi_position=self.best_position_value,  # Extract only the best position
                )
            
            # Extract activations for the best layer from the best position
            # When specified_eoi_position is used, shape is [N, L, D] (position dimension squeezed)
            # Index [:, best_layer_idx, :] to get [N, D]
            # activations_data is a tuple: (activations, labels, layer_indices, positions, n_eoi, d_model)
            activations = activations_data[0]  # [N, L, D] when specified_eoi_position used
            x_pred = activations[:, self.best_layer_idx, :].numpy()
        else:
            # Assume already extracted activations
            x_pred = np.array(prompts)
        
        # Make predictions
        if self.task_type == "regression":
            predictions = self.best_probe.predict(x_pred)
        else:  # classification
            predictions = self.best_probe.predict_proba(x_pred)[:, 1]
        
        # Return as formatted tensors for consistency with attn_probe
        return (
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor(predictions, dtype=torch.float32)
        )

    def save_probe(self, results_path: Path | str, metadata: dict | None = None) -> None:
        """
        Save the trained probe weights and metadata to disk.
        
        Args:
            results_path: Directory where to save the probe files
            metadata: Optional full metadata dict (e.g., with test metrics). 
                     If not provided, creates minimal metadata.
        
        Saves:
            - best_probe.joblib: The sklearn model with trained weights
            - probe_metadata.json: Probe configuration and layer/position info
        """
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        
        if self.best_probe is None:
            raise RuntimeError("Must call train() before save_probe()")
        
        # Save the trained sklearn model
        model_file = results_path / "best_probe.joblib"
        joblib.dump(self.best_probe, model_file)
        
        # If no metadata provided, create minimal metadata for loading
        if metadata is None:
            metadata = {
                "best_layer_idx": self.best_layer_idx,
                "best_position_value": self.best_position_value,
                "best_position_idx": self.best_pos_idx,
                "best_alpha": self.best_alpha,
                "best_val_score": float(self.best_val_score),
                "task_type": self.task_type,
                "model_name": self.model_name,
                "d_model": self.d_model,
            }
        
        metadata_file = results_path / "probe_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_from_checkpoint(cls, results_path: Path | str, device: str = "cuda") -> "SklearnProbe":
        """
        Load a trained probe from saved checkpoint.
        
        Args:
            results_path: Directory containing saved probe files
            device: Device to load the model on (for activation extraction)
        
        Returns:
            SklearnProbe instance ready for prediction
        """
        results_path = Path(results_path)
        
        # Load metadata
        metadata_file = results_path / "probe_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Load the sklearn model
        model_file = results_path / "best_probe.joblib"
        best_probe = joblib.load(model_file)
        
        # Create probe instance
        probe = cls(config={})
        
        # Restore state
        probe.best_probe = best_probe
        probe.best_layer_idx = metadata["best_layer_idx"]
        probe.best_position_value = metadata["best_position_value"]
        probe.best_pos_idx = metadata["best_position_idx"]
        probe.best_alpha = metadata["best_alpha"]
        probe.best_val_score = metadata["best_val_score"]
        probe.task_type = metadata["task_type"]
        probe.model_name = metadata["model_name"]
        probe.d_model = metadata["d_model"]
        
        # Setup the model for activation extraction
        if torch.cuda.is_available() and device == "cuda":
            device = "cuda"
        else:
            device = "cpu"
        
        probe.setup(model_name=probe.model_name, device=device)
        
        return probe
