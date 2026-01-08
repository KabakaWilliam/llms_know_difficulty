import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import math
import einops
import numpy as np
import os

from .base_probe import Probe
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from itertools import product

from typing import List, Tuple, Optional
from ..config import ROOT_ACTIVATION_DATA_DIR
from . import sk_activation_utils, sk_train_utils
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT_ACTIVATION_DATA_DIR = os.path.join(ROOT_ACTIVATION_DATA_DIR,"sklearn_probe")


class SklearnProbe(Probe):
    def __init__(self, config):
        super().__init__(config)
        self._has_setup_run = False
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.d_model = None
        
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
        self.best_val_score = None
        self.test_score = None
        self.task_type = None
        self.metric_name = None

    def name(self) -> str:
        """The name of the probe."""
        return "sklearn_probe"

    def init_model(self, config: dict):
        """
        Load a prior training checkpoint.
        """
        pass

    def setup(self, model_name: str, device: str) -> None:
        """
        Any pre-training loading steps, run before .fit or .predict.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.d_model = self.model.config.hidden_size
        self.device = device
        self._has_setup_run = True

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
                batch_size=16,
                max_length=512,
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
            val_score = self._evaluate_probe(probe, x_val, y_val)
            alpha_scores[alpha] = val_score
        
        # Select alpha with best validation score
        best_alpha = max(alpha_scores.keys(), key=lambda a: alpha_scores[a])
        
        return best_alpha, alpha_scores
    
    def _evaluate_probe(self, probe, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate a probe on data and return metric score.
        
        Args:
            probe: Trained sklearn model (Ridge or LogisticRegression)
            x: Activations [N, D]
            y: Labels [N]
        
        Returns:
            score (float): Spearman correlation for regression, ROC-AUC for classification
        """
        if self.task_type == "regression":
            y_pred = probe.predict(x)
            corr_result = spearmanr(y, y_pred)
            score = corr_result[0] if isinstance(corr_result, tuple) else corr_result.correlation
        else:  # classification
            y_proba = probe.predict_proba(x)[:, 1]
            score = roc_auc_score(y, y_proba)
        
        return float(score)
    
    def _train_and_select_probes(self, alpha_grid: Optional[List[float]] = None) -> None:
        """
        Train candidate probes for all (position, layer) combinations, select best by validation score.
        
        Args:
            alpha_grid: List of alpha values to search over. If None or length 1, uses single alpha.
        """
        n_positions = len(self.positions)
        n_layers = len(self.layer_indices)
        
        if alpha_grid is None:
            alpha_grid = [1.0]
        
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
                val_score = self._evaluate_probe(probe, x_val, y_val)
                
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
            self.test_score = self._evaluate_probe(self.best_probe, x_test_best, y_test)
        
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

    def train(self, train_data: Tuple[List[str], List[float]], val_data: Tuple[List[str], List[float]], test_data: Optional[Tuple[List[str], List[float]]] = None, alpha_grid: Optional[List[float]] = None) -> "SklearnProbe":
        """
        Train probes on train data, select best using validation data, optionally evaluate on test data.
        
        Extracts activations for train and val splits (test is optional).
        Trains a candidate probe at each (position, layer) combination with optional alpha grid search.
        Selects the single best probe based on validation set performance.
        If test data is provided, reports final performance on held-out test set.
        
        Args:
            train_data: Tuple of (prompts, targets) for training
            val_data: Tuple of (prompts, targets) for validation
            test_data: Optional tuple of (prompts, targets) for testing
            alpha_grid: Optional list of alpha values to search. If None, uses [1.0]
        
        Returns:
            self (for method chaining)
        """
        if not self._has_setup_run:
            raise RuntimeError("Must call setup() before train()")
        
        # Convert data tuples to lists
        train_texts, train_labels = list(train_data[0]), list(train_data[1])
        val_texts, val_labels = list(val_data[0]), list(val_data[1])
        test_texts = None
        test_labels = None
        if test_data is not None:
            test_texts, test_labels = list(test_data[0]), list(test_data[1])
        
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
        self.metric_name = "spearman" if self.task_type == "regression" else "roc_auc"
        print(f"Task type: {self.task_type}")
        print(f"Metric: {self.metric_name}")
        
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

    def predict(self, prompts: List[str] | np.ndarray) -> np.ndarray:
        """
        Make predictions on new prompts using the best trained probe.
        
        If prompts are strings, extracts activations first.
        If prompts are numpy arrays (pre-extracted activations), uses them directly.
        
        Args:
            prompts: Either list of prompt strings or pre-extracted activations array
        
        Returns:
            Predictions from best probe (logits for regression, probabilities for classification)
        """
        if self.best_probe is None:
            raise RuntimeError("Must call train() before predict()")
        
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
                    batch_size=16,
                    max_length=512,
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
        
        return predictions
