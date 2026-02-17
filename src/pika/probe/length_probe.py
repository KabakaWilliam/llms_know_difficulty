"""
Length Baseline Probe for difficulty prediction.

A lightweight baseline using Length features extracted from problem text,
with spaCy preprocessing (lemmatization + stopword removal) and Ridge regression.
"""
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from transformers import AutoTokenizer
import spacy
import torch.optim

from .base_probe import Probe
from pika.metrics import compute_metrics
from pika.probe.probe_utils.linear_eoi_probe import linear_eoi_probe_train_utils
from pika.config import ROOT_ACTIVATION_DATA_DIR, TfidfProbeConfig

ROOT_ACTIVATION_DATA_DIR = os.path.join(ROOT_ACTIVATION_DATA_DIR,"length_probe")


class LengthProbe(Probe):
    """Simple length-based baseline that uses total_input_tokens to predict targets.
    
    Uses the 'total_input_tokens' field when available in the data,
    otherwise falls back to whitespace-based token counting.
    Performs temperature scaling for classification tasks like linear_eoi_probe.
    """
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.config = config or {}
        self._has_setup_run = True
        
        # Model & metadata
        self.best_model = None
        self.best_alpha = None
        self.best_val_score = None
        self.test_score = None
        self.task_type = None
        self.metric_name = None
        self.model_type = None
        
        # Calibration (Platt scaling for classification)
        self.platt_scaler: Optional[LogisticRegression] = None
        self.val_metrics_raw: Optional[dict] = None
        self.val_metrics_cal: Optional[dict] = None
    
    def name(self) -> str:
        return "length_probe"
    
    def init_model(self, config: dict):
        raise NotImplementedError("LengthProbe does not support loading from checkpoint.")
    
    def setup(self) -> "LengthProbe":
        """No setup needed for length probe."""
        self._has_setup_run = True
        return self
    
    def _get_lengths_from_data(self, data) -> np.ndarray:
        """Extract length features from data.
        
        Priority:
          1. If data is tuple with 4th element (total_input_tokens), use it
          2. If data is a 3-tuple and the 3rd element looks like token counts
             (all values > 1), use it (supports target-free inference tuples)
          3. If prompts are dicts with 'total_input_tokens' key, use that
          4. Fallback: whitespace token count
        """
        # Handle tuple input with optional token counts
        if isinstance(data, tuple):
            prompts = data[1] if len(data) > 1 else data[0]
            
            # Check data[3] first (standard 4-tuple with targets)
            token_counts = data[3] if len(data) > 3 else None
            
            # If no data[3], check data[2] — it might be token counts
            # when targets are omitted: (indices, prompts, token_counts)
            if token_counts is None and len(data) == 3:
                candidate = data[2]
                if (isinstance(candidate, (list, tuple)) and len(candidate) > 0
                        and isinstance(candidate[0], (int, float))
                        and all(v > 1 for v in candidate)):
                    token_counts = candidate
            
            if token_counts is not None:
                return np.array(token_counts, dtype=float)
        else:
            prompts = data
        
        # Check if prompts contain total_input_tokens
        if prompts and isinstance(prompts[0], dict) and 'total_input_tokens' in prompts[0]:
            return np.array([p.get('total_input_tokens', 0) for p in prompts], dtype=float)
        
        # Fallback: whitespace token count
        return np.array([len(str(p).split()) for p in prompts], dtype=float)
    
    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Fit Platt scaling calibration for classification probes.
        Uses logistic regression on validation logits to calibrate probabilities."""
        assert self.task_type == "classification"
        assert self.best_model is not None
        
        # Get logits (NOT probabilities)
        logits = self.best_model.decision_function(x_cal.reshape(-1, 1)).reshape(-1, 1)
        
        # Fit logistic regression: sigmoid(logits) = sigmoid(a*logits + b)
        self.platt_scaler = LogisticRegression(random_state=42, max_iter=1000)
        self.platt_scaler.fit(logits, y_cal)
    
    def train(self,
              train_data: Tuple[List[int], List[str], List[float]],
              val_data: Tuple[List[int], List[str], List[float]],
              test_data: Optional[Tuple[List[int], List[str], List[float]]] = None,
              alpha_grid: Optional[List[float]] = None) -> "LengthProbe":
        """Train length baseline with alpha grid search.
        
        Args:
            train_data: Tuple of (indices, prompts, targets, [optional: total_input_tokens])
            val_data: Tuple of (indices, prompts, targets, [optional: total_input_tokens])
            test_data: Optional tuple of (indices, prompts, targets, [optional: total_input_tokens])
            alpha_grid: Optional list of alpha values to search
        """
        # Extract features
        X_train = self._get_lengths_from_data(train_data).reshape(-1, 1)
        X_val = self._get_lengths_from_data(val_data).reshape(-1, 1)
        X_test = self._get_lengths_from_data(test_data).reshape(-1, 1) if test_data is not None else None
        
        y_train = np.array(train_data[2])
        y_val = np.array(val_data[2])
        y_test = np.array(test_data[2]) if test_data is not None else None
        
        # Infer task type
        self.task_type = linear_eoi_probe_train_utils.infer_task_type(train_data[2])
        self.metric_name = "spearman" if self.task_type == "regression" else "auc"
        self.model_type = "ridge" if self.task_type == "regression" else "logistic"
        
        print(f"\n{'='*80}")
        print(f"LengthProbe Training")
        print(f"{'='*80}")
        print(f"Task type: {self.task_type}")
        print(f"Train samples: {len(y_train)}")
        print(f"Val samples: {len(y_val)}")
        if test_data is not None:
            print(f"Test samples: {len(y_test)}")
        
        # Grid search over alpha
        if alpha_grid is None:
            alpha_grid = getattr(self.config, 'alpha_grid', [0.001, 0.01, 0.1, 1, 10, 100, 1000])
        
        print(f"\nGrid search over alpha values: {alpha_grid}")
        alpha_results = {}
        
        for alpha in alpha_grid:
            if self.task_type == "regression":
                model = Ridge(alpha=alpha, random_state=42)
            else:
                C = 1.0 / alpha if alpha != 0 else 1.0
                model = LogisticRegression(C=C, random_state=42, max_iter=1000)
            
            model.fit(X_train, y_train)
            
            # Get predictions
            if self.task_type == "regression":
                val_preds = model.predict(X_val)
            else:
                val_preds = model.predict_proba(X_val)[:, 1]
            
            val_metrics = compute_metrics(y_val, val_preds, task_type=self.task_type, full_metrics=False)
            alpha_results[alpha] = val_metrics[self.metric_name]
            
            print(f"  Alpha: {alpha:8.3f} | Val {self.metric_name}: {val_metrics[self.metric_name]:.4f}")
        
        # Select best alpha
        valid_alphas = {a: s for a, s in alpha_results.items() if not np.isnan(s)}
        if not valid_alphas:
            raise ValueError("All alpha values produced NaN scores.")
        
        self.best_alpha = max(valid_alphas.keys(), key=lambda a: valid_alphas[a])
        self.best_val_score = alpha_results[self.best_alpha]
        
        print(f"\n{'='*50}")
        print(f"Best alpha: {self.best_alpha}")
        print(f"Best val {self.metric_name}: {self.best_val_score:.4f}")
        print(f"{'='*50}")
        
        # Retrain on train+val with best alpha
        print(f"\nRetraining on full train+val set with best alpha...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        
        if self.task_type == "regression":
            self.best_model = Ridge(alpha=self.best_alpha, random_state=42)
        else:
            C = 1.0 / self.best_alpha if self.best_alpha != 0 else 1.0
            self.best_model = LogisticRegression(C=C, random_state=42, max_iter=1000)
        
        self.best_model.fit(X_full, y_full)
        print(f"✓ Model trained on {len(y_full)} samples")
        
        # Calibration (Platt scaling) for classification
        if self.task_type == "classification" and self.platt_scaler is None:
            # Raw probabilities before calibration
            raw_val_probs = self.best_model.predict_proba(X_val)[:, 1]
            raw_val_metrics = compute_metrics(
                y_val, raw_val_probs,
                task_type="classification",
                full_metrics=True
            )
            
            # Fit Platt scaling on validation logits
            self.calibrate(X_val, y_val)
            
            # Calibrated probabilities using Platt scaling
            logits = self.best_model.decision_function(X_val).reshape(-1, 1)
            cal_val_probs = self.platt_scaler.predict_proba(logits)[:, 1]
            
            cal_val_metrics = compute_metrics(
                y_val, cal_val_probs,
                task_type="classification",
                full_metrics=True
            )
            
            # Store for logging
            self.val_metrics_raw = raw_val_metrics
            self.val_metrics_cal = cal_val_metrics
            
            print("\nCalibration (Platt scaling on validation set):")
            print(f"  Platt scaling fitted on {len(y_val)} validation samples")
            print(f"  ECE before: {raw_val_metrics.get('ece', 'N/A')}")
            print(f"  ECE after : {cal_val_metrics.get('ece', 'N/A')}")
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            print(f"\nEvaluating on test set...")
            
            if self.task_type == "regression":
                test_preds = self.best_model.predict(X_test)
            else:
                logits = self.best_model.decision_function(X_test).reshape(-1, 1)
                if self.platt_scaler is not None:
                    # Use Platt scaling if fitted
                    test_preds = self.platt_scaler.predict_proba(logits)[:, 1]
                else:
                    # Fallback to uncalibrated probabilities
                    test_preds = 1 / (1 + np.exp(-logits.ravel()))
            
            test_metrics = compute_metrics(y_test, test_preds, task_type=self.task_type, full_metrics=True)
            self.test_score = test_metrics[self.metric_name]
            
            print(f"\n{'='*50}")
            print(f"Test Results")
            print(f"{'='*50}")
            print(f"Test {self.metric_name}: {self.test_score:.4f}")
            if self.task_type == "regression":
                print(f"Test MSE: {test_metrics.get('mse', 'N/A'):.4f}")
                print(f"Test MAE: {test_metrics.get('mae', 'N/A'):.4f}")
            print(f"{'='*50}")
        
        return self
    
    def fit(self):
        raise NotImplementedError("Use train() method instead of fit().")
    
    def predict(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on new data.
        
        Args:
            data: Either:
                - Tuple of (indices, prompts, [optional: targets], [optional: total_input_tokens])
                - List of prompts
              Targets are NOT required for prediction.
        
        Returns:
            Tuple of (indices_tensor, predictions_tensor)
        """
        if self.best_model is None:
            raise RuntimeError("Must call train() before predict()")
        
        # Handle tuple input — targets (data[2]) are optional and never used
        if isinstance(data, tuple) and len(data) >= 2:
            indices = list(data[0])
        else:
            prompts = data if isinstance(data, list) else list(data)
            indices = list(range(len(prompts)))
        
        # Extract features
        X = self._get_lengths_from_data(data).reshape(-1, 1)
        
        # Predict
        if self.task_type == "regression":
            predictions = self.best_model.predict(X)
        else:
            logits = self.best_model.decision_function(X).reshape(-1, 1)
            if self.platt_scaler is not None:
                # Use Platt scaling if fitted
                predictions = self.platt_scaler.predict_proba(logits)[:, 1]
            else:
                # Fallback to uncalibrated probabilities
                predictions = 1 / (1 + np.exp(-logits.ravel()))
        
        return (
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor(predictions, dtype=torch.float32)
        )
    
    def save_probe(self, results_path: Path | str, metadata: dict | None = None) -> None:
        """Save the trained probe to disk.
        
        Args:
            results_path: Directory where to save the probe files
            metadata: Optional full metadata dict (e.g., with test metrics)
        
        Saves:
            - model.joblib: The Ridge/Logistic regression model
            - platt_scaler.joblib: The Platt scaling calibrator (if classification)
            - probe_metadata.json: Probe configuration and results
        """
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        
        if self.best_model is None:
            raise RuntimeError("Must call train() before save_probe()")
        
        # Save model
        joblib.dump(self.best_model, results_path / "model.joblib")
        
        # Save Platt scaler if available (classification only)
        if self.platt_scaler is not None:
            joblib.dump(self.platt_scaler, results_path / "platt_scaler.joblib")
        
        # Create metadata if not provided
        if metadata is None:
            metadata = self.get_probe_metadata()
        
        # Save metadata
        with open(results_path / "probe_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_from_checkpoint(cls, results_path: Path | str) -> "LengthProbe":
        """Load a trained probe from checkpoint.
        
        Args:
            results_path: Directory containing saved probe files
        
        Returns:
            LengthProbe instance ready for prediction
        """
        results_path = Path(results_path)
        
        # Load metadata
        with open(results_path / "probe_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load model
        model = joblib.load(results_path / "model.joblib")
        
        # Load Platt scaler if available (classification only)
        platt_scaler_path = results_path / "platt_scaler.joblib"
        platt_scaler = joblib.load(platt_scaler_path) if platt_scaler_path.exists() else None
        
        # Create probe instance
        probe = cls(config={})
        probe.setup()
        
        # Restore state
        probe.best_model = model
        probe.platt_scaler = platt_scaler
        probe.best_alpha = metadata.get("best_alpha")
        probe.best_val_score = metadata.get("best_val_score")
        probe.test_score = metadata.get("test_score")
        probe.task_type = metadata.get("task_type")
        probe.model_type = metadata.get("model_type")
        probe.metric_name = "spearman" if probe.task_type == "regression" else "auc"
        
        return probe
    
    def get_probe_metadata(self) -> dict:
        """Return probe metadata for saving."""
        metadata = {
            'best_alpha': self.best_alpha,
            'best_val_score': float(self.best_val_score) if self.best_val_score is not None else None,
            'test_score': float(self.test_score) if self.test_score is not None else None,
            'task_type': self.task_type,
            'model_type': self.model_type,
            'metric_name': self.metric_name,
            'probe_type': self.name(),
        }
        
        # Add calibration metrics for classification
        if self.task_type == "classification":
            if self.val_metrics_raw is not None:
                metadata['ece_before_calibration'] = self.val_metrics_raw.get('ece')
            if self.val_metrics_cal is not None:
                metadata['ece_after_calibration'] = self.val_metrics_cal.get('ece')
            if self.platt_scaler is not None:
                metadata['calibration_method'] = 'platt_scaling'
        
        return metadata