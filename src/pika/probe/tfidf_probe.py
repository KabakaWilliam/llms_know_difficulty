"""
TF-IDF Baseline Probe for difficulty prediction.

A lightweight baseline using TF-IDF features extracted from problem text,
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
import spacy
import torch.optim

from .base_probe import Probe
from pika.metrics import compute_metrics
from pika.probe.probe_utils.linear_eoi_probe import linear_eoi_probe_train_utils
from pika.config import ROOT_ACTIVATION_DATA_DIR, TfidfProbeConfig

ROOT_ACTIVATION_DATA_DIR = os.path.join(ROOT_ACTIVATION_DATA_DIR,"tfidf_probe")


class TfidfProbe(Probe):
    """TF-IDF baseline probe with spaCy preprocessing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._has_setup_run = False
        
        # spaCy model
        self.nlp = None
        
        # Vectorizer and model
        self.vectorizer = None
        self.best_model = None
        self.fit_intercept = getattr(self.config, "fit_intercept", False)

        self.alpha_grid = getattr(self.config, "alpha_grid", [1.0])
        
        # Metadata
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.best_alpha = None
        self.best_val_score = None
        self.test_score = None
        self.task_type = None
        self.metric_name = None
        self.model_type = None  # "ridge" for regression, "logistic" for classification
        
        # Calibration (Platt scaling for classification)
        self.platt_scaler: Optional[LogisticRegression] = None
        self.val_metrics_raw: Optional[dict] = None
        self.val_metrics_cal: Optional[dict] = None
    
    def name(self) -> str:
        """The name of the probe."""
        return "tfidf_probe"
    
    
    def init_model(self, config: dict):
        """
        Load a prior training checkpoint.
        """
        raise NotImplementedError("Initializing a model from a checkpoint is not implemented for the eoi probe.")
    
    def fit(self):
        raise NotImplementedError("Initializing a model from a checkpoint is not implemented for the eoi probe.")    
    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Fit Platt scaling calibration for classification probes.
        Uses logistic regression on validation logits to calibrate probabilities.
        """
        assert self.task_type == "classification"
        assert self.best_model is not None

        # Get logits (NOT probabilities)
        logits = self.best_model.decision_function(x_cal).reshape(-1, 1)
        
        # Fit logistic regression: sigmoid(logits) = sigmoid(a*logits + b)
        self.platt_scaler = LogisticRegression(random_state=42, max_iter=1000)
        self.platt_scaler.fit(logits, y_cal)            

    def setup(self) -> "TfidfProbe":
        """
        Load spaCy model for text preprocessing.
        
        Args:
            model_name: Model name (not used for TfidfProbe, kept for schema compatibility)
            device: Device (not used for TfidfProbe, kept for schema compatibility)
        
        Returns:
            self (for method chaining)
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer()
        
        self._has_setup_run = True
        return self
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts: lemmatization + stopword removal with spaCy.
        
        Args:
            texts: List of raw text strings
        
        Returns:
            List of preprocessed text strings
        """
        processed = []
        for text in texts:
            doc = self.nlp(text)
            # Lemmatize and remove stopwords
            lemmas = [token.lemma_ for token in doc if not token.is_stop]
            processed.append(' '.join(lemmas))
        return processed
    
    def train(self,
              train_data: Tuple[List[int], List[str], List[float]],
              val_data: Tuple[List[int], List[str], List[float]],
              test_data: Optional[Tuple[List[int], List[str], List[float]]] = None,
              alpha_grid: Optional[List[float]] = None) -> "TfidfProbe":
        """
        Train TF-IDF baseline with alpha grid search on validation set.
        
        Args:
            train_data: Tuple of (indices, prompts, targets) for training
            val_data: Tuple of (indices, prompts, targets) for validation
            test_data: Optional tuple of (indices, prompts, targets) for testing
            alpha_grid: Optional list of alpha values to search. If None, uses sensible defaults.
        
        Returns:
            self (for method chaining)
        """
        if not self._has_setup_run:
            raise RuntimeError("Must call setup() before train()")
        
        # Unpack data
        train_idxs, train_texts, train_labels = train_data
        val_idxs, val_texts, val_labels = val_data
        test_idxs = test_texts = test_labels = None
        if test_data is not None:
            test_idxs, test_texts, test_labels = test_data
        
        # Store indices
        self.train_indices = train_idxs
        self.val_indices = val_idxs
        self.test_indices = test_idxs
        
        # Print info
        print(f"\n{'='*80}")
        print(f"TfidfProbe Training")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}")
        if test_texts is not None:
            print(f"Test samples: {len(test_texts)}")
        
        # Preprocess texts
        print(f"\nPreprocessing texts with spaCy...")
        train_processed = self._preprocess_texts(train_texts)
        val_processed = self._preprocess_texts(val_texts)
        if test_texts is not None:
            test_processed = self._preprocess_texts(test_texts)
        else:
            test_processed = None
        
        # Fit vectorizer on training data
        print(f"Fitting TF-IDF vectorizer...")
        X_train = self.vectorizer.fit_transform(train_processed)
        X_val = self.vectorizer.transform(val_processed)
        X_test = self.vectorizer.transform(test_processed) if test_processed is not None else None
        
        y_train = np.array(train_labels)
        y_val = np.array(val_labels)
        y_test = np.array(test_labels) if test_labels is not None else None
        
        print(f"Train features shape: {X_train.shape}")
        print(f"Val features shape: {X_val.shape}")
        
        # Infer task type
        self.task_type = linear_eoi_probe_train_utils.infer_task_type(train_labels)
        self.metric_name = "spearman" if self.task_type == "regression" else "auc"
        self.model_type = "ridge" if self.task_type == "regression" else "logistic"
        print(f"Task type: {self.task_type}")
        print(f"Model type: {self.model_type}")
        
        print(f"\nGrid search over alpha values: {self.alpha_grid}")
        alpha_results = {}
        
        for alpha in self.alpha_grid:
            # Select model based on task type
            if self.task_type == "regression":
                model = Ridge(alpha=alpha, fit_intercept=self.fit_intercept, random_state=42)
            else:  # classification
                C = 1.0 / alpha if alpha != 0 else 1.0
                model = LogisticRegression(C=C, fit_intercept=self.fit_intercept, random_state=42, max_iter=1000)
            
            model.fit(X_train, y_train)
            
            # Get predictions (different approach for regression vs classification)
            if self.task_type == "regression":
                val_preds = model.predict(X_val)
            else:  # classification - use probability of positive class
                val_preds = model.predict_proba(X_val)[:, 1]
            
            val_metrics = compute_metrics(y_val, val_preds, task_type=self.task_type, full_metrics=False)
            alpha_results[alpha] = val_metrics[self.metric_name]
            
            print(f"  Alpha: {alpha:8.3f} | Val {self.metric_name}: {val_metrics[self.metric_name]:.4f}")
        
        # Select best alpha (filter out NaN values)
        valid_alphas = {a: score for a, score in alpha_results.items() if not np.isnan(score)}
        if not valid_alphas:
            raise ValueError("All alpha values produced NaN scores. Check numerical stability of features/labels.")
        
        self.best_alpha = max(valid_alphas.keys(), key=lambda a: valid_alphas[a])
        self.best_val_score = alpha_results[self.best_alpha]
        
        print(f"\n{'='*50}")
        print(f"Best alpha: {self.best_alpha}")
        print(f"Best val {self.metric_name}: {self.best_val_score:.4f}")
        print(f"{'='*50}")
        
        # Retrain on full train+val set with best alpha
        print(f"\nRetraining on full train+val set with best alpha...")
        X_full_train = self.vectorizer.fit_transform(train_processed + val_processed)
        y_full_train = np.concatenate([y_train, y_val])
        
        # Create appropriate model based on task type
        if self.task_type == "regression":
            self.best_model = Ridge(alpha=self.best_alpha, fit_intercept=self.fit_intercept, random_state=42)
        else:  # classification
            C = 1.0 / self.best_alpha if self.best_alpha != 0 else 1.0
            self.best_model = LogisticRegression(C=C, fit_intercept=self.fit_intercept, random_state=42, max_iter=1000)
        
        self.best_model.fit(X_full_train, y_full_train)
        print(f"✓ Model trained on {len(y_full_train)} samples")
        
        # === Calibration step (Platt scaling) ===
        if self.task_type == "classification" and self.platt_scaler is None:
            # Get validation predictions for calibration using val data transformed with current vectorizer
            X_val_recalc = self.vectorizer.transform(val_processed)
            
            # Raw probabilities before calibration
            raw_val_probs = self.best_model.predict_proba(X_val_recalc)[:, 1]
            raw_val_metrics = compute_metrics(
                y_val, raw_val_probs,
                task_type="classification",
                full_metrics=True
            )
            
            # Fit Platt scaling on validation logits
            self.calibrate(X_val_recalc, y_val)
            
            # Calibrated probabilities using Platt scaling
            logits = self.best_model.decision_function(X_val_recalc).reshape(-1, 1)
            cal_val_probs = self.platt_scaler.predict_proba(logits)[:, 1]
            
            assert np.all((cal_val_probs >= 0) & (cal_val_probs <= 1))
            
            cal_val_metrics = compute_metrics(
                y_val, cal_val_probs,
                task_type="classification",
                full_metrics=True
            )
            
            # Store for logging / saving
            self.val_metrics_raw = raw_val_metrics
            self.val_metrics_cal = cal_val_metrics
            
            print("\nCalibration (Platt scaling on validation set):")
            print(f"  Platt scaling fitted on {len(y_val)} validation samples")
            print(f"  ECE before: {raw_val_metrics.get('ece', 'N/A')}")
            print(f"  ECE after : {cal_val_metrics.get('ece', 'N/A')}")
        
        # Evaluate on test set if provided
        if test_texts is not None and test_labels is not None:
            print(f"\nEvaluating on test set...")
            # Re-fit vectorizer on training data for consistent features
            self.vectorizer.fit(train_processed)
            X_test = self.vectorizer.transform(test_processed)
            
            # Get predictions using appropriate method
            if self.task_type == "regression":
                test_preds = self.best_model.predict(X_test)
            else:  # classification - use probability of positive class
                test_preds = self.best_model.predict_proba(X_test)[:, 1]
            
            test_metrics = compute_metrics(y_test, test_preds, task_type=self.task_type, full_metrics=True)
            self.test_score = test_metrics[self.metric_name]
            
            print(f"\n{'='*50}")
            print(f"Test Results")
            print(f"{'='*50}")
            print(f"Test {self.metric_name}: {self.test_score:.4f}")
            print(f"Test MSE: {test_metrics.get('mse', 'N/A'):.4f}")
            print(f"Test MAE: {test_metrics.get('mae', 'N/A'):.4f}")
            print(f"{'='*50}")
        
        return self
    
    def predict(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on new data.
        
        Args:
            data: Either:
                - Tuple of (indices, prompts, ...) — targets and other trailing
                  elements are optional and ignored.
                - List of prompt strings
        
        Returns:
            Tuple of (indices_tensor, predictions_tensor):
                - indices_tensor: torch.Tensor of dtype int32 with shape [N]
                - predictions_tensor: torch.Tensor of dtype float32 with shape [N]
        """
        if self.best_model is None:
            raise RuntimeError("Must call train() before predict()")
        
        # Handle tuple input — only indices and prompts are used; targets are optional
        if isinstance(data, tuple) and len(data) >= 2:
            indices = list(data[0])
            prompts = list(data[1])
        else:
            # For non-tuple inputs, generate sequential indices
            prompts = data if isinstance(data, list) else list(data)
            indices = list(range(len(prompts)))
        
        # Preprocess prompts
        processed = self._preprocess_texts(prompts)
        
        # Vectorize
        X = self.vectorizer.transform(processed)
        
        # Predict (handle both Ridge and LogisticRegression)
        if self.task_type == "regression":
            predictions = self.best_model.predict(X)
        else:  # classification - use probability of positive class
            logits = self.best_model.decision_function(X).reshape(-1, 1)
            if self.platt_scaler is not None:
                # Use Platt scaling if fitted
                predictions = self.platt_scaler.predict_proba(logits)[:, 1]
            else:
                # Fallback to uncalibrated probabilities
                predictions = 1 / (1 + np.exp(-logits.ravel()))
        
        # Return as formatted tensors
        return (
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor(predictions, dtype=torch.float32)
        )
    
    def save_probe(self, results_path: Path | str, metadata: dict | None = None) -> None:
        """
        Save the trained probe (vectorizer, model, and Platt scaler) to disk.
        
        Args:
            results_path: Directory where to save the probe files
            metadata: Optional full metadata dict (e.g., with test metrics)
        
        Saves:
            - vectorizer.joblib: The TF-IDF vectorizer
            - model.joblib: The Ridge regression model
            - platt_scaler.joblib: The Platt scaling calibrator (if classification)
            - probe_metadata.json: Probe configuration and results
        """
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        
        if self.best_model is None:
            raise RuntimeError("Must call train() before save_probe()")
        
        # Save vectorizer and model
        joblib.dump(self.vectorizer, results_path / "vectorizer.joblib")
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
    def load_from_checkpoint(cls, results_path: Path | str) -> "TfidfProbe":
        """
        Load a trained probe from checkpoint.
        
        Args:
            results_path: Directory containing saved probe files
        
        Returns:
            TfidfProbe instance ready for prediction
        """
        results_path = Path(results_path)
        
        # Load metadata
        with open(results_path / "probe_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load vectorizer and model
        vectorizer = joblib.load(results_path / "vectorizer.joblib")
        model = joblib.load(results_path / "model.joblib")
        
        # Load Platt scaler if available (classification only)
        platt_scaler_path = results_path / "platt_scaler.joblib"
        platt_scaler = joblib.load(platt_scaler_path) if platt_scaler_path.exists() else None
        
        # Create probe instance
        probe = cls(config={})
        probe.setup()  # Setup spaCy
        
        # Restore state
        probe.vectorizer = vectorizer
        probe.best_model = model
        probe.platt_scaler = platt_scaler
        probe.best_alpha = metadata.get("best_alpha")
        probe.best_val_score = metadata.get("best_val_score")
        probe.test_score = metadata.get("test_score")
        probe.task_type = metadata.get("task_type")
        probe.metric_name = "spearman" if probe.task_type == "regression" else "auc"
        
        return probe
    
    def get_probe_metadata(self) -> dict:
        """
        Return probe metadata for saving.
        
        Returns:
            Dictionary with all probe attributes
        """
        metadata = {
            'best_alpha': self.best_alpha,
            'best_val_score': float(self.best_val_score) if self.best_val_score is not None else None,
            'test_score': float(self.test_score) if self.test_score is not None else None,
            'task_type': self.task_type,
            'model_type': self.model_type,
            'metric_name': self.metric_name,
            'probe_type': self.name(),
        }
        
        # Add calibration metrics for classification probes
        if self.task_type == "classification":
            if self.val_metrics_raw is not None:
                metadata['ece_before_calibration'] = self.val_metrics_raw.get('ece')
            if self.val_metrics_cal is not None:
                metadata['ece_after_calibration'] = self.val_metrics_cal.get('ece')
            if self.platt_scaler is not None:
                metadata['calibration_method'] = 'platt_scaling'
        
        return metadata
    

