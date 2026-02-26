"""
MLP Probe - Multi-Layer Perceptron probe for difficulty prediction.

Similar to LinearEoiProbe but uses PyTorch neural networks instead of sklearn models.
Extracts activations at specific (layer, position) and trains MLP on top.
"""

import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

from .base_probe import Probe
from pika.config import ROOT_ACTIVATION_DATA_DIR
from pika.probe.probe_utils.linear_eoi_probe import linear_eoi_probe_activation_utils, linear_eoi_probe_train_utils
from tqdm import tqdm
from pika.metrics import compute_metrics

ROOT_ACTIVATION_DATA_DIR = os.path.join(ROOT_ACTIVATION_DATA_DIR, "mlp_probe")


class MLPNetwork(nn.Module):
    """Multi-layer perceptron neural network for probing."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1, 
                 dropout_rate: float = 0.0, task_type: str = "regression"):
        """
        Args:
            input_dim: Dimension of input features (e.g., 768 for LLM hidden states)
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
            output_dim: Output dimension (1 for both regression and binary classification)
            dropout_rate: Dropout probability between layers
            task_type: "regression" or "classification"
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.task_type = task_type
        
        layers = []
        
        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            logits: Raw output [batch_size, output_dim] (or [batch_size] if squeezed)
        """
        return self.network(x).squeeze(-1)


class MLPProbe(Probe):
    """MLP probe for learning difficulty/success rate prediction."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._has_setup_run = False
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.d_model = None
        self.batch_size = getattr(self.config, "batch_size", 16)
        self.batch_size_train = getattr(self.config, "batch_size_train", 32)
        self.max_length = getattr(self.config, "max_length", 512)
        self.alpha_grid = getattr(self.config, "alpha_grid", [1.0])
        
        # MLP-specific hyperparameters (one-layer only)
        self.hidden_dims_grid = getattr(self.config, "hidden_dims", [[256], [512]])
        self.learning_rates_grid = getattr(self.config, "learning_rates", [1e-3, 5e-4])
        self.num_epochs_grid = getattr(self.config, "num_epochs", [10, 20])
        self.dropout_rates_grid = getattr(self.config, "dropout_rates", [0.0, 0.1])
        
        # Data storage
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        self.train_activations = None
        self.train_labels = None
        self.val_activations = None
        self.val_labels = None
        self.test_activations = None
        self.test_labels = None
        self.layer_indices = None
        self.positions: List[int] = [-1]
        
        # Model storage
        self.best_model = None
        self.best_pos_idx = None
        self.best_position_value = None
        self.best_layer_idx = None
        self.best_hidden_dims = None
        self.best_learning_rate = None
        self.best_num_epochs = None
        self.best_dropout_rate = None
        self.best_alpha = None
        self.best_val_score: Optional[float] = None
        
        # Calibration (Platt scaling for classification)
        self.platt_scaler = None
        
        # Task-specific attributes
        self.test_score = None
        self.task_type: Optional[str] = None
        self.metric_name: Optional[str] = None
        self.val_metrics_raw: Optional[dict] = None
        self.val_metrics_cal: Optional[dict] = None

    @property
    def name(self) -> str:
        """The name of the probe."""
        return "mlp_probe"

    def init_model(self, checkpoint_path: str):
        """Load a prior training checkpoint."""
        raise NotImplementedError("Initializing from checkpoint not yet implemented for MLP probe.")

    def setup(self, model_name: str, device: str) -> "MLPProbe":
        """Setup: load the base model for activation extraction."""
        self.model_name = model_name
        
        # Extract base model name (remove thinking mode suffix)
        base_model_name = model_name
        for thinking_mode in ['_low', '_medium', '_high', '_reasoning']:
            if base_model_name.endswith(thinking_mode):
                base_model_name = base_model_name[:-len(thinking_mode)]
                break
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use device_map="auto" to shard large models across all available GPUs
        if device == "cpu":
            dm = "cpu"
        else:
            dm = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, attn_implementation="eager", 
                                                          torch_dtype="auto", device_map=dm, 
                                                          low_cpu_mem_usage=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.d_model = self.model.config.hidden_size
        # Resolve actual input device (where the embedding layer lives)
        from pika.probe.probe_utils.linear_eoi_probe.linear_eoi_probe_activation_utils import get_model_input_device
        self.device = str(get_model_input_device(self.model))
        self._has_setup_run = True
        
        return self

    def _extract_and_load_all_activations(self, train_texts: List[str], train_labels: List[float], 
                                          val_texts: List[str], val_labels: List[float], 
                                          test_texts: Optional[List[str]] = None, 
                                          test_labels: Optional[List[float]] = None) -> None:
        """Extract activations for train/val/test splits."""
        print(f"\nExtracting activations...")
        
        splits = [("train", train_texts, train_labels), ("val", val_texts, val_labels)]
        if test_texts is not None:
            splits.append(("test", test_texts, test_labels))
        
        activation_data = {}
        
        for split_name, texts, labels in splits:
            print(f"  {split_name.capitalize()}...", end=" ")
            
            activation_data[split_name] = linear_eoi_probe_activation_utils.extract_or_load_activations(
                model=self.model,
                tokenizer=self.tokenizer,
                texts=texts,
                labels=labels,
                model_name=self.model_name,
                split=split_name,
                device=self.device,
                batch_size=self.batch_size,
                max_length=self.max_length,
                eoi_tokens=None,
                layer_indices=None,
                cache_dir=ROOT_ACTIVATION_DATA_DIR,
                use_cache=True,
            )
            
            shape = activation_data[split_name]['activations'].shape
            from_cache = activation_data[split_name]['from_cache']
            print(f"Shape {shape} ({'cached' if from_cache else 'extracted'})")
        
        # Store activations
        self.train_activations = activation_data['train']['activations']
        self.train_labels = activation_data['train']['labels']
        self.val_activations = activation_data['val']['activations']
        self.val_labels = activation_data['val']['labels']
        if "test" in activation_data:
            self.test_activations = activation_data['test']['activations']
            self.test_labels = activation_data['test']['labels']
        
        self.layer_indices = activation_data['train']['layer_indices']
        self.positions = activation_data['train']['positions']

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            hidden_dims: List[int], learning_rate: float, num_epochs: int, dropout_rate: float,
            alpha: float, task_type: str, batch_size: Optional[int] = None) -> Tuple[MLPNetwork, float]:
        """
        Train an MLP network and return the best model.
        
        Args:
            x_train: Training activations [N_train, D]
            y_train: Training labels [N_train]
            x_val: Validation activations [N_val, D]
            y_val: Validation labels [N_val]
            hidden_dims: List of hidden layer dimensions (one-layer MLP)
            learning_rate: Learning rate for Adam optimizer
            num_epochs: Number of training epochs
            dropout_rate: Dropout probability
            alpha: L2 regularization strength (weight decay)
            task_type: "regression" or "classification"
            batch_size: Batch size for mini-batch training (if None, uses self.batch_size_train)
        
        Returns:
            best_model: Trained MLPNetwork
            best_val_score: Best validation metric score
        """
        if batch_size is None:
            batch_size = self.batch_size_train
        
        # Convert to torch tensors
        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        x_val_t = torch.tensor(x_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        
        # Create model
        model = MLPNetwork(
            input_dim=x_train.shape[1],
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout_rate=dropout_rate,
            task_type=task_type
        ).to(self.device)
        
        # Setup optimizer with L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
        
        # Loss function
        if task_type == "regression":
            criterion = nn.MSELoss()
        else:  # classification
            criterion = nn.BCEWithLogitsLoss()
        
        best_val_score = -np.inf
        best_model_state = None
        patience = 5
        patience_counter = 0
        
        # Create mini-batch dataset
        train_dataset = torch.utils.data.TensorDataset(x_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training with mini-batches
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass on mini-batch
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val_t)
                val_loss = criterion(val_logits, y_val_t)
                
                # Compute metric
                if task_type == "regression":
                    val_preds = val_logits.cpu().numpy()
                else:
                    val_preds = torch.sigmoid(val_logits).cpu().numpy()
                
                val_score = self._compute_metric(y_val, val_preds, task_type)
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model, best_val_score

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> float:
        """Compute validation metric."""
        metrics = compute_metrics(y_true, y_pred, task_type=task_type, full_metrics=False)
        return metrics["spearman"] if task_type == "regression" else metrics["auc"]

    def _calibrate_platt(self, model: MLPNetwork, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Fit Platt scaling for classification probes using LogisticRegression.
        
        Args:
            model: Trained MLP model
            x_cal: Calibration activations [N, D]
            y_cal: Calibration labels [N]
        """
        if self.task_type != "classification":
            return
        
        # Get logits from model
        model.eval()
        x_t = torch.tensor(x_cal, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = model(x_t).cpu().numpy()
        
        logits_2d = logits.reshape(-1, 1)
        
        # Fit LogisticRegression for Platt scaling: P(y=1|z) = sigmoid(A*z + B)
        from sklearn.linear_model import LogisticRegression
        self.platt_scaler = LogisticRegression(max_iter=1000)
        self.platt_scaler.fit(logits_2d, y_cal)

    def _evaluate_probe(self, model: MLPNetwork, x: np.ndarray, y: np.ndarray, 
                        full_metrics: bool = False) -> float:
        """Evaluate MLP model on data."""
        model.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = model(x_t)
            
            if self.task_type == "regression":
                y_pred = logits.cpu().numpy()
            else:
                logits_np = logits.cpu().numpy()
                
                # Apply Platt scaling if available
                if self.platt_scaler is not None:
                    logits_2d = logits_np.reshape(-1, 1)
                    y_pred = self.platt_scaler.predict_proba(logits_2d)[:, 1]
                else:
                    y_pred = torch.sigmoid(logits).cpu().numpy()
        
        metrics = compute_metrics(y, y_pred, task_type=self.task_type, full_metrics=full_metrics)
        score = metrics["spearman"] if self.task_type == "regression" else metrics["auc"]
        return float(score)

    def _train_and_select_probes(self) -> None:
        """
        Train and select best MLP probe across all (layer, position, hyperparameter) combinations.
        """
        n_positions = len(self.positions)
        n_layers = len(self.layer_indices)
        
        print(f"\n{'='*80}")
        print(f"MLP Probe Training: {n_positions} positions × {n_layers} layers")
        print(f"Hyperparameter grid size: {len(self.hidden_dims_grid)} × {len(self.learning_rates_grid)} ×"
              f" {len(self.num_epochs_grid)} × {len(self.dropout_rates_grid)}")
        print(f"{'='*80}")
        
        best_val_score = -np.inf
        
        for pos_idx in range(n_positions):
            pos = self.positions[pos_idx]
            print(f"\nPosition {pos}:")
            
            for layer_idx in tqdm(self.layer_indices, desc="  Layers"):
                # Extract activations for this (position, layer)
                x_train = linear_eoi_probe_train_utils.to_numpy(self.train_activations[:, layer_idx, pos_idx, :])
                x_val = linear_eoi_probe_train_utils.to_numpy(self.val_activations[:, layer_idx, pos_idx, :])
                y_train = np.array(self.train_labels)
                y_val = np.array(self.val_labels)
                
                # Hyperparameter search for this (position, layer)
                for hidden_dims in self.hidden_dims_grid:
                    for lr in self.learning_rates_grid:
                        for num_epochs in self.num_epochs_grid:
                            for dropout_rate in self.dropout_rates_grid:
                                for alpha in self.alpha_grid:
                                    # Train and evaluate
                                    model, val_score = self.fit(
                                        x_train, y_train, x_val, y_val,
                                        hidden_dims=hidden_dims,
                                        learning_rate=lr,
                                        num_epochs=num_epochs,
                                        dropout_rate=dropout_rate,
                                        alpha=alpha,
                                        task_type=self.task_type,
                                        batch_size=self.batch_size_train
                                    )
                                    
                                    # Update best if better
                                    if val_score > best_val_score:
                                        best_val_score = val_score
                                        self.best_model = model
                                        self.best_pos_idx = pos_idx
                                        self.best_position_value = pos
                                        self.best_layer_idx = layer_idx
                                        self.best_hidden_dims = hidden_dims
                                        self.best_learning_rate = lr
                                        self.best_num_epochs = num_epochs
                                        self.best_dropout_rate = dropout_rate
                                        self.best_alpha = alpha
                                        self.best_val_score = best_val_score
        
        # Calibrate best model using validation set (Platt scaling for classification)
        if self.best_model is not None:
            x_val_best = linear_eoi_probe_train_utils.to_numpy(self.val_activations[:, self.best_layer_idx, self.best_pos_idx, :])
            y_val_best = np.array(self.val_labels)
            self._calibrate_platt(self.best_model, x_val_best, y_val_best)
        
        print(f"\nBest validation {self.metric_name}: {self.best_val_score:.4f}")
        print(f"  Position: {self.best_position_value}")
        print(f"  Layer: {self.best_layer_idx}")
        print(f"  Hidden dims: {self.best_hidden_dims}")
        print(f"  Learning rate: {self.best_learning_rate}")

    def train(self, train_data: Tuple[List[int], List[str], List[float]], 
              val_data: Tuple[List[int], List[str], List[float]],
              test_data: Optional[Tuple[List[int], List[str], List[float]]] = None,
              alpha_grid: Optional[List[float]] = None) -> "MLPProbe":
        """Train MLP probe on train/val data, optionally evaluate on test."""
        if not self._has_setup_run:
            raise RuntimeError("Must call setup() before train()")
        
        # Unpack data
        train_idxs, train_texts, train_labels = train_data
        val_idxs, val_texts, val_labels = val_data
        test_idxs, test_texts, test_labels = None, None, None
        if test_data is not None:
            test_idxs, test_texts, test_labels = test_data
        
        self.train_indices = train_idxs
        self.val_indices = val_idxs
        self.test_indices = test_idxs
        
        print(f"\n{'='*80}")
        print(f"MLP Probe Training: Extracting activations")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}")
        if test_texts is not None:
            print(f"Test samples: {len(test_texts)}")
        
        # Extract activations
        self._extract_and_load_all_activations(train_texts, train_labels, val_texts, val_labels,
                                                test_texts, test_labels)
        
        # Infer task type
        self.task_type = linear_eoi_probe_train_utils.infer_task_type(train_labels)
        self.metric_name = "spearman" if self.task_type == "regression" else "auc"
        print(f"Task type: {self.task_type}")
        print(f"Metric: {self.metric_name}")
        
        # Update alpha grid if provided
        if alpha_grid is not None:
            self.alpha_grid = alpha_grid
        
        # Train and select best probe
        self._train_and_select_probes()
        
        # Evaluate on test set if available
        if test_texts is not None:
            x_test = linear_eoi_probe_train_utils.to_numpy(self.test_activations[:, self.best_layer_idx, self.best_pos_idx, :])
            y_test = np.array(self.test_labels)
            self.test_score = self._evaluate_probe(self.best_model, x_test, y_test, full_metrics=True)
            print(f"Test {self.metric_name}: {self.test_score:.4f}")
        
        return self

    def predict(self, data: Tuple[List[int], List[str], List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on data."""
        indices, prompts, labels = data
        
        # Extract activations
        activations_data = linear_eoi_probe_activation_utils.extract_or_load_activations(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=prompts,
            labels=labels,
            model_name=self.model_name,
            split="test",
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            eoi_tokens=None,
            layer_indices=None,
            specified_eoi_position=self.best_position_value,
        )
        
        # Extract and prepare activations (activations_data is a dict with 'activations' key)
        activations = activations_data['activations']
        x_pred = linear_eoi_probe_train_utils.to_numpy(activations[:, self.best_layer_idx, :])
        
        # Forward pass
        self.best_model.eval()
        x_t = torch.tensor(x_pred, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.best_model(x_t)
            
            if self.task_type == "regression":
                predictions = logits.cpu().numpy()
            else:
                logits_np = logits.cpu().numpy()
                
                # Apply Platt scaling if available
                if self.platt_scaler is not None:
                    logits_2d = logits_np.reshape(-1, 1)
                    predictions = self.platt_scaler.predict_proba(logits_2d)[:, 1]
                else:
                    predictions = 1 / (1 + np.exp(-logits_np))  # sigmoid
        
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
            - best_model.pt: The PyTorch model state dict
            - platt_scaler.joblib: Platt scaler if classification task
            - probe_metadata.json: Probe configuration and layer/position info
        """
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        
        if self.best_model is None:
            raise RuntimeError("Must call train() before save_probe()")
        
        # Save the trained PyTorch model
        model_file = results_path / "best_model.pt"
        torch.save(self.best_model.state_dict(), model_file)
        
        # Save Platt scaler if it exists
        if self.platt_scaler is not None:
            import joblib
            platt_file = results_path / "platt_scaler.joblib"
            joblib.dump(self.platt_scaler, platt_file)
        
        # If no metadata provided, create minimal metadata for loading
        if metadata is None:
            metadata = {
                "best_layer_idx": self.best_layer_idx,
                "best_position_value": self.best_position_value,
                "best_hidden_dims": self.best_hidden_dims,
                "best_learning_rate": self.best_learning_rate,
                "best_dropout_rate": self.best_dropout_rate,
                "best_alpha": self.best_alpha,
                "best_val_score": float(self.best_val_score) if self.best_val_score else None,
                "task_type": self.task_type,
                "model_name": self.model_name,
                "d_model": self.d_model,
            }
            
            # Add calibration info if available
            if self.task_type == "classification" and self.platt_scaler is not None:
                metadata["has_platt_scaler"] = True
                metadata["platt_coef"] = float(self.platt_scaler.coef_[0][0])
                metadata["platt_intercept"] = float(self.platt_scaler.intercept_[0])
            elif self.task_type == "classification":
                metadata["has_platt_scaler"] = False
        
        # Save metadata as JSON
        metadata_file = results_path / "probe_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def get_probe_metadata(self) -> dict:
        """Return probe metadata for saving."""
        metadata = {
            "probe_type": "mlp_probe",
            "best_layer_idx": self.best_layer_idx,
            "best_position_value": self.best_position_value,
            "best_hidden_dims": self.best_hidden_dims,
            "best_learning_rate": self.best_learning_rate,
            "best_dropout_rate": self.best_dropout_rate,
            "best_alpha": self.best_alpha,
            "best_val_score": float(self.best_val_score) if self.best_val_score else None,
            "test_score": float(self.test_score) if self.test_score else None,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "model_name": self.model_name,
            "d_model": self.d_model,
        }
        
        # Add calibration info for classification
        if self.task_type == "classification" and self.platt_scaler is not None:
            metadata["has_platt_scaler"] = True
            metadata["platt_coef"] = float(self.platt_scaler.coef_[0][0])
            metadata["platt_intercept"] = float(self.platt_scaler.intercept_[0])
        elif self.task_type == "classification":
            metadata["has_platt_scaler"] = False
        
        return metadata
