import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import math
import einops
import numpy as np

from base_probe import Probe
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from itertools import product

from typing import List, Tuple, Optional
from base_probe import Probe
from ..config import ROOT_ACTIVATION_DATA_DIR
from . import sk_activation_utils, sk_train_utils
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm


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
        self.layer_indices = None
        self.positions: List[int] = [-1] #default to last token if none here

        # Probe storage
        self.weights = None
        self.probing_layer = None
        self.probing_position = None

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

    def _extract_and_load_all_activations(self, train_texts: List[str], train_labels: List[float], val_texts: List[str], val_labels: List[float]) -> None:
        """
        Extract and cache activations for both train and val splits, with logging.
        Stores results in instance variables for use in fit/predict methods.
        
        Args:
            train_texts: List of training prompts
            train_labels: List of training labels
            val_texts: List of validation prompts
            val_labels: List of validation labels
        """
        print(f"\nProcessing activations for {len(train_texts)} train and {len(val_texts)} val samples...")
        activation_data = {}
        
        for split_name, texts, labels in [("train", train_texts, train_labels), ("val", val_texts, val_labels)]:
            print(f"  {split_name.capitalize()} split...", end=" ")
            
            activation_data[split_name] = sk_activation_utils.extract_or_load_activations(
                model=self.model,
                tokenizer=self.tokenizer,
                texts=texts,
                labels=labels,
                model_name=self.model_name,
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
        self.layer_indices = activation_data['train']['layer_indices']
        self.positions = activation_data['train']['positions']

    def train(self, train_data: Tuple[List[str], List[float]], val_data: Tuple[List[str], List[float]]) -> "SklearnProbe":
        """
        Train a probe on the training data, evaluate on the validation data. We will do our corss validation here by repeatedy calling .fit()
        
        Extracts activations for train and val data, caches them in ROOT_ACTIVATION_DATA_DIR.
        
        Args:
            train_data: Tuple of (prompts, targets) for training
            val_data: Tuple of (prompts, targets) for validation
        
        Returns:
            self (for method chaining)
        """
        if not self._has_setup_run:
            raise RuntimeError("Must call setup() before train()")
        
        # Convert data tuples to lists
        train_texts, train_labels = list(train_data[0]), list(train_data[1])
        val_texts, val_labels = list(val_data[0]), list(val_data[1])
        
        print(f"\n{'='*80}")
        print(f"SklearnProbe Training: Extracting activations")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Extract and load activations with logging
        self._extract_and_load_all_activations(train_texts, train_labels, val_texts, val_labels)

        # use the train activations+labels to perform CV and validate on the val labels.
        # training_loop
        # train a probe at each position, for each layer
        for pos_idx in range(self.positions):
            for layer_idx in tqdm((self.layer_indices), desc=f"Layer progress for position {pos_idx}"):
                try:
                    x_train = self.train_activations[:, layer_idx, pos_idx, :].numpy()  # [N, D]
                    y_val = self.val_activations[:, layer_idx, pos_idx, :].numpy()  # [M, D]
                except:
                    x_train = self.train_activations[:, layer_idx, pos_idx, :].detach().to(torch.float16).cpu().numpy()  # [N, D]
                    y_val = self.val_activations[:, layer_idx, pos_idx, :].detach().to(torch.float16).cpu().numpy()  # [M, D]

                y_train = self.train_labels
                y_val = self.val_labels

                

                

        return self

    def fit(self, prompt_activations: np.ndarray,
                targets: List[float],
                selected_alpha: int,
                ) -> Ridge | LogisticRegression:
        """
        Run a single training loop for the probe. Very simple no validation or logging ortracking etc...
        Args:
            prompt_activations: A list of activations to train on.
            targets: A list of targets to train on.
            layer: The layer to train on.
        """
        task_type = sk_train_utils.infer_task_type(targets)
        if task_type == "regression":
            probe_model = Ridge(
                alpha=selected_alpha,
                fit_intercept=False
            )
        else:
            probe_model = LogisticRegression(
                penalty="l2",
                fit_intercept=False,
                solver="lbfgs",
                max_iter=1000
            )
        
        probe_model.fit(prompt_activations, targets)

        return probe_model


    def predict(self, prompts: List[str] | np.ndarray) -> None:
        """
        Returns logits or probits
        """
        pass
