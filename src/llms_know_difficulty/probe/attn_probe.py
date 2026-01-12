import itertools
from typing import Any, Tuple
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import math
import einops
from llms_know_difficulty.probe.base_probe import Probe
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from itertools import product
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

class AttnLite(nn.Module):
    
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.context_query = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, xs): # xs: [B, T, D]
        """
        Forward pass through the attention probe.

        Args:
            xs: A tensor of shape [B, T, D] where:
                B - batch size
                T - sequence length
                D - hidden size
        Returns:
            A tensor of shape [B] where each element is the logit for the next token in the sequence.

        """

        attn_scores = self.context_query(xs).squeeze(-1) / self.scale        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = einops.einsum(
            attn_weights, xs, "batch seq, batch seq embed -> batch embed"
        )
        sequence_logits = self.classifier(context).squeeze(-1)

        return sequence_logits

@dataclass
class CollateConfig:
    max_length: int = 256

def make_collate_fn(tokenizer: AutoTokenizer, cfg: CollateConfig):
    """
    Creates a collate function that tokenizes the prompt text input.
    """

    def collate(batch):
        ids, texts, ys = zip(*batch)
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        ys = torch.stack(list(ys), dim=0)
        ids = torch.stack(list(ids), dim=0)
        return ids, enc, ys
    return collate

class TextNumberDataset(Dataset):
    def __init__(
            self, 
            idx: list[int],
            prompts: list[str],
            targets: list[float]
    ):

        assert len(idx) == len(prompts) == len(targets), \
        f"Idx: {len(idx)}, prompts: {len(prompts)}, and targets: {len(targets)} must have the same length."
        self.idx = idx
        self.prompts = prompts
        self.targets = targets

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        ids, text, y = self.idx[idx], self.prompts[idx], self.targets[idx]
        return torch.tensor(ids, dtype=torch.int32), text, torch.tensor(y, dtype=torch.float32)


class AttnProbe(Probe):
    def __init__(self, config):
        super().__init__(config)
        self.config = config.model_dump()

    @property
    def name(self) -> str:
        """The name of the probe."""
        return "attn_probe"

    def setup(self, model_name: str, device: str) -> "AttnProbe":
        """
        Any pre-training loading steps, run before .fit or .predict.

        TODO: Automatic discovery of number of layers...
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        model_config = AutoConfig.from_pretrained(model_name)
        self.n_layers = model_config.num_hidden_layers

        self.model_name = model_name  # Store model name
        self.d_model = self.model.config.hidden_size
        self.device = self.config.get('device', 'cuda:0')

     
        # Variables that store the best probe after training:
        self.best_probe = None
        self.best_hyperparameters = None
        self.best_layer_id = None

        return self

    def init_model(self, config: dict):
        """
        Load a prior training checkpoint.
        """
        raise NotImplementedError("Initializing a model from a checkpoint is not implemented for the attention probe.")

    def train(self, train_data: tuple[list[str], list[float]],
              val_data: tuple[list[str], list[float]]) -> "AttnProbe":
        """
        Train a probe on the training data, evaluate on the validation data, and repeat, returning the best probe.
        """

        # Unpack the training and validation data:
        train_idx,train_prompts, train_targets = train_data
        val_idx, val_prompts, val_targets = val_data

        # Setup cross validation on layers and other hyperparameters:        
        # Use itertools to create list of all combinations of layers and other hyperparameters:
        cv_hyper_list_names = self.config.get('cross_validated_hyperparameters', [])
        cv_hyper_values = []

        # Make sure the config provides a grid of values:
        for hyper_name in cv_hyper_list_names:
            hyper_values = self.config.get(hyper_name, None)
            if hyper_values is None or not isinstance(hyper_values, list):
                raise ValueError(
                    f"""Attention probe config error! Cross_validated_hyperparameter includes {hyper_name} but {hyper_name} does not provide a list of values.
                    {hyper_name} set to {hyper_values} in AttentionProbeConfig inconfig.py""")
            
            if hyper_name == 'layer':
                if hyper_values[0] == -1:
                    hyper_values = list(range(0, self.n_layers, 2))
                else:
                    hyper_values = [int(layer) for layer in hyper_values]

            cv_hyper_values.append(hyper_values)

        # Create a grid of all combinations of the hyperparameters:
        cv_hypers = product(*cv_hyper_values)    
        cv_results = []

        for hyper_setup in cv_hypers:

            # Create a dictionary input:
            hyperparams = {}
            for i, name in enumerate(cv_hyper_list_names):
                hyperparams[name] = hyper_setup[i]

            print('-' * 80)
            print(f'Hyperparams: {hyperparams}')
            print('-' * 80)
            # Fit the probe to the training data, with these specific hyperparameters:
            probe = self.fit(train_idx, train_prompts, train_targets, **hyperparams)

            # Evaluate the probe's performance on the validation data:
            evaluate_metrics, _ = self.evaluate(probe, val_idx,val_prompts, val_targets, hyperparams)
            cv_metric = evaluate_metrics[self.config.get('cv_metric', 'mse')]

            print('-' * 80)
            print(f'CV metric: {cv_metric:.4f}')
            print('-' * 80)
            
            cv_results.append([cv_metric, hyperparams, probe])

        # After training we select the best cv hyperparameters and evaluate the probe on the test_data
        max_or_min_cv_metric = self.config.get('max_or_min_cv_metric', 'max')
        metrics, hyperparameters, probes = zip(*cv_results)
        if max_or_min_cv_metric == 'max':
            best_cv_probe_idx = np.argmax(metrics)
        else:
            best_cv_probe_idx = np.argmin(metrics)

        self.best_probe = probes[best_cv_probe_idx]
        self.best_hyperparameters = hyperparameters[best_cv_probe_idx]
        self.best_layer_id = self.best_hyperparameters['layer']
        self.best_metrics = metrics[best_cv_probe_idx]

        return self


    def evaluate(self, probe: AttnLite, idx:list[int], prompts: list[str], targets: list[float], hyperparameters: dict) -> dict:
        """
        Evaluate the probe's performance on some eval dataset.
        """
        
        # Setup data loader:
        if self.config.get('test_mode', False):
            batch_size_approx = self.config.get('batch_size',128)[0]
            idx = idx[:batch_size_approx]
            prompts = prompts[:batch_size_approx]
            targets = targets[:batch_size_approx]
            eval_dataset = TextNumberDataset(idx, prompts, targets)
        else:
            eval_dataset = TextNumberDataset(idx, prompts, targets)

        collate_fn = make_collate_fn(self.tokenizer, CollateConfig(max_length=hyperparameters.get('max_length', 256)))
        eval_loader = DataLoader(eval_dataset,
            batch_size=hyperparameters.get('batch_size', 128),
            shuffle=False,
            collate_fn=collate_fn)

        eval_layer = hyperparameters.get('layer', None)
        assert eval_layer is not None, "Evaluation layer must be specified."

        # Evaluate the probe:
        ids = []
        outputs = []
        targets = []

        eval_loader_iter = tqdm(eval_loader, desc="Evaluating", leave=False)
        for idx, enc, ys in eval_loader_iter:
            ys = ys.to(self.device)

            # Extract the activations from the model:
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            
            activations = self._extract_layer_features(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=[eval_layer],
            )

            preds = probe(activations)
            ids.append(idx)
            outputs.append(preds)
            targets.append(ys)

        ids = torch.cat(ids, dim=0)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        spearman_corr, _ = spearmanr(outputs_np, targets_np)
        return {'spearmanr': spearman_corr}, outputs

    def predict(self, test_data: tuple[list[int], list[str], list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the success rate using the probe.
        
        Args:
            test_data: Tuple of (indices, prompts, targets)
            
        Returns:
            Tuple of (indices_tensor[int32], predictions_tensor[float32])
        """
        assert self.best_probe is not None, \
            "Best probe not found. Please train the probe first."
        
        assert self.best_layer_id is not None, \
            "Evaluation layer must be specified."

        idx, prompts, _ = test_data

        # For test_mode, sample a subset of data for quick debugging
        if self.config.get('test_mode', False):
            test_sample_size = self.config.get('test_sample_size', 256)
            idx = idx[:test_sample_size]
            prompts = prompts[:test_sample_size]

        # Setup data loader - uses full dataset or subset if test_mode
        eval_dataset = TextNumberDataset(idx, prompts, [0]*len(prompts))
        collate_fn = make_collate_fn(self.tokenizer, CollateConfig(max_length=self.best_hyperparameters.get('max_length', 256)))
        eval_loader = DataLoader(eval_dataset,
         batch_size=self.best_hyperparameters.get('batch_size', 128),
         shuffle=False,
         collate_fn=collate_fn)

        # Evaluate the probe:
        outputs = []
        ids_list = []
        # Use tqdm with manual set_postfix to update with the number of predictions
        pbar = tqdm(eval_loader)
        num_pred = 0
        for ids, enc, ys in pbar:
            # We'll count predictions as the number of examples processed so far
            n_batch = ys.size(0)
            num_pred += n_batch
            pbar.set_postfix({'probe.predict': num_pred})
            # Extract the activations from the model:
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            
            activations = self._extract_layer_features(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=[self.best_layer_id],
            )

            preds = self.best_probe(activations)
            outputs.append(preds.cpu().detach())
            ids_list.append(ids)
        
        # Return formatted tensors: (int32 indices, float32 predictions)
        indices_tensor = torch.cat(ids_list, dim=0).int()
        predictions_tensor = torch.cat(outputs, dim=0).float()
        return indices_tensor, predictions_tensor

    def fit(self, idx: list[int],
                prompts: list[str],
                targets: list[float],
                layer: int,
                batch_size:int,
                learning_rate:float,
                num_epochs:int,
                weight_decay:float,
                max_length:int) -> None:
        """
        Run a single training loop for the attention probe. Very simple no validation or logging ortracking etc...

        Args:
            prompts: A list of prompts to train on.
            targets: A list of targets to train on.
            layer: The layer to train on.

        TODO: Add some validation loss during training?
        """

        # Setup train loader:
        if self.config.get('test_mode', False):
            batch_size_approx = self.config.get('batch_size',128)[0]
            idx = idx[:batch_size_approx]
            prompts = prompts[:batch_size_approx]
            targets = targets[:batch_size_approx]
            train_dataset = TextNumberDataset(idx, prompts, targets)
        else:
            train_dataset = TextNumberDataset(idx, prompts, targets)

        collate_fn = make_collate_fn(self.tokenizer, CollateConfig(max_length=max_length))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Setup the probe:
        probe = AttnLite(self.d_model).to(self.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Train the probe:        
        for epoch in range(1, num_epochs + 1):
            
            probe.train()
            running = 0.0
            step = 0
            n = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", dynamic_ncols=True)
            for idx, enc, ys in pbar:
                ys = ys.to(self.device)

                # Extract the activations from the model:
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                
                activations = self._extract_layer_features(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer_indices=[layer],
                )

                preds = probe(activations)
                loss = loss_fn(preds, ys)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                bs = ys.size(0)
                running += loss.item() * bs
                n += bs

                # Update progress bar with step, batch_mse, and epoch
                pbar.set_postfix({
                    'step': step,
                    'epoch': epoch,
                    'batch_mse': f'{loss.item():.6f}'
                })
                step += 1

        return probe

    @torch.no_grad()
    def _extract_layer_features(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_indices: list[int],
    )-> torch.Tensor:
        """
        Given an input ids, attention mask and model return the activations of a specific layer.

        TODO: More efficient layer hidden state extraction, using hooks.
        TODO: Re-write to extract only one layer not using a list...

        Returns: A tensor of activations of shape [B, T, D] where
            B - batch size
            T - sequence length
            D - hidden size
        """

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple: (emb, layer1, ..., layerN), each [B,T,D]

        layer_activations = [hidden_states[layer_idx] * attention_mask.unsqueeze(-1) for layer_idx in layer_indices]
        return torch.stack(layer_activations, dim=1)[:,0,...] # [Batch size, Sequence length, hidden size]

    def save_probe(self, path: Path, metadata: dict | None = None):
        """
        Save the probe state dictionary and other metadata to disk.

        Args:
            path: The path to save the probe to.
            metadata: Optional full metadata dict (e.g., with test metrics).
        """

        assert self.best_probe is not None, "Best probe not found. Please train the probe first."
        assert self.best_hyperparameters is not None, "Best hyperparameters not found. Please train the probe first."
        assert self.best_layer_id is not None, "Best layer id not found. Please train the probe first."

        # Save the probe state dictionary:
        state_dict = {
            'best_probe_state_dict': self.best_probe.state_dict(),
            'best_hyperparameters': self.best_hyperparameters,
            'best_layer_id': self.best_layer_id,
            'best_metrics': self.best_metrics if hasattr(self, 'best_metrics') else None}

        torch.save(state_dict, path / "best_probe.pt")
        
        # Save metadata to JSON if provided
        if metadata is not None:
            import json
            metadata_file = path / "probe_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    def load_from_checkpoint(self, path: Path):
        """
        Load a trained probe from a saved checkpoint.

        Args:
            path: The path to load the probe from.
        """

        if self.best_probe is not None:
            warning("Best probe already loaded. Overwriting...")

        state_dict = torch.load(path / "best_probe.pt")
        self.best_probe = AttnLite(self.d_model).to(self.device).load_state_dict(state_dict['best_probe_state_dict'])
        self.best_hyperparameters = state_dict['best_hyperparameters']
        self.best_layer_id = state_dict['best_layer_id']
        self.best_metrics = state_dict['best_metrics']

        return self

    def get_metadata(self) -> dict:
        """
        Dump the config setup into a dictionary.
        """
        return self.config
    def get_probe_metadata(self) -> dict:
        """
        Return probe metadata for saving (attn_probe specific format).
        
        Returns:
            Dictionary with attn probe metadata (uses best_layer_id internally but returns standardized format)
        """
        return {
            'best_layer_idx': self.best_layer_id,
            'best_val_score': self.best_metrics if hasattr(self, 'best_metrics') else None,
            'model_name': getattr(self, 'model_name', 'unknown'),
            'd_model': self.d_model,
            'task_type': 'regression',  # AttnProbe is always regression
            'best_pos_idx': None,  # AttnProbe doesn't use positions
            'best_position_value': None,
            'best_alpha': None,  # AttnProbe doesn't use regularization
            'test_score': None,  # Computed during evaluation
        }
