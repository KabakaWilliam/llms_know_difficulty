import itertools
from logging import warning
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
    
    def __init__(self, d_model: int, **kwargs: Any):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.context_query = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, xs: torch.Tensor, mask: torch.Tensor): # xs: [B, T, D]
        """
        Forward pass through the attention probe.

        Args:
            xs: A tensor of shape [B, T, D] where:
                B - batch size
                T - sequence length
                D - hidden size
            mask: A tensor of shape [B, T] where:
                B - batch size
                T - sequence length
                The mask is used to mask out the padding tokens.
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

class SigmoidAttnLite(AttnLite):
    def __init__(self, d_model: int, **kwargs: Any):
        super().__init__(d_model, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs: torch.Tensor, mask: torch.Tensor):
        sequence_logits = super().forward(xs, mask)
        return self.sigmoid(sequence_logits)

class LinearThenAgg(nn.Module):
    """
    Base class for linear then aggregation probes.
    Subclasses must implement the agg method.
    """

    def __init__(
        self,
        embed_dim: int,
        **kwargs: Any,
    ):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through a linear layer then aggregate the outputs.

        Args:
            x: A tensor of shape [B, T, D] where:
                B - batch size
                T - sequence length
                D - hidden size
            mask: A tensor of shape [B, T] where:
                B - batch size
                T - sequence length
                The mask is used to mask out the padding tokens.
        Returns:
            A tensor of shape [B] where each element is the aggregated output of the linear layer.
        """

        x = self.linear(x).squeeze(-1)
        x = self.agg(x, mask)
        return x

    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")


class LinearThenMax(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values


class LinearThenSoftmax(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        mask = mask.to(torch.bool)
        temperature = self.kwargs["temperature"]
        x_for_softmax = x.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(x_for_softmax / temperature, dim=1)
        return (x * weights).sum(dim=1)


class LinearThenRollingMax(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        window_size = self.kwargs["window_size"]
        windows = x.unfold(1, window_size, 1)
        window_means = windows.mean(dim=2)
        return window_means.max(dim=1).values

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


class TorchProbe(Probe):
    def __init__(self, config):
        """
        TorchProbe trains a vareity of probes using a standard torch training loop.

        Args:
            config: The configuration for the probe.
        """


        super().__init__(config)
        self.config = config.model_dump()

    @property
    def name(self) -> str:
        """The name of the probe."""
        return "torch_probe"

    def setup(self, model_name: str, device: str, ProbeClass: nn.Module) -> "TorchProbe":
        """
        Any pre-training loading steps, run before .fit or .predict.

        TODO: Automatic discovery of number of layers...
        """

        self.ProbeClass = ProbeClass

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
         use_fast=True,
         padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # n_layers important for inhereted probe classes.
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
              val_data: tuple[list[str], list[float]]) -> "TorchProbe":
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
            with torch.no_grad():
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

    @torch.no_grad()
    def evaluate(self, probe: AttnLite,
                        idx:list[int],
                        prompts: list[str],
                        targets: list[float],
                        hyperparameters: dict) -> dict:
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

            preds = probe(activations, mask=attention_mask)
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

    @torch.no_grad()
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

            preds = self.best_probe(activations, mask=attention_mask)
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
                max_length:int,
                **kwargs) -> nn.Module:
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
            batch_size_approx = self.config.get('batch_size', 128)[0]
            idx = idx[:batch_size_approx]
            prompts = prompts[:batch_size_approx]
            targets = targets[:batch_size_approx]
            train_dataset = TextNumberDataset(idx, prompts, targets)
        else:
            train_dataset = TextNumberDataset(idx, prompts, targets)

        collate_fn = make_collate_fn(self.tokenizer, CollateConfig(max_length=max_length))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Setup the probe:
        probe = self.ProbeClass(self.d_model, **kwargs).to(self.device)
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
            
                preds = probe(activations, mask=attention_mask)
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
                del activations

        # Empty the cache to free up memory for the evaluation step.
        torch.cuda.empty_cache()
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
        Extract the activations of a specific layer using the layer_features method.

        Returns: A tensor of activations of shape [B, T, D] where
            B - batch size
            T - sequence length
            D - hidden size
        """

        assert len(layer_indices) == 1, "Only one layer can be extracted at a time."

        if self.config.get('use_hooks', False):
                    activations = self._extract_hooked_activations(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer_idx=layer_indices,
                )
        else:
            activations = self._extract_hidden_outputs(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=[layer_indices[0]],
            )

        return activations

    @torch.no_grad()
    def _extract_hidden_outputs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_indices: list[int],
    )-> torch.Tensor:
        """
        Given an input ids, attention mask and model return the activations of a specific layer.
        Do this through the hidden_states tuple, returned by the model.

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

    def _extract_residual(self, model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Extract the residual output at a specific transformer layer using a hook.

        Args:
            model: The transformer model.
            input_ids: Input tensor [B, T].
            attention_mask: Attention mask tensor [B, T].
            layer_idx: Index of the transformer block/layer to extract the residual from.

        Returns:
            Tensor of shape [B, T, D] for the specified layer residual.
        """
        residuals = {}

        def hook_fn(module, input, output):
            # output is typically [B, T, D]
            # Store the output of the forward hook
            if isinstance(output, tuple):
                residuals['value'] = output[0].detach()
            else:
                residuals['value'] = output.detach()

        # Find the transformer block module (most models: model.transformer.h[layer_idx] or model.transformer.blocks[layer_idx])
        # Support common naming
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            block_module = model.transformer.h[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            block_module = model.transformer.blocks[layer_idx]
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            block_module = model.gpt_neox.layers[layer_idx]
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            block_module = model.model.layers[layer_idx]
        else:
            raise ValueError("Cannot find transformer block in model for hooks.")

        # Register a forward hook on the block; typically, the output of the whole block is the residual before it goes into the next block
        hook = block_module.register_forward_hook(hook_fn)

        # Forward pass (with gradients off)
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )

        hook.remove()

        # The stored 'value' should be [B, T, D]
        return residuals['value'] * attention_mask.unsqueeze(-1)

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
        self.best_hyperparameters = state_dict['best_hyperparameters']
        self.best_probe = self.ProbeClass(self.d_model, **self.best_hyperparameters).to(self.device).load_state_dict(state_dict['best_probe_state_dict'])
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


class TorchLayerProbe(TorchProbe):

    def __init__(self, config: dict):
        """
        The torch layer probe inherits from the torch probe, it adapts the activations to 
        allow for layer wise probing rather than probing along the sequence dimension.
        """
        super().__init__(config)
        

    @property
    def name(self) -> str:
        return "torch_layer_probe"

    def _extract_layer_features(self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_indices: list[int]) -> torch.Tensor:
        """
        Extract the activations of a specific layer

        Args:
            model: The transformer model.
            input_ids: The input ids tensor.
            attention_mask: The attention mask tensor.
            layer_indices: The layer indices to extract.

        Returns:
            Returns: A tensor of activations of shape [B, T, D] where
            B - batch size
            L - Layer dimension
            D - hidden size
            Here the layer_indices is adapted and used to select a single sequence position working backwards
            from the last token of the sequence.
        """

        if self.config.get('use_hooks', False):
            raise NotImplementedError("Hooks are not supported for layer probing.")

        else:
            return self._extract_hidden_outputs(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=layer_indices,
            )
    
    
    @torch.no_grad()
    def _extract_hidden_outputs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_indices: list[int],
    )-> torch.Tensor:
        """
        Given an input ids, attention mask and model return the activations of a specific layer.
        Do this through the hidden_states tuple, returned by the model.

        Args:
            model: The transformer model.
            input_ids: The input ids tensor.
            attention_mask: The attention mask tensor.
            layer_indices: The sequence position from the end of the input ids to extract.

        Returns:
            A tensor of activations of shape [B, L, D] where
            B - batch size
            L: - Layer dimension
            D - hidden size
        """

        assert len(layer_indices) == 1, "Only one layer can be extracted at a time."

        sequence_position = layer_indices[0] * -1

        # Extract the hidden states from the model.
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # To avoid OOM errors we extract the sequence position before stacking the hidden states.
        hidden_states = out.hidden_states  # tuple: (emb, layer1, ..., layerN), each [B,T,D]
        layer_activations = [hidden_state[:, sequence_position, ...] * attention_mask.unsqueeze(-1)[:, sequence_position, ...] \
            for hidden_state in hidden_states]
        
        return torch.stack(layer_activations, dim=1) # [Batch size, Layer dimension, Seq position, hidden size]