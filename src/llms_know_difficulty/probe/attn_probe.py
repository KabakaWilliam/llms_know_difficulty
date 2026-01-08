import itertools
from typing import Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import math
import einops
from base_probe import Probe
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from itertools import product
import numpy as np

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
        texts, ys = zip(*batch)
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        ys = torch.stack(list(ys), dim=0)
        return enc, ys
    return collate

class TextNumberDataset(Dataset):
    def __init__(
            self, 
            prompts: list[str],
            targets: list[float]
    ):

        assert len(prompts) == len(targets), \
        f"Prompts: {len(prompts)} and targets: {len(targets)} must have the same length."
        self.prompts = prompts
        self.targets = targets

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text, y = self.prompts[idx], self.targets[idx]
        return text, torch.tensor(y, dtype=torch.float32)


class AttnProbe(Probe):
    def __init__(self, config):
        super().__init__(config)
        self._has_setup_run = False

    @property
    def name(self) -> str:
        """The name of the probe."""
        return "attn_probe"

    def setup(self, model_name: str, device: str) -> None:
        """
        Any pre-training loading steps, run before .fit or .predict.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.d_model = self.model.config.hidden_size
        self.device = self.config.get('device', 'cuda:0')
        self.cv_layers = self.config.get('layer_indices', list(range(29))) # Hard coded for now, TODO: Make this dynamic or stored in config.
        

    def init_model(self, config: dict):
        """
        Load a prior training checkpoint.
        """
        raise NotImplementedError("Initializing a model from a checkpoint is not implemented for the attention probe.")

    def train(self, train_data: tuple[list[str], list[float]], val_data: tuple[list[str], list[float]]) -> Tuple[Probe, dict[str, Any]]:
        """
        Train a probe on the training data, evaluate on the validation data, and repeat, returning the best probe.
        """

        # Unpack the training and validation data:
        train_prompts, train_targets = train_data
        val_prompts, val_targets = val_data

        # Setup cross validation on layers and other hyperparameters:        
        # Use itertools to create list of all combinations of layers and other hyperparameters:
        cv_hyper_list = self.config.get('cross_validated_hyperparameters', [])
        
        if len(cv_hyper_list) > 0:

            # Make sure the config provides a grid of values:
            for hyper in cv_hyper_list:
                hyper_values = self.config.get(hyper, None)
                if hyper_values is None or not isinstance(hyper_values, list):
                    raise ValueError(
                        f"""Attention probe config error!
                        Cross_validated_hyperparameter includes {hyper} but {hyper} does not provide a list of values.
                        hyper set to {hyper_values} in config.py""")

            # Create a grid of all combinations of the hyperparameters:
            cv_hypers = product(*[self.config.get(hyper, None) for hyper in cv_hyper_list] + [self.cv_layers])
        
            cv_results = []

            #TODO: Move hyper_setup from list to dict so it can be passed to fit... This is likely a bug
            for hyper_setup in cv_hypers:
                
                # Fit the probe to the training data, with these specific hyperparameters:
                probe = self.fit(train_prompts, train_targets, **hyper_setup)

                # Evaluate the probe's performance on the validation data:
                evaluate_metrics = self.evaluate(probe, val_prompts, val_targets, hyperparameters)
                cv_metric = evaluate_metrics[self.config.get('cv_metric', 'mse')]
                cv_results.append([cv_metric, hyper_setup, probe])

            # After training we select the best cv hyperparameters and evaluate the probe on the test_data
            max_or_min_cv_metric = self.config.get('max_or_min_cv_metric', 'max')
            metrics, hyperparameters, probes = zip(*cv_results)
            if max_or_min_cv_metric == 'max':
                best_cv_probe_idx = np.argmax(metrics)
            else:
                best_cv_probe_idx = np.argmin(metrics)

            return probes[best_cv_probe_idx], hyperparameters[best_cv_probe_idx]


    def evaluate(self, probe: AttnLite, prompts: list[str], targets: list[float], hyperparameters: dict) -> dict:
        """
        Evaluate the probe's performance on the validation data.
        """
        
        # Setup data loader:
        eval_dataset = TextNumberDataset(prompts, targets)
        collate_fn = make_collate_fn(self.tokenizer, CollateConfig(max_length=self.config.get('max_length', 256)))
        train_loader = DataLoader(eval_dataset, batch_size=self.config.get('batch_size', 128), shuffle=True, collate_fn=collate_fn)

        eval_layer = hyperparameters.get('layer', None)
        assert eval_layer is not None, "Evaluation layer must be specified."

        # Evaluate the probe:
        outputs = []
        targets = []
        for enc, ys in train_loader:
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
            outputs.append(preds)
            targets.append(ys)

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        

    def fit(self, prompts: list[str],
                targets: list[float],
                layer: int,
                batch_size:int,
                learning_rate:float,
                num_epochs:int,
                weight_decay:float) -> None:
        """
        Run a single training loop for the attention probe. Very simple no validation or logging ortracking etc...

        Args:
            prompts: A list of prompts to train on.
            targets: A list of targets to train on.
            layer: The layer to train on.

        TODO: Add setup where the activations are extracted via a hook, not using hidden_outputs.
        TODO: Add some validation loss eventually.
        """

        # Setup train loader:
        train_dataset = TextNumberDataset(prompts, targets)
        collate_fn = make_collate_fn(self.tokenizer, CollateConfig(max_length=self.config.get('max_length', 256)))
        train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 128), shuffle=True, collate_fn=collate_fn)

        # Setup the probe:
        probe = AttnLite(self.d_model).to(self.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=self.config.get('learning_rate', 1e-3))
        loss_fn = nn.MSELoss()

        # Train the probe:
        for epoch in range(1, self.config.get('num_epochs', 2) + 1):
            
            probe.train()
            running = 0.0
            step = 0
            n = 0

            for enc, ys in train_loader:
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

                if step % 25 == 0:
                    print(f"  epoch {epoch:03d} | step {step:04d} | batch_mse={loss.item():.6f}")
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

        Returns: A tensor of activations of shape [B, L, T, D] where
            B - batch size
            L - layer dimension
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
        return torch.stack(layer_activations, dim=1) # [Batch size, N_layers, Sequence length, hidden size]

        