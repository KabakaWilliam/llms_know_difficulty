# linear_probe_hf_llm_multilayer.py
import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -----------------------------
# Data
# -----------------------------
class TextNumberDataset(Dataset):
    def __init__(
            self, 
            hf_dataset: str,
            hf_dataset_split: str,
            scores_path: str
    ):
        text_dataset = load_dataset(hf_dataset)[hf_dataset_split]
        scores_dataset = load_dataset('parquet', data_files=scores_path)['train']
        assert len(text_dataset) == len(scores_dataset), "Text and scores dataset must have the same length."

        self.items = []
        for text_item, score_item in zip(text_dataset, scores_dataset):
            assert score_item['ground_truth'] in text_item['solution'], "Ground truth not found in solution text."
            self.items.append((text_item['problem'], float(score_item['success_rate'])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, y = self.items[idx]
        return text, torch.tensor(y, dtype=torch.float32)


@dataclass
class CollateConfig:
    max_length: int = 256


def make_collate_fn(tokenizer: AutoTokenizer, cfg: CollateConfig):
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


# -----------------------------
# Probes
# -----------------------------
class LinearProbeSingle(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, D]
        return self.linear(x).squeeze(-1)  # [B]


class LinearProbeConcat(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.linear = nn.Linear(d_model * n_layers, 1)

    def forward(self, xs):  # xs: [B, n_layers, D]
        B, L, D = xs.shape
        x = xs.reshape(B, L * D)
        x = self.linear(x).squeeze(-1)
        return x  # [B]

class LinearProbeConcatAllSequencePositions(nn.Module):
    def __init__(self, d_model: int, n_layers: int, seq_length: int):
        super().__init__()
        self.linear = nn.Linear(d_model * n_layers * seq_length, 1)

    def forward(self, xs):  # xs: [B, n_layers, seq_length, D]
        B, L, T, D = xs.shape
        x = xs.reshape(B, L * T * D)
        x = self.linear(x).squeeze(-1)
        return x  # [B]

class LinearProbeWeightedSum(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        # Learn layer weights (softmaxed)
        self.logits = nn.Parameter(torch.zeros(n_layers))
        self.linear = nn.Linear(d_model, 1)

    def forward(self, xs):  # xs: [B, n_layers, D]
        w = torch.softmax(self.logits, dim=0)             # [n_layers]
        x = (xs * w.view(1, -1, 1)).sum(dim=1)            # [B, D]
        return self.linear(x).squeeze(-1)


# -----------------------------
# Activation extraction
# -----------------------------
@torch.no_grad()
def extract_layer_features(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
    pool: str = "last_token",
):
    """
    Returns: feats [B, L, D] where L=len(layer_indices)
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = out.hidden_states  # tuple: (emb, layer1, ..., layerN), each [B,T,D]

    # Normalize negative indices against tuple length
    num_hs = len(hidden_states)
    norm_idxs = []
    for i in layer_indices:
        norm_idxs.append(i if i >= 0 else (num_hs + i))

    hs_list = [hidden_states[i] for i in norm_idxs]  # list of [B,T,D]

    if pool == "last_token":
        lengths = attention_mask.sum(dim=1)               # [B]
        idx = (lengths - 1).clamp(min=0)                  # [B]
        feats = []
        for hs in hs_list:
            B, T, D = hs.shape
            feats.append(hs[torch.arange(B, device=hs.device), idx])  # [B,D]
        return torch.stack(feats, dim=1)  # [B, L, D]

    elif pool == "mean":
        mask = attention_mask.unsqueeze(-1)  # [B,T,1]
        feats = []
        for hs in hs_list:
            summed = (hs * mask).sum(dim=1)                  # [B,D]
            denom = mask.sum(dim=1).clamp(min=1)             # [B,1]
            feats.append(summed / denom)
        return torch.stack(feats, dim=1)  # [B, L, D]

    else:
        raise ValueError(f"Unknown pool={pool}")


def parse_layers_arg(layers_arg: str, num_hidden_states: int) -> List[int]:
    """
    layers_arg:
      - "all" -> [0..num_hidden_states-1]
      - "1,2,3" -> [1,2,3]
      - "-1" -> [-1]
    Note: hidden_states includes embeddings at index 0.
    """
    s = layers_arg.strip().lower()
    if s == "all":
        return list(range(num_hidden_states))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# -----------------------------
# Train / Eval
# -----------------------------
def train_probe(
    model_name: str,
    hf_dataset: str,
    train_scores_path: str,
    val_scores_path: str,
    layers_arg: str,
    layer_mode: str,
    pool: str,
    max_length: int,
    batch_size: int,
    lr: float,
    epochs: int,
    weight_decay: float,
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size

    # Determine how many hidden_states the model returns (embeddings + layers)
    # Do a tiny forward pass to discover tuple length robustly.
    with torch.no_grad():
        dummy = tokenizer("hello", return_tensors="pt")
        dummy = {k: v.to(device) for k, v in dummy.items()}
        out = model(**dummy, output_hidden_states=True, use_cache=False)
        num_hidden_states = len(out.hidden_states)

    layer_indices = parse_layers_arg(layers_arg, num_hidden_states)

    if layer_mode == "single":
        if len(layer_indices) != 1:
            raise ValueError("layer_mode=single requires exactly one layer index (e.g. --layers -1).")
        probe = LinearProbeSingle(d_model).to(device)
    elif layer_mode == "concat":
        probe = LinearProbeConcat(d_model, n_layers=len(layer_indices)).to(device)
    elif layer_mode == "weighted_sum":
        probe = LinearProbeWeightedSum(d_model, n_layers=len(layer_indices)).to(device)
    else:
        raise ValueError(f"Unknown layer_mode={layer_mode}")

    train_ds = TextNumberDataset(hf_dataset=hf_dataset, hf_dataset_split="train", scores_path=train_scores_path)
    val_ds = TextNumberDataset(hf_dataset=hf_dataset, hf_dataset_split="test", scores_path=val_scores_path)
    val_ds.items = val_ds.items[:1000]  # limit val set size for speed

    print(f"Loaded train set of size {len(train_ds)} and val set of size {len(val_ds)}")

    collate_fn = make_collate_fn(tokenizer, CollateConfig(max_length=max_length))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_ds else None

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def forward_probe(enc):
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            feats = extract_layer_features(
                model, input_ids, attention_mask,
                layer_indices=layer_indices,
                pool=pool
            )  # [B,L,D] if L>1 else still [B,1,D]

        if layer_mode == "single":
            x = feats[:, 0, :]          # [B,D]
            preds = probe(x)
        else:
            preds = probe(feats)        # [B]
        return preds

    def bin(y, n_bins=10, min_val=0.0, max_val=1.0):
        # Use torch bucketize to bin y into n_bins between min_val and max_val
        bins = torch.linspace(min_val, max_val, n_bins + 1).to(y.device)  # [n_bins+1]
        # clamp y to [min_val, max_val]
        y = y.clamp(min=min_val, max=max_val)
        # add small epsilon to handle edge case where y == min_val
        return torch.bucketize(y + 1e-8, bins) - 1  # [B], bin indices

    @torch.no_grad()
    def evaluate():
        probe.eval()

        # collect all targets and predictions
        all_ys, all_preds = [], []
        for enc, ys in tqdm(val_loader, desc="evaluating..."):
            ys = ys.to(device)
            preds = forward_probe(enc)
            all_ys.append(ys.detach())  
            all_preds.append(preds.detach())

        ys = torch.cat(all_ys, dim=0)
        preds = torch.cat(all_preds, dim=0)

        # global regression metrics
        mse = loss_fn(preds, ys).item()
        mae = (preds - ys).abs().mean().item()

        # Spearman's rank correlation (Pearson corr of ranks)
        y = ys.view(-1).float()
        p = preds.view(-1).float()

        # ranks via stable double-argsort (ties get arbitrary order; see note below)
        y_rank = torch.argsort(torch.argsort(y, stable=True), stable=True).float()
        p_rank = torch.argsort(torch.argsort(p, stable=True), stable=True).float()

        y_rank = y_rank - y_rank.mean()
        p_rank = p_rank - p_rank.mean()
        spearman = (y_rank * p_rank).sum() / (torch.sqrt((y_rank**2).sum()) * torch.sqrt((p_rank**2).sum()) + 1e-12)
        spearman = spearman.item()

        # binning + accuracies
        binned_ys = bin(ys, n_bins=5, min_val=0.0, max_val=1.0)
        binned_preds = bin(preds, n_bins=5, min_val=0.0, max_val=1.0)

        acc_all = (binned_ys == binned_preds).float().mean().item()

        metrics = {
            "mse": mse,
            "mae": mae,
            "spearman": spearman,
            "acc_all": acc_all,
        }

        for b in range(5):
            mask = (binned_ys == b)
            correct = (binned_preds[mask] == b).float().sum()
            count = mask.sum()
            metrics[f"count_bin_{b}"] = count.item()
            metrics[f"acc_bin_{b}"] = (correct / count.clamp_min(1)).item()

        # do precision, recall and num predictions for each class
        for b in range(5):
            true_positives = ((binned_preds == b) & (binned_ys == b)).float().sum()
            predicted_positives = (binned_preds == b).float().sum()
            actual_positives = (binned_ys == b).float().sum()

            metrics[f"num_predicted_bin_{b}"] = predicted_positives.item()

            precision = (true_positives / predicted_positives.clamp_min(1)).item()
            recall = (true_positives / actual_positives.clamp_min(1)).item()

            metrics[f"precision_bin_{b}"] = precision
            metrics[f"recall_bin_{b}"] = recall

        return metrics


    step = 0
    for epoch in range(1, epochs + 1):
        probe.train()
        running = 0.0
        n = 0
        for enc, ys in train_loader:
            ys = ys.to(device)
            preds = forward_probe(enc)
            loss = loss_fn(preds, ys)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = ys.size(0)
            running += loss.item() * bs
            n += bs
            step += 1
            print(f"{step=}")

            if step % 25 == 0:
                print(f"  step {step} | batch_mse={loss.item():.6f}")
                metrics = evaluate()
                pprint.pprint(metrics)

        train_mse = running / max(n, 1)
        msg = f"epoch {epoch:03d} | train_mse={train_mse:.6f}"

        if val_loader is not None:
            metrics = evaluate()
            pprint.pprint(metrics)

        if layer_mode == "weighted_sum":
            with torch.no_grad():
                w = torch.softmax(probe.logits, dim=0).detach().cpu()
            msg += f" | top_layer_weight={w.max().item():.3f}"

        print(msg)

    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--hf_dataset", type=str, required=True)
    ap.add_argument("--train_scores_path", type=str, required=True)
    ap.add_argument("--val_scores_path", type=str, required=True)

    ap.add_argument("--layers", type=str, default="-1",
                    help='Layer indices over hidden_states tuple. "all" or e.g. "-1" or "0,1,2". Note 0=embeddings.')
    ap.add_argument("--layer_mode", type=str, default="single",
                    choices=["single", "concat", "weighted_sum"])

    ap.add_argument("--pool", type=str, default="last_token", choices=["last_token", "mean"])
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # example usage: python predict_success_rate.py --model gpt2 --

    train_probe(
        model_name=args.model,
        hf_dataset=args.hf_dataset,
        train_scores_path=args.train_scores_path,
        val_scores_path=args.val_scores_path,
        layers_arg=args.layers,
        layer_mode=args.layer_mode,
        pool=args.pool,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
