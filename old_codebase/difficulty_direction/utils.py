import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from einops import rearrange
from torchtyping import TensorType


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def ceildiv(a, b):
    return -(a // -b)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean


def kl_div_fn(
    logits_a: TensorType["batch_size", "seq_len", "vocab_size"],
    logits_b: TensorType["batch_size", "seq_len", "vocab_size"],
    mask: Optional[TensorType[int, "batch_size", "seq_len"]] = None,
    epsilon: float=1e-6
) -> TensorType['batch_size']:
    """Compute the KL divergence loss between two tensors of logits."""
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)


def orthogonal_rejection(activation: TensorType[..., -1], unit_direction: TensorType[-1]) -> TensorType[..., -1]:
    ablated_act = activation - (activation @ unit_direction).unsqueeze(-1) * unit_direction
    return ablated_act


def save_to_json_file(filepath: Path, results: List[Dict]):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df

def fen_to_ascii_grid(fen: str) -> str:
    board_part, side_to_move, castling, en_passant, halfmove, fullmove = fen.split()

    ranks = board_part.split("/")  # 8 ranks from 8 to 1
    ascii_rows = []

    # Parse each rank
    for rank_index, rank in enumerate(ranks):
        row = []
        for ch in rank:
            if ch.isdigit():
                # Digit = that many empty squares
                row.extend(["."] * int(ch))
            else:
                # Piece character as-is (K,Q,R,B,N,P / k,q,r,b,n,p)
                row.append(ch)
        # Sanity check: each row should be 8 squares
        assert len(row) == 8, f"Bad FEN row length: {row}"
        rank_num = 8 - rank_index
        ascii_rows.append(f"{rank_num} | " + " ".join(row))

    # File labels
    files_line = "    a b c d e f g h"

    # Human-friendly metadata
    side_text = "White" if side_to_move == "w" else "Black"
    castling_desc = []
    if "K" in castling: castling_desc.append("White king-side")
    if "Q" in castling: castling_desc.append("White queen-side")
    if "k" in castling: castling_desc.append("Black king-side")
    if "q" in castling: castling_desc.append("Black queen-side")
    if castling == "-":
        castling_str = "No castling rights"
    else:
        castling_str = ", ".join(castling_desc)
    enp_str = "none" if en_passant == "-" else en_passant

    board_str = "BOARD:\n" + "\n".join(ascii_rows) + "\n" + files_line
    meta_str = (
        f"\nSide to move: {side_text}\n"
        f"Castling rights: {castling_str}\n"
        f"En passant square: {enp_str}\n"
        f"Halfmove clock: {halfmove}\n"
        f"Fullmove number: {fullmove}"
    )

    return board_str + meta_str
