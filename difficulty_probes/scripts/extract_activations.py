"""
Extract activations from LLM layers at post-instruction token positions.

This script replicates the activation extraction methodology from
difficulty_direction/get_directions_store_preds.py where activations are
extracted only from the tokens that follow the instruction in the chat template.
"""

import argparse
import os
from typing import List, Optional
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset


def get_eoi_toks(tokenizer: AutoTokenizer) -> List[int]:
    """
    Get end-of-instruction tokens by applying chat template and extracting
    the tokens that come after the instruction placeholder.
    
    This replicates ModelBase._get_eoi_toks() from difficulty_direction.
    """
    # Apply chat template with a placeholder instruction
    messages = [{"role": "user", "content": "{instruction}"}]
    templated = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Split by the placeholder and get the part after it
    parts = templated.split("{instruction}")
    if len(parts) < 2:
        raise ValueError("Chat template did not contain the instruction placeholder")
    
    post_instruction_text = parts[-1]
    
    # Tokenize the post-instruction part (without special tokens)
    eoi_toks = tokenizer.encode(post_instruction_text, add_special_tokens=False)
    
    return eoi_toks


@torch.no_grad()
def extract_layer_features(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
    positions: List[int],
):
    """
    Extract activations from specified layers and token positions.
    
    Args:
        model: The language model
        input_ids: Token IDs [B, T]
        attention_mask: Attention mask [B, T]
        layer_indices: Which layers to extract from
        positions: Token positions to extract (negative indices from end)
    
    Returns:
        feats: Extracted features [B, L, P, D] where L=len(layer_indices), P=len(positions)
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = out.hidden_states  # tuple: (emb, layer1, ..., layerN), each [B,T,D]

    # Normalize negative indices
    num_hs = len(hidden_states)
    norm_idxs = []
    for i in layer_indices:
        norm_idxs.append(i if i >= 0 else (num_hs + i))

    hs_list = [hidden_states[i] for i in norm_idxs]  # list of [B,T,D]

    # Extract features for each position
    lengths = attention_mask.sum(dim=1)  # [B]
    feats = []
    
    for hs in hs_list:
        B, T, D = hs.shape
        layer_feats = []
        for pos in positions:
            # Negative position: count from the end (last valid token)
            if pos < 0:
                idx = (lengths + pos).clamp(min=0, max=T-1)  # [B]
            else:
                # Positive position: use directly
                idx = torch.full((B,), pos, device=hs.device).clamp(min=0, max=T-1)
            
            pos_feats = hs[torch.arange(B, device=hs.device), idx]  # [B,D]
            layer_feats.append(pos_feats)
        
        feats.append(torch.stack(layer_feats, dim=1))  # [B, P, D]
    
    result = torch.stack(feats, dim=1)  # [B, L, P, D]
    return result


def parse_layers_arg(layers_arg: str, num_hidden_states: int) -> List[int]:
    """
    Parse layer specification string.
    
    Examples:
        "all" -> [0..num_hidden_states-1]
        "1,2,3" -> [1,2,3]
        "-1" -> [-1]
    """
    s = layers_arg.strip().lower()
    if s == "all":
        return list(range(num_hidden_states))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def extract_and_save_activations(
    model_name: str,
    dataset_path: str,
    output_path: str,
    label_column: str = "success_rate",
    question_column: str = "question",
    layers_arg: str = "all",
    eoi_tokens: Optional[int] = None,
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    use_chat_template: bool = False,
):
    """
    Extract activations from post-instruction tokens at specified layers and save to disk.
    
    This replicates get_directions_store_preds.py where positions are determined by
    the number of end-of-instruction (EOI) tokens in the chat template.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_path: Path to parquet file with questions and labels
        output_path: Where to save extracted activations
        label_column: Name of column containing labels/scores
        question_column: Name of column containing questions/prompts
        layers_arg: Layer specification string
        eoi_tokens: Number of end-of-instruction tokens. If None, auto-detect from chat template.
        max_length: Maximum sequence length
        batch_size: Batch size for extraction
        device: Device to use
        max_samples: Maximum number of samples to process
        use_chat_template: Whether to apply chat template (False if already formatted)
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get end-of-instruction tokens
    if eoi_tokens is None:
        print("Auto-detecting end-of-instruction tokens from chat template...")
        eoi_tok_ids = get_eoi_toks(tokenizer)
        n_eoi = len(eoi_tok_ids)
        print(f"Detected {n_eoi} post-instruction tokens: {tokenizer.convert_ids_to_tokens(eoi_tok_ids)}")
    else:
        n_eoi = eoi_tokens
        print(f"Using {n_eoi} post-instruction tokens (manually specified)")
    
    # Positions are the last n_eoi tokens
    # This matches: positions = list(range(-len(model.eoi_toks), 0))
    positions = list(range(-n_eoi, 0))
    print(f"Extracting from positions: {positions}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size

    # Determine number of layers
    with torch.no_grad():
        dummy = tokenizer("hello", return_tensors="pt")
        dummy = {k: v.to(device) for k, v in dummy.items()}
        out = model(**dummy, output_hidden_states=True, use_cache=False)
        num_hidden_states = len(out.hidden_states)

    layer_indices = parse_layers_arg(layers_arg, num_hidden_states)
    print(f"Extracting from {len(layer_indices)} layers: {layer_indices}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    # df = pd.read_parquet(dataset_path)
    df = load_dataset(dataset_path)['train'].to_pandas()
    
    # Check required columns exist
    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in dataset. Available: {df.columns.tolist()}")
    # if label_column not in df.columns:
        # raise ValueError(f"Column '{label_column}' not found in dataset. Available: {df.columns.tolist()}")
    
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Using '{question_column}' as questions, '{label_column}' as labels")

    # Prepare data
    texts = []
    # labels = []
    
    n_samples = min(len(df), max_samples) if max_samples is not None else len(df)
    
    for idx in range(n_samples):
        question = df[question_column].iloc[idx]
        # label = df[label_column].iloc[idx]
        
        # Apply chat template if needed (though usually already formatted)
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": question}]
            question = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        texts.append(question)
        # labels.append(float(label))

    breakpoint()
    print(f"Processing {len(texts)} samples...")

    # Extract activations in batches
    all_activations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        
        # Extract features
        feats = extract_layer_features(
            model, input_ids, attention_mask,
            layer_indices=layer_indices,
            positions=positions
        )  # [B, L, P, D]
        
        all_activations.append(feats.cpu())

    # Concatenate all batches
    all_activations = torch.cat(all_activations, dim=0)  # [N, L, P, D]
    # labels = torch.tensor(labels, dtype=torch.float32)  # [N]

    print(f"Activations shape: {all_activations.shape}")
    # print(f"Labels shape: {labels.shape}")
    print(f"  N={all_activations.shape[0]} samples")
    print(f"  L={all_activations.shape[1]} layers")
    print(f"  P={all_activations.shape[2]} positions")
    print(f"  D={all_activations.shape[3]} hidden_size")
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save({
        'activations': all_activations,
        # 'labels': labels,
        'layer_indices': layer_indices,
        'positions': positions,
        'n_eoi_tokens': n_eoi,
        'd_model': d_model,
        'model_name': model_name,
    }, output_path)
    
    print(f"Saved activations to {output_path}")

    
def main():
    parser = argparse.ArgumentParser(description="Extract activations from post-instruction tokens")
    
    # Model args
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--layers", type=str, default="all",
                        help='Layer specification: "all", "-1", or "0,1,2"')
    parser.add_argument("--eoi_tokens", type=int, default=None,
                        help="Number of end-of-instruction tokens. If not specified, auto-detect from chat template.")
    
    # Data args
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to parquet file with questions and labels")
    parser.add_argument("--label_column", type=str, default="success_rate",
                        help="Name of column containing labels/scores")
    parser.add_argument("--question_column", type=str, default="question",
                        help="Name of column containing questions/prompts")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save extracted activations (.pt file)")
    
    # Processing args
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for extraction")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Apply chat template to questions (use if not already formatted)")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    extract_and_save_activations(
        model_name=args.model,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        label_column=args.label_column,
        question_column=args.question_column,
        layers_arg=args.layers,
        eoi_tokens=args.eoi_tokens,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
        use_chat_template=args.use_chat_template,
    )


if __name__ == "__main__":
    main()
