import os
import hashlib
from typing import List, Optional, Tuple, Dict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from ....config import ROOT_ACTIVATION_DATA_DIR

ROOT_ACTIVATION_DATA_DIR = os.path.join(ROOT_ACTIVATION_DATA_DIR,"Linear_EOI_probe")

def get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    """
    Get the device where the model expects its inputs (i.e. the embedding layer device).
    Works for both single-GPU and multi-GPU (device_map="auto") models.
    """
    try:
        # For models loaded with device_map, get the device of the first parameter
        # (the embedding layer, which is where inputs need to go)
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
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

def get_eoi_toks(tokenizer: AutoTokenizer) -> List[int]:
    """
    Get end-of-instruction tokens by applying chat template and extracting
    the tokens that come after the instruction placeholder.
    
    This replicates ModelBase._get_eoi_toks() from difficulty_direction.
    
    For models without a chat template (like gpt2), defaults to last token only.
    """
    # Check if tokenizer has a chat template
    if tokenizer.chat_template is None:
        # Default for models without chat template (e.g., gpt2)
        # Return a single token at the end of sequence
        return [tokenizer.eos_token_id if tokenizer.eos_token_id else 0]
    
    # Apply chat template with a placeholder instruction
    messages = [{"role": "user", "content": "{instruction}"}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    except (ValueError, KeyError):
        # Fallback if chat template application fails
        print("No chat template found, returing eos token")
        return [tokenizer.eos_token_id if tokenizer.eos_token_id else 0]
    
    # Split by the placeholder and get the part after it
    parts = templated.split("{instruction}")
    if len(parts) < 2:
        # Fallback if placeholder not found
        return [tokenizer.eos_token_id if tokenizer.eos_token_id else 0]
    
    post_instruction_text = parts[-1]
    
    # Tokenize the post-instruction part (without special tokens)
    eoi_toks = tokenizer.encode(post_instruction_text, add_special_tokens=False)
    
    return eoi_toks

@torch.no_grad()
def _extract_layer_features(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: list,
    positions: List[int]
) -> torch.Tensor:
    """
    Extract activations from specified layers and token positions.

    Returns: A tensor of activations of shape [B, L, P, D] where
        B - batch size
        L - layer dimension
        P - Token positions to extract (negative indices from end)
        D - hidden size
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

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size
    input_device = get_model_input_device(model)

    # Determine number of layers
    with torch.no_grad():
        dummy = tokenizer("hello", return_tensors="pt")
        dummy = {k: v.to(input_device) for k, v in dummy.items()}
        out = model(**dummy, output_hidden_states=True, use_cache=False)
        num_hidden_states = len(out.hidden_states)

    layer_indices = parse_layers_arg(layers_arg, num_hidden_states)
    print(f"Extracting from {len(layer_indices)} layers: {layer_indices}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # Check required columns exist
    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in dataset. Available: {df.columns.tolist()}")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset. Available: {df.columns.tolist()}")
    
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Using '{question_column}' as questions, '{label_column}' as labels")

    # Prepare data
    texts = []
    labels = []
    
    n_samples = min(len(df), max_samples) if max_samples is not None else len(df)
    
    for idx in range(n_samples):
        question = df[question_column].iloc[idx]
        label = df[label_column].iloc[idx]
        
        # Apply chat template if needed (though usually already formatted)
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": question}]
            question = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        texts.append(question)
        labels.append(float(label))

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
        input_ids = enc['input_ids'].to(input_device)
        attention_mask = enc['attention_mask'].to(input_device)
        
        # Extract features
        feats = _extract_layer_features(
            model, input_ids, attention_mask,
            layer_indices=layer_indices,
            positions=positions
        )  # [B, L, P, D]
        
        all_activations.append(feats.cpu())

    # Concatenate all batches
    all_activations = torch.cat(all_activations, dim=0)  # [N, L, P, D]
    labels = torch.tensor(labels, dtype=torch.float32)  # [N]

    print(f"Activations shape: {all_activations.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"  N={all_activations.shape[0]} samples")
    print(f"  L={all_activations.shape[1]} layers")
    print(f"  P={all_activations.shape[2]} positions")
    print(f"  D={all_activations.shape[3]} hidden_size")
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save({
        'activations': all_activations,
        'labels': labels,
        'layer_indices': layer_indices,
        'positions': positions,
        'n_eoi_tokens': n_eoi,
        'd_model': d_model,
        'model_name': model_name,
    }, output_path)
    
    print(f"Saved activations to {output_path}")


@torch.no_grad()
def extract_activations_from_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    labels: List[float] | None,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
    eoi_tokens: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    specified_eoi_position: Optional[int] = None,
) -> tuple:
    """
    Extract activations from texts without saving to disk.
    
    Used by SklearnProbe.train() and predict() to extract activations in-memory.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        texts: List of input texts/prompts
        labels: List of float labels corresponding to texts
        device: Device to use ("cuda", "cpu", etc.)
        batch_size: Batch size for extraction
        max_length: Maximum sequence length
        eoi_tokens: Number of end-of-instruction tokens. If None, auto-detect from chat template.
        layer_indices: Which layers to extract. If None, extracts all layers.
        specified_eoi_position: If provided, extract only this specific EOI position (e.g., -1 for last token).
                                If None, extracts all EOI positions from -n_eoi to 0.
    
    Returns:
        Tuple of (activations, labels_tensor, layer_indices, positions, n_eoi, d_model)
        where:
            activations: Tensor of shape [N, L, P, D] or [N, L, D] if specified_eoi_position is used
            labels_tensor: Tensor of shape [N]
            layer_indices: List of extracted layer indices
            positions: List of token positions extracted
            n_eoi: Number of end-of-instruction tokens
            d_model: Hidden dimension size
    """
    # Get end-of-instruction tokens
    if eoi_tokens is None:
        eoi_tok_ids = get_eoi_toks(tokenizer)
        n_eoi = len(eoi_tok_ids)
    else:
        n_eoi = eoi_tokens
    
    # Determine positions to extract
    if specified_eoi_position is not None:
        # Extract only the specified position
        positions = [specified_eoi_position]
    else:
        # Extract all EOI positions (default behavior)
        positions = list(range(-n_eoi, 0))
    
    # Resolve the actual input device from the model (supports multi-GPU device_map="auto")
    input_device = get_model_input_device(model)
    
    # Determine number of layers if not provided
    if layer_indices is None:
        with torch.no_grad():
            dummy = tokenizer("hello", return_tensors="pt")
            dummy = {k: v.to(input_device) for k, v in dummy.items()}
            out = model(**dummy, output_hidden_states=True, use_cache=False)
            num_hidden_states = len(out.hidden_states)
        layer_indices = list(range(num_hidden_states))
    
    d_model = model.config.hidden_size
    
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
        input_ids = enc['input_ids'].to(input_device)
        attention_mask = enc['attention_mask'].to(input_device)
        
        # Extract features
        feats = _extract_layer_features(
            model, input_ids, attention_mask,
            layer_indices=layer_indices,
            positions=positions
        )  # [B, L, P, D]
        
        all_activations.append(feats.cpu())
    
    # Concatenate all batches
    all_activations = torch.cat(all_activations, dim=0)  # [N, L, P, D]
    labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels is not None else None  # [N]
    
    # Squeeze position dimension if only extracting one position
    if specified_eoi_position is not None:
        all_activations = all_activations.squeeze(2)  # [N, L, P, D] -> [N, L, D]
    return all_activations, labels_tensor, layer_indices, positions, n_eoi, d_model


def get_cache_path(
    model_name: str,
    texts: List[str],
    split: str = "train",
    label_column: str = "labels",
    layers_arg: str = "all",
    eoi_tokens: Optional[int] = None,
    max_length: int = 512,
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
    labels: Optional[List[float]] = None,
) -> str:
    """
    Generate a cache path for activations based on model name and data.
    Uses a hash of the first prompt, labels hash, and explicit split type to ensure unique and traceable filenames.
    
    Args:
        model_name: HuggingFace model identifier
        texts: List of input texts (first one hashed to create unique filename)
        split: Data split type ("train", "val", or "test") - makes it explicit what data is cached
        label_column: Column name for labels (used in filename)
        layers_arg: Layer specification string
        eoi_tokens: Number of end-of-instruction tokens
        max_length: Maximum sequence length
        batch_size: Batch size used for extraction
        cache_dir: Directory to cache in. If None, uses ROOT_ACTIVATION_DATA_DIR
        labels: List of labels to include in cache path hash (ensures different label sets get different caches)
    
    Returns:
        Full path to cache file
        
    Example:
        For train split with gpt2 model, texts starting with "Hello world", batch_size 16:
        {cache_dir}/gpt2_cache/activations_train_a1b2c3d4_xyz789ab_len512_bs16.pt
    """
    import hashlib
    
    if cache_dir is None:
        cache_dir = ROOT_ACTIVATION_DATA_DIR
    
    # Validate split type
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
    
    # Create a hash of the first prompt to ensure unique paths for different datasets
    # Using only first prompt (not first 100) for more predictable hashing
    first_text = texts[0] if texts else ""
    text_hash = hashlib.md5(first_text.encode()).hexdigest()[:8]
    
    # Create a hash of labels to distinguish datasets with different k values (e.g., k=1 vs k=50)
    # This ensures that different label distributions get different cache files
    labels_hash = ""
    if labels is not None:
        import numpy as np
        labels_array = np.array(labels)
        # Hash based on: number of unique labels + min/max/mean values
        labels_str = f"{len(np.unique(labels_array))}-{labels_array.min():.6f}-{labels_array.max():.6f}-{labels_array.mean():.6f}"
        labels_hash = hashlib.md5(labels_str.encode()).hexdigest()[:8]
    
    model_cache_dir = os.path.join(cache_dir, f"{model_name.replace('/', '-')}_cache")
    
    # New naming scheme: explicit split type + hash of first prompt + hash of labels
    # Example: activations_train_a1b2c3d4_xyz789ab_len512_bs16.pt
    if labels_hash:
        cache_filename = f"activations_{split}_{text_hash}_{labels_hash}_len{max_length}_bs{batch_size}.pt"
    else:
        cache_filename = f"activations_{split}_{text_hash}_len{max_length}_bs{batch_size}.pt"
    
    return os.path.join(model_cache_dir, cache_filename)

def extract_or_load_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    labels: List[float],
    model_name: str,
    split: str = "train",
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
    eoi_tokens: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    specified_eoi_position: Optional[int] = None,
) -> dict:
    """
    Extract activations from texts, using cache if available.
    
    This is a high-level wrapper that:
    1. Checks if activations are cached (using split type for explicit identification)
    2. Loads from cache if available (and use_cache=True)
    3. Extracts fresh activations if not cached
    4. Saves to cache for future use
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        texts: List of input texts/prompts
        labels: List of float labels corresponding to texts
        model_name: Model identifier (used for cache path)
        split: Data split type ("train", "val", or "test") - must be specified for proper caching
        device: Device to use
        batch_size: Batch size for extraction
        max_length: Maximum sequence length
        eoi_tokens: Number of end-of-instruction tokens. If None, auto-detect.
        layer_indices: Which layers to extract. If None, extracts all layers.
        cache_dir: Directory to cache in. If None, uses ROOT_ACTIVATION_DATA_DIR
        use_cache: Whether to use cached activations if available
        specified_eoi_position: If provided, extract only this specific EOI position (e.g., -1).
                                If None, extracts all EOI positions.
    
    Returns:
        Dictionary with keys:
            'activations': Tensor of shape [N, L, P, D] or [N, L, D] if specified_eoi_position is used
            'labels': Tensor of shape [N]
            'layer_indices': List of extracted layer indices
            'positions': List of token positions extracted
            'n_eoi_tokens': Number of end-of-instruction tokens
            'd_model': Hidden dimension size
            'model_name': The model identifier
            'split': The data split used
            'from_cache': Boolean indicating if loaded from cache
    """
    import os
    
    if cache_dir is None:
        cache_dir = ROOT_ACTIVATION_DATA_DIR
    
    # Validate split
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
    
    # Generate cache path with explicit split type and labels hash
    cache_path = get_cache_path(
        model_name=model_name,
        texts=texts,
        split=split,
        label_column="labels",
        layers_arg="all",
        eoi_tokens=eoi_tokens,
        max_length=max_length,
        batch_size=batch_size,
        cache_dir=cache_dir,
        labels=labels,
    )
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached {split} activations from: {cache_path}")
        cached_data = torch.load(cache_path)
        cached_data['from_cache'] = True
        return cached_data
    
    # Extract fresh activations
    print(f"Extracting {split} activations (not found in cache)...")
    activations, labels_tensor, extracted_layer_indices, positions, n_eoi, d_model = \
        extract_activations_from_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            labels=labels,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            eoi_tokens=eoi_tokens,
            layer_indices=layer_indices,
            specified_eoi_position=specified_eoi_position,
        )
    
    # Save to cache
    print(f"Caching {split} activations to: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({
        'activations': activations,
        'labels': labels_tensor,
        'layer_indices': extracted_layer_indices,
        'positions': positions,
        'n_eoi_tokens': n_eoi,
        'd_model': d_model,
        'model_name': model_name,
        'split': split,
    }, cache_path)
    
    return {
        'activations': activations,
        'labels': labels_tensor,
        'layer_indices': extracted_layer_indices,
        'positions': positions,
        'n_eoi_tokens': n_eoi,
        'd_model': d_model,
        'model_name': model_name,
        'split': split,
        'from_cache': False,
    }
