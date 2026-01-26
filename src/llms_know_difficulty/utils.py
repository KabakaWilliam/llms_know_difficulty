import os
import pandas as pd
import torch
import json

from pathlib import Path
from llms_know_difficulty.probe.base_probe import Probe
from datetime import datetime


import numpy as np
from sklearn.utils.multiclass import type_of_target

def infer_task_type(y: np.ndarray, task_type: str = "auto") -> str:
    """Infer whether task is regression or classification."""
    if task_type in ("regression", "classification"):
        return task_type

    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return "classification"
    else:
        return "regression"

def create_results_path(root_data_dir: str, dataset_name: str, model_name: str, probe_name: str, gen_str: str = None) -> Path:
    """
    Create a path for saving probe results
    
    Args:
        dataset_name: Name of the dataset (e.g., "DigitalLearningGmbH_MATH-lighteval")
        model_name: Name of the model (e.g., "gpt2")
        probe_name: Name of the probe (e.g., "sklearn_probe")
        gen_str: Optional generation settings string (e.g., "maxlen_3000_k_8_temp_0.7")
    
    Returns:
        Path to results directory with structure:
        data/results/{model_name}/{dataset_name}/{probe_name}/{gen_str}/{timestamp}
        or
        data/results/{model_name}/{dataset_name}/{probe_name}/{timestamp} if gen_str is None
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if gen_str:
        results_path = os.path.join(root_data_dir, "results", model_name, dataset_name, probe_name, gen_str, timestamp)
    else:
        results_path = os.path.join(root_data_dir, "results", model_name, dataset_name, probe_name, timestamp)

    os.makedirs(results_path, exist_ok=True)
    return Path(results_path)

def save_probe_predictions(probe_preds: dict, probe_metadata: dict, best_probe: Probe, results_path: Path):
    """
    Save the probe predictions to the results directory
    """

    assert os.path.exists(results_path), "Results path does not exist"
    
    print(f'Saving probe results to {results_path}')

    # Save the probe predictions as a jsonl file and parquet:
    # assertions on the probe preds:
    assert probe_preds[0].dtype == torch.int32, \
        f"probe_preds[0] is the idx must be of type torch.int32, got {probe_preds[0].dtype}"
    assert probe_preds[1].dtype == torch.float32, \
        f"probe_preds[1] is the predictions must be of type torch.float32, got {probe_preds[1].dtype}"
    

    # Combine them as a pandas dataframe:
    df = pd.DataFrame({"idx": probe_preds[0].tolist(), "pred": probe_preds[1].tolist()})
    
    # Save as parquet
    df.to_parquet(results_path / "probe_preds.parquet")
    
    # Save as jsonl
    with open(results_path / "probe_preds.jsonl", "w") as f:
        for _, row in df.iterrows():
            json.dump({"idx": int(row["idx"]), "pred": float(row["pred"])}, f)
            f.write("\n")
    
    # Save the best probe (and pass full metadata to ensure it's saved with test metrics)
    best_probe.save_probe(results_path, metadata=probe_metadata)

def parse_dataset_str(dataset_str: str) -> dict:
    """
    Parse dataset string to extract components.
    
    Format: {DS_NAME}/{SPLIT}-{HF_MODEL_WITH_DASHES}_maxlen_{MAXLEN}_k_{K}_temp_{TEMP}.parquet
    
    Example:
        "DigitalLearningGmbH_MATH-lighteval/test-openai-gpt-oss-20b_maxlen_3000_k_1_temp_1.0.parquet"
        Returns:
        {
            "dataset_name": "DigitalLearningGmbH_MATH-lighteval",
            "split": "test",
            "model_name": "openai/gpt-oss-20b",
            "maxlen": 3000,
            "k": 1,
            "temp": 1.0
        }
    
    Notes:
        - Split names are case-insensitive (Test, test, TEST all become 'test')
        - Model names are converted to HuggingFace format (org/model)
        - Handles organizations with dashes (e.g., miromind-ai)
    """
    import re
    
    # Remove .parquet extension
    if dataset_str.endswith('.parquet'):
        dataset_str = dataset_str[:-8]
    
    # Split by first /
    parts = dataset_str.split('/', 1)
    dataset_name = parts[0]
    rest = parts[1]
    
    # Parse the rest: {SPLIT}-{MODEL}_maxlen_{MAXLEN}_k_{K}_temp_{TEMP}
    # Use case-insensitive regex to handle any case variation in split names
    pattern = r'^([a-z]+)-(.+?)_maxlen_(\d+)_k_(\d+)_temp_([\d.]+)$'
    match = re.match(pattern, rest, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Invalid dataset string format: {dataset_str}.parquet")
    
    # Normalize split to lowercase
    split = match.group(1).lower()
    model_with_dashes = match.group(2)
    maxlen = int(match.group(3))
    k = int(match.group(4))
    temp = float(match.group(5))
    
    # Convert model name from dashes to HuggingFace format (org/model)
    model_name = extract_model_name(model_with_dashes)
    
    return {
        "dataset_name": dataset_name,
        "split": split,
        "model_name": model_name,
        "maxlen": maxlen,
        "k": k,
        "temp": temp
    }

def extract_model_name(model_with_dashes: str) -> str:
    """
    Convert model string to HuggingFace format: org/model
    
    Handles cases where org names contain dashes (like miromind-ai).
    
    Examples:
        openai-gpt-oss-20b -> openai/gpt-oss-20b
        miromind-ai-MiroThinker-v1.5-235B -> miromind-ai/MiroThinker-v1.5-235B
        Qwen-Qwen2.5-Math-1.5B-Instruct -> Qwen/Qwen2.5-Math-1.5B-Instruct
    """
    import re
    
    # Strategy 1: Find dash followed by uppercase letter (likely start of model name)
    match = re.search(r'-(?=[A-Z])', model_with_dashes)
    if match:
        pos = match.start()
        return model_with_dashes[:pos] + '/' + model_with_dashes[pos+1:]
    
    # Strategy 2: Pattern where org is all lowercase (possibly with dashes)
    match = re.match(r'^([a-z]+(?:-[a-z]+)*)-(.+)$', model_with_dashes)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    
    # Fallback: just replace first dash
    return model_with_dashes.replace('-', '/', 1)

def parse_dataset_from_path(full_path: str) -> dict:
    """
    Parse dataset from a full file path.
    
    Extracts the dataset string (everything from dataset name onwards) and parses it.
    
    Args:
        full_path: Full path like "/VData/linna4335/llms_know_difficult/data/SR_DATA/DigitalLearningGmbH_MATH-lighteval/test-openai-gpt-oss-20b_maxlen_3000_k_1_temp_1.0.parquet"
    
    Returns:
        Dictionary with parsed components (dataset_name, split, model_name, maxlen, k, temp)
    """
    from pathlib import Path
    
    # Convert to Path object
    path = Path(full_path)
    
    # Get the file name with extension
    filename = path.name
    
    # Get the parent directory (dataset name directory)
    dataset_dir = path.parent.name
    
    # Reconstruct the dataset string: dataset_dir/filename
    dataset_str = f"{dataset_dir}/{filename}"
    
    # Use the existing parser
    return parse_dataset_str(dataset_str)
