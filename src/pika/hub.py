"""
HuggingFace Hub integration for PIKA probes.

Download and load pre-trained probes from HuggingFace Hub.

Usage:
    from pika.hub import download_probe, list_available_probes
    
    # List available probes
    probes = list_available_probes("KabakaWilliam/pika-probes")
    
    # Download and load a probe
    probe = download_probe(
        repo_id="KabakaWilliam/pika-probes",
        probe_name="Qwen2.5-Math-7B-Instruct--MATH--linear-eoi-probe--success-rate--k5-t0.7"
    )
    
    # Use the probe
    predictions = probe.predict(texts=["Solve: 2x + 3 = 7"])
"""

import json
from pathlib import Path
from typing import Optional, Union

# Default repo for PIKA probes
DEFAULT_REPO_ID = "KabakaWilliam/pika-probes"


def download_probe(
    repo_id: str = DEFAULT_REPO_ID,
    probe_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    probe_type: str = "linear_eoi_probe",
    label_type: str = "success_rate",
    cache_dir: Optional[str] = None,
    revision: str = "main",
    device: str = "cuda",
):
    """
    Download a probe from HuggingFace Hub and return a loaded probe instance.
    
    Args:
        repo_id: HuggingFace repo ID (default: KabakaWilliam/pika-probes)
        probe_name: Exact probe folder name (if known). If provided, other filters are ignored.
        model_name: Filter by model name (e.g., "Qwen2.5-Math-7B-Instruct")
        dataset: Filter by dataset name (e.g., "MATH")
        probe_type: Type of probe (default: "linear_eoi_probe")
        label_type: Label type (default: "success_rate")
        cache_dir: Local cache directory (default: HF cache)
        revision: Git revision to use (default: "main")
        device: Device for model loading (default: "cuda")
    
    Returns:
        Loaded probe instance ready for inference
    
    Examples:
        # By exact name
        probe = download_probe(probe_name="Qwen2.5-Math-7B--MATH--linear-eoi-probe--success-rate--k5-t0.7")
        
        # By filters
        probe = download_probe(model_name="Qwen2.5-Math-7B-Instruct", dataset="MATH")
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading probes. "
            "Install with: pip install huggingface_hub"
        )
    
    if probe_name:
        # Download specific probe folder
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{probe_name}/*"],
            cache_dir=cache_dir,
            revision=revision,
        )
        probe_path = Path(local_dir) / probe_name
    else:
        # Need to find matching probe from manifest
        manifest = get_manifest(repo_id, cache_dir=cache_dir, revision=revision)
        
        matching = find_matching_probe(
            manifest,
            model_name=model_name,
            dataset=dataset,
            probe_type=probe_type,
            label_type=label_type,
        )
        
        if not matching:
            raise ValueError(
                f"No probe found matching: model={model_name}, dataset={dataset}, "
                f"probe_type={probe_type}, label_type={label_type}"
            )
        
        probe_name = matching["name"]
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{probe_name}/*"],
            cache_dir=cache_dir,
            revision=revision,
        )
        probe_path = Path(local_dir) / probe_name
    
    # Load the probe
    return load_probe_from_path(probe_path, device=device)


def download_probe_path(
    repo_id: str = DEFAULT_REPO_ID,
    probe_name: str = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> Path:
    """
    Download a probe and return the local path (without loading).
    
    Args:
        repo_id: HuggingFace repo ID
        probe_name: Exact probe folder name
        cache_dir: Local cache directory
        revision: Git revision
    
    Returns:
        Path to downloaded probe folder
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{probe_name}/*"],
        cache_dir=cache_dir,
        revision=revision,
    )
    return Path(local_dir) / probe_name


def get_manifest(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> dict:
    """
    Download and return the manifest.json listing all available probes.
    
    Args:
        repo_id: HuggingFace repo ID
        cache_dir: Local cache directory
        revision: Git revision
    
    Returns:
        Manifest dict with probe listings
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    manifest_path = hf_hub_download(
        repo_id=repo_id,
        filename="manifest.json",
        cache_dir=cache_dir,
        revision=revision,
    )
    
    with open(manifest_path) as f:
        return json.load(f)


def list_available_probes(
    repo_id: str = DEFAULT_REPO_ID,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    probe_type: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """
    List all available probes, optionally filtered.
    
    Args:
        repo_id: HuggingFace repo ID
        model_name: Filter by model name (partial match)
        dataset: Filter by dataset name (partial match)
        probe_type: Filter by probe type (exact match)
        cache_dir: Local cache directory
    
    Returns:
        List of probe metadata dicts
    """
    manifest = get_manifest(repo_id, cache_dir=cache_dir)
    probes = manifest.get("probes", [])
    
    results = []
    for p in probes:
        if model_name and model_name.lower() not in p.get("model", "").lower():
            continue
        if dataset and dataset.lower() not in p.get("dataset", "").lower():
            continue
        if probe_type and p.get("probe_type") != probe_type:
            continue
        results.append(p)
    
    return results


def find_matching_probe(
    manifest: dict,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    probe_type: str = "linear_eoi_probe",
    label_type: str = "success_rate",
) -> Optional[dict]:
    """Find a probe matching the given criteria."""
    probes = manifest.get("probes", [])
    
    for p in probes:
        # Check model name (partial match)
        if model_name:
            if model_name.lower() not in p.get("model", "").lower():
                continue
        
        # Check dataset (partial match)
        if dataset:
            if dataset.lower() not in p.get("dataset", "").lower():
                continue
        
        # Check probe type (exact)
        if p.get("probe_type") != probe_type:
            continue
        
        # Check label type (exact)
        if p.get("label_type") != label_type:
            continue
        
        return p
    
    return None


def load_probe_from_path(probe_path: Union[str, Path], device: str = "cuda"):
    """
    Load a probe from a local path.
    
    Args:
        probe_path: Path to probe folder containing config.json and model files
        device: Device for model loading
    
    Returns:
        Loaded probe instance
    """
    probe_path = Path(probe_path)
    
    # Read config to determine probe type
    config_file = probe_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        probe_type = config.get("probe_type", "linear_eoi_probe")
    else:
        # Infer from files present
        if (probe_path / "best_probe.joblib").exists():
            probe_type = "linear_eoi_probe"
        elif (probe_path / "model.joblib").exists():
            probe_type = "tfidf_probe"
        else:
            raise ValueError(f"Cannot determine probe type for {probe_path}")
    
    # Load the appropriate probe class
    if probe_type == "linear_eoi_probe":
        from pika.probe.linear_eoi_probe import LinearEoiProbe
        return LinearEoiProbe.load_from_checkpoint(probe_path, device=device)
    elif probe_type == "tfidf_probe":
        from pika.probe.tfidf_probe import TfidfProbe
        return TfidfProbe.load_from_checkpoint(probe_path)
    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")
