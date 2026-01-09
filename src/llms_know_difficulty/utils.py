import os
from pathlib import Path
from datetime import datetime
from .config import ROOT_DATA_DIR

def create_results_path(dataset_name: str, model_name: str, probe_name: str, gen_str: str = None) -> Path:
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
        results_path = os.path.join(ROOT_DATA_DIR, "results", model_name, dataset_name, probe_name, gen_str, timestamp)
    else:
        results_path = os.path.join(ROOT_DATA_DIR, "results", model_name, dataset_name, probe_name, timestamp)

    os.makedirs(results_path, exist_ok=True)
    return results_path

def save_probe_predictions(probe_preds: dict, results_path: Path):
    """
    Save the probe predictions to the results directory
    """
    pass