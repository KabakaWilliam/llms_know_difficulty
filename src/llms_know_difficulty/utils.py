import os
from pathlib import Path
from datetime import datetime
from config import ROOT_DATA_DIR

def create_results_path(dataset_name: str, model_name: str, probe_name: str, ) -> Path:
    """
    Create a path to load the dataset from the config
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = os.path.join(ROOT_DATA_DIR, "results", model_name, dataset_name, probe_name, timestamp)

    os.path.makedirs(results_path, exist_ok=True)
    return results_path

def save_probe_predictions(probe_preds: dict, results_path: Path):
    """
    Save the probe predictions to the results directory
    """
    pass