import os
import pandas as pd

from pathlib import Path
from datetime import datetime
from .config import (
    ROOT_DATA_DIR,
    SEED,
    VAL_TRAIN_SPLIT_RATIO)

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

class DataIngestionWorkflow:

    @staticmethod
    def load_dataset(dataset_name: str, model_name: str, max_len: int, k: int, temperature: float):
        """
        1. Check if the dataset exists at the expected directory.

        2. If not, run the download workflow.

        3. Load all splits of the dataset, if no val split create one and return to user.

        TODO: We might need to adapt this if certain datasets don't have a test split...

        Args:
            dataset_name: Name of the dataset (e.g., "DigitalLearningGmbH_MATH-lighteval")
            model_name: Name of the model (e.g., "gpt2")
            max_len: Maximum length of the response
            k: Number of rollouts per question
            temperature: Temperature for the model

        Returns:
            Tuple with 'train', 'val' and 'test' DataFrames
        """

        outputs = {}
        for split in ["train", "test", "val"]:

            dataset_path = DataIngestionWorkflow.create_dataset_path(dataset_name, model_name, split, max_len, k, temperature)

            if os.path.exists(dataset_path):
                df = pd.read_parquet(dataset_path)
            elif not os.path.exists(dataset_path) and split == "val":
                # If a specific dataset doesn't have a validation split make one from the train split:
                if "train" not in outputs:
                    raise ValueError("Train split must be loaded before creating a validation split.")
                
                df = outputs["train"].iloc[:-int(len(outputs["train"]) * VAL_TRAIN_SPLIT_RATIO)]
                df.to_parquet(dataset_path)

            else:
                DataIngestionWorkflow.download(dataset_name, model_name, split, max_len, k, temperature)

                # Reload the data if successful
                df = pd.read_parquet(dataset_path)

            # Shuffle the dataframe with the SEED:
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            outputs[split] = df

        return outputs['train'], outputs['val'], outputs['test']

    @staticmethod
    def create_dataset_path(dataset_name: str, model_name: str, split:str, max_len: int, k: int, temperature: float):
        """
        Create the path to the dataset:
        """

        # Break down the model name into parts:
        name_split = model_name.split("/")
        model_family, specific_model_name = name_split[0], name_split[1]
        file_name = f"{split}_maxlen_{max_len}_k_{k}_temp_{temperature}.parquet"

        return os.path.join(ROOT_DATA_DIR,
                                model_family,
                                specific_model_name,
                                dataset_name,
                                file_name)


    def download(file_id: str, output_dir: str):
        pass