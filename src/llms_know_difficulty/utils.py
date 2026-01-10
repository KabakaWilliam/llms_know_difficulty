import os
import pandas as pd

from pathlib import Path
from datetime import datetime
from llms_know_difficulty.config import (
    ROOT_DATA_DIR,
    SEED,
    VAL_TRAIN_SPLIT_RATIO,
    PROMPT_COLUMN_NAME,
    LABEL_COLUMN_NAME)

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
                print(f"Loading {split} data from {dataset_path}")
                df = pd.read_parquet(dataset_path)
            elif not os.path.exists(dataset_path) and split == "val":

                print(f"Creating validation split from train split")
                # If a specific dataset doesn't have a validation split make one from the train split:
                if "train" not in outputs:
                    raise ValueError("Train split must be loaded before creating a validation split.")
                
                df = outputs["train"].iloc[:-int(len(outputs["train"]) * VAL_TRAIN_SPLIT_RATIO)]
                df.to_parquet(dataset_path)

            else:
                print(f"Dataset {dataset_name} does not exist, attempting to download...")
                DataIngestionWorkflow.download(dataset_name, model_name, split, max_len, k, temperature)

                # Reload the data if successful
                df = pd.read_parquet(dataset_path)

            # Shuffle the dataframe with the SEED:
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)            
            outputs[split] = df

        # turn it into a tuple of prompts and labels:
        for key, value in outputs.items():
            outputs[key] = (value[PROMPT_COLUMN_NAME].tolist(), value[LABEL_COLUMN_NAME].tolist())

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