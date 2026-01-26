from pathlib import Path
import os
import pandas as pd


class DataIngestionWorkflow:

    @staticmethod
    def load_dataset(dataset_name: str,
     model_name: str,
     max_len: int,
     k: int,
     temperature: float,
     root_data_dir: Path,
     seed: int = 42,
     val_train_split_ratio: float = 0.2,
     prompt_column_name: str = "formatted_prompt",
     label_column_name: str = "success_rate",
     idx_column_name: str = "idx"):
        """
        1. Check if the dataset exists at the expected directory.

        2. If not, run the download workflow.

        3. Load all splits of the dataset, if no val split create one and return to user.

        TODO: We might need to adapt this if certain datasets don't have a test split...
        TODO: Write an optional flag which processes a directory of downloaded datasets into the correct file structure.

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

            dataset_path = DataIngestionWorkflow.create_dataset_path(
                root_data_dir=root_data_dir,
                dataset_name=dataset_name,
                model_name=model_name,
                split=split,
                max_len=max_len,
                k=k,
                temperature=temperature
            )

            if os.path.exists(dataset_path):
                print(f"Loading {split} data from {dataset_path}")
                df = pd.read_parquet(dataset_path)
            elif not os.path.exists(dataset_path) and split == "val":

                print(f"Creating validation split from train split")
                # If a specific dataset doesn't have a validation split make one from the train split:
                if "train" not in outputs:
                    raise ValueError("Train split must be loaded before creating a validation split.")
                
                df = outputs["train"].iloc[-int(len(outputs["train"]) * val_train_split_ratio):]

                # Save the new train split:
                outputs["train"] = outputs["train"].iloc[:-int(len(outputs["train"]) * val_train_split_ratio)]
                new_train_path = DataIngestionWorkflow.create_dataset_path(dataset_name, model_name, "train", max_len, k, temperature)
                outputs["train"].to_parquet(new_train_path)

                df.to_parquet(dataset_path)

            else:
                print(f"Checked dataset path {dataset_path}")
                print(f"Dataset {dataset_name} does not exist, attempting to download...")
                DataIngestionWorkflow.download(dataset_name, model_name, split, max_len, k, temperature)

                # Reload the data if successful
                df = pd.read_parquet(dataset_path)

            # Shuffle the dataframe with the SEED:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)            
            outputs[split] = df
            
        # turn it into a tuple of prompts and labels:
        for key, value in outputs.items():
            outputs[key] = (value[idx_column_name].tolist(), value[prompt_column_name].tolist(), value[label_column_name].tolist())

        print(f"Dataset sizes:")
        print(f"Train data: {len(outputs['train'][0])}")
        print(f"Val data: {len(outputs['val'][0])}")
        print(f"Test data: {len(outputs['test'][0])}")

        return outputs['train'], outputs['val'], outputs['test']
    
    @staticmethod
    def get_dataset_size(dataset_name: str, model_name: str, max_len: int, k: int, temperature: float, root_data_dir: Path) -> int:
        """
        Get the size of the dataset:
        """
        return len(pd.read_parquet(DataIngestionWorkflow.create_dataset_path(root_data_dir, dataset_name, model_name, "train", max_len, k, temperature)))

    @staticmethod
    def create_dataset_path(
                    root_data_dir: str,
                    dataset_name: str,
                    model_name: str,
                    split:str,
                    max_len: int,
                    k: int,
                    temperature: float
                    ) -> str:
        """
        Create the path to the dataset:
        """

        # Break down the model name into parts:
        name_split = model_name.split("/")
        model_family, specific_model_name = name_split[0], name_split[1]
        file_name = f"{split}_maxlen_{max_len}_k_{k}_temp_{temperature}.parquet"

        return os.path.join(root_data_dir,
                                model_family,
                                specific_model_name,
                                dataset_name,
                                file_name)

    def download(**kwargs):
        raise NotImplementedError("Download functionality not implemented yet")