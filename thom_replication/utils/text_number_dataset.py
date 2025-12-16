from torch.utils.data import Dataset
import torch
from datasets import load_dataset

class TextNumberDataset(Dataset):
    def __init__(
            self, 
            hf_dataset: str,
            hf_dataset_split: str,
            scores_path: str
    ):
        text_dataset = load_dataset(hf_dataset)[hf_dataset_split]
        scores_dataset = load_dataset('parquet', data_files=scores_path)['train']
        assert len(text_dataset) == len(scores_dataset), "Text and scores dataset must have the same length."

        self.items = []
        for text_item, score_item in zip(text_dataset, scores_dataset):
            assert score_item['ground_truth'] in text_item['solution'], "Ground truth not found in solution text."
            self.items.append((text_item['problem'], float(score_item['success_rate'])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, y = self.items[idx]
        return text, torch.tensor(y, dtype=torch.float32)