import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard

# Standalone dataset configuration that can be imported independently
DATASET_CONFIGS = {
    "predicting_learnability": {
        "dataset_type": "local",  # 'local' or 'huggingface'
        "local_path": "/VData/linna4335/llms_know_difficult/predicting_learnability/data",
        "file_pattern": "MATH_{model_alias}-SR_{split}.parquet",  # {split} will be replaced with train/test
        "file_format": "parquet",  # parquet, json, csv, etc.
        "splits": ["train", "test"],
        "prompt_column": "prompt",
        "answer_column": "ground_truth",
        "has_train_split": True,
        "max_n_train": 12000,
        "max_n_test": 500,
        "default_n_train": 10000,
        "default_n_test": 500,
        "difficulty_column": "success_rate"
    },
    "E2H-Lichess": {
        "dataset_type": "huggingface",
        "hf_dataset": "furonghuang-lab/Easy2Hard-Bench",
        "subset_name": "E2H-Lichess",
        "splits": ["train", "eval"],
        "prompt_column": "fen",
        "answer_column": "answer_uci",
        "has_train_split": True,
        "max_n_train": 71800,
        "max_n_test": 5000,
        "default_n_train": 1000,
        "default_n_test": 500,
        "difficulty_column": "rating"
    },
    "E2H-AMC": {
        "dataset_type": "huggingface",
        "hf_dataset": "furonghuang-lab/Easy2Hard-Bench",
        "subset_name": "E2H-AMC", 
        "splits": ["train", "eval"],
        "prompt_column": "problem",
        "answer_column": "answer",
        "has_train_split": True,
        "max_n_train": 1000,
        "max_n_test": 2980,
        "default_n_train": 1000,
        "default_n_test": 600,
        "difficulty_column": "rating"
    },
    "E2H-GSM8K": {
        "dataset_type": "huggingface",
        "hf_dataset": "furonghuang-lab/Easy2Hard-Bench",
        "subset_name": "E2H-GSM8K",
        "splits": ["eval"],  # Only eval split available
        "prompt_column": "question",
        "answer_column": "answer",
        "has_train_split": False,
        "max_n_train": None,
        "max_n_test": 1320,
        "default_n_train": 792,
        "default_n_test": 528,
        "train_split_ratio": 0.6,  # Use 80% for training
        "difficulty_column": "rating"
    },

    "E2H-Codeforces": {
        "dataset_type": "huggingface",
        "hf_dataset": "furonghuang-lab/Easy2Hard-Bench",
        "subset_name": "E2H-Codeforces",
        "splits": ["train", "eval"], 
        "prompt_column": "raw_formatted_prompt",
        "answer_column": "solution_0",
        "has_train_split": True,
        "max_n_train": 3660,
        "max_n_test": 4000,
        "default_n_train": 2400,
        "default_n_test": 1600,
        "train_split_ratio": 0.6,  # Use 80% for training
        "difficulty_column": "rating"
    },

    "AMC_test": {
        "dataset_type": "huggingface",
        "hf_dataset": "furonghuang-lab/Easy2Hard-Bench",
        "subset_name": "E2H-AMC",
        "splits": ["test"],
        "prompt_column": "problem",
        "answer_column": "answer",
        "difficulty_column": "rating"
    },
    "GSM8K_test": {
        "dataset_type": "huggingface",
        "hf_dataset": "furonghuang-lab/Easy2Hard-Bench", 
        "subset_name": "E2H-GSM8K",
        "splits": ["test"], 
        "prompt_column": "question",
        "answer_column": "answer",
        "difficulty_column": "rating"
    }
}

CODEFORCES_TEMPLATE = """"Please generate executable Python 3.10 code that directly solves the problem described below.
The code should be ready to run without any modifications or additional comments. It must
strictly follow Python 3.10 syntax and be formatted correctly for direct execution. Do not
include explanations or comments within the code.
[PROBLEM_MAIN_START]
{problem_main}
[PROBLEM_MAIN_END]
[PROBLEM_NOTE_START]
{problem_note}
[PROBLEM_NOTE_END]
[INPUT_SPEC_START]
{input_spec}
[INPUT_SPEC_END]
[OUTPUT_SPEC_START]
{output_spec}
[OUTPUT_SPEC_END]
[SAMPLE_INPUTS_START]
{sample_inputs}
[SAMPLE_INPUTS_END]
[SAMPLE_OUTPUTS_START]
{sample_outputs}
[SAMPLE_OUTPUTS_END]
1. Please make sure to include correct import statements for any Python packages required by
the solution at the start of the script.
2. When handling input within the code, utilize ‘sys.stdin.readline()’ instead of the ‘input()’
function.
3. The code should begin with "“‘python" and conclude with "“‘". Everything between these
markers must be Python 3.10 code that is ready to execute as is. This code should be directly
savable as a *.py file and fully functional to address the specified problem when run.
"""

LICHESS_TEMPLATE = """You are a chess engine that outputs moves in UCI format.
BOARD ENCODING:
- White pieces: K, Q, R, B, N, P (uppercase)
- Black pieces: k, q, r, b, n, p (lowercase)  
- Empty squares: .
- Ranks: 8 (top) to 1 (bottom)
- Files: a-h (left to right)

MOVE FORMAT (UCI):
You must output moves in UCI format whereby it is in the form <from_square><to_square>. Some examples below:
- a1g1 (from a1 to g1)
- e2e4 (from e2 to e4)
- f8e8 (from f8 to e8)
{BOARD}\nGiven the chess board above, find the best move that immediately checkmates the opponent.The final answer MUST be a single legal move in UCI format.
"""

PROMPT_TEMPLATES = {
    "E2H-AMC": {
        "template": "Please take your time to thoroughly analyze and solve the following math problem step by step. Your approach should be detailed, ensuring that each step of your reasoning is clearly explained to minimize errors and maximize understanding.\n[PROBLEM_START]\n{problem}\n[PROBLEM_END]\n While solving, consider all possible scenarios and subtleties involved in the problem. Each step should build upon the previous one logically, leading to a cohesive solution.\nOnce you arrive at the solution, please present the final answer enclosed in ' '.Ensure the answer is displayed using appropriate LaTeX formatting to maintain mathematical precision and clarity.",
        "prompt_column": "problem"
    },
    "E2H-GSM8K": {
        "template": "Please take your time to thoroughly analyze and solve the following math problem step by step. Your approach should be detailed, ensuring that each step of your reasoning is clearly explained to minimize errors and maximize understanding.\n[PROBLEM_START]\n{problem}\n[PROBLEM_END]\n While solving, consider all possible scenarios and subtleties involved in the problem. Each step should build upon the previous one logically, leading to a cohesive solution.\nOnce you arrive at the solution, please present the final answer enclosed in ' '.Ensure the answer is displayed using appropriate LaTeX formatting to maintain mathematical precision and clarity.",
        "prompt_column": "question"
    },
    "E2H-Codeforces": {
        "template": CODEFORCES_TEMPLATE,
        "prompt_column": DATASET_CONFIGS["E2H-Codeforces"]["prompt_column"]
    },
    "E2H-Lichess":{
        "template": LICHESS_TEMPLATE,
        "prompt_column": DATASET_CONFIGS["E2H-Lichess"]["prompt_column"]
    },
    "predicting_learnability": {
        "template": None,
        "prompt_column": DATASET_CONFIGS["predicting_learnability"]["prompt_column"]
    }
}


@dataclass
class Config(YAMLWizard):
    model_path: str
    model_alias: str
    save_dir: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "runs"))

    # Per subset
    n_train: int = 1000
    n_test: int = 500
    
    # Dataset configuration
    # subset_datasets: List[str] = field(default_factory=lambda: ["E2H-AMC", "E2H-GSM8K", "E2H-Codeforces"])
    # evaluation_datasets: List[str] = field(default_factory=lambda: ["E2H-AMC", "E2H-GSM8K", "E2H-Codeforces"])
    subset_datasets: List[str] = field(default_factory=lambda: ["predicting_learnability"])
    evaluation_datasets: List[str] = field(default_factory=lambda: ["predicting_learnability"])
    # subset_datasets: List[str] = field(default_factory=lambda: ["E2H-Lichess"])
    # evaluation_datasets: List[str] = field(default_factory=lambda: ["E2H-Lichess"])

    # subset_datasets: List[str] = field(default_factory=lambda: ["E2H-Codeforces"])
    # evaluation_datasets: List[str] = field(default_factory=lambda: ["E2H-Codeforces"])
    
    # Dataset split configuration
    # Defines which datasets have train/eval splits and how to handle them
    dataset_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: DATASET_CONFIGS)
    
    # Generation parameters
    max_new_tokens: int = 3000
    # batch_size: int = 32
    batch_size: int = 32
    generation_batch_size: int = 24
    
    # Cross-validation parameters
    use_k_fold: bool = False
    n_folds: int = 5
    cv_seed: int = 42
    do_sample: bool = True

    def artifact_path(self) -> Path:
        return Path(self.save_dir) / self.model_alias
    
    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')

    @classmethod
    def load(cls, filepath: str):
        try:
            return cls.from_yaml_file(filepath)
        except FileNotFoundError:
            return None
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset."""
        return self.dataset_config.get(dataset_name, {})
    
    def needs_manual_train_split(self, dataset_name: str) -> bool:
        """Check if a dataset needs manual train/test split creation."""
        config = self.get_dataset_config(dataset_name)
        return not config.get("has_train_split", True)
    
    def get_train_split_ratio(self, dataset_name: str) -> float:
        """Get the train split ratio for datasets that need manual splitting."""
        config = self.get_dataset_config(dataset_name)
        return config.get("train_split_ratio", 0.8)

# Additional dataset configurations that can be added:
# 
# "E2H-Codeforces": {
#     "splits": ["train", "eval"],
#     "prompt_column": "problem_main",
#     "has_train_split": True
# },
# "E2H-Lichess": {
#     "splits": ["train", "eval"], 
#     "prompt_column": "pgn",
#     "has_train_split": True
# },
# "E2H-ARC": {
#     "splits": ["eval"],
#     "prompt_column": "question",
#     "has_train_split": False,
#     "train_split_ratio": 0.8
# },
# "E2H-Winogrande": {
#     "splits": ["eval"],
#     "prompt_column": "sentence", 
#     "has_train_split": False,
#     "train_split_ratio": 0.8
# }