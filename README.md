# LLMs Know What's Difficult

This repository contains pipelines for predicting problem learnability and training difficulty-aware probes on language models.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Predicting Learnability Pipeline](#predicting-learnability-pipeline)
- [Difficulty Direction Pipeline](#difficulty-direction-pipeline)

---

## Environment Setup

```bash
conda env create -f difficulty_direction/probe_environment.yml
conda activate diff-direction
```

---

## Predicting Learnability Pipeline

Compute success rates for a model on the MATH dataset by running multiple rollouts per question.

### Quick Start

```bash
cd predicting_learnability
python rollout_success_rate.py
```

### Configuration

Edit the variables at the top of `rollout_success_rate.py`:

```python
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"  # Model to evaluate
NUM_ROLLOUTS_PER_QUESTION = 50         # Number of rollouts per problem
MAX_RESPONSE_LEN = 1024                # Max tokens per response
```

### Output

Results are saved as parquet files in `predicting_learnability/`:

```
predicting_learnability/
├── MATH_{MODEL_ALIAS}-SR_train.parquet   # Success rates for train split
├── MATH_{MODEL_ALIAS}-SR_test.parquet    # Success rates for test split
├── success_rate_hist_{MODEL_ALIAS}_train.png
└── success_rate_hist_{MODEL_ALIAS}_test.png
```

Each parquet contains columns: `prompt`, `correct`, `total`, `success_rate`, `ground_truth`

---

## Difficulty Direction Pipeline

Train probes to predict problem difficulty from model activations.

### Quick Start

```bash
# Train probes on a model
CUDA_VISIBLE_DEVICES=0 python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-1.5B

# With response generation (slower)
CUDA_VISIBLE_DEVICES=0 python3 -m difficulty_direction.run \
    --model_path Qwen/Qwen2.5-Math-1.5B \
    --generate_responses
```

### Command Line Options

```bash
--model_path          # HuggingFace model path (required if no config)
--config_file         # Load from existing config.yaml
--resume_from_step    # Resume from step 0, 1, or 2
--generate_responses  # Generate model completions (optional)
--use_k_fold          # Enable K-Fold cross-validation
--n_folds             # Number of folds (default: 5)
```

### Pipeline Steps

| Step | Description |
|------|-------------|
| 0 | Load datasets (E2H-AMC, E2H-GSM8K), apply chat templates, save datasplits |
| 1 | Extract activations, train probes across all layers/positions |
| 2 | Select best probe, save steering vectors |

### Results Location

All results are saved in `runs/{model_alias}/`:

```
runs/Qwen2.5-Math-1.5B/
├── config.yaml                              # Configuration used
├── datasplits/                              # Formatted datasets
│   ├── E2H-AMC_train.json
│   ├── E2H-AMC_test.json
│   ├── E2H-GSM8K_train.json
│   └── E2H-GSM8K_test.json
└── generate_svs/                            # Probe results
    ├── E2H-AMC/
    │   ├── probe_directions.pt              # All trained probes
    │   ├── probe_performance.json           # Accuracy per layer/position
    │   ├── best_direction.pt                # Best steering vector
    │   └── best_direction_metadata.json     # Best probe info
    ├── E2H-GSM8K/
    └── predicting_learnability/
```

### Key Output Files

| File | Description |
|------|-------------|
| `probe_performance.json` | Test accuracy for each (layer, position) combination |
| `best_direction.pt` | The probe direction with highest accuracy |
| `best_direction_metadata.json` | Layer index, position, and accuracy of best probe |
| `probe_directions.pt` | All probe weights for every layer/position |

### Example: Inspecting Results

```python
import torch
import json

# Load best probe metadata
with open("runs/Qwen2.5-Math-1.5B/generate_svs/E2H-AMC/best_direction_metadata.json") as f:
    meta = json.load(f)
print(f"Best layer: {meta['layer']}, position: {meta['position']}, accuracy: {meta['accuracy']}")

# Load probe performance across all layers
with open("runs/Qwen2.5-Math-1.5B/generate_svs/E2H-AMC/probe_performance.json") as f:
    perf = json.load(f)

# Load the best steering vector
direction = torch.load("runs/Qwen2.5-Math-1.5B/generate_svs/E2H-AMC/best_direction.pt")
```
