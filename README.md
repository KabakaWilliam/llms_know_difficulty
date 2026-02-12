# PIKA: Probe-Informed K-Aware Routing

[![arXiv](https://img.shields.io/badge/arXiv-2602.09924-b31b1b.svg)](https://arxiv.org/abs/2602.09924)
[![HuggingFace Probes](https://img.shields.io/badge/🤗%20HuggingFace-Probes-yellow)](https://huggingface.co/CoffeeGitta/pika-probes)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-blue)](https://huggingface.co/datasets/CoffeeGitta/pika-math-generations)

**Predict LLM success before generation and route queries efficiently across model pools.**

PIKA trains lightweight probes on LLM internal representations to predict per-problem difficulty *before generation begins*. These predictions enable intelligent routing across a pool of models, balancing accuracy against inference cost — achieving up to 70% cost reduction on MATH while maintaining or exceeding the accuracy of the best-performing model.

## What This Does

Running LLMs with extended reasoning on every problem is expensive. PIKA solves this by:

1. **Predicting success before generation** — Linear probes on pre-generation activations predict whether a model will succeed on a specific problem
2. **Learning model-specific difficulty** — Models encode their own notion of difficulty (distinct from human difficulty) in their internal representations
3. **Routing intelligently** — Use probe predictions to route queries to appropriate models, reserving expensive models for hard problems

## Quick Start

### Installation

Requires **conda** and a CUDA-capable GPU.

```bash
# Clone repository
git clone https://github.com/KabakaWilliam/llms_know_difficulty.git
cd llms_know_difficulty

# Create conda environment and install PIKA
bash setup.sh

# Activate environment
conda activate pika
```

### Basic Usage

**Train a probe:**
```bash
python src/pika/main.py \
    --probe linear_eoi_probe \
    --dataset DigitalLearningGmbH_MATH-lighteval \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --max_len 3000 --k 50 --temperature 0.7
```

**Run inference with a trained probe:**
```bash
python src/pika/predict_with_probe.py \
    --probe_path path/to/trained_probe.pkl \
    --dataset DigitalLearningGmbH_MATH-lighteval \
    --split test
```

**Label a dataset split:**
```bash
bash label_with_probe.sh
```

**Run probe sweep:**
```bash
bash run_flexible_probe_sweep.sh
```

## Key Features

### Supported Probes

| Probe Type | Description |
|------------|-------------|
| `linear_eoi_probe` | Linear probe on end-of-input hidden states (recommended baseline) |
| `tfidf_probe` | TF-IDF features → linear model |
| `attn_probe` | Attention-based probe architecture |
| `length_probe` | Token-length baseline |
| `linear_then_max_probe` | Linear layer → max-pooling |
| `linear_then_softmax_probe` | Linear layer → softmax-pooling |
| `linear_then_rolling_max_probe` | Linear layer → rolling max-pooling |

### Supported Benchmarks

- **[MATH](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)** — Competition mathematics problems
- **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)** — Grade school math word problems
- **[AIME](https://huggingface.co/datasets/gneubig/aime-1983-2024)** — AMC/AIME competition problems

### Routing Strategies

Two routing approaches are provided in the `notebooks/` directory:

- **Utility Router** (`utility_router.ipynb`) — λ-sweep utility optimization with Pareto frontier visualization and LaTeX table generation
- **Cascade Router** (`cascade.ipynb`) — Threshold-based escalation to more capable models when probe confidence is low

Both notebooks use shared utilities from `notebooks/router_utils.py`.

## Repository Structure

```
├── src/pika/                  # Core library (pip-installable)
│   ├── main.py                # Train a probe on a dataset
│   ├── predict_with_probe.py  # Run inference with trained probe
│   ├── probe/                 # Probe implementations
│   │   ├── probe_factory.py   # Factory for all probe types
│   │   ├── linear_eoi_probe.py
│   │   ├── tfidf_probe.py
│   │   ├── torch_probe.py     # Attention, LinearThenMax, etc.
│   │   └── length_probe.py    # Token-length baseline
│   ├── config.py              # Paths and probe configurations
│   ├── metrics.py             # Spearman correlation & AUC evaluation
│   └── utils.py               # Data ingestion utilities
│
├── notebooks/                 # Analysis & routing experiments
│   ├── router_utils.py        # Shared routing & plotting utilities
│   ├── utility_router.ipynb   # λ-sweep utility router (Pareto analysis)
│   ├── cascade.ipynb          # Threshold-based cascade router
│   └── test_probe_library.ipynb
│
├── create_sr_datasets/        # Success-rate dataset generation
│   ├── create_pareto_sr_datasets.py
│   ├── create_code_pareto_sr_datasets.py
│   └── utils/                 # Verification & evaluation helpers
│
├── setup.sh                   # Environment setup script
├── pika.yml                   # Conda environment specification
├── pyproject.toml             # Package metadata & dependencies
├── label_with_probe.sh        # Dataset labeling script
└── run_flexible_probe_sweep.sh # Probe sweep script
```

## Advanced Usage

### Generate Custom Success-Rate Datasets

Requires `vllm` (install with `pip install vllm`):

```bash
cd create_sr_datasets
python create_pareto_sr_datasets.py
```

This generates datasets with model success rates across different problems, which are used for probe training.

### Routing Analysis

The routing notebooks demonstrate how to:
- Train probes on multiple models
- Analyze probe performance (Spearman correlation, AUC)
- Generate Pareto frontiers for accuracy vs. cost trade-offs
- Export results as LaTeX tables

See `notebooks/utility_router.ipynb` and `notebooks/cascade.ipynb` for complete examples.

## Pre-trained Resources

### Probes
Pre-trained probes are available at [CoffeeGitta/pika-probes](https://huggingface.co/CoffeeGitta/pika-probes):
- Probes for GPT-OSS-20B (high/low/medium difficulty variants)
- Probes for Qwen2.5-Math-7B and other models
- Includes layer and position metadata for easy inference

*Note: We continuously upload new probes as experiments complete.*

### Generation Datasets
Model generations on benchmark tasks at [CoffeeGitta/pika-math-generations](https://huggingface.co/datasets/CoffeeGitta/pika-math-generations):
- MATH dataset generations with multiple model configurations
- Correctness annotations and generation hyperparameters
- Train/validation/test splits

*Note: Additional datasets are being uploaded regularly.*

## Papers

This work is based on two papers:

**[LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](https://arxiv.org/abs/2602.09924)** (February 2026)
- Demonstrates that LLMs encode success likelihood in pre-generation activations
- Shows model-specific difficulty diverges from human difficulty
- Achieves 70% cost reduction on MATH via probe-based routing

**[LLMs Encode How Difficult Problems Are](https://arxiv.org/abs/2510.18147)** (October 2025)
- Initial investigation of difficulty encoding in LLM representations
- Establishes baseline probe architectures and evaluation metrics

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{lugoloobi_llms_2026,
    title = {{LLMs} {Encode} {Their} {Failures}: {Predicting} {Success} from {Pre}-{Generation} {Activations}},
    url = {http://arxiv.org/abs/2602.09924},
    doi = {10.48550/arXiv.2602.09924},
    publisher = {arXiv},
    author = {Lugoloobi, William and Foster, Thomas and Bankes, William and Russell, Chris},
    month = feb,
    year = {2026},
    note = {arXiv:2602.09924 [cs]},
}

@misc{lugoloobi_llms_2025,
    title = {{LLMs} {Encode} {How} {Difficult} {Problems} {Are}},
    url = {http://arxiv.org/abs/2510.18147},
    doi = {10.48550/arXiv.2510.18147},
    publisher = {arXiv},
    author = {Lugoloobi, William and Russell, Chris},
    month = oct,
    year = {2025},
    note = {arXiv:2510.18147 [cs]},
}
```

## License

MIT