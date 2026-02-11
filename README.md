# LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations

## Setup

Requires **conda** and a CUDA-capable GPU.

```bash
# Clone and enter the repo
git clone https://github.com/KabakaWilliam/llms_know_difficulty.git
cd llms_know_difficulty

# Create the conda env and install pika in editable mode
bash setup.sh

# Activate
conda activate pika
```

### Generate success-rate datasets

Requires `vllm` (`pip install vllm`):

```bash
cd create_sr_datasets
python create_pareto_sr_datasets.py
```

## PIKA — Probe-Informed K-Aware Routing

PIKA trains lightweight probes on LLM internal representations to predict per-problem difficulty, then uses those predictions to route queries across a pool of models — balancing accuracy against inference cost.

## Repository Structure

```
├── src/pika/                  # Core library (pip-installable)
│   ├── main.py                # CLI: train a probe on a dataset
│   ├── predict_with_probe.py  # CLI: run inference with a trained probe
│   ├── probe/                 # Probe implementations
│   │   ├── probe_factory.py   #   Factory for all probe types
│   │   ├── linear_eoi_probe.py
│   │   ├── tfidf_probe.py
│   │   ├── torch_probe.py     #   Attn, LinearThenMax, etc.
│   │   └── length_probe.py    #   Token-length baseline
│   ├── config.py              # Paths and probe configs
│   ├── metrics.py             # Spearman / AUC evaluation
│   └── utils.py               # Data ingestion helpers
│
├── notebooks/                 # Analysis & routing notebooks
│   ├── router_utils.py        #   Shared routing + plotting utilities
│   ├── utility_router.ipynb   #   λ-sweep utility router (Pareto analysis)
│   ├── cascade.ipynb          #   Threshold-based cascade router
│   └── test_probe_library.ipynb
│
├── create_sr_datasets/        # Success-rate dataset generation (vLLM)
│   ├── create_pareto_sr_datasets.py
│   ├── create_code_pareto_sr_datasets.py
│   └── utils/                 #   Verification & evaluation helpers
│
├── setup.sh                   # One-shot env creation + editable install
├── pika.yml                   # Conda environment spec
├── pyproject.toml             # Package metadata & dependencies
├── label_with_probe.sh        # Label a dataset split with a trained probe
└── run_flexible_probe_sweep.sh # Sweep probes across models/datasets
```


## Supported Probes

| Probe | Description |
|---|---|
| `linear_eoi_probe` | Linear probe on end-of-input hidden states |
| `tfidf_probe` | TF-IDF features → linear model |
| `attn_probe` | Attention-based probe (via `TorchProbe`) |
| `length_probe` | Token-length baseline |
| `linear_then_max_probe` | Linear → max-pool architecture |
| `linear_then_softmax_probe` | Linear → softmax-pool architecture |
| `linear_then_rolling_max_probe` | Linear → rolling-max-pool architecture |

## Usage

### Train a probe

```bash
python src/pika/main.py \
    --probe linear_eoi_probe \
    --dataset DigitalLearningGmbH_MATH-lighteval \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --max_len 3000 --k 50 --temperature 0.7
```

### Label data with a trained probe

```bash
bash label_with_probe.sh
```

### Run a probe sweep across models and datasets

```bash
bash run_flexible_probe_sweep.sh
```

### Routing analysis

Open the notebooks in `notebooks/`:

- **`utility_router.ipynb`** — Probe-based utility router with λ-sweep, Pareto frontier visualisation, and LaTeX table generation.
- **`cascade.ipynb`** — Threshold-based cascade router that escalates to more capable models when probe confidence is low.

Both notebooks import shared functions from `notebooks/router_utils.py`.

## Benchmarks

The routing experiments support three benchmarks:

- [MATH](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) — competition mathematics
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — grade school math
- [AIME](https://huggingface.co/datasets/gneubig/aime-1983-2024) — AMC/AIME problems

## Citation

If you use this code, please cite:

```bibtex

@misc{lugoloobi_llms_2026,
	title = {{LLMs} {Encode} {Their} {Failures}: {Predicting} {Success} from {Pre}-{Generation} {Activations}},
	shorttitle = {{LLMs} {Encode} {Their} {Failures}},
	url = {http://arxiv.org/abs/2602.09924},
	doi = {10.48550/arXiv.2602.09924},
	abstract = {Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70{\textbackslash}\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: https://github.com/KabakaWilliam/llms\_know\_difficulty},
	urldate = {2026-02-11},
	publisher = {arXiv},
	author = {Lugoloobi, William and Foster, Thomas and Bankes, William and Russell, Chris},
	month = feb,
	year = {2026},
	note = {arXiv:2602.09924 [cs]},
	keywords = {Computer Science - Artificial Intelligence, Computer Science - Computation and Language, Computer Science - Machine Learning},
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
