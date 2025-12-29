import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import time
import gc
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------
# Experiment config (MODULAR)
# ----------------------------
LABELLED_DATASETS_LIST = [
    # "opencompass/AIME2025",
    # "gneubig/aime-1983-2024",
    # "openai/gsm8k",
    "DigitalLearningGmbH/MATH-lighteval",
]

PROBING_DATASET = "DigitalLearningGmbH_MATH-lighteval"
PROBE_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

# Load probe-labelled files from THIS config:
PROBE_K = 1
PROBE_TEMP = 0.0

# Run routed inference with THIS config:
ROUTING_K = 5          # bookkeeping / output naming
ROUTING_TEMP = 0.6

# Baselines: (k,t) used for OG points NOW (you can extend later)
BASELINE_K = 1
BASELINE_T = 0.0


# Self-consistency by routing tier
if ROUTING_TEMP == 0.0:
    SC_POLICY = {"easy": 1, "medium": 1, "hard": 1}
else:
    SC_POLICY = {"easy": 3, "medium": 5, "hard": 1}

# Inference params
MAX_TOKENS = 3000

# Cost semantics
CHARGE_INPUT_PER_SAMPLE = False

# Optional debug (big!)
STORE_ALL_SAMPLES_COL = None  # e.g. "all_samples_json"

batch_size_by_model = {
    "Qwen/Qwen2.5-Math-1.5B-Instruct": 256,
    "Qwen/Qwen2.5-Math-7B-Instruct":  128,
    "Qwen/Qwen2.5-Math-72B-Instruct":  64,
}


# Where OG SR_DATA lives
SR_DATA_BASEDIR = "../will_replication/DATA/SR_DATA"

# Where to save figures
FIG_DIR = "pareto_figures"