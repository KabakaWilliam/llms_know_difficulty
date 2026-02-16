import json
import numpy as np
import pandas as pd
import sys
import base64
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from math_verify import parse, verify
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import Counter
from tqdm import tqdm

from .verification_math import try_extract_solution, compute_score

def encode_str(prob_str):
    return base64.b64encode(prob_str.encode()).decode()

def decode_str(encoded_prob_str):
    return base64.b64decode(encoded_prob_str).decode()

def _make_hashable(obj):
    """Convert unhashable types (like SymPy matrices) to hashable equivalents."""
    try:
        # Check if it's a SymPy matrix
        if hasattr(obj, '__class__') and 'Matrix' in obj.__class__.__name__:
            # Convert matrix to tuple of tuples
            return tuple(tuple(row) for row in obj.tolist())
        # If it's a list, convert to tuple recursively
        elif isinstance(obj, list):
            return tuple(_make_hashable(item) for item in obj)
        # Otherwise return as-is
        return obj
    except Exception:
        return obj

def parse_answers(responses):
    PARSED_ANSWER_LIST = []
    for response in responses:
        try:
            parsed_anwer = parse(response)
            # Make the parsed answer hashable if needed
            parsed_anwer = _make_hashable(parsed_anwer)
            PARSED_ANSWER_LIST.append(parsed_anwer)
        except Exception:
            # Append empty list instead of unparsed response to avoid verification bottleneck
            PARSED_ANSWER_LIST.append([])
    return PARSED_ANSWER_LIST


def verify_answer(ground_truth, response):
    """
    Verify if a response matches the ground truth.
    
    Args:
        ground_truth: The correct answer (will be parsed with $ delimiters)
        response: The model's response to verify
        
    Returns:
        1 if correct, 0 if incorrect
    """
    try:
        parsed_ans = parse(response)
        # Make the parsed answer hashable if needed
        parsed_ans = _make_hashable(parsed_ans)
    except Exception:
        parsed_ans = response
    
    try:
        parsed_gt = parse(f"${ground_truth}$")
        # Make the ground truth hashable if needed
        parsed_gt = _make_hashable(parsed_gt)
        
        if verify(parsed_gt, parsed_ans):
            return 1
        else:
            return 0
    except Exception:
        return 0

def evaluate_responses(responses, ground_truths):
    """
    Evaluate a list of responses against ground truths.
    
    Args:
        responses: List of model responses
        ground_truths: List of correct answers
        
    Returns:
        Tuple of (is_correct_list, accuracy_percentage, num_correct, total)
    """
    is_correct = [verify_answer(gt, resp) for gt, resp in zip(ground_truths, responses)]
    accuracy = sum(is_correct) / len(is_correct) * 100 if is_correct else 0.0
    return is_correct, accuracy, sum(is_correct), len(is_correct)

def load_probe_data(MODEL_NAME, PROBING_DATASET="MATH", K=1, TEMPERATURE=0.0, DATA_PATH="../probe_results/DATA/SR_DATA", MAXLEN=3000, IS_MAJORITY_VOTE=False):
    MODEL_ALIAS= "-".join(MODEL_NAME.split("/"))
    if IS_MAJORITY_VOTE == False:
        GEN_STR=f"maxlen_{MAXLEN}_k_{K}_temp_{TEMPERATURE}"
    else:
        GEN_STR=f"maxlen_{MAXLEN}_k_{K}_temp_{TEMPERATURE}_labelcol_majority_vote_is_correct"
    PROBE_PATH = f"{DATA_PATH}/{PROBING_DATASET}/{MODEL_ALIAS}_{GEN_STR}/best_probe_predictions.json"

    with open(PROBE_PATH, "r") as f:
        probe_data = json.load(f)
    return probe_data


def sigmoid_np(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))

def load_labelled_probe_dataset(MODEL_NAME, PROBE_SOURCE_DATASET="MATH", LABELLED_DATASET="openai_gsm8k", K=1, TEMPERATURE=0.0, DATA_PATH="../probe_results/DATA/Labelled_SR"):
    MODEL_ALIAS= "-".join(MODEL_NAME.split("/"))
    GEN_STR = f"maxlen_3000_k_{K}_temp_{TEMPERATURE}"
    LABELLED_PROBE_DATASET_PATH = f"{DATA_PATH}/{PROBE_SOURCE_DATASET}_probe/{LABELLED_DATASET}/{MODEL_ALIAS}_{GEN_STR}/scored.parquet"

    print(f"... loading labelled dataset at: {LABELLED_PROBE_DATASET_PATH}")

    return pd.read_parquet(LABELLED_PROBE_DATASET_PATH)

prompt_sfx = "Let's think step by step and output the final answer within \\boxed{}."


@dataclass
class ModeSettings:
    prompt_sfx: str
    default_temperature: float
    default_max_tokens: int

@dataclass
class ModelCosts:
    input_per_mill: float
    output_per_mill: float

@dataclass
class ModelConfig:
    model_base: str
    api_key: str
    default_temperature: float
    default_max_tokens: int
    mode_settings: Dict[str, ModeSettings]
    model_costs: ModelCosts

TOKENS_PER_MILLION = 1_000_000


SIMPLE_MODEL_POOL_CONFIG = {
    "mistralai/Ministral-3-3B-Instruct-2512": {
        "model_base": "http://localhost:8001/v1",
        "api_key": "token-abc1235",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: <4B params = $0.10 / 1M tokens, cached input $0.05 / 1M
        "model_costs": {
            "input_per_mill": 0.10,
            "cached_input_per_mill": 0.05,
            "output_per_mill": 0.10,
        }
    },
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": {
        "model_base": "http://localhost:8001/v1",
        "api_key": "token-abc1235",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: <4B params = $0.10 / 1M tokens, cached input $0.05 / 1M
        "model_costs": {
            "input_per_mill": 0.10,
            "cached_input_per_mill": 0.05,
            "output_per_mill": 0.10,
        }
    },
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {
        "model_base": "http://localhost:8001/v1",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: <4B params = $0.10 / 1M tokens, cached input $0.05 / 1M
        "model_costs": {
            "input_per_mill": 0.10,
            "cached_input_per_mill": 0.05,
            "output_per_mill": 0.10,
        }
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "model_base": "http://localhost:8001/v1",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: <4B params = $0.10 / 1M tokens, cached input $0.05 / 1M
        "model_costs": {
            "input_per_mill": 0.10,
            "cached_input_per_mill": 0.05,
            "output_per_mill": 0.10,
        }
    },

    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.20,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.20,
        }
    },
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.20,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.20,
        }
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.90,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.90,
        }
    },
    "Qwen/Qwen2.5-Math-1.5B-Instruct": {
        "model_base": "http://localhost:8001/v1",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: <4B params = $0.10 / 1M tokens, cached input $0.05 / 1M
        "model_costs": {
            "input_per_mill": 0.10,
            "cached_input_per_mill": 0.05,
            "output_per_mill": 0.10,
        }
    },

    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.20,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.20,
        }
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "model_base": "http://localhost:8001/v1",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: <4B params = $0.10 / 1M tokens, cached input $0.05 / 1M
        "model_costs": {
            "input_per_mill": 0.10,
            "cached_input_per_mill": 0.05,
            "output_per_mill": 0.10,
        }
    },

    "Qwen/Qwen2.5-7B-Instruct": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.20,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.20,
        }
    },
    "Qwen/Qwen3-8B": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 32768,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 32768,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.20,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.20,
        }
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "model_base": "http://localhost:8001/v2",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: 4B–16B params = $0.20 / 1M tokens, cached input $0.10 / 1M
        "model_costs": {
            "input_per_mill": 0.20,
            "cached_input_per_mill": 0.10,
            "output_per_mill": 0.20,
        }
    },

    "Qwen/Qwen2.5-Math-72B-Instruct": {
        "model_base": "http://localhost:8001/v3",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks serverless tier: >16B params = $0.90 / 1M tokens, cached input $0.45 / 1M
        "model_costs": {
            "input_per_mill": 0.90,
            "cached_input_per_mill": 0.45,
            "output_per_mill": 0.90,
        }
    },
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
        "model_base": "http://localhost:8001/v3",
        "api_key": "token-abc123",
        "default_temperature": 0.7,
        "default_max_tokens": 65536,
        "mode_settings": {
            "Code": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.7,
                "default_max_tokens": 65536,
                "top_p":0.8,
                 "top_k":20,
                 "repetition_penalty":1.05

            }
        },
        # Fireworks serverless tier: >16B params = $0.90 / 1M tokens, cached input $0.45 / 1M
        "model_costs": {
            "input_per_mill": 0.15,
            "output_per_mill": 0.60,
        }
    },

    "openai/gpt-oss-20b": {
        "model_base": "http://localhost:8001/v4",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks explicit pricing for gpt-oss-20b
        "model_costs": {
            "input_per_mill": 0.07,
            "cached_input_per_mill": 0.04,
            "output_per_mill": 0.30,
        }
    },

    "openai/gpt-oss-120b": {
        "model_base": "http://localhost:8001/v4",
        "api_key": "token-abc123",
        "default_temperature": 0.6,
        "default_max_tokens": 3000,
        "mode_settings": {
            "MATH": {
                "prompt_sfx": prompt_sfx,
                "default_temperature": 0.6,
                "default_max_tokens": 3000,
            }
        },
        # Fireworks explicit pricing for gpt-oss-120b
        "model_costs": {
            "input_per_mill": 0.15,
            "cached_input_per_mill": 0.07,
            "output_per_mill": 0.60,
        }
    },
}

def run_greedy_baseline(llm, prompts, gts, MAX_TOKENS=3000):
    """Get baseline greedy performance on all questions"""
    from vllm import SamplingParams
    
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(prompts, params)
    
    responses = []
    token_lengths = []
    is_correct = []
    
    for output, gt in zip(outputs, gts):
        response = output.outputs[0].text
        token_length = len(output.outputs[0].token_ids)
        
        responses.append(response)
        token_lengths.append(token_length)
        
        # Verify answer
        parsed_answers = parse_answers([response])
        if parsed_answers and len(parsed_answers[0]) > 0:
            final_answer = parsed_answers[0][0]
        else:
            final_answer = ""
        
        correct = verify(parse(f"${gt}$"), final_answer)
        is_correct.append(int(correct))
    
    accuracy = np.mean(is_correct)
    print(f"Greedy Baseline Accuracy: {accuracy:.2%} ({sum(is_correct)}/{len(is_correct)})")
    
    return responses, token_lengths, is_correct


# VLLM Specific Utils

def check_gpu_memory():
    """Check available GPU memory and warn if low"""
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_mem = float(result.stdout.strip().split('\n')[0])  # First GPU
            free_mem_gb = free_mem / 1024
            print(f"💾 GPU Memory Available: {free_mem_gb:.2f} GB")
            return free_mem_gb
    except:
        pass
    return None

# ----------------------------
# vLLM helpers
# ----------------------------
@dataclass
class VLLMModelRunCfg:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: Optional[int] = 64
    max_num_batched_tokens: Optional[int] = 8192

def unload_model(llm) -> None:
    import gc
    import torch
    try:
        if hasattr(llm, "llm_engine"):
            del llm.llm_engine
    except Exception:
        pass
    del llm
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def batch_apply_chat_template(problems, tokenizer):
    prompt_store = []
    for problem in problems:
        messages = [{"role": "user", "content": problem}]
        prompt_store.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompt_store


def count_input_tokens_batch(prompts: List[str], tokenizer) -> List[int]:
    enc = tokenizer(prompts, add_special_tokens=False)
    return [len(ids) for ids in enc["input_ids"]]

def _normalize_answer_for_vote(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    ans = str(ans).strip()
    return ans if ans else None

def majority_vote_from_samples(
    sample_texts: List[str],
    *,
    extract_answer_fn: Callable[[str], Optional[str]],
) -> Tuple[str, Optional[str], int]:
    answers = [_normalize_answer_for_vote(extract_answer_fn(t)) for t in sample_texts]

    if any(a is not None for a in answers):
        keys = [a if a is not None else "__NO_ANSWER__" for a in answers]
        winner_key, _ = Counter(keys).most_common(1)[0]
        winner_idx = next(i for i, k in enumerate(keys) if k == winner_key)
        chosen_text = sample_texts[winner_idx]
        chosen_answer = None if winner_key == "__NO_ANSWER__" else winner_key
        return chosen_text, chosen_answer, winner_idx

    norm_texts = [t.strip() for t in sample_texts]
    winner_text, _ = Counter(norm_texts).most_common(1)[0]
    winner_idx = next(i for i, nt in enumerate(norm_texts) if nt == winner_text)
    return sample_texts[winner_idx], None, winner_idx

def get_output_cost(solutions):
    total_output_cost = 0
    for sol in solutions:
        total_output_cost += sol["output_cost_usd"]
    return total_output_cost

def get_output_tokens(solutions):
    total_output_tokens = 0
    for sol in solutions:
        total_output_tokens += sol["output_tokens"]
    return total_output_tokens

def get_output_text(solutions):
    all_responses = []
    for sol in solutions:
        all_responses.append(sol["text"])
    return all_responses

def add_majority_vote_answer(solutions):
    response_list = get_output_text(solutions)
    return majority_vote_from_samples(response_list, extract_answer_fn=try_extract_solution)[1]

def pareto_efficient_min_cost_max_score(costs, scores, eps=0):
    costs = np.asarray(costs, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(costs)
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_eff[i]:
            continue
        dominated = (costs <= costs[i] + eps) & (scores >= scores[i] - eps) & (
            (costs < costs[i] - eps) | (scores > scores[i] + eps)
        )
        if np.any(dominated):
            is_eff[i] = False
    return is_eff


def plot_pareto_frontier(points, title, score_col="score", x_label="Total eval cost (USD)", y_label="Pass@1"):
    """
    points: list of dicts with keys:
      - name: str  (legend entry; e.g. model alias or "ROUTER")
      - cost: float
      - score: float
      - kind: str  ("model" or "router")
    """
    names = [p["name"] for p in points]
    costs = np.array([p["cost"] for p in points], dtype=float)
    scores = np.array([p[score_col] for p in points], dtype=float)

    mask = pareto_efficient_min_cost_max_score(costs, scores)
    pareto_pts = sorted([p for p, m in zip(points, mask) if m], key=lambda d: d["cost"])

    uniq_names = list(dict.fromkeys(names))  # stable order
    cmap = plt.get_cmap("tab20")
    color_map = {n: cmap(i % cmap.N) for i, n in enumerate(uniq_names)}

    fig, ax = plt.subplots(figsize=(9, 5))

    for n in uniq_names:
        pts_n = [p for p in points if p["name"] == n]
        kind = pts_n[0].get("kind", "model")
        marker = "*" if kind == "router" else "o"
        size = 180 if kind == "router" else 70
        edge = "black" if kind == "router" else None

        ax.scatter(
            [p["cost"] for p in pts_n],
            [p[score_col] for p in pts_n],
            label=n,
            marker=marker,
            s=size,
            c=[color_map[n]],
            edgecolors=edge,
            linewidths=1.0 if kind == "router" else 0.0,
            alpha=0.9,
            zorder=3 if kind == "router" else 2,
        )

    # Pareto frontier line
    if len(pareto_pts) >= 2:
        ax.plot([p["cost"] for p in pareto_pts], [p[score_col] for p in pareto_pts], linewidth=1.5, zorder=1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    plt.show()

# CALIBRATION CODE
def sr_to_counts(sr, n_trials=50):
    sr = np.asarray(sr).reshape(-1)
    k = np.rint(sr * n_trials).astype(int)
    k = np.clip(k, 0, n_trials)
    n = np.full_like(k, n_trials, dtype=int)
    return k, n

def fit_platt_binomial(
    logits_cal,
    sr50_cal,
    n_trials=50,
    max_iter=300,
    device="cpu",
):
    import torch
    import torch.nn.functional as F
    """
    Fit Platt scaling parameters (a, b) for:
        p = sigmoid(a * logit + b)
    minimizing binomial NLL using SR@50 -> (k successes out of n trials).
    """
    k_np, n_np = sr_to_counts(sr50_cal, n_trials=n_trials)

    z = torch.as_tensor(np.asarray(logits_cal).reshape(-1), dtype=torch.float32, device=device)
    k = torch.as_tensor(k_np, dtype=torch.float32, device=device)
    n = torch.as_tensor(n_np, dtype=torch.float32, device=device)

    # Initialize: a=1, b=0 (identity-ish)
    a = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = a * z + b
        # Binomial NLL per example: -[k*logsigmoid(s) + (n-k)*logsigmoid(-s)]
        loss = -(k * F.logsigmoid(s) + (n - k) * F.logsigmoid(-s)).mean()
        loss.backward()
        return loss

    opt.step(closure)

    a_hat = float(a.detach().cpu().item())
    b_hat = float(b.detach().cpu().item())
    return a_hat, b_hat

def apply_platt(logits, a, b):
    logits = np.asarray(logits)
    return sigmoid_np(a * logits + b)

def compute_ece_soft(probs, labels, n_bins=10):
    """
    ECE for probabilistic (soft) labels in [0,1].
    probs: predicted probabilities in [0,1], shape [N]
    labels: target probabilities in [0,1], shape [N]
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"Shape mismatch: probs {probs.shape}, labels {labels.shape}")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probs must be in [0,1]. Apply sigmoid if you have logits.")
    if np.any(labels < 0) or np.any(labels > 1):
        raise ValueError("labels must be in [0,1] for soft-label ECE.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        # include 1.0 in last bin
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)

        if np.any(mask):
            avg_conf = probs[mask].mean()
            avg_true = labels[mask].mean()   # mean target probability
            bin_frac = mask.mean()
            ece += np.abs(avg_conf - avg_true) * bin_frac

    return float(ece)


def reliability_diagram_soft(probs, labels, n_bins=10, title="Reliability diagram (soft labels)", show_hist=True):
    """
    Reliability diagram for soft labels:
      x = mean predicted probability per bin
      y = mean target probability per bin
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_pred = np.full(n_bins, np.nan)
    mean_true = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)

        counts[i] = int(mask.sum())
        if counts[i] > 0:
            mean_pred[i] = probs[mask].mean()
            mean_true[i] = labels[mask].mean()

    if show_hist:
        fig, (ax, axh) = plt.subplots(
            2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        axh = None

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    valid = ~np.isnan(mean_true)
    ax.plot(mean_pred[valid], mean_true[valid], marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Mean target probability")
    ax.set_title(title)

    if axh is not None:
        axh.bar(centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9)
        axh.set_ylabel("Count")
        axh.set_xlabel("Predicted probability bin")

    plt.tight_layout()
    return fig

def get_json_responses(json_sols):
    json_sols = json.loads(json_sols)
    sols_list = []
    for sol in json_sols:
        
        sols_list.append(sol["text"])
    return sols_list

def extract_answers_from_json_solutions(gen_sols_obj):
    return [try_extract_solution(sol) for sol in get_json_responses(gen_sols_obj)]

def compute_passk_from_extracted_answers(extracted_answer_list, ground_truth):
    total_passk = 0
    for extracted_ans in extracted_answer_list:
        total_passk += compute_score(solution_str=f"\\boxed{{{extracted_ans}}}", ground_truth=ground_truth)
    return float(total_passk/len(extracted_answer_list))

def compute_passk_from_json_solutions(generated_sols_obj, ground_truth):
    extracted_answer_list = extract_answers_from_json_solutions(generated_sols_obj)
    return compute_passk_from_extracted_answers(extracted_answer_list, ground_truth)

def run_routed_vllm_inference(
    df: pd.DataFrame,
    *,
    route_col: str,
    prompt_col: str,
    out_text_col: str = "routed_response_text",
    out_model_col: str = "response_model",
    prompt_text_col: str = "prompt_text",
    input_num_tokens_col: str = "input_num_tokens",
    out_tok_col: str = "response_num_tokens",          # TOTAL output tokens across all samples
    out_latency_col: str = "response_latency_s",
    out_err_col: str = "response_error",
    input_cost_col: str = "input_cost_usd",
    output_cost_col: str = "output_cost_usd",
    total_cost_col: str = "total_cost_usd",
    gt_col: str = "original_solution",
    pricing_config: Optional[dict] = None,
    max_tokens: int = 3000,
    batch_size_by_model: Optional[dict] = None,
    checkpoint_path: Optional[str] = None,
    model_run_cfgs: Optional[Dict[str, VLLMModelRunCfg]] = None,
    n_col: str = "sc_n",
    temperature_col: str = "sc_temp",
    majority_vote: bool = True,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    store_all_samples_col: Optional[str] = None,
    charge_input_per_sample: bool = False,
    cascade_tiers_col: str = "cascade_tiers",
    target_confidence: float = 0.90,
    allow_escalation: bool = True,
) -> pd.DataFrame:
    import time
    from vllm import LLM, SamplingParams

    if model_run_cfgs is None:
        model_run_cfgs = {}
    if pricing_config is None:
        pricing_config = {}
    if majority_vote and extract_answer_fn is None:
        raise ValueError("majority_vote=True requires extract_answer_fn")
    if batch_size_by_model is None:
        batch_size_by_model = {}

    for col, default in [
        (prompt_text_col, None),
        (out_text_col, None),
        (out_model_col, None),
        (input_num_tokens_col, np.nan),
        (out_tok_col, np.nan),
        (out_latency_col, np.nan),
        (out_err_col, None),
        (input_cost_col, np.nan),
        (output_cost_col, np.nan),
        (total_cost_col, np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    if store_all_samples_col is not None and store_all_samples_col not in df.columns:
        df[store_all_samples_col] = None
    
    # Initialize cascade escalation output columns
    for col, default in [
        ("tiers_executed", None),
        ("escalation_path", None),
        ("final_confidence", np.nan),
        ("tier_confidences", None),
        ("cost_per_tier", None),
    ]:
        if col not in df.columns:
            df[col] = default

    pending_mask = df[out_text_col].isna()
    if pending_mask.sum() == 0:
        print("✅ Nothing to do: all rows already have responses.")
        return df

    # Detect cascade vs non-cascade routes
    cascade_mask = pending_mask & (df[cascade_tiers_col].notna())
    non_cascade_mask = pending_mask & (df[cascade_tiers_col].isna())
    
    if cascade_mask.any():
        print(f"📊 Routes: {cascade_mask.sum()} cascade, {non_cascade_mask.sum()} non-cascade")
    
    # Process non-cascade routes first (using existing logic unchanged)
    if non_cascade_mask.any():
        print("\n=== PROCESSING NON-CASCADE ROUTES ===")
        routes = df.loc[non_cascade_mask, route_col].dropna().unique().tolist()
        print(f"Routes to run: {routes}")

        for model_name in routes:
            default_bs = 256
            if "72" in model_name:
                default_bs = 32
            batch_size = batch_size_by_model.get(model_name, default_bs)

            model_mask = non_cascade_mask & (df[route_col] == model_name)
            idxs = df.index[model_mask].tolist()
            if not idxs:
                continue

            cfg = model_run_cfgs.get(model_name, VLLMModelRunCfg())
            print(f"\n=== Running model: {model_name} | rows: {len(idxs)} | batch_size={batch_size} ===")
            print(f"vLLM cfg: {cfg}")

            model_costs = pricing_config.get(model_name, {}).get("model_costs", {})
            in_rate = model_costs.get("input_per_mill", None)
            out_rate = model_costs.get("output_per_mill", None)
            has_pricing = (in_rate is not None) and (out_rate is not None)

            # Build LLM kwargs from config
            llm_kwargs = {
                "model": model_name,
                "tensor_parallel_size": cfg.tensor_parallel_size,
                "gpu_memory_utilization": cfg.gpu_memory_utilization,
                "max_model_len": cfg.max_model_len,
            }
            
            # Add optional parameters if specified
            if cfg.max_num_seqs is not None:
                llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
            if cfg.max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
            
            llm = LLM(**llm_kwargs)

            tokenizer = llm.llm_engine.tokenizer.tokenizer

            sub = df.loc[idxs, [n_col, temperature_col]].copy()
            sub[n_col] = sub[n_col].astype(int)
            sub[temperature_col] = sub[temperature_col].astype(float)

            for (n_val, temp_val), sub_idx_df in sub.groupby([n_col, temperature_col]):
                group_idxs = sub_idx_df.index.tolist()
                sampling = SamplingParams(
                    temperature=float(temp_val),
                    max_tokens=max_tokens,
                    n=int(n_val),
                )

                print(f"  -> group n={n_val} temp={temp_val} rows={len(group_idxs)}")

                for start in tqdm(range(0, len(group_idxs), batch_size), desc=f"{model_name} n={n_val}"):
                    batch_idxs = group_idxs[start : start + batch_size]
                    problems = df.loc[batch_idxs, prompt_col].tolist()

                    prompts = batch_apply_chat_template(problems, tokenizer)
                    input_tok_counts = count_input_tokens_batch(prompts, tokenizer)
                    df.loc[batch_idxs, prompt_text_col] = prompts

                    t0 = time.time()
                    try:
                        outputs = llm.generate(prompts, sampling_params=sampling)
                        latency = time.time() - t0

                        chosen_texts: List[str] = []
                        total_out_tok_counts: List[int] = []
                        all_samples_json: List[str] = []
                        errs = [None] * len(outputs)

                        for req_out in outputs:
                            sample_texts = [o.text for o in req_out.outputs]
                            detailed_sample_data = [
                                {
                                    "text": o.text,
                                    "output_tokens": int(len(o.token_ids or [])),
                                    "output_cost_usd": (
                                        (len(o.token_ids or []) / TOKENS_PER_MILLION) * float(out_rate)
                                        if out_rate is not None
                                        else None
                                    ),
                                }
                                for o in req_out.outputs
                            ]

                            sample_tok_counts = [
                                (len(o.token_ids) if o.token_ids is not None else 0) for o in req_out.outputs
                            ]
                            total_out_tok_counts.append(int(np.sum(sample_tok_counts)))

                            if store_all_samples_col is not None:
                                all_samples_json.append(json.dumps(detailed_sample_data, ensure_ascii=False))

                            if majority_vote and n_val > 1:
                                chosen_text, _, _ = majority_vote_from_samples(
                                    sample_texts, extract_answer_fn=extract_answer_fn
                                )
                                chosen_texts.append(chosen_text)
                            else:
                                chosen_texts.append(sample_texts[0])

                        df.loc[batch_idxs, out_text_col] = chosen_texts
                        df.loc[batch_idxs, out_model_col] = model_name
                        df.loc[batch_idxs, input_num_tokens_col] = input_tok_counts
                        df.loc[batch_idxs, out_tok_col] = total_out_tok_counts
                        df.loc[batch_idxs, out_latency_col] = latency
                        df.loc[batch_idxs, out_err_col] = errs
                        if store_all_samples_col is not None:
                            df.loc[batch_idxs, store_all_samples_col] = all_samples_json

                        if has_pricing:
                            in_arr = np.array(input_tok_counts, dtype=float)
                            out_arr = np.array(total_out_tok_counts, dtype=float)
                            if charge_input_per_sample and n_val > 1:
                                in_arr = in_arr * float(n_val)

                            input_costs = (in_arr / TOKENS_PER_MILLION) * float(in_rate)
                            output_costs = (out_arr / TOKENS_PER_MILLION) * float(out_rate)
                            df.loc[batch_idxs, input_cost_col] = input_costs
                            df.loc[batch_idxs, output_cost_col] = output_costs
                            df.loc[batch_idxs, total_cost_col] = input_costs + output_costs

                    except Exception as e:
                        latency = time.time() - t0
                        df.loc[batch_idxs, out_text_col] = None
                        df.loc[batch_idxs, out_model_col] = model_name
                        df.loc[batch_idxs, input_num_tokens_col] = np.nan
                        df.loc[batch_idxs, out_tok_col] = np.nan
                        df.loc[batch_idxs, out_latency_col] = latency
                        df.loc[batch_idxs, out_err_col] = repr(e)
                        df.loc[batch_idxs, input_cost_col] = np.nan
                        df.loc[batch_idxs, output_cost_col] = np.nan
                        df.loc[batch_idxs, total_cost_col] = np.nan
                        if store_all_samples_col is not None:
                            df.loc[batch_idxs, store_all_samples_col] = None

                    if checkpoint_path is not None:
                        df.to_parquet(checkpoint_path, index=True)

            unload_model(llm)
            pending_mask = df[out_text_col].isna()
            if checkpoint_path is not None:
                df.to_parquet(checkpoint_path, index=True)

    # Process cascade routes with escalation logic (BATCHED: tier-first approach)
    if cascade_mask.any() and allow_escalation:
        print("\n=== PROCESSING CASCADE ROUTES WITH ESCALATION (BATCHED) ===")
        
        tokenizer_cache = {}
        total_cascade_samples = cascade_mask.sum()
        print(f"Processing {total_cascade_samples} cascade samples with tier-wise batching")
        
        # Initialize per-sample tracking dictionaries
        accumulated_samples_per_idx = {idx: [] for idx in df[cascade_mask].index}
        accumulated_json_per_idx = {idx: [] for idx in df[cascade_mask].index}
        tier_confidences_per_idx = {idx: {} for idx in df[cascade_mask].index}
        cost_per_tier_per_idx = {idx: {} for idx in df[cascade_mask].index}
        total_cost_per_idx = {idx: 0.0 for idx in df[cascade_mask].index}
        escalation_path_per_idx = {idx: [] for idx in df[cascade_mask].index}
        tiers_executed_per_idx = {idx: [] for idx in df[cascade_mask].index}
        input_costs_by_model_per_idx = {idx: {} for idx in df[cascade_mask].index}
        
        # Extract all unique cascade tier configurations
        all_tier_configs = []
        for idx in df[cascade_mask].index:
            cascade_tiers = df.loc[idx, cascade_tiers_col]
            if cascade_tiers:
                all_tier_configs.extend(cascade_tiers)
        
        # Get unique tier configs (model, k, temp tuples)
        unique_tiers = {}  # model_name -> [(k, temp), ...]
        for model_name, k_val, temp_val in all_tier_configs:
            if model_name not in unique_tiers:
                unique_tiers[model_name] = []
            if (k_val, temp_val) not in unique_tiers[model_name]:
                unique_tiers[model_name].append((k_val, temp_val))
        
        # Track which samples still need processing
        pending_indices = set(df[cascade_mask].index)
        
        # TIER-FIRST LOOP: Process one tier at a time, batching all samples for that tier
        tier_global_idx = 0
        while pending_indices:
            # Collect all unique tier configs from pending samples
            tier_configs_needed = {}  # (model, k, temp) -> list of indices that need it
            
            for idx in pending_indices:
                cascade_tiers = df.loc[idx, cascade_tiers_col]
                if cascade_tiers:
                    # Find next tier for this sample (based on tiers_executed count)
                    next_tier_pos = len(tiers_executed_per_idx[idx])
                    if next_tier_pos < len(cascade_tiers):
                        model_name, k_val, temp_val = cascade_tiers[next_tier_pos]
                        tier_key = (model_name, k_val, temp_val)
                        if tier_key not in tier_configs_needed:
                            tier_configs_needed[tier_key] = []
                        tier_configs_needed[tier_key].append(idx)
            
            if not tier_configs_needed:
                break
            
            # Process each tier configuration
            for (model_name, k_val, temp_val), indices_for_tier in tier_configs_needed.items():
                print(f"\n  Tier {tier_global_idx}: Processing {len(indices_for_tier)} samples with {model_name.split('/')[-1]} (k={k_val}, t={temp_val})")
                
                # Get or create tokenizer
                if model_name not in tokenizer_cache:
                    cfg = model_run_cfgs.get(model_name, VLLMModelRunCfg())
                    llm_kwargs = {
                        "model": model_name,
                        "tensor_parallel_size": cfg.tensor_parallel_size,
                        "gpu_memory_utilization": cfg.gpu_memory_utilization,
                        "max_model_len": cfg.max_model_len,
                    }
                    if cfg.max_num_seqs is not None:
                        llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
                    if cfg.max_num_batched_tokens is not None:
                        llm_kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
                    
                    temp_llm = LLM(**llm_kwargs)
                    tokenizer_cache[model_name] = temp_llm.llm_engine.tokenizer.tokenizer
                    unload_model(temp_llm)
                
                tokenizer = tokenizer_cache[model_name]
                
                # BATCH: Extract all prompts for this tier
                problems = [df.loc[idx, prompt_col] for idx in indices_for_tier]
                prompts = batch_apply_chat_template(problems, tokenizer)
                input_tok_counts = count_input_tokens_batch(prompts, tokenizer)
                
                # Load LLM once for all samples in this tier
                cfg = model_run_cfgs.get(model_name, VLLMModelRunCfg())
                llm_kwargs = {
                    "model": model_name,
                    "tensor_parallel_size": cfg.tensor_parallel_size,
                    "gpu_memory_utilization": cfg.gpu_memory_utilization,
                    "max_model_len": cfg.max_model_len,
                }
                if cfg.max_num_seqs is not None:
                    llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
                if cfg.max_num_batched_tokens is not None:
                    llm_kwargs["max_num_batched_tokens"] = cfg.max_num_batched_tokens
                
                llm = LLM(**llm_kwargs)
                
                try:
                    # BATCH: Run inference on all prompts at once
                    sampling = SamplingParams(temperature=float(temp_val), max_tokens=max_tokens, n=int(k_val))
                    outputs = llm.generate(prompts, sampling_params=sampling)
                    
                    # BATCH: Extract samples for each sample and accumulate
                    newly_confident_indices = []
                    for sample_pos, (idx, req_out) in enumerate(zip(indices_for_tier, outputs)):
                        tier_samples = [o.text for o in req_out.outputs]
                        tier_detailed_data = [
                            {
                                "text": o.text,
                                "output_tokens": int(len(o.token_ids or [])),
                                "tier": len(tiers_executed_per_idx[idx]),
                                "model": model_name,
                            }
                            for o in req_out.outputs
                        ]
                        
                        # Accumulate samples
                        accumulated_samples_per_idx[idx].extend(tier_samples)
                        accumulated_json_per_idx[idx].extend(tier_detailed_data)
                        tiers_executed_per_idx[idx].append((model_name, k_val, temp_val))
                        escalation_path_per_idx[idx].append(model_name.split('/')[-1].replace('-Instruct', ''))
                        
                        # BATCH: Calculate confidence for this sample with all accumulated samples
                        if len(accumulated_samples_per_idx[idx]) > 0:
                            mv_answer, chosen_answer, _ = majority_vote_from_samples(
                                accumulated_samples_per_idx[idx], extract_answer_fn=extract_answer_fn
                            )
                            # Calculate confidence
                            extracted_answers = [extract_answer_fn(t) for t in accumulated_samples_per_idx[idx]]
                            normalized_answers = [_normalize_answer_for_vote(a) for a in extracted_answers]
                            chosen_normalized = _normalize_answer_for_vote(chosen_answer) if chosen_answer else "__NO_ANSWER__"
                            vote_count = sum(1 for a in normalized_answers if a == chosen_normalized)
                            confidence = vote_count / len(accumulated_samples_per_idx[idx])
                            tier_confidences_per_idx[idx][len(tiers_executed_per_idx[idx]) - 1] = float(confidence)
                        
                        # Cost calculation (only charge input once per model per sample)
                        model_costs = pricing_config.get(model_name, {}).get("model_costs", {})
                        in_rate = model_costs.get("input_per_mill", None)
                        out_rate = model_costs.get("output_per_mill", None)
                        
                        if in_rate and out_rate:
                            # Only charge input cost once per model
                            if model_name not in input_costs_by_model_per_idx[idx]:
                                input_cost = (input_tok_counts[sample_pos] / TOKENS_PER_MILLION) * in_rate
                                input_costs_by_model_per_idx[idx][model_name] = input_cost
                                total_cost_per_idx[idx] += input_cost
                            
                            # Count output tokens for this sample
                            sample_output_tokens = sum(int(len(o.token_ids or [])) for o in req_out.outputs)
                            tier_output_cost = (sample_output_tokens / TOKENS_PER_MILLION) * out_rate
                            tier_idx_for_sample = len(tiers_executed_per_idx[idx]) - 1
                            cost_per_tier_per_idx[idx][tier_idx_for_sample] = float(
                                input_costs_by_model_per_idx[idx][model_name] + tier_output_cost
                            )
                            total_cost_per_idx[idx] += tier_output_cost
                        
                        # Check if confident (early stopping)
                        if confidence >= target_confidence:
                            # This sample is done
                            df.at[idx, out_text_col] = mv_answer
                            df.at[idx, out_model_col] = escalation_path_per_idx[idx][-1]
                            df.at[idx, "final_confidence"] = confidence
                            newly_confident_indices.append(idx)
                    
                    # BATCH: Log progress
                    confident_count = len(newly_confident_indices)
                    still_pending = len(indices_for_tier) - confident_count
                    print(f"      ✓ {confident_count} samples reached confidence threshold")
                    if still_pending > 0:
                        print(f"      ↗ {still_pending} samples will escalate to next tier")
                    
                    # Remove confident samples from pending
                    for idx in newly_confident_indices:
                        pending_indices.discard(idx)
                
                except Exception as e:
                    print(f"      ✗ Error processing tier: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use what we have for affected samples
                    for idx in indices_for_tier:
                        if len(accumulated_samples_per_idx[idx]) > 0:
                            mv_answer, _, _ = majority_vote_from_samples(
                                accumulated_samples_per_idx[idx], extract_answer_fn=extract_answer_fn
                            )
                            df.at[idx, out_text_col] = mv_answer
                            df.at[idx, out_model_col] = escalation_path_per_idx[idx][-1] if escalation_path_per_idx[idx] else "unknown"
                            pending_indices.discard(idx)
                
                finally:
                    unload_model(llm)
                
                tier_global_idx += 1
        
        # Store final results for remaining samples (those that didn't reach confidence)
        for idx in df[cascade_mask].index:
            if df.at[idx, out_text_col] is None and accumulated_samples_per_idx[idx]:
                # Use best available answer
                mv_answer, _, _ = majority_vote_from_samples(
                    accumulated_samples_per_idx[idx], extract_answer_fn=extract_answer_fn
                )
                df.at[idx, out_text_col] = mv_answer
                df.at[idx, out_model_col] = escalation_path_per_idx[idx][-1] if escalation_path_per_idx[idx] else "unknown"
                
                # Calculate final confidence
                extracted_answers = [extract_answer_fn(t) for t in accumulated_samples_per_idx[idx]]
                normalized_answers = [_normalize_answer_for_vote(a) for a in extracted_answers]
                chosen_answer = majority_vote_from_samples(
                    accumulated_samples_per_idx[idx], extract_answer_fn=extract_answer_fn
                )[1]
                chosen_normalized = _normalize_answer_for_vote(chosen_answer) if chosen_answer else "__NO_ANSWER__"
                vote_count = sum(1 for a in normalized_answers if a == chosen_normalized)
                df.at[idx, "final_confidence"] = vote_count / len(accumulated_samples_per_idx[idx])
            
            # Store remaining metadata (use pd.isna instead of is None to handle np.nan)
            if pd.isna(df.at[idx, "tiers_executed"]):
                df.at[idx, "tiers_executed"] = tiers_executed_per_idx[idx]
            if pd.isna(df.at[idx, "escalation_path"]):
                df.at[idx, "escalation_path"] = "→".join(escalation_path_per_idx[idx])
            if pd.isna(df.at[idx, "tier_confidences"]):
                df.at[idx, "tier_confidences"] = json.dumps(tier_confidences_per_idx[idx], ensure_ascii=False)
            if pd.isna(df.at[idx, "cost_per_tier"]):
                df.at[idx, "cost_per_tier"] = json.dumps(cost_per_tier_per_idx[idx], ensure_ascii=False)
            if pd.isna(df.at[idx, total_cost_col]):
                df.at[idx, total_cost_col] = total_cost_per_idx[idx]
            
            if store_all_samples_col is not None:
                if df.at[idx, store_all_samples_col] is None:
                    df.at[idx, store_all_samples_col] = json.dumps(accumulated_json_per_idx[idx], ensure_ascii=False)
        
        print(f"\n✓ Batched cascade processing complete")
        if checkpoint_path is not None:
            df.to_parquet(checkpoint_path, index=True)
    
    return df
