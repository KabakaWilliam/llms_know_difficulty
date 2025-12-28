import json
import numpy as np
import pandas as pd
import sys
import base64

from math_verify import parse, verify
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import Counter

from pathlib import Path

repo_root = Path.cwd().parent.parent
sys.path.insert(0, str(repo_root))
from thom_replication.utils.verification_math import try_extract_solution

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

def load_probe_data(MODEL_NAME, PROBING_DATASET="MATH", K=1, TEMPERATURE=0.0, DATA_PATH="../probe_results/DATA/SR_DATA"):
    MODEL_ALIAS= "-".join(MODEL_NAME.split("/"))
    GEN_STR = f"maxlen_3000_k_{K}_temp_{TEMPERATURE}"
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

SIMPLE_MODEL_POOL_CONFIG = {
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
        "model_costs": { # Based on Alibaba CLoud Pricing
            "input_per_mill": 0.10,
            "output_per_mill": 0.10
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
        "model_costs": {
            "input_per_mill": 0.144,
            "output_per_mill": 0.287
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
        "model_costs": {
            "input_per_mill": 0.574,
            "output_per_mill": 1.721
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
        all_responses .append(sol["text"])
    return all_responses

def add_majority_vote_answer(solutions):
    response_list = get_output_text(solutions)
    return majority_vote_from_samples(response_list, extract_answer_fn=try_extract_solution)[1]