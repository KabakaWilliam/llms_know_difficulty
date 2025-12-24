import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from datasets import load_dataset
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset


import sys

repo_root = Path.cwd().parent.parent
sys.path.insert(0, str(repo_root))

print(f"Added to path: {repo_root}")
print(f"Checking if thom_replication exists: {(repo_root / 'thom_replication').exists()}")

from thom_replication.create_success_rate_datasets import get_task
# -----------------------------
# Probe loading
# -----------------------------
@dataclass
class BestProbe:
    w: torch.Tensor          # [D]
    best_layer: int          # int index into hidden_states tuple
    best_position: int       # token position (can be negative: -1 = last valid token)
    test_score: float


def load_best_probe_json(probe_path: str, device: Optional[torch.device] = None) -> BestProbe:
    with open(probe_path, "r") as f:
        data = json.load(f)

    w = torch.tensor(np.array(data["probe_weights"]), dtype=torch.float32)
    best_layer = int(data["best_layer"])
    best_position = int(data["best_position"])
    test_score = float(data.get("test_score", float("nan")))
    benchmark_score = float(data.get("avg_benchmark_score", float("nan")))

    if device is not None:
        w = w.to(device)

    print(f"✅ Loaded best probe")
    print(f"   benchmark     : {benchmark_score}")
    print(f"   test_score     : {test_score}")
    print(f"   best_layer     : {best_layer}")
    print(f"   best_position  : {best_position}")
    print(f"   weights shape  : {tuple(w.shape)}")

    return BestProbe(w=w, best_layer=best_layer, best_position=best_position, test_score=test_score)


# -----------------------------
# Token index selection
# -----------------------------
def _pick_token_index(
    attention_mask: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    """
    Returns idx: [B] with the token index per example.
    - If pos < 0: counts from the end of *valid* tokens (masked length)
      e.g. -1 = last valid token
    - If pos >= 0: uses absolute position
    """
    # lengths = number of valid tokens per row
    lengths = attention_mask.sum(dim=1)  # [B], dtype long
    B, T = attention_mask.shape

    if pos < 0:
        # last valid token idx is lengths-1, so pos=-1 -> lengths-1
        idx = (lengths + pos).clamp(min=0, max=T - 1)
    else:
        idx = torch.full((B,), pos, device=attention_mask.device, dtype=lengths.dtype).clamp(min=0, max=T - 1)

    return idx


# -----------------------------
# Activation extraction (fast, simple)
# -----------------------------
@torch.no_grad()
def extract_acts_at_layer_and_pos(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer_idx: int,
    pos_idx: int,
    batch_size: int = 8,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns activations: [N, D] from (layer_idx, pos_idx) for each prompt.
    layer_idx refers to hidden_states tuple indexing:
      hidden_states[0] = embeddings, hidden_states[1] = layer1, ..., hidden_states[N] = final layer
    """
    device = next(model.parameters()).device
    acts_all = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple of [B, T, D]

        # Normalize negative layer indices
        num_hs = len(hidden_states)
        if layer_idx < 0:
            li = num_hs + layer_idx
        else:
            li = layer_idx
        if not (0 <= li < num_hs):
            raise ValueError(f"layer_idx {layer_idx} (normalized {li}) out of range for hidden_states len={num_hs}")

        hs = hidden_states[li]  # [B, T, D]
        idx = _pick_token_index(attention_mask, pos_idx)  # [B]

        B = hs.shape[0]
        batch_acts = hs[torch.arange(B, device=device), idx]  # [B, D]
        acts_all.append(batch_acts.detach().cpu())

    return torch.cat(acts_all, dim=0)  # [N, D]


# -----------------------------
# Scoring / labeling
# -----------------------------
def score_prompts_with_probe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts_raw: List[Union[str, Dict]],
    probe: BestProbe,
    batch_size: int = 8,
    apply_chat_template: bool = False,
    max_length: Optional[int] = None,
    return_sigmoid: bool = True,
) -> List[Dict]:
    """
    Produces a list of dicts:
      { "prompt": <original prompt>, "formatted": <formatted string>, "score_raw": float, "score": float }
    """
    # 1) Format prompts
    if apply_chat_template:
        # prompts_raw must be list of chat turns OR raw user strings; we’ll handle both.
        formatted = []
        for p in prompts_raw:
            if isinstance(p, str):
                messages = [{"role": "user", "content": p}]
            else:
                # assume already a messages list or a dict structure
                # if you pass a dict like {"role":..,"content":..} wrap into list
                messages = p if isinstance(p, list) else [p]

            formatted_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted.append(formatted_str)
    else:
        formatted = [p if isinstance(p, str) else json.dumps(p) for p in prompts_raw]

    # 2) Extract activations at best (layer, pos)
    acts = extract_acts_at_layer_and_pos(
        model=model,
        tokenizer=tokenizer,
        prompts=formatted,
        layer_idx=probe.best_layer,
        pos_idx=probe.best_position,
        batch_size=batch_size,
        max_length=max_length,
    )  # [N, D]

    # 3) Score
    w = probe.w.detach().cpu().float()  # [D]
    acts = acts.float()                 # [N, D]
    score_raw = acts @ w                # [N]

    if return_sigmoid:
        score = torch.sigmoid(score_raw)
    else:
        score = score_raw

    # 4) Pack results
    results = []
    for i in range(len(prompts_raw)):
        results.append({
            "prompt": prompts_raw[i],
            "formatted": formatted[i],
            "score_raw": float(score_raw[i].item()),
            "score": float(score[i].item()),
            "layer": probe.best_layer,
            "pos": probe.best_position,
        })

    # Optional quick stats
    sr = score.detach().cpu()
    print("📊 Probe score stats")
    print(f"   mean: {sr.mean().item():.4f}")
    print(f"   std : {sr.std().item():.4f}")
    print(f"   min : {sr.min().item():.4f}")
    print(f"   max : {sr.max().item():.4f}")

    return results


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    K=50
    TEMP=0.6
    GEN_STR=f"maxlen_3000_k_{K}_temp_{TEMP}"
    MODEL_ALIAS = "-".join(model_name.split("/"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    probe = load_best_probe_json(f"../probe_results/DATA/SR_DATA/MATH/{MODEL_ALIAS}_{GEN_STR}/best_probe_predictions.json", device=None)  # keep on CPU for scoring

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")


    test_split = ds["test"]
    prompts = test_split["problem"]

    # Robustly find the "solution" column (different MATH variants name it differently)
    candidate_solution_cols = ["solution", "answer", "final_answer", "target", "ground_truth"]
    solution_col = next((c for c in candidate_solution_cols if c in test_split.column_names), None)

#     prompts = [
#         "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
#         "Compute the integral of x^2 from 0 to 1",
#         "What is 2 + 2 ?",
#         "Six points $ A, B, C, D, E, $ and $ F $ lie in a straight line in that order. Suppose that $ G $ is a point not on the line and that $ AC = 26 $, $ BD = 22 $, $ CE = 31 $, $ DF = 33 $, $ AF = 73 $, $ CG = 40 $, and $ DG = 30 $. Find the area of $ \triangle BGE $.",
#         """Let $ A_1A_2 \ldots A_{11} $ be an 11-sided non-convex simple polygon with the following properties:
# * The area of $ A_iA_1A_{i+1} $ is 1 for each $ 2 \leq i \leq 10 $,
# * $ \cos(\angle A_iA_1A_{i+1}) = \frac{12}{13} $ for each $ 2 \leq i \leq 10 $,
# * The perimeter of $ A_1A_2 \ldots A_{11} $ is 20.
# If $ A_1A_2 + A_1A_{11} $ can be expressed as $ \frac{m\sqrt{n} - p}{q} $ for positive integers $ m, n, p, q $ with $ n $ squarefree and no prime divides all of $ m, p, q$, find $ m + n + p + q $."""
#     ]


    prompt_sffx = " Let's think step by step and output the final answer within \\boxed{}."

    semi_formatted_prompts = [prompt + prompt_sffx for prompt in prompts]

    labeled = score_prompts_with_probe(
        model=model,
        tokenizer=tokenizer,
        prompts_raw=semi_formatted_prompts,
        probe=probe,
        batch_size=8,
        apply_chat_template=True,
        max_length=2048,
        return_sigmoid=True,
    )

    print(f"\nTemperature 🌡 = {TEMP}, K 🧮= {K}\n")
    for data_point in labeled[:5]:
        print(data_point["score_raw"])
        print(data_point["score"])
        print("========================\n")
    out_records = []
    for i in range(len(test_split)):
        ex = test_split[i]

        # labeled[i]["prompt"] is your semi_formatted prompt string (problem + suffix)
        out_records.append({
            "idx": i,
            "problem": ex.get("problem"),
            "original_solution": ex.get(solution_col),

            # optional extra metadata if present
            "level": ex.get("level"),
            "type": ex.get("type"),

            # what you actually scored
            "prompt_scored": labeled[i]["prompt"],
            "formatted": labeled[i]["formatted"],

            # probe outputs
            "score_raw": labeled[i]["score_raw"],
            "score": labeled[i]["score"],
            "layer": labeled[i]["layer"],
            "pos": labeled[i]["pos"],
        })

    # Write JSONL (easy to stream / grep)
    out_dir = Path(f"../probe_results/DATA/SR_DATA/MATH/{MODEL_ALIAS}_{GEN_STR}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "math_lighteval_scored.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(out_records)} rows to: {jsonl_path}")

    # Optional: also save as Parquet (faster to load later)
    parquet_path = out_dir / "math_lighteval_scored.parquet"
    Dataset.from_list(out_records).to_parquet(str(parquet_path))
    print(f"✅ Wrote Parquet to: {parquet_path}")
