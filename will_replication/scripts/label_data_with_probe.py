import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, Callable
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

# from thom_replication.create_success_rate_datasets import get_task
from will_replication.my_utils.utils import encode_str
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
            "problem": prompts_raw[i],
            "formatted_prompt": formatted[i],
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

# -----------------------------------------------------------------------------
# Dataset adapter (THE KEY ADDITION)
# -----------------------------------------------------------------------------
@dataclass
class DatasetAdapter:
    name: str
    split: str
    prompt_fn: Callable[[Dict], str]
    target_cols: List[str]
    metadata_cols: List[str]
    subset: Optional[str] = None


def load_dataset_with_adapter(adapter: DatasetAdapter):
    if adapter.subset:
        ds = load_dataset(adapter.name, adapter.subset)
    else:
        ds = load_dataset(adapter.name)
    split = ds[adapter.split]

    prompts = [adapter.prompt_fn(ex) for ex in split]

    solution_col = next(
        (c for c in adapter.target_cols if c in split.column_names),
        None,
    )

    return split, prompts, solution_col

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"

    DATASETS = ["opencompass/AIME2025", "gneubig/aime-1983-2024", "DigitalLearningGmbH/MATH-lighteval", "openai/gsm8k"]

    DS_ALIASES = ["_".join(DATASET.split("/")) for DATASET in DATASETS]
    
    K=5
    TEMP=0.6
    GEN_STR=f"maxlen_3000_k_{K}_temp_{TEMP}"
    TARGET_PROBE_DATASET = 'DigitalLearningGmbH_MATH-lighteval'
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

    probe = load_best_probe_json(f"../probe_results/DATA/SR_DATA/{TARGET_PROBE_DATASET}/{MODEL_ALIAS}_{GEN_STR}/best_probe_predictions.json", device=None)  # keep on CPU for scoring

    for DS_NAME in DATASETS:

        if "gneubig/aime" in DS_NAME:

            ADAPTER = DatasetAdapter(
                name="gneubig/aime-1983-2024",
                split="train",
                prompt_fn=lambda ex: ex["Question"],
                target_cols=["Answer"],
                metadata_cols=["Year", "Problem Number"],
            )
        elif "MATH-lighteval" in DS_NAME:
            ADAPTER = DatasetAdapter(
                name="DigitalLearningGmbH/MATH-lighteval",
                split="test",
                prompt_fn=lambda ex: ex["problem"],
                target_cols=["solution"],
                metadata_cols=["level", "type"],
            )
        elif "gsm8k" in DS_NAME:
            ADAPTER = DatasetAdapter(
                name="openai/gsm8k",
                split="test",
                prompt_fn=lambda ex: ex["question"],
                target_cols=["answer"],
                metadata_cols=[],
                subset="main"
            )
        elif "aime2025" in DS_NAME.lower():
            ADAPTER = DatasetAdapter(
                name="opencompass/AIME2025",
                split="test",
                prompt_fn=lambda ex: ex["question"],
                target_cols=["answer"],
                metadata_cols=[],
                subset="AIME2025-I"
            )
        else:
            raise ValueError(f"{DS_NAME} isn't configured present")

        dataset_tag = ADAPTER.name.replace("/", "_")

        print(f"LABELLING DATASET: {dataset_tag}\n")

        out_dir = Path(
            f"../probe_results/DATA/Labelled_SR/{TARGET_PROBE_DATASET}_probe/{dataset_tag}/{MODEL_ALIAS}_{GEN_STR}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        split, prompts, solution_col = load_dataset_with_adapter(ADAPTER)

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
        for data_point in labeled[:2]:
            print(f"PROMPT: {data_point["problem"]}\n")
            print(f"Pred raw score: {data_point["score_raw"]}\n")
            print(f"Pred score: {data_point["score"]}\n")
            print("========================\n")

        out_records = []
        for i, ex in enumerate(split):
            rec = {
            "idx": i,
            "problem_id": encode_str(labeled[i]["problem"]),
            "dataset": ADAPTER.name,
            "problem": labeled[i]["problem"],
            "formatted_prompt": labeled[i]["formatted_prompt"],
            "score_raw": labeled[i]["score_raw"],
            "score": labeled[i]["score"],
            "layer": labeled[i]["layer"],
            "pos": labeled[i]["pos"],
            }

            if solution_col:
                rec["original_solution"] = ex.get(solution_col)

            for k in ADAPTER.metadata_cols:
                rec[k] = ex.get(k)

            out_records.append(rec)

        jsonl_path = out_dir / "scored.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"✅ Wrote {len(out_records)} rows to: {jsonl_path}")

        # Optional: also save as Parquet (faster to load later)
        parquet_path = out_dir / "scored.parquet"
        Dataset.from_list(out_records).to_parquet(str(parquet_path))
        print(f"✅ Wrote Parquet to: {parquet_path}")
