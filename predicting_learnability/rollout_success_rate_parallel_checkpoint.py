import os
from concurrent.futures import ProcessPoolExecutor

from vllm import LLM, SamplingParams
from math_verify import parse, verify

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

CHOSEN_DEVICE=0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{CHOSEN_DEVICE}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct" ##3000
# MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "HuggingFaceTB/FineMath-Llama-3B"
MODEL_NAME = "allenai/Olmo-3-7B-Think"

MODEL_ALIAS = MODEL_NAME.split("/")[-1]
DATA_PATH = "../hard_rl/data/MATH/{}.parquet"
NUM_ROLLOUTS_PER_QUESTION = 50
MAX_RESPONSE_LEN = 32768#1024 #32768 #3000


# tune these to fit GPU/memory
BATCH_QUESTIONS = 24  # number of distinct questions per model call batch
MODEL_CHUNK = 1000   #500  # number of prompts sent to the model at once (after repetition)

# Checkpointing settings
CHECKPOINT_EVERY_N_QUESTIONS = 100  # Save checkpoint every N questions
CHECKPOINT_DIR = "checkpoints"

assert(MODEL_CHUNK % NUM_ROLLOUTS_PER_QUESTION == 0)

llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.98)

train_df = pd.read_parquet(DATA_PATH.format("train"))
test_df = pd.read_parquet(DATA_PATH.format("test"))



TOTAL_GENERATIONS = NUM_ROLLOUTS_PER_QUESTION * len(train_df)

STORE_DF = {"train": train_df, "test": test_df}

PARAMS = SamplingParams(temperature=1, max_tokens=MAX_RESPONSE_LEN)


def _extract_prompt_and_gt_from_row(row):
    # try several common shapes that the dataset used earlier appears to have
    # prompt may be a list with dict at [0], or a dict
    try:
        prompt = row["prompt"][0]["content"]
    except Exception:
        try:
            prompt = row["prompt"]["content"]
        except Exception:
            prompt = str(row.get("prompt", ""))

    try:
        gt = row["reward_model"][0]["ground_truth"]
    except Exception:
        try:
            gt = row["reward_model"]["ground_truth"]
        except Exception:
            gt = row.get("ground_truth")
    return prompt, gt


def _eval_pair(item):
    # top-level function so it can be pickled for ProcessPoolExecutor
    gt, resp = item
    try:
        ans = parse(resp)
    except Exception:
        ans = resp
    try:
        return 1 if verify(gt, ans) else 0
    except Exception:
        return 0


def save_checkpoint(subset, correct_counts, total_counts, prompts, gts, last_processed_idx, model_alias):
    """Save current progress to a checkpoint file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_data = {
        "correct_counts": correct_counts,
        "total_counts": total_counts,
        "prompts": prompts,
        "gts": gts,
        "last_processed_idx": last_processed_idx,
    }
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{model_alias}_{subset}.pkl")
    pd.to_pickle(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved at question {last_processed_idx} to {checkpoint_path}")


def load_checkpoint(subset, model_alias):
    """Load checkpoint if it exists, return None otherwise."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{model_alias}_{subset}.pkl")
    if os.path.exists(checkpoint_path):
        checkpoint_data = pd.read_pickle(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}, resuming from question {checkpoint_data['last_processed_idx']}")
        return checkpoint_data
    return None


def save_intermediate_results(subset, correct_counts, total_counts, prompts, gts, model_alias, suffix="intermediate"):
    """Save intermediate results as parquet file."""
    n_q = len(prompts)
    RESULTS = {}
    for i in range(n_q):
        total = total_counts[i]
        correct = correct_counts[i]
        rate = correct / total if total > 0 else 0.0
        RESULTS[i] = {"correct": correct, "total": total, "success_rate": rate}

    results_df = pd.DataFrame.from_dict(RESULTS, orient='index')
    results_df["prompt"] = prompts
    results_df["ground_truth"] = gts
    
    output_path = f"MATH_{model_alias}-SR_{subset}_{suffix}.parquet"
    results_df.to_parquet(output_path)
    print(f"Intermediate results saved to {output_path}")


for SUBSET in ["train", "test"]:
    df = STORE_DF[SUBSET]
    n_q = len(df)

    # Try to load existing checkpoint
    checkpoint = load_checkpoint(SUBSET, MODEL_ALIAS)
    
    if checkpoint is not None:
        # Resume from checkpoint
        prompts = checkpoint["prompts"]
        gts = checkpoint["gts"]
        correct_counts = checkpoint["correct_counts"]
        total_counts = checkpoint["total_counts"]
        start_q_idx = checkpoint["last_processed_idx"]
        print(f"Resuming {SUBSET} from question {start_q_idx}")
    else:
        # Start fresh - pre-extract prompts and ground-truths
        prompts = [None] * n_q
        gts = [None] * n_q
        for i, (_, row) in enumerate(df.iterrows()):
            p, gt = _extract_prompt_and_gt_from_row(row)
            try:
                p = llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            except:
                print("Continuing without chat template")
            prompts[i] = p
            gts[i] = gt

        # Initialize counters
        correct_counts = [0] * n_q
        total_counts = [0] * n_q
        start_q_idx = 0

    # Track questions processed since last checkpoint
    questions_since_checkpoint = 0

    # iterate over question batches to build repeated prompts, then send to model in chunks
    for q_start in tqdm(range(start_q_idx, n_q, BATCH_QUESTIONS), desc=f"gen:{SUBSET}"):
        q_end = min(q_start + BATCH_QUESTIONS, n_q)
        batch_indices = list(range(q_start, q_end))

        # build flattened list of repeated prompts and a mapping to question indices
        batch_prompts = []
        mapping = []
        for qi in batch_indices:
            p = prompts[qi]
            for _ in range(NUM_ROLLOUTS_PER_QUESTION):
                batch_prompts.append(p)
                mapping.append(qi)

        # generate in model-sized chunks to avoid memory blowups
        for chunk_start in range(0, len(batch_prompts), MODEL_CHUNK):
            chunk_end = min(chunk_start + MODEL_CHUNK, len(batch_prompts))
            chunk_prompts = batch_prompts[chunk_start:chunk_end]
            chunk_mapping = mapping[chunk_start:chunk_end]

            outputs = llm.generate(chunk_prompts, PARAMS)

            # collect responses in same order as mapping
            responses = []
            for out in outputs:
                try:
                    text = out.outputs[0].text
                except Exception:
                    text = ""
                responses.append(text)

            # prepare items for parallel evaluation: (gt, resp)
            items = [(gts[qidx], resp) for qidx, resp in zip(chunk_mapping, responses)]

            # parallel parse+verify; small overhead but speeds up CPU-bound verification
            with ProcessPoolExecutor(max_workers=min(8, (os.cpu_count() or 1))) as exc:
                for qidx, res in zip(chunk_mapping, exc.map(_eval_pair, items)):
                    correct_counts[qidx] += res
                    total_counts[qidx] += 1

        # Update questions processed count
        questions_since_checkpoint += (q_end - q_start)

        # Save checkpoint periodically
        if questions_since_checkpoint >= CHECKPOINT_EVERY_N_QUESTIONS:
            save_checkpoint(SUBSET, correct_counts, total_counts, prompts, gts, q_end, MODEL_ALIAS)
            save_intermediate_results(SUBSET, correct_counts, total_counts, prompts, gts, MODEL_ALIAS, suffix=f"checkpoint_q{q_end}")
            questions_since_checkpoint = 0

    # Build final results
    RESULTS = {}
    for i in range(n_q):
        total = total_counts[i]
        correct = correct_counts[i]
        rate = correct / total if total > 0 else 0.0
        RESULTS[i] = {"correct": correct, "total": total, "success_rate": rate}

    results_df = pd.DataFrame.from_dict(RESULTS, orient='index')
    results_df["prompt"] = prompts
    results_df["ground_truth"] = gts

    results_df.to_parquet(f"MATH_{MODEL_ALIAS}-SR_{SUBSET}.parquet")

    # Clean up checkpoint file after successful completion
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{MODEL_ALIAS}_{SUBSET}.pkl")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file {checkpoint_path} after successful completion")

    fig, ax = plt.subplots(figsize=(6, 4))
    results_df["success_rate"].hist(ax=ax, bins=20)
    ax.set_xlabel("Success rate")
    ax.set_ylabel("Count")
    ax.set_title(f"success_rate: {SUBSET} [{MODEL_ALIAS}]")

    fig.savefig(f"success_rate_hist_{MODEL_ALIAS}_{SUBSET}.png", dpi=300, bbox_inches="tight")
    plt.show()
