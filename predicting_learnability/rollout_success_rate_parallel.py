
import os
from concurrent.futures import ProcessPoolExecutor

from vllm import LLM, SamplingParams
from math_verify import parse, verify

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

CHOSEN_DEVICE=3
os.environ["CUDA_VISIBLE_DEVICES"] = f"{CHOSEN_DEVICE}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct" ##3000
# MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "HuggingFaceTB/FineMath-Llama-3B"
MODEL_NAME = "allenai/Olmo-3-7B-Think"

MODEL_ALIAS = MODEL_NAME.split("/")[-1]
DATA_PATH = "../hard_rl/data/MATH/{}.parquet"
MEMORY_UTIL = 0.90
NUM_ROLLOUTS_PER_QUESTION = 50
MAX_RESPONSE_LEN = 32768#1024 #32768 #3000


# tune these to fit GPU/memory
BATCH_QUESTIONS = 24  # number of distinct questions per model call batch
MODEL_CHUNK = 1000   #500  # number of prompts sent to the model at once (after repetition)

assert(MODEL_CHUNK % NUM_ROLLOUTS_PER_QUESTION == 0)

llm = LLM(model=MODEL_NAME, gpu_memory_utilization=MEMORY_UTIL)

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
        return 1 if verify(parse(f"${gt}$"), ans) else 0
    except Exception:
        return 0


for SUBSET in ["train", "test"]:
    df = STORE_DF[SUBSET]
    n_q = len(df)

    # pre-extract prompts and ground-truths to avoid repeated iloc access
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

    # We'll keep counters per-question (index-based) to avoid storing all responses
    correct_counts = [0] * n_q
    total_counts = [0] * n_q

    # iterate over question batches to build repeated prompts, then send to model in chunks
    for q_start in tqdm(range(0, n_q, BATCH_QUESTIONS), desc=f"gen:{SUBSET}"):
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

    # Build results
    RESULTS = {}
    for i in range(n_q):
        total = total_counts[i]
        correct = correct_counts[i]
        rate = correct / total if total > 0 else 0.0
        RESULTS[i] = {"correct": correct, "total": total, "success_rate": rate}

    results_df = pd.DataFrame.from_dict(RESULTS, orient='index')
    results_df["prompt"] = prompts
    results_df["ground_truth"] = gts

    results_df.to_parquet(f"MATH_{MODEL_ALIAS}-SR_{SUBSET}_{MAX_RESPONSE_LEN}.parquet")

    fig, ax = plt.subplots(figsize=(6, 4))
    results_df["success_rate"].hist(ax=ax, bins=20)
    ax.set_xlabel("Success rate")
    ax.set_ylabel("Count")
    ax.set_title(f"success_rate: {SUBSET} [{MODEL_ALIAS}]")

    fig.savefig(f"success_rate_hist_{MODEL_ALIAS}_{SUBSET}.png", dpi=300, bbox_inches="tight")
    plt.show()




