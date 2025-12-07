
import os
import torch

from vllm import LLM, SamplingParams
from math_verify import parse, verify

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHOSEN_DEVICE = "cuda:2"
MODEL_NAME = "HuggingFaceTB/FineMath-Llama-3B"
# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_ALIAS = MODEL_NAME.split("/")[-1]
DATA_PATH = "../hard_rl/data/MATH/{}.parquet"
NUM_ROLLOUTS_PER_QUESTION = 50
MAX_RESPONSE_LEN = 1024 #3000

llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.4)

train_df = pd.read_parquet(DATA_PATH.format("train"))
test_df = pd.read_parquet(DATA_PATH.format("test"))

STORE_DF = {"train": train_df, "test": test_df}

PARAMS = SamplingParams(temperature=1, max_tokens=MAX_RESPONSE_LEN)


for SUBSET in ["train", "test"]:
    QA_STORE = dict()
    for i in tqdm(range(len(STORE_DF[SUBSET]))):
        QUESTION_PROMPT = STORE_DF[SUBSET]["prompt"].iloc[i][0]["content"]
        GROUND_TRUTH = STORE_DF[SUBSET]["reward_model"].iloc[0]["ground_truth"]

        PER_QUESTION_RESP_STORE_LOCAL = []
        outputs = llm.generate([QUESTION_PROMPT]*NUM_ROLLOUTS_PER_QUESTION, PARAMS)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            PER_QUESTION_RESP_STORE_LOCAL.append(generated_text)
            
        QA_STORE[QUESTION_PROMPT] = {"responses": PER_QUESTION_RESP_STORE_LOCAL, "ground_truth": GROUND_TRUTH }

    RESULTS = {}
    for prompt, v in tqdm(QA_STORE.items()):
        responses = v.get("responses", [])
        gt = v.get("ground_truth")
        correct = 0
        for resp in responses:
            try:
                ans = parse(resp)
            except Exception:
                ans = resp  # fallback to raw text
            try:
                if verify(gt, ans):
                    correct += 1
            except Exception:
                pass
        total = len(responses)
        rate = correct / total if total > 0 else 0.0
        RESULTS[prompt] = {"correct": correct, "total": total, "success_rate": rate, "ground_truth": gt}

    # View as a DataFrame sorted by success_rate
    results_df = pd.DataFrame.from_dict(RESULTS, orient='index').sort_values("success_rate", ascending=False)

    results_df.to_parquet(f"MATH_{MODEL_ALIAS}-SR_{SUBSET}.parquet")
    

    fig, ax = plt.subplots(figsize=(6, 4))
    results_df["success_rate"].hist(ax=ax, bins=20)
    ax.set_xlabel("Success rate")
    ax.set_ylabel("Count")
    ax.set_title(f"success_rate: {SUBSET} [{MODEL_ALIAS}]")

    fig.savefig(f"success_rate_hist_{MODEL_ALIAS}_{SUBSET}.png", dpi=300, bbox_inches="tight")
    plt.show()




