# Largely the same as predicting_learnability/rollout_success_rate.py but a bit cleaner
# and uses our own verification functions

import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer


# MODEL_NAME = "HuggingFaceTB/FineMath-Llama-3B"
# MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

MEMORY_UTIL=0.6
NUM_ROLLOUTS_PER_QUESTION = 50
MAX_QUESTIONS_PER_SPLIT = None
MAX_RESPONSE_LEN = 3000 #3000
DATASET_NAME = "MATH"
OUTPUT_DIR = f"../will_replication/DATA/SR_DATA/{DATASET_NAME}"
TEMPERATURE=1.0
TOP_P=1
TOP_K=-1
N=1
CONFIG_STR = f"maxlen_{MAX_RESPONSE_LEN}_k_{NUM_ROLLOUTS_PER_QUESTION}_temp_{TEMPERATURE}"

PROMPT = "Let's think step by step and output the final answer within \\boxed{}."

def get_task(name):
    if name == 'MATH':

        from utils.verification_math import compute_score, extract_solution
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval")
        # extract the solution from the chain of thought
        def add_extracted_solution(example):
            example['extracted_solution'] = extract_solution(example['solution'])
            return example
        for split in ds:
            ds[split] = ds[split].map(add_extracted_solution)

        return ds, compute_score
    
    raise ValueError(f"Unknown task {name}")


def main():

    sampling_params = SamplingParams(max_tokens=MAX_RESPONSE_LEN, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, n=N)
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=MEMORY_UTIL)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    ds, compute_score = get_task('MATH')

    # for eaach split, generate NUM_ROLLOUTS_PER_QUESTION solutions
    inputs = []
    for split in ds:
        for idx, item in enumerate(ds[split]):
            # prompt = item['problem'] + ' ' + PROMPT
            messages = [{"role": "user", "content": item['problem'] + ' ' + PROMPT}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            for _ in range(NUM_ROLLOUTS_PER_QUESTION):
                inputs.append({
                    'split': split,
                    'idx': idx,
                    'problem': item['problem'],
                    'formatted_prompt': formatted_prompt,
                })
            if MAX_QUESTIONS_PER_SPLIT is not None and idx > MAX_QUESTIONS_PER_SPLIT:
                break

    outputs = llm.generate([inp['formatted_prompt'] for inp in inputs], sampling_params)

    # collect results
    results = {split: dict() for split in ds}
    for inp, output in zip(inputs, outputs):
        split = inp['split']
        idx = inp['idx']
        ground_truth = ds[split][idx]['extracted_solution']
        formatted_prompt = inp['formatted_prompt']
        generated_text = output.outputs[0].text

        if idx not in results[split]:
            results[split][idx] = {
                'generated_solutions': [],
                'ground_truth': ground_truth,
                'formatted_prompt': formatted_prompt,
                'problem': inp['problem']
            }

        score = compute_score(generated_text, ground_truth)
        results[split][idx]['generated_solutions'].append({
            'text': generated_text,
            'score': score
        })

    # then compute success rate
    for split in results:
        for idx in results[split]:
            gen_sols = results[split][idx]['generated_solutions']
            correct = sum(gs['score'] for gs in gen_sols)
            total = len(gen_sols)
            success_rate = correct / total if total > 0 else 0.0
            results[split][idx]['success_rate'] = success_rate

    # and overall success rate per split
    for split in results:
        total_questions = len(results[split])
        total_success = sum(results[split][idx]['success_rate'] for idx in results[split])
        overall_success_rate = total_success / total_questions if total_questions > 0 else 0.0
        print(f"Overall success rate for split {split}: {overall_success_rate:.4f}")

    # then save to disk as parquet
    import pandas as pd

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in results:
        results_df = pd.DataFrame.from_dict(results[split], orient='index')
        MODEL_ALIAS = MODEL_NAME.replace("/", "-")
        output_filename = f"{split}-{MODEL_ALIAS}_maxlen_{MAX_RESPONSE_LEN}_k_{NUM_ROLLOUTS_PER_QUESTION}_temp_{TEMPERATURE}.parquet"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        results_df.to_parquet(output_path)
        print(f"Saved {split} split to {output_path}")

if __name__ == "__main__":
    main()