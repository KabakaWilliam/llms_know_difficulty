# Greedy (deterministic) variant for generating success rate datasets
# Uses temperature=0 for deterministic generation, no sampling

import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

# MODEL_NAME = "HuggingFaceTB/FineMath-Llama-3B"
# MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

NUM_ROLLOUTS_PER_QUESTION = 1  # Greedy: only one generation per question
MEMORY_UTIL = 0.9
MAX_QUESTIONS_PER_SPLIT = None
MAX_RESPONSE_LEN = 3000
DATASET_NAME = "MATH"  # Used for output directory structure
OUTPUT_DIR = f"../will_replication/DATA/SR_DATA/{DATASET_NAME}"

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
    # Greedy sampling: temperature=0 for deterministic generation
    sampling_params = SamplingParams(
        max_tokens=MAX_RESPONSE_LEN, 
        temperature=0.0,  # Greedy decoding
        n=1
    )
    
    llm = LLM(model=MODEL_NAME, gpu_memory_utilization=MEMORY_UTIL)
    
    # Load tokenizer for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    ds, compute_score = get_task('MATH')

    # For each split, generate one greedy solution per question
    inputs = []

    for split in ds:
        for idx, item in enumerate(ds[split]):
            # Format prompt with chat template
            messages = [{"role": "user", "content": item['problem'] + ' ' + PROMPT}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            
            inputs.append({
                'split': split,
                'idx': idx,
                'problem': item['problem'],
                'answer': item['extracted_solution'],
                'formatted_prompt': formatted_prompt
            })
            
            if MAX_QUESTIONS_PER_SPLIT is not None and idx >= MAX_QUESTIONS_PER_SPLIT:
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

    # Save to disk as parquet with naming convention: {DATASET}_{split}-{MODEL_ALIAS}_maxlen_{MAX_RESPONSE_LEN}_k_0.parquet
    import pandas as pd
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in results:
        results_df = pd.DataFrame.from_dict(results[split], orient='index')
        MODEL_ALIAS = MODEL_NAME.split("/")[-1]  # Extract just the model name
        output_filename = f"{split}-{MODEL_ALIAS}_maxlen_{MAX_RESPONSE_LEN}_k_1_temp_0.0.parquet"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        results_df.to_parquet(output_path)
        print(f"Saved {split} split to {output_path}")

if __name__ == "__main__":
    main()