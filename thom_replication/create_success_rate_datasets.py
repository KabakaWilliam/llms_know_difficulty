# Largely the same as predicting_learnability/rollout_success_rate.py but a bit cleaner
# and uses our own verification functions


def get_task(name):
    import os
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from transformers import AutoTokenizer

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


def main(
        model_name,
        max_response_len=3000,
        temperature=1.0,
        prompt_suffix="Let's think step by step and output the final answer within \\boxed{}.",
        num_rollouts_per_question=50,
        max_questions_per_split=None,
        tensor_parallel_size=1,
):
    import os
    import random
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from transformers import AutoTokenizer
    seed = 42                        
    rng = random.Random(seed)
    sampling_params = SamplingParams(max_tokens=max_response_len, temperature=temperature, top_p=1, top_k=-1, n=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    ds, compute_score = get_task('MATH')

    selected_indices = {}
    for split in ds:
        n = len(ds[split])
        if max_questions_per_split is None:
            selected_indices[split] = list(range(n))          # full split
        else:
            k = min(max_questions_per_split, n)
            selected_indices[split] = rng.sample(range(n), k) # random subset

    inputs = []
    for split in ds:
        for idx in selected_indices[split]:
            item = ds[split][idx]


            prompt = item['problem'] + ' ' + prompt_suffix
            messages = [
                {"role": "user", "content": prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for _ in range(num_rollouts_per_question):
                inputs.append({
                    'split': split,
                    'idx': idx,
                    'problem': item['problem'],
                    'formatted_prompt': prompt,
                })

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
    OUTPUT_DIR="../will_replication/DATA/SR_DATA/MATH"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in results:
        results_df = pd.DataFrame.from_dict(results[split], orient='index')
        model_alias = model_name.replace("/", "-")
        
        # Include number of questions in filename only if max_questions_per_split was specified
        if max_questions_per_split is not None:
            FILEPATH = f"{OUTPUT_DIR}/{split}-{len(results[split])}-{model_alias}_maxlen_{max_response_len}_k_{num_rollouts_per_question}_temp_{temperature}.parquet"
        else:
            FILEPATH = f"{OUTPUT_DIR}/{split}-{model_alias}_maxlen_{max_response_len}_k_{num_rollouts_per_question}_temp_{temperature}.parquet"
        
        results_df.to_parquet(FILEPATH)
        print(f"Saved {split} split to: {FILEPATH}")

if __name__ == "__main__":
    import time
    import gc
    
    MODELS_TO_RUN = [
     "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2-1.5B-Instruct",
    # "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-Math-7B-Instruct"
    # "openai/gpt-oss-20b"
    ]
    
    for i, MODEL_TO_ROLLOUT in enumerate(MODELS_TO_RUN):
        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(MODELS_TO_RUN)}: {MODEL_TO_ROLLOUT}")
        print(f"{'='*60}\n")
        
        main(
            model_name=MODEL_TO_ROLLOUT,
            max_questions_per_split=1000,
            tensor_parallel_size=1,
        )
        
        print(f"\nFinished processing {MODEL_TO_ROLLOUT}")
        
        # Clean up and wait for vLLM to die before loading next model
        if i < len(MODELS_TO_RUN) - 1:  # Don't wait after the last model
            print("Cleaning up and waiting for vLLM to release resources...")
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(10)  # Wait 10 seconds for vLLM to fully shutdown
            print(f"Ready to load next model: {MODELS_TO_RUN[i+1]}\n")


# Add command line args
# Make as a slurm script
# Run all the other permutations on h200_lowest
# - temperature 0, 0.5, 1.0
# - gpt-oss, qwen base models
# - other datasets