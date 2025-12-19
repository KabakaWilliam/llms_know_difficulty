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
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from transformers import AutoTokenizer

    sampling_params = SamplingParams(max_tokens=max_response_len, temperature=temperature, top_p=1, top_k=-1, n=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    ds, compute_score = get_task('MATH')
    # # Sample 1k examples per split
    # for split in ds:
    #     if len(ds[split]) > 1000:
    #         ds[split] = ds[split].shuffle(seed=42).select(range(1000))

    # for eaach split, generate NUM_ROLLOUTS_PER_QUESTION solutions
    inputs = []
    for split in ds:
        for idx, item in enumerate(ds[split]):
            if max_questions_per_split is not None and idx >= max_questions_per_split:
                break

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
                    'formatted_prompt': formatted_prompt,
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in results:
        results_df = pd.DataFrame.from_dict(results[split], orient='index')
        model_alias = model_name.replace("/", "-")
        results_df.to_parquet(f"data/MATH_{split}_{len(results[split])}-{model_alias}-maxlen_{max_response_len}_k_{num_rollouts_per_question}_temp_{temperature}.parquet")


if __name__ == "__main__":

    main(
        model_name="Qwen/Qwen2.5-Math-72B-Instruct",
        max_questions_per_split=100,
        tensor_parallel_size=8
    )


# Add command line args
# Make as a slurm script
# Run all the other permutations on h200_lowest
# - temperature 0, 0.5, 1.0
# - gpt-oss, qwen base models
# - other datasets