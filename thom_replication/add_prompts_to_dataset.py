from datasets import load_dataset

SR_DATASET = '/home/thomfoster/llms_know_difficulty/difficulty_probes/data/MATH_train_1000-Qwen-Qwen2.5-Math-7B-Instruct-TEMPERATURE=1.0.parquet'
OG_DATASET = 'DigitalLearningGmbH/MATH-lighteval'
OG_SPLIT = 'train'

ds_sr = load_dataset('parquet', data_files=SR_DATASET)['train']
ds_og = load_dataset(OG_DATASET)[OG_SPLIT]

ds_og = ds_og.select(list(range(len(ds_sr))))

problems = [d['problem'] for d in ds_og]
ds_sr = ds_sr.map(lambda x, i: {'problem': problems[i]}, with_indices=True)

solutions = [d['solution'] for d in ds_og]
ds_sr = ds_sr.map(lambda x, i: {'solution': solutions[i]}, with_indices=True)

output_dataset = SR_DATASET.replace('.parquet', '-with_prompt.parquet')
ds_sr.to_parquet(output_dataset)