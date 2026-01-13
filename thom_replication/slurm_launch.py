import submitit
from itertools import product
from create_success_rate_datasets import main
import pprint

if __name__ == "__main__":
    temperatures = [1.0]
    models = [
        # "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        # "Qwen/Qwen2.5-1.5B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
    ]

    executor = submitit.AutoExecutor(folder="logs")

    executor.update_parameters(
        qos="h200_lowest",
        slurm_time="72:00:00",
        cpus_per_task=96,
        gpus_per_node=8,
        mem=128000,
        slurm_array_parallelism=4,  # max concurrent jobs
    )

    jobs = []
    for model, temp in product(models, temperatures):
        if temp == 0.0:
            num_rollouts_per_question = 1
        else:
            num_rollouts_per_question = 50
        if "72B" in model:
            tensor_parallel_size = 8
        else:
            tensor_parallel_size = 1

        max_questions_per_split = None

        job = executor.submit(
            main, 
            model_name=model, 
            temperature=temp,
            num_rollouts_per_question=num_rollouts_per_question,
            max_questions_per_split=max_questions_per_split,
            tensor_parallel_size=tensor_parallel_size
         )
        jobs.append(job)

        pprint.pprint(f"JOB: {job.job_id} - Model: {model}, Temp: {temp}, Rollouts: {num_rollouts_per_question}, Max Qs: {max_questions_per_split}, TP Size: {tensor_parallel_size}")

    print(f"Submitted {len(jobs)} jobs")
