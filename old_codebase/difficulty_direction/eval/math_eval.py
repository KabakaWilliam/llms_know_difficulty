import os
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from datasets import load_dataset
import pandas as pd
import torch
import torch.nn.functional as F

from difficulty_direction.config import Config
from difficulty_direction.model_wrapper import ModelBase
from difficulty_direction.utils import ceildiv, chunks, save_to_json_file
from math_verify import parse, verify


def verify_math_answer(gold_raw, answer_raw):
    gold = parse(f"${gold_raw}$")
    answer = parse(f"${answer_raw}$")

    # Order here is important!
    # True if correct
    return verify(gold, answer)


def batch_verify_math_answers(answers: List[str], completions: List[str], batch_size: int = 32) -> List[int]:
    is_correct_list = []
    # Loop over the inputs in batches
    for i in tqdm(range(0, len(answers), batch_size), desc="Processing Batches"):
        batch_answers = answers[i:i + batch_size]
        batch_completions = completions[i:i + batch_size]
        # Process each batch element one by one
        for gold, answer in zip(batch_answers, batch_completions):
            is_correct = verify_math_answer(gold, answer)
            is_correct_list.append(is_correct)
    return is_correct_list



def evaluate_Math(
    cfg: Config, model: ModelBase, 
    intervention_label: str,
    tasks: List[str] = ["AMC",], #Add GSM8K later

):
    # inside  runs/{model_alias}/evaluation
    save_dir = cfg.artifact_path() / "evaluation"
    completions_dir =  ".." / cfg.artifact_path() / "completions"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(completions_dir, exist_ok=True)

    results = {}

    for task in tasks:
        COMPLETION_PATH = f"{task}_{intervention_label}_completions.json"

        # check if we already have the completions. If so, no need to generate any new ones
        if not os.path.exists(COMPLETION_PATH):
            # task_inputs is essentially json loaded into a dictionaty
            # if there are no completions, we have to generate the completions
            task_inputs = prepare_task_inputs(model, task)
        else:
            pass
            # need function to load the json 

        RESULT_LIST = batch_verify_math_answers(answers=task_inputs["answers"], completions=task_inputs["completions"], batch_size = cfg.batch_size)
        acc = sum(RESULT_LIST)/len(RESULT_LIST)
        results[task] = acc
        print(f"Task: {task}, Intervention: {intervention_label}, Accuracy={acc:.3f}")

        # need to update the is_correct field for each completion
        task_inputs["is_correct"] = RESULT_LIST
        
        save_to_json_file(save_dir / f'{intervention_label}.json', results)


# the eval results should be saved in runs/{model_alias}/evaluation
# as {eval_dataset_name}_{intervention_label}_eval_results.json


# the completion results should also be saved in runs/{model_alias}/completions
# as {eval_dataset_name}_{intervention_label}_completions.json