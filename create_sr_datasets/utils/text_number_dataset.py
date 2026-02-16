from torch.utils.data import Dataset
import torch
from datasets import load_dataset
import random
import hashlib

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

# Copied over from verl.utils.reward_score.math_reward for easier imports
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def try_extract_solution(sol):
    try:
        return extract_solution(sol)
    except:
        return ""

def extract_gsm8k_solution(solution_str):
    return solution_str.split("\n####")[-1].strip()

def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

LABELS = ["A", "B", "C", "D"]

def _stable_seed(s: str) -> int:
    """Deterministic seed from a string (Record ID)."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def to_mcq_with_shuffle(ex, *, global_seed: int = 0):
    """
    Produces:
      - problem: string
      - options: dict {A,B,C,D} -> answer text
      - gold_label: "A"/"B"/"C"/"D"
      - extracted_solution: gold_label (for your verifier)
    """
    q = ex["Question"]
    correct = ex["Correct Answer"]
    incorrects = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]

    # Build choice list and shuffle deterministically per question
    choices = [correct] + incorrects
    rid = str(ex.get("Record ID", ""))

    rng = random.Random(_stable_seed(f"{global_seed}:{rid}"))
    rng.shuffle(choices)

    # Assign labels
    options = {lab: txt for lab, txt in zip(LABELS, choices)}
    gold_label = next(lab for lab, txt in options.items() if txt == correct)

    problem = (
        "You are an expert scientist.\n\n"
        "Answer the following multiple choice question.\n\n"
        f"Question:\n{q}\n\n"
        "Choices:\n"
        f"(A) {options['A']}\n"
        f"(B) {options['B']}\n"
        f"(C) {options['C']}\n"
        f"(D) {options['D']}\n\n"
        "Please reason step-by-step.\n\n"
        "After your reasoning, write your final answer as:\n"
        "Final Answer: \\boxed{X}\n\n"
        "Where X is one of A, B, C, or D.\n"
        "Do not write anything after the boxed answer."
    )

    return {
        "problem": problem,
        "option_A": options["A"],
        "option_B": options["B"],
        "option_C": options["C"],
        "option_D": options["D"],
        "gold_label": gold_label,
        # 👇 Your verifier expects the correct chosen answer here
        "extracted_solution": gold_label,
        # Optional: keep the correct answer text around for debugging
        "correct_answer_text": correct,
    }


class TextNumberDataset(Dataset):
    """
    A dataset that pairs text inputs with numerical targets.

    Usage:
    ```
    train_ds = TextNumberDataset(hf_dataset=hf_dataset, hf_dataset_split="train", scores_path=train_scores_path)
    collate_fn = make_collate_fn(tokenizer, CollateConfig(max_length=max_length))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    ```
    """
    def __init__(
            self, 
            hf_dataset: str,
            hf_dataset_split: str,
            scores_path: str
    ):
        text_dataset = load_dataset(hf_dataset)[hf_dataset_split]
        scores_dataset = load_dataset('parquet', data_files=scores_path)['train']
        assert len(text_dataset) == len(scores_dataset), "Text and scores dataset must have the same length."

        self.items = []
        for text_item, score_item in zip(text_dataset, scores_dataset):
            assert score_item['ground_truth'] in text_item['solution'], "Ground truth not found in solution text."
            self.items.append((text_item['problem'], float(score_item['success_rate'])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, y = self.items[idx]
        return text, torch.tensor(y, dtype=torch.float32)