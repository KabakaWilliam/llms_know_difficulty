from math_verify import parse, verify
import json
def _make_hashable(obj):
    """Convert unhashable types (like SymPy matrices) to hashable equivalents."""
    try:
        # Check if it's a SymPy matrix
        if hasattr(obj, '__class__') and 'Matrix' in obj.__class__.__name__:
            # Convert matrix to tuple of tuples
            return tuple(tuple(row) for row in obj.tolist())
        # If it's a list, convert to tuple recursively
        elif isinstance(obj, list):
            return tuple(_make_hashable(item) for item in obj)
        # Otherwise return as-is
        return obj
    except Exception:
        return obj

def parse_answers(responses):
    PARSED_ANSWER_LIST = []
    for response in responses:
        try:
            parsed_anwer = parse(response)
            # Make the parsed answer hashable if needed
            parsed_anwer = _make_hashable(parsed_anwer)
            PARSED_ANSWER_LIST.append(parsed_anwer)
        except Exception:
            # Append empty list instead of unparsed response to avoid verification bottleneck
            PARSED_ANSWER_LIST.append([])
    return PARSED_ANSWER_LIST


def verify_answer(ground_truth, response):
    """
    Verify if a response matches the ground truth.
    
    Args:
        ground_truth: The correct answer (will be parsed with $ delimiters)
        response: The model's response to verify
        
    Returns:
        1 if correct, 0 if incorrect
    """
    try:
        parsed_ans = parse(response)
        # Make the parsed answer hashable if needed
        parsed_ans = _make_hashable(parsed_ans)
    except Exception:
        parsed_ans = response
    
    try:
        parsed_gt = parse(f"${ground_truth}$")
        # Make the ground truth hashable if needed
        parsed_gt = _make_hashable(parsed_gt)
        
        if verify(parsed_gt, parsed_ans):
            return 1
        else:
            return 0
    except Exception:
        return 0

def evaluate_responses(responses, ground_truths):
    """
    Evaluate a list of responses against ground truths.
    
    Args:
        responses: List of model responses
        ground_truths: List of correct answers
        
    Returns:
        Tuple of (is_correct_list, accuracy_percentage, num_correct, total)
    """
    is_correct = [verify_answer(gt, resp) for gt, resp in zip(ground_truths, responses)]
    accuracy = sum(is_correct) / len(is_correct) * 100 if is_correct else 0.0
    return is_correct, accuracy, sum(is_correct), len(is_correct)

def load_probe_data(MODEL_NAME, PROBING_DATASET="MATH", K=1, TEMPERATURE=0.0):
    MODEL_ALIAS= "-".join(MODEL_NAME.split("/"))
    GEN_STR = f"maxlen_3000_k_{K}_temp_{TEMPERATURE}"
    PROBE_PATH = f"../probe_results/DATA/SR_DATA/{PROBING_DATASET}/{MODEL_ALIAS}_{GEN_STR}/best_probe_predictions.json"

    with open(PROBE_PATH, "r") as f:
        probe_data = json.load(f)
    return probe_data