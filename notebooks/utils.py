from math_verify import parse, verify

def parse_answers(responses):
    PARSED_ANSWER_LIST = []
    for response in responses:
        try:
            parsed_anwer = parse(response)
            PARSED_ANSWER_LIST.append(parsed_anwer)
        except Exception:
            PARSED_ANSWER_LIST.append(response)
            print("\n***** can't parse answer so appending whole response ****** \n")
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
    except Exception:
        parsed_ans = response
    
    try:
        if verify(parse(f"${ground_truth}$"), parsed_ans):
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