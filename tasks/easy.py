def grade(pred_action: dict, truth: dict) -> float:
    """
    Evaluates the prediction based on the Easy task criteria.
    Objective: Evaluate ONLY category classification.
    Score: 1.0 if correct, 0.0 otherwise.
    """
    if pred_action.get("category") == truth.get("category"):
        return 1.0
    return 0.0
