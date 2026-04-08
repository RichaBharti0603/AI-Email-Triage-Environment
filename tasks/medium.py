def grade(pred_action: dict, truth: dict) -> float:
    """
    Evaluates the prediction based on the Medium task criteria.
    Objective: Evaluate category + priority.
    Score: Category (0.5) + Priority (0.5).
    """
    score = 0.0
    if pred_action.get("category") == truth.get("category"):
        score += 0.5
    if pred_action.get("priority") == truth.get("priority"):
        score += 0.5
    return score
