def grade(pred_action: dict, truth: dict) -> float:
    """
    Evaluates the prediction based on the Hard task criteria.
    Objective: Full triage evaluation (category + priority + department).
    Score: Category (0.5) + Priority (0.3) + Department (0.2).
    """
    score = 0.0
    if pred_action.get("category") == truth.get("category"):
        score += 0.5
    if pred_action.get("priority") == truth.get("priority"):
        score += 0.3
    if pred_action.get("department") == truth.get("department"):
        score += 0.2
    return score
