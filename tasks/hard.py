def grade(pred_action: dict, truth: dict) -> float:
    """
    HARD Task: Classify category + priority + department.
    Score: (category_match + priority_match + department_match) / 3.
    Normalizes to Title Case before comparison.
    """
    def normalize(val):
        return str(val).strip().title() if val else ""

    matches = 0
    if normalize(pred_action.get("category")) == normalize(truth.get("category")):
        matches += 1
    if normalize(pred_action.get("priority")) == normalize(truth.get("priority")):
        matches += 1
    if normalize(pred_action.get("department")) == normalize(truth.get("department")):
        matches += 1
    
    return matches / 3.0
