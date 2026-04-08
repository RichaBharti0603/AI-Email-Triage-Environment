def grade(pred_action: dict, truth: dict) -> float:
    """
    MEDIUM Task: Classify category + priority.
    Score: 0.5 * category_match + 0.5 * priority_match.
    Normalizes to Title Case before comparison.
    """
    def normalize(val):
        return str(val).strip().title() if val else ""

    score = 0.0
    if normalize(pred_action.get("category")) == normalize(truth.get("category")):
        score += 0.5
    if normalize(pred_action.get("priority")) == normalize(truth.get("priority")):
        score += 0.5
    return score
