def grade(pred_action: dict, truth: dict) -> float:
    """
    EASY Task: Classify category (Spam, Inquiry, Complaint, Request).
    Score: 1.0 if correct, 0.0 otherwise.
    Normalizes to Title Case before comparison.
    """
    def normalize(val):
        return str(val).strip().title() if val else ""

    if normalize(pred_action.get("category")) == normalize(truth.get("category")):
        return 1.0
    return 0.0
