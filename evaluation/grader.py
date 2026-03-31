def evaluate(preds, gts):
    total = len(preds)

    score = 0
    max_score = total * 10

    for p, g in zip(preds, gts):
        if p["label"] == g["label"]:
            score += 5
        if p["priority"] == g["priority"]:
            score += 3
        if p["department"] == g["department"]:
            score += 2

    return {
        "final_score": score / max_score
    }