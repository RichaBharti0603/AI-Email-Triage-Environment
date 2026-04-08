from typing import Dict, Any

def normalize(val: str) -> str:
    return val.strip().title()

def grade_easy(prediction: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    return grade_task(prediction, ground_truth, "easy")

def grade_medium(prediction: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    return grade_task(prediction, ground_truth, "medium")

def grade_hard(prediction: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    return grade_task(prediction, ground_truth, "hard")

def grade_task(prediction: Dict[str, str], ground_truth: Dict[str, str], difficulty: str) -> float:
    """
    Deterministically grades a prediction based on ground truth.
    Weights: Category (0.4), Priority (0.3), Department (0.3)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    score = 0.0
    
    # Category (0.4)
    if normalize(prediction.get("category", "")) == normalize(ground_truth.get("category", "")):
        score += 0.4
    
    # Priority (0.3)
    if normalize(prediction.get("priority", "")) == normalize(ground_truth.get("priority", "")):
        score += 0.3
        
    # Department (0.3)
    if normalize(prediction.get("department", "")) == normalize(ground_truth.get("department", "")):
        score += 0.3
        
    return round(score, 2)
