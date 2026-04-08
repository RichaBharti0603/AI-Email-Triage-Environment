from typing import Dict, Any

def normalize(val: Any) -> str:
    return str(val).strip().lower()

def grade_task(prediction: Dict[str, Any], ground_truth: Dict[str, Any], difficulty: str = "easy") -> float:
    """
    Unified grading interface for all tasks.
    Weights: Category (0.4), Priority (0.3), Department (0.3)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    score = 0.0
    
    # Category (0.4)
    if normalize(prediction.get("category")) == normalize(ground_truth.get("category")):
        score += 0.4
    
    # Priority (0.3)
    if normalize(prediction.get("priority")) == normalize(ground_truth.get("priority")):
        score += 0.3
        
    # Department (0.3)
    if normalize(prediction.get("department")) == normalize(ground_truth.get("department")):
        score += 0.3
        
    return round(score, 2)

def grade_easy(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    return grade_task(prediction, ground_truth, "easy")

def grade_medium(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    return grade_task(prediction, ground_truth, "medium")

def grade_hard(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    return grade_task(prediction, ground_truth, "hard")
