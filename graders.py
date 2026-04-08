from typing import Dict, Any

def grade_task(prediction: Dict[str, Any], ground_truth: Dict[str, Any], difficulty: str = "easy") -> float:
    """
    Unified grading logic for Email Triage tasks.
    Weights: Category (0.4), Priority (0.3), Department (0.3)
    """
    if not prediction or not ground_truth:
        return 0.0
    
    score = 0.0
    
    def norm(v): return str(v).strip().lower()

    # Category matching
    if norm(prediction.get("category")) == norm(ground_truth.get("category")):
        score += 0.4
    
    # Priority matching
    if norm(prediction.get("priority")) == norm(ground_truth.get("priority")):
        score += 0.3
        
    # Department matching
    if norm(prediction.get("department")) == norm(ground_truth.get("department")):
        score += 0.3
        
    return round(score, 2)

def grade_easy(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    return grade_task(prediction, ground_truth, "easy")

def grade_medium(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    return grade_task(prediction, ground_truth, "medium")

def grade_hard(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    return grade_task(prediction, ground_truth, "hard")
