from typing import Dict, Any

def normalize(val: Any) -> str:
    """Standardize strings for comparison."""
    if val is None: return ""
    return str(val).strip().title()

def calculate_score(pred: Dict[str, Any], gt: Dict[str, Any], 
                   weights: Dict[str, float]) -> float:
    """Calculate weighted partial correctness score."""
    score = 0.0
    fields = ['category', 'priority', 'department']
    
    for field in fields:
        if normalize(pred.get(field)) == normalize(gt.get(field)):
            score += weights.get(field, 0.0)
            
    return round(score, 2)

def grade_easy(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Easy: Focus on category and priority matching."""
    weights = {'category': 0.5, 'priority': 0.5, 'department': 0.0}
    return calculate_score(pred, gt, weights)

def grade_medium(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Medium: Equal weighting across all fields."""
    weights = {'category': 0.34, 'priority': 0.33, 'department': 0.33}
    return calculate_score(pred, gt, weights)

def grade_hard(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Hard: Emphasis on Department (correct routing is critical)."""
    weights = {'category': 0.3, 'priority': 0.2, 'department': 0.5}
    return calculate_score(pred, gt, weights)

def grade_task(prediction: Dict[str, Any], ground_truth: Dict[str, Any], difficulty: str) -> float:
    """Unified grading entry point."""
    diff = normalize(difficulty).lower()
    if diff == 'easy':
        return grade_easy(prediction, ground_truth)
    elif diff == 'medium':
        return grade_medium(prediction, ground_truth)
    elif diff == 'hard':
        return grade_hard(prediction, ground_truth)
    else:
        # Default to neutral weighting
        return grade_medium(prediction, ground_truth)
