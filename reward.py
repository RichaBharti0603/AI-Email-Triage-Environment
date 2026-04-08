from typing import Dict, Tuple, Any
from schemas import Category, Priority, Department

class MultiObjectiveReward:
    """
    Advanced reward function for multi-objective email triage.
    Weights: Category (0.5), Priority (0.3), Department (0.2).
    Includes penalties for missing or invalid fields.
    """
    
    def __init__(self, config: Dict[str, float] = None):
        self.config = config or {
            'category_weight': 0.5,
            'priority_weight': 0.3,
            'department_weight': 0.2
        }
        # Define allowed values from schemas to identify invalid actions
        self.allowed_categories = {e.value for e in Category}
        self.allowed_priorities = {e.value for e in Priority}
        self.allowed_departments = {e.value for e in Department}

    def calculate(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate scalar reward and return a structured breakdown.
        
        Args:
            prediction (dict): The action predicted by the agent.
            ground_truth (dict): The correct labels for the email.
            
        Returns:
            tuple: (final_reward: float, breakdown: dict)
        """
        if not prediction:
            # Completely empty action
            penalty = -0.3
            breakdown = {
                "components": {"category": 0.0, "priority": 0.0, "department": 0.0},
                "penalty": penalty,
                "total": penalty
            }
            return max(0.0, min(1.0, penalty)), breakdown

        components = {"category": 0.0, "priority": 0.0, "department": 0.0}
        penalty = 0.0
        
        fields = [
            ('category', 'category_weight', self.allowed_categories),
            ('priority', 'priority_weight', self.allowed_priorities),
            ('department', 'department_weight', self.allowed_departments)
        ]
        
        for field, weight_key, allowed_values in fields:
            val = prediction.get(field)
            
            # Check for missing field
            if field not in prediction or val is None:
                penalty -= 0.1
                continue
                
            # Check for invalid field
            if val not in allowed_values:
                penalty -= 0.2
                continue
            
            # Check for correct value (case-insensitive)
            truth_val = ground_truth.get(field)
            if str(val).lower() == str(truth_val).lower():
                components[field] = self.config[weight_key]

        total_score = sum(components.values())
        final_total = total_score + penalty
        
        # Clip to [0.0, 1.0]
        final_reward = max(0.0, min(1.0, final_total))
        
        breakdown = {
            "components": components,
            "penalty": round(penalty, 2),
            "total": round(final_total, 2)
        }
        
        return float(final_reward), breakdown