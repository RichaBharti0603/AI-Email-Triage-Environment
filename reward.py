import re
import json
from typing import Dict, Tuple, Any, Optional
from schemas import Category, Priority, Department

class MultiObjectiveReward:
    """
    Production-grade reward function for AI Email Triage.
    Weights: Category (0.3), Priority (0.3), Department (0.4).
    Includes strict penalties for format and structural errors.
    """
    
    def __init__(self, config: Dict[str, float] = None):
        self.config = config or {
            'category_weight': 0.3,
            'priority_weight': 0.3,
            'department_weight': 0.4
        }
        self.allowed_categories = {e.value for e in Category}
        self.allowed_priorities = {e.value for e in Priority}
        self.allowed_departments = {e.value for e in Department}

    def parse_and_normalize(self, raw_output: str) -> Tuple[Dict[str, Any], float]:
        """
        Extract JSON using regex and normalize fields to Title Case.
        Returns (parsed_dict, penalty).
        """
        penalty = 0.0
        
        if not raw_output or not raw_output.strip():
            return {}, -0.5 # Empty response penalty
            
        try:
            # Regex extraction: r"\{.*\}"
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if not match:
                return {}, -1.0 # Invalid JSON penalty
            
            parsed = json.loads(match.group())
            
            # Normalize to Title Case
            normalized = {}
            for k, v in parsed.items():
                if isinstance(v, str):
                    normalized[k.lower()] = v.strip().title()
                else:
                    normalized[k.lower()] = v
            
            return normalized, 0.0
        except Exception:
            return {}, -1.0 # Parsing/JSON failure penalty

    def calculate(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                  format_penalty: float = 0.0, repeat_penalty: float = 0.0) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate scalar reward and return a structured breakdown.
        
        Args:
            prediction (dict): The parsed and normalized action.
            ground_truth (dict): Correct labels.
            format_penalty (float): Penalty from parsing (Invalid JSON or Empty).
            repeat_penalty (float): Penalty for repetitive attempts.
            
        Returns:
            tuple: (final_reward: float, breakdown: dict)
        """
        components = {"category": 0.0, "priority": 0.0, "department": 0.0}
        
        # Weighted matching
        if prediction.get("category") == ground_truth.get("category"):
            components["category"] = self.config['category_weight']
            
        if prediction.get("priority") == ground_truth.get("priority"):
            components["priority"] = self.config['priority_weight']
            
        if prediction.get("department") == ground_truth.get("department"):
            components["department"] = self.config['department_weight']

        total_score = sum(components.values())
        total_penalty = format_penalty + repeat_penalty
        
        final_total = total_score + total_penalty
        
        # Reward is usually clipped to [0, 1] but penalties can make it negative in some systems.
        # User specified "Ensure reward is always in [0, 1]" in previous turns, 
        # but here they just say "Return reward at each step".
        # I'll clip it to [0, 1] for safety as per standard OpenEnv/RL practices.
        final_reward = max(0.0, min(1.0, final_total))
        
        breakdown = {
            "components": components,
            "penalty": round(total_penalty, 2),
            "total": round(final_total, 2)
        }
        
        return float(final_reward), breakdown