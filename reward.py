# reward.py
import numpy as np
from typing import Dict, List, Any

class MultiObjectiveReward:
    """Advanced reward function for multi-objective email triage.
    Normalized to [0, 1] for OpenEnv compliance.
    Weights: Category (0.5), Priority (0.3), Department (0.2).
    """
    
    def __init__(self, config: Dict[str, float] = None):
        self.config = config or {
            'category_weight': 0.5,
            'priority_weight': 0.3,
            'department_weight': 0.2
        }
        self.history = []

    def calculate(self, prediction: Dict[str, str], ground_truth: Dict[str, str]) -> float:
        """Calculate total reward based on weighted objective matches"""
        reward = 0.0
        
        # 1. Category Reward (0.5)
        if str(prediction.get('category')).lower() == str(ground_truth.get('category')).lower():
            reward += self.config['category_weight']
            
        # 2. Priority Reward (0.3)
        if str(prediction.get('priority')).lower() == str(ground_truth.get('priority')).lower():
            reward += self.config['priority_weight']
            
        # 3. Department Reward (0.2)
        if str(prediction.get('department')).lower() == str(ground_truth.get('department')).lower():
            reward += self.config['department_weight']
            
        self.history.append({
            'prediction': prediction,
            'ground_truth': ground_truth,
            'reward': float(reward)
        })
        return float(reward)

    def get_statistics(self) -> Dict:
        """Get reward statistics"""
        if not self.history:
            return {}
        rewards = [h['reward'] for h in self.history]
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'total_steps': len(self.history)
        }