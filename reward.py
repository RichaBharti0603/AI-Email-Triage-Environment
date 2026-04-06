# reward.py
import numpy as np
from typing import Dict, List, Any

class MultiObjectiveReward:
    """Advanced reward function for multi-objective email triage"""
    
    def __init__(self, config: Dict[str, float] = None):
        self.config = config or {
            'category_weight': 0.6,
            'priority_weight': 0.2,
            'department_weight': 0.2,
            'critical_miss_penalty': -10.0
        }
        self.history = []

    def calculate(self, prediction: Dict[str, str], ground_truth: Dict[str, str]) -> float:
        """Calculate total reward based on all objectives"""
        reward = 0.0
        
        # 1. Category Reward
        if prediction.get('category') == ground_truth.get('category'):
            reward += 10.0 * self.config['category_weight']
        elif ground_truth.get('category') == 'urgent':
            reward += self.config['critical_miss_penalty'] # Heavy penalty for failing urgent
        else:
            reward -= 2.0
            
        # 2. Priority Reward
        if prediction.get('priority') == ground_truth.get('priority'):
            reward += 10.0 * self.config['priority_weight']
        else:
            reward -= 1.0
            
        # 3. Department Reward
        if prediction.get('department') == ground_truth.get('department'):
            reward += 10.0 * self.config['department_weight']
        else:
            reward -= 0.5
            
        self.history.append({
            'prediction': prediction,
            'ground_truth': ground_truth,
            'reward': reward
        })
        return reward

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