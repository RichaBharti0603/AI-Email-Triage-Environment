# reward.py
import numpy as np
from typing import Dict, List, Any
class RewardFunction:
    """Advanced reward function with multiple components"""
    def __init__(self, config: Dict[str, float] = None):
        self.config = config or {
            'accuracy_weight': 0.7,
            'speed_weight': 0.2,
            'consistency_weight': 0.1
        }
        self.history = []
    def calculate(self, prediction: str, ground_truth: str, 
                  processing_time: float = None) -> float:
        """Calculate total reward"""
        accuracy_reward = self._accuracy_reward(prediction, ground_truth)
        speed_reward = self._speed_reward(processing_time) if processing_time else 0
        consistency_reward = self._consistency_reward(prediction)
        total_reward = (
            self.config['accuracy_weight'] * accuracy_reward +
            self.config['speed_weight'] * speed_reward +
            self.config['consistency_weight'] * consistency_reward
        )
        self.history.append({
            'prediction': prediction,
            'ground_truth': ground_truth,
            'accuracy_reward': accuracy_reward,
            'total_reward': total_reward
        })
        return total_reward
    def _accuracy_reward(self, prediction: str, ground_truth: str) -> float:
        """Calculate accuracy-based reward"""
        if prediction == ground_truth:
            if ground_truth == 'urgent':
                return 15
            return 10
        elif prediction == 'urgent' and ground_truth == 'normal':
            return 2
        elif prediction == 'spam' and ground_truth in ['urgent', 'normal']:
            return -10
        return -5
    def _speed_reward(self, processing_time: float) -> float:
        """Calculate speed-based reward"""
        optimal_time = 0.5  
        if processing_time < optimal_time:
            return -2
        elif processing_time > 2.0:
            return -1
        else:
            return 5
    def _consistency_reward(self, prediction: str) -> float:
        """Reward consistent predictions"""
        if len(self.history) < 3:
            return 0
        recent_predictions = [h['prediction'] for h in self.history[-3:]]
        if len(set(recent_predictions)) == 1:
            return 3
        return 0
    def get_statistics(self) -> Dict:
        """Get reward statistics"""
        if not self.history:
            return {}
        rewards = [h['total_reward'] for h in self.history]
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_steps': len(self.history)
        }