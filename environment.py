# environment_fixed.py (replace your environment.py with this)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional
import random

class EmailTriageEnv(gym.Env):
    """Fixed Email Triage Environment with Correct Rewards"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Configuration
        self.config = config or {
            'categories': ['urgent', 'normal', 'spam'],
            'reward_weights': {
                'correct': 10,      # +10 for correct classification
                'wrong': -5,        # -5 for wrong classification
                'critical_miss': -15 # Extra penalty for missing urgent emails
            }
        }
        
        # Action space: 0=urgent, 1=normal, 2=spam
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Dict({
            'email_text': spaces.Text(max_length=1000),
            'sender': spaces.Text(max_length=100),
            'timestamp': spaces.Text(max_length=50),
            'email_length': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            'contains_urgent_words': spaces.Discrete(2),
            'contains_spam_words': spaces.Discrete(2)
        })
        
        # Internal state
        self.current_email_index = 0
        self.total_reward = 0
        self.done = False
        
        # Email database
        self.email_database = self._create_email_database()
        
    def _create_email_database(self):
        """Create email database with clear categories"""
        return [
            {
                'text': 'URGENT: Server down! Need immediate assistance.',
                'sender': 'admin@company.com',
                'category': 'urgent'
            },
            {
                'text': 'Team meeting at 3 PM tomorrow. Please confirm attendance.',
                'sender': 'manager@company.com',
                'category': 'normal'
            },
            {
                'text': 'WIN $1000 FREE!!! Claim your prize now! Click here!',
                'sender': 'noreply@prizes.com',
                'category': 'spam'
            },
            {
                'text': 'Project deadline extended to Friday. Great news!',
                'sender': 'pm@company.com',
                'category': 'normal'
            },
            {
                'text': 'CRITICAL: Security breach detected! Immediate action required!',
                'sender': 'security@company.com',
                'category': 'urgent'
            },
            {
                'text': 'Buy cheap watches online! 70% off!!! Limited time offer!',
                'sender': 'sales@cheapstuff.com',
                'category': 'spam'
            },
            {
                'text': 'Weekly report for review. Please check the attached file.',
                'sender': 'analyst@company.com',
                'category': 'normal'
            },
            {
                'text': 'EMERGENCY: Database corrupted! Immediate action required!',
                'sender': 'dba@company.com',
                'category': 'urgent'
            }
        ]
    
    def _extract_features(self, email):
        """Extract features from email"""
        if email is None:
            return None
            
        text = email['text'].lower()
        
        # Keyword detection
        urgent_words = ['urgent', 'critical', 'emergency', 'immediate', 'important', 'asap']
        spam_words = ['free', 'win', 'prize', 'cheap', '!!!', '$$$', 'click', 'offer']
        
        contains_urgent = any(word in text for word in urgent_words)
        contains_spam = any(word in text for word in spam_words)
        
        return {
            'email_text': email['text'],
            'sender': email['sender'],
            'timestamp': '2024-01-15 10:00:00',
            'email_length': np.array([len(text)], dtype=np.int32),
            'contains_urgent_words': int(contains_urgent),
            'contains_spam_words': int(contains_spam)
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_email_index = 0
        self.total_reward = 0
        self.done = False
        
        # Get first email
        current_email = self.email_database[self.current_email_index]
        observation = self._extract_features(current_email)
        
        return observation, {'step': 0, 'total_emails': len(self.email_database)}
    
    def step(self, action):
        """Execute one step"""
        # Check if episode is done
        if self.done:
            return None, 0, True, False, {'error': 'Episode finished'}
        
        # Get current email
        current_email = self.email_database[self.current_email_index]
        ground_truth = current_email['category']
        
        # Map action to category
        action_map = {0: 'urgent', 1: 'normal', 2: 'spam'}
        predicted = action_map[action]
        
        # Calculate reward
        reward = self._calculate_reward(predicted, ground_truth)
        self.total_reward += reward
        
        # Move to next email
        self.current_email_index += 1
        self.done = self.current_email_index >= len(self.email_database)
        
        # Get next observation
        next_observation = None
        if not self.done:
            next_email = self.email_database[self.current_email_index]
            next_observation = self._extract_features(next_email)
        
        # Info dict
        info = {
            'ground_truth': ground_truth,
            'predicted': predicted,
            'step': self.current_email_index,
            'total_reward': self.total_reward,
            'reward': reward,
            'correct': predicted == ground_truth
        }
        
        return next_observation, reward, self.done, False, info
    
    def _calculate_reward(self, predicted, ground_truth):
        """Calculate reward based on prediction"""
        if predicted == ground_truth:
            return self.config['reward_weights']['correct']
        
        # Extra penalty for missing urgent emails
        if ground_truth == 'urgent' and predicted != 'urgent':
            return self.config['reward_weights']['critical_miss']
        
        # Standard wrong prediction penalty
        return self.config['reward_weights']['wrong']