import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional
import random
from reward import MultiObjectiveReward

class EmailTriageEnv(gym.Env):
    """Modular Email Triage Environment using external Reward Function"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Configuration
        self.config = config or {
            'categories': ['urgent', 'normal', 'spam'],
            'priorities': ['High', 'Medium', 'Low'],
            'departments': ['Tech', 'HR', 'Sales', 'Billing']
        }
        
        # Action space: Discrete action space for simple RL (Urgent, Normal, Spam)
        # But step() prefers a dict for multi-field agents like DistilBART
        self.action_space = spaces.Discrete(3) 
        
        # Observation space
        self.observation_space = spaces.Dict({
            'email_text': spaces.Text(max_length=1000),
            'sender': spaces.Text(max_length=100),
            'timestamp': spaces.Text(max_length=50)
        })
        
        # Reward Function
        self.reward_fn = MultiObjectiveReward()
        
        # Internal state
        self.current_email_index = 0
        self.total_reward = 0
        self.done = False
        
        # Email database
        self.email_database = self._create_email_database()
        
    def _create_email_database(self):
        """Create email database with clear categories, priorities, and departments"""
        return [
            {
                'text': 'URGENT: Server down! Need immediate assistance.',
                'sender': 'admin@company.com',
                'category': 'urgent', 'priority': 'High', 'department': 'Tech'
            },
            {
                'text': 'Team meeting at 3 PM tomorrow. Please confirm attendance.',
                'sender': 'manager@company.com',
                'category': 'normal', 'priority': 'Medium', 'department': 'HR'
            },
            {
                'text': 'WIN $1000 FREE!!! Claim your prize now! Click here!',
                'sender': 'noreply@prizes.com',
                'category': 'spam', 'priority': 'Low', 'department': 'Sales'
            },
            {
                'text': 'Project deadline extended to Friday. Great news!',
                'sender': 'pm@company.com',
                'category': 'normal', 'priority': 'Medium', 'department': 'Tech'
            },
            {
                'text': 'CRITICAL: Security breach detected! Immediate action required!',
                'sender': 'security@company.com',
                'category': 'urgent', 'priority': 'High', 'department': 'Tech'
            },
            {
                'text': 'Buy cheap watches online! 70% off!!! Limited time offer!',
                'sender': 'sales@cheapstuff.com',
                'category': 'spam', 'priority': 'Low', 'department': 'Sales'
            },
            {
                'text': 'Weekly report for review. Please check the attached file.',
                'sender': 'analyst@company.com',
                'category': 'normal', 'priority': 'Medium', 'department': 'Billing'
            },
            {
                'text': 'EMERGENCY: Database corrupted! Immediate action required!',
                'sender': 'dba@company.com',
                'category': 'urgent', 'priority': 'High', 'department': 'Tech'
            }
        ]
    
    def _extract_features(self, email):
        """Extract features from email"""
        if email is None:
            return None
        return {
            'email_text': email['text'],
            'sender': email['sender'],
            'timestamp': '2024-01-15 10:00:00'
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_email_index = 0
        self.total_reward = 0
        self.done = False
        current_email = self.email_database[self.current_email_index]
        observation = self._extract_features(current_email)
        return observation, {'step': 0, 'total_emails': len(self.email_database)}
    
    def step(self, action_dict):
        """Execute one step
        action_dict: {category, priority, department} as predicted by the agent
        """
        if self.done:
            return None, 0, True, False, {'error': 'Episode finished'}
        
        current_email = self.email_database[self.current_email_index]
        
        # Calculate Reward using external modular function
        reward = self.reward_fn.calculate(action_dict, current_email)
        
        self.total_reward += reward
        self.current_email_index += 1
        self.done = self.current_email_index >= len(self.email_database)
        
        next_observation = None
        if not self.done:
            next_email = self.email_database[self.current_email_index]
            next_observation = self._extract_features(next_email)
            
        info = {
            'ground_truth': {
                'category': current_email['category'],
                'priority': current_email['priority'],
                'department': current_email['department']
            },
            'predicted': action_dict,
            'reward': reward,
            'total_reward': self.total_reward,
            'is_correct': action_dict.get('category') == current_email['category']
        }
        
        return next_observation, reward, self.done, False, info