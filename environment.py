import random
from typing import Dict, Any, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from reward import MultiObjectiveReward
from schemas import Category, Priority, Department, EmailObservation, TriageAction

class EmailTriageEnv(gym.Env):
    """Modular Email Triage Environment using external Reward Function"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # OpenEnv Metadata
        self.spec_name = "email-triage-env"
        self.spec_version = "1.0.0"
        
        # Configuration
        self.config = config or {
            'categories': [e.value for e in Category],
            'priorities': [e.value for e in Priority],
            'departments': [e.value for e in Department]
        }
        
        # Action space: Mapped from TriageAction schema
        # We also support legacy int actions (0=Urgent, 1=Normal, 2=Spam)
        self.action_space = spaces.Dict({
            'category': spaces.Discrete(len(Category)),
            'priority': spaces.Discrete(len(Priority)),
            'department': spaces.Discrete(len(Department))
        })
        
        # Observation space: matching openenv.yaml
        self.observation_space = spaces.Dict({
            'email_text': spaces.Text(max_length=1000),
            'previous_action': spaces.Text(max_length=50) # Nullable in spec
        })
        
        # Reward Function
        self.reward_fn = MultiObjectiveReward()
        
        # Internal state
        self.current_email = None
        self.previous_action = None
        self.steps = 0
        self.max_steps = 1 # Single-step episodes as per OpenEnv spec
        
        # Email database
        self.email_database = self._create_email_database()
        
    def _create_email_database(self):
        """Create email database with standardized labels"""
        return [
            {
                'text': 'URGENT: Server down! Need immediate assistance.',
                'category': 'Urgent', 'priority': 'High', 'department': 'Tech'
            },
            {
                'text': 'Team meeting at 3 PM tomorrow. Please confirm attendance.',
                'category': 'Normal', 'priority': 'Medium', 'department': 'HR'
            },
            {
                'text': 'WIN $1000 FREE!!! Claim your prize now! Click here!',
                'category': 'Spam', 'priority': 'Low', 'department': 'Sales'
            },
            {
                'text': 'Project deadline extended to Friday. Great news!',
                'category': 'Normal', 'priority': 'Medium', 'department': 'Tech'
            },
            {
                'text': 'CRITICAL: Security breach detected! Immediate action required!',
                'category': 'Urgent', 'priority': 'High', 'department': 'Tech'
            },
            {
                'text': 'Buy cheap watches online! 70% off!!! Limited time offer!',
                'category': 'Spam', 'priority': 'Low', 'department': 'Sales'
            },
            {
                'text': 'Weekly report for review. Please check the attached file.',
                'category': 'Normal', 'priority': 'Medium', 'department': 'Billing'
            },
            {
                'text': 'EMERGENCY: Database corrupted! Immediate action required!',
                'category': 'Urgent', 'priority': 'High', 'department': 'Tech'
            }
        ]
    
    def _extract_features(self, email):
        """Extract features matching EmailObservation schema"""
        if email is None:
            return None
        return EmailObservation(
            email_text=email['text'],
            previous_action=self.previous_action
        ).dict()
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new single-step episode"""
        super().reset(seed=seed)
        
        # Pick a random email for the new episode
        self.current_email = random.choice(self.email_database)
        self.steps = 0
        
        # We persist previous_action across episodes but it's None for new runs
        observation = self._extract_features(self.current_email)
        return observation, {'step': 0}
    
    def step(self, action: Union[Dict, int, TriageAction]):
        """Execute one step with backward compatibility for legacy agents"""
        
        # 1. Handle Backward Compatibility for Legacy Agents
        if isinstance(action, int):
            # Map legacy discrete action to full TriageAction
            cat_list = [e.value for e in Category]
            category_val = cat_list[action] if action < len(cat_list) else Category.NORMAL.value
            action_dict = {
                'category': category_val,
                'priority': Priority.LOW.value,
                'department': Department.HR.value
            }
        elif isinstance(action, TriageAction):
            action_dict = action.dict()
        else:
            action_dict = action

        # 2. Process Step
        reward = self.reward_fn.calculate(action_dict, self.current_email)
        self.steps += 1
        done = True # Enforce single-step
        
        # Update persistent state
        self.previous_action = action_dict.get('category')
        
        info = {
            'ground_truth': {
                'category': self.current_email['category'],
                'priority': self.current_email['priority'],
                'department': self.current_email['department']
            },
            'predicted': action_dict,
            'reward': reward,
            'is_correct': str(action_dict.get('category')).lower() == str(self.current_email['category']).lower()
        }
        
        return None, reward, done, False, info