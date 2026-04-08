import random
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Union
import gymnasium as gym
from gymnasium import spaces

from reward import MultiObjectiveReward
from schemas import Category, Priority, Department, EmailObservation, TriageAction

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmailTriageEnv")

class EmailTriageEnv(gym.Env):
    """
    Production-grade Email Triage Environment.
    Features strict determinism, robust parsing, and observability.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 1. Determinism & Reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # OpenEnv Metadata
        self.spec_name = "email-triage-env"
        self.spec_version = "2.0.0"
        
        # Configuration
        self.config = config or {
            'categories': [e.value for e in Category],
            'priorities': [e.value for e in Priority],
            'departments': [e.value for e in Department]
        }
        
        # Action space
        self.action_space = spaces.Dict({
            'category': spaces.Discrete(len(Category)),
            'priority': spaces.Discrete(len(Priority)),
            'department': spaces.Discrete(len(Department))
        })
        
        # Observation space
        self.observation_space = spaces.Dict({
            'email_text': spaces.Text(max_length=1000),
            'previous_action': spaces.Text(max_length=50)
        })
        
        # Reward Function
        self.reward_fn = MultiObjectiveReward()
        
        # Internal state
        self.current_email = None
        self.previous_action = None
        self.steps = 0
        self.max_steps = 1
        
        # 2. Dataset Setup (Updated Labels)
        self.email_database = [
            {'text': 'I want to buy 100 units of your pro plan.', 'category': 'Inquiry', 'priority': 'Medium', 'department': 'Sales'},
            {'text': 'My server is melting down! Help!', 'category': 'Request', 'priority': 'Urgent', 'department': 'Tech'},
            {'text': 'I am unhappy with the service last week.', 'category': 'Complaint', 'priority': 'High', 'department': 'Support'},
            {'text': 'Win a free cruise by clicking here!', 'category': 'Spam', 'priority': 'Low', 'department': 'Sales'},
            {'text': 'How do I change my billing address?', 'category': 'Inquiry', 'priority': 'Medium', 'department': 'Finance'},
            {'text': 'Can you reset my password for the HR portal?', 'category': 'Request', 'priority': 'Medium', 'department': 'HR'},
            {'text': 'I want to cancel my subscription immediately.', 'category': 'Complaint', 'priority': 'High', 'department': 'Finance'},
            {'text': 'Great job on the new update, guys!', 'category': 'Inquiry', 'priority': 'Low', 'department': 'Tech'}
        ]
        
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        if seed is not None:
             random.seed(seed)
             np.random.seed(seed)
             torch.manual_seed(seed)
             
        self.current_email = random.choice(self.email_database)
        self.steps = 0
        
        observation = EmailObservation(
            email_text=self.current_email['text'],
            previous_action=self.previous_action
        ).dict()
        
        logger.info(f"Environment reset. New Email: {self.current_email['text'][:50]}...")
        return observation, {'step': 0}
    
    def step(self, action_input: Union[Dict, str, TriageAction]):
        """Execute one step with robust parsing and logging."""
        
        # 3. Robust Parsing
        if isinstance(action_input, str):
            # If raw string (e.g. from LLM), parse it
            action_dict, format_penalty = self.reward_fn.parse_and_normalize(action_input)
        elif isinstance(action_input, dict):
            action_dict = action_input
            format_penalty = 0.0
        elif isinstance(action_input, TriageAction):
            action_dict = action_input.dict()
            format_penalty = 0.0
        else:
            action_dict = {}
            format_penalty = -1.0
            
        # 4. Reward Calculation
        # For simplicity, we assume no repeat penalty in this single-step env
        reward, breakdown = self.reward_fn.calculate(
            action_dict, self.current_email, format_penalty=format_penalty
        )
        
        self.steps += 1
        done = True
        self.previous_action = action_dict.get('category')
        
        info = {
            'ground_truth': self.current_email,
            'predicted': action_dict,
            'reward_breakdown': breakdown,
            'step': self.steps
        }
        
        logger.info(f"Step {self.steps}: Reward={reward}, Predicted={action_dict}")
        
        return None, reward, done, False, info