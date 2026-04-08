import json
import logging
from typing import Dict, Any, Optional, Union, List

import gymnasium as gym
from gymnasium import spaces

from reward import MultiObjectiveReward
from schemas import Category, Priority, Department, EmailObservation, TriageAction

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmailTriageEnv")

class EmailTriageEnv(gym.Env):
    """
    OpenEnv-compliant Email Triage Environment.
    Supports Easy, Medium, and Hard task difficulties.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.spec_name = "email-triage-env"
        self.spec_version = "1.0.0"
        
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 5)
        self.task_difficulty = self.config.get('difficulty', 'medium').lower()
        
        self.action_space = spaces.Dict({
            'category': spaces.Discrete(len(Category)),
            'priority': spaces.Discrete(len(Priority)),
            'department': spaces.Discrete(len(Department))
        })
        
        self.observation_space = spaces.Dict({
            'email_text': spaces.Text(max_length=1000),
            'previous_action': spaces.Text(max_length=50)
        })
        
        self.reward_fn = MultiObjectiveReward()
        self.email_database = self._get_full_dataset()
        self.active_emails = self._filter_dataset()
        
        self.current_email = None
        self.previous_action = "None"
        self.steps = 0
        self.episode_reward = 0.0

    def _get_full_dataset(self) -> List[Dict[str, str]]:
        """Complete dataset categorized by difficulty."""
        return [
            # EASY: Clear, single intent
            {'text': 'I want to buy 100 units of your pro plan.', 'category': 'Inquiry', 'priority': 'Medium', 'department': 'Sales', 'difficulty': 'easy'},
            {'text': 'My server is down! Help me fix it!', 'category': 'Request', 'priority': 'Urgent', 'department': 'Tech', 'difficulty': 'easy'},
            
            # MEDIUM: Slightly ambiguous
            {'text': 'I am unhappy with the service last week and want a refund.', 'category': 'Complaint', 'priority': 'High', 'department': 'Support', 'difficulty': 'medium'},
            {'text': 'How do I change my billing address in the HR portal?', 'category': 'Inquiry', 'priority': 'Medium', 'department': 'Finance', 'difficulty': 'medium'},
            
            # HARD: Noisy, multi-intent, or deceptive
            {'text': 'Subject: WINNER! You have been selected for a free cruise. Please verify your billing details at our finance office.', 'category': 'Spam', 'priority': 'Low', 'department': 'Finance', 'difficulty': 'hard'},
            {'text': 'RE: Invoice #001 - The tech support was great, but the billing department is overcharging me for the sales consultation.', 'category': 'Complaint', 'priority': 'Medium', 'department': 'Finance', 'difficulty': 'hard'}
        ]

    def _filter_dataset(self) -> List[Dict[str, str]]:
        """Filter data based on current task difficulty."""
        filtered = [e for e in self.email_database if e['difficulty'] == self.task_difficulty]
        if not filtered:
            logger.warning(f"No emails found for difficulty '{self.task_difficulty}', using full dataset.")
            return self.email_database
        return filtered

    def load_dataset(self, path: str):
        """Dynamic dataset loading from external file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.email_database = data
                self.active_emails = self._filter_dataset()
            logger.info(f"Loaded {len(data)} emails from {path}")
        except Exception as e:
            logger.error(f"Dataset load failed: {e}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.active_emails))
        self.current_email = self.active_emails[idx]
        self.steps = 0
        self.previous_action = "None"
        self.episode_reward = 0.0
        return self._get_observation(), self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "email_text": self.current_email['text'] if self.current_email else "",
            "previous_action": self.previous_action,
            "current_step": self.steps,
            "difficulty": self.task_difficulty
        }

    def _get_observation(self) -> Dict[str, str]:
        return EmailObservation(
            email_text=self.current_email['text'] if self.current_email else "",
            previous_action=self.previous_action
        ).dict()

    def step(self, action_input: Union[Dict, str, TriageAction]):
        format_penalty = 0.0
        if isinstance(action_input, str):
            action_dict, format_penalty = self.reward_fn.parse_and_normalize(action_input)
        elif isinstance(action_input, dict):
            action_dict = action_input
        else:
            action_dict = {}
            format_penalty = -0.5

        # Check validity manually vs config if needed
        reward, breakdown = self.reward_fn.calculate(action_dict, self.current_email, format_penalty=format_penalty)
        
        gt_copy = self.current_email.copy()
        self.steps += 1
        self.previous_action = str(action_dict.get('category', 'None'))
        self.episode_reward += reward
        terminated = self.steps >= self.max_steps
        
        if not terminated:
            idx = self.np_random.integers(0, len(self.active_emails))
            self.current_email = self.active_emails[idx]
        
        info = {
            'ground_truth': gt_copy,
            'predicted': action_dict,
            'reward_breakdown': breakdown,
            'step': self.steps,
            'difficulty': self.task_difficulty,
            'env_spec': f"{self.spec_name}-v{self.spec_version}"
        }
        
        return self._get_observation(), reward, terminated, False, info