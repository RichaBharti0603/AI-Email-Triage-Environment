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
    Production-grade Email Triage Environment.
    Compliant with OpenEnv and Gymnasium specifications.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.spec_name = "email-triage-env"
        self.spec_version = "3.1.0"
        
        self.config = config or {
            'categories': [e.value for e in Category],
            'priorities': [e.value for e in Priority],
            'departments': [e.value for e in Department],
            'max_steps': 5,
            'invalid_action_penalty': -0.5
        }
        
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
        self.email_database: List[Dict[str, str]] = self._get_default_dataset()
        self.current_email: Optional[Dict[str, str]] = None
        self.previous_action: str = "None"
        self.steps: int = 0
        self.max_steps: int = self.config['max_steps']
        self.episode_reward: float = 0.0

    def _get_default_dataset(self) -> List[Dict[str, str]]:
        return [
            {'text': 'I want to buy 100 units of your pro plan.', 'category': 'Inquiry', 'priority': 'Medium', 'department': 'Sales'},
            {'text': 'My server is melting down! Help!', 'category': 'Request', 'priority': 'Urgent', 'department': 'Tech'},
            {'text': 'I am unhappy with the service last week.', 'category': 'Complaint', 'priority': 'High', 'department': 'Support'},
            {'text': 'Win a free cruise by clicking here!', 'category': 'Spam', 'priority': 'Low', 'department': 'Sales'},
            {'text': 'How do I change my billing address?', 'category': 'Inquiry', 'priority': 'Medium', 'department': 'Finance'},
            {'text': 'Can you reset my password for the HR portal?', 'category': 'Request', 'priority': 'Medium', 'department': 'HR'},
            {'text': 'I want to cancel my subscription immediately.', 'category': 'Complaint', 'priority': 'High', 'department': 'Finance'},
            {'text': 'Great job on the new update, guys!', 'category': 'Inquiry', 'priority': 'Low', 'department': 'Tech'}
        ]

    def load_dataset(self, path: str):
        try:
            with open(path, 'r') as f:
                self.email_database = json.load(f)
            logger.info(f"Loaded dataset: {path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")

    def state(self) -> Dict[str, Any]:
        return {
            "email_text": self.current_email['text'] if self.current_email else "",
            "previous_action": self.previous_action,
            "current_step": self.steps,
            "episode_reward": self.episode_reward
        }

    def _get_observation(self) -> Dict[str, str]:
        return EmailObservation(
            email_text=self.current_email['text'] if self.current_email else "",
            previous_action=self.previous_action
        ).dict()

    def render(self):
        if self.current_email:
            print(f"\n[Step {self.steps}] Text: {self.current_email['text'][:100]}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.email_database))
        self.current_email = self.email_database[idx]
        self.steps = 0
        self.previous_action = "None"
        self.episode_reward = 0.0
        return self._get_observation(), self.state()

    def _validate_action(self, action: Dict[str, Any]) -> bool:
        return (action.get('category') in self.config['categories'] and
                action.get('priority') in self.config['priorities'] and
                action.get('department') in self.config['departments'])

    def step(self, action_input: Union[Dict, str, TriageAction]):
        format_penalty = 0.0
        # 1. Pipeline: Robust Parsing
        if isinstance(action_input, str):
            action_dict, format_penalty = self.reward_fn.parse_and_normalize(action_input)
        elif isinstance(action_input, dict):
            action_dict = action_input
        elif isinstance(action_input, TriageAction):
            action_dict = action_input.dict()
        else:
            action_dict = {}
            format_penalty = self.config['invalid_action_penalty']

        # 2. Pipeline: Validation & Reward
        valid_action = self._validate_action(action_dict)
        if valid_action:
            reward, breakdown = self.reward_fn.calculate(action_dict, self.current_email, format_penalty=format_penalty)
        else:
            reward = self.config['invalid_action_penalty']
            breakdown = {"invalid_action": reward}

        # 3. Pipeline: State Context Capture (BEFORE updating current_email)
        ground_truth_handled = self.current_email.copy() if self.current_email else {}
        
        # 4. Pipeline: Update State
        self.steps += 1
        self.previous_action = str(action_dict.get('category', 'None'))
        self.episode_reward += reward
        terminated = self.steps >= self.max_steps
        
        # 5. Pipeline: Transition (Sample next email ONLY if not terminated)
        if not terminated:
            idx = self.np_random.integers(0, len(self.email_database))
            self.current_email = self.email_database[idx]
        
        # 6. Pipeline: Building Returns
        observation = self._get_observation()
        info = {
            'ground_truth': ground_truth_handled,
            'predicted': action_dict,
            'reward_breakdown': breakdown,
            'valid_action': valid_action,
            'step_number': self.steps,
            'episode_reward': self.episode_reward,
            'env_spec': f"{self.spec_name}-v{self.spec_version}"
        }
        
        logger.info(f"Step {self.steps}: Reward {reward}")
        return observation, reward, terminated, False, info