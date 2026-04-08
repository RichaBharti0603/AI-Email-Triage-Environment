import json
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Tuple, Union
from pydantic import BaseModel, Field

# OpenEnv Spec Models
class Observation(BaseModel):
    subject: str = Field(..., description="Subject of the email")
    body: str = Field(..., description="Body of the email")
    sender: str = Field(..., description="Sender of the email")
    urgency_hint: str = Field(..., description="A hint about the email urgency")
    intent_hint: str = Field(..., description="A hint about the email intent")

class Action(BaseModel):
    category: str = Field(..., description="Spam, Inquiry, Complaint, Request")
    priority: str = Field(..., description="Low, Medium, High, Urgent")
    department: str = Field(..., description="Sales, Support, HR, Finance, Tech")

class EmailTriageEnv(gym.Env):
    """
    OpenEnv Compliant Email Triage Environment.
    Strictly aligned with Phase 1 reset/step requirements.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, target_difficulty: str = "easy"):
        super().__init__()
        self.target_difficulty = target_difficulty
        self.dataset_path = f"datasets/{target_difficulty}.json"
        self.email_database = self._load_dataset()
        
        # Attribute alignment with user example
        self.steps = 0
        self.episode_reward = 0.0
        self.previous_action: str = "None"
        self.current_email: Dict[str, Any] = {}
        
        # Valid label sets for penalty logic
        self.valid_labels = {
            "category": {"Spam", "Inquiry", "Complaint", "Request"},
            "priority": {"Low", "Medium", "High", "Urgent"},
            "department": {"Sales", "Support", "HR", "Finance", "Tech"}
        }

        # Define spaces
        self.observation_space = spaces.Dict({
            "subject": spaces.Text(min_length=0, max_length=1000),
            "body": spaces.Text(min_length=0, max_length=5000),
            "sender": spaces.Text(min_length=0, max_length=100),
            "urgency_hint": spaces.Text(min_length=0, max_length=100),
            "intent_hint": spaces.Text(min_length=0, max_length=100)
        })
        self.action_space = spaces.Dict({
            "category": spaces.Text(min_length=1, max_length=50),
            "priority": spaces.Text(min_length=1, max_length=50),
            "department": spaces.Text(min_length=1, max_length=50)
        })

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            return [{
                "subject": "System Test",
                "body": "Placeholder for missing dataset.",
                "sender": "system@openenv.ai",
                "ground_truth": {"category": "Inquiry", "priority": "Low", "department": "Support"}
            }]
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Required seeding call
        super().reset(seed=seed)
        
        # State alignment with user example
        self.steps = 0
        self.episode_reward = 0.0
        self.previous_action = "None"
        
        # Deterministic seeding using self.np_random as requested
        idx = self.np_random.integers(0, len(self.email_database))
        self.current_email = self.email_database[idx]
        
        observation = self._get_observation()
        info = self.state()
        
        # MUST return tuple of dicts
        return observation, info

    def _get_observation(self) -> Dict[str, Any]:
        obs_obj = Observation(
            subject=self.current_email.get("subject", ""),
            body=self.current_email.get("body", ""),
            sender=self.current_email.get("sender", ""),
            urgency_hint="Pending analysis..." if self.target_difficulty != "easy" else "Clear",
            intent_hint="Primary intent detection..."
        )
        return obs_obj.dict()

    def state(self) -> Dict[str, Any]:
        """Returns structured internal state."""
        return {
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "target_difficulty": self.target_difficulty,
            "previous_action": self.previous_action,
            "current_email_id": self.current_email.get("id", "unknown")
        }

    def step(self, action: Union[Dict[str, Any], Action]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # Robust action handling
        if isinstance(action, dict):
            try:
                action_obj = Action(**action)
                penalty = 0.0
            except Exception:
                # Penalty for invalid schema
                penalty = -0.1
                action_obj = Action(category="Inquiry", priority="Low", department="Support") # Fallback
        else:
            action_obj = action
            penalty = 0.0

        self.previous_action = json.dumps(action_obj.dict())
        
        gt = self.current_email.get("ground_truth", {})
        
        # Grading logic
        step_reward = 0.0
        def norm(v): return str(v).strip().lower()

        # Category/Priority/Dept checks
        if norm(action_obj.category) == norm(gt.get("category", "")):
            step_reward += 0.4
        if norm(action_obj.priority) == norm(gt.get("priority", "")):
            step_reward += 0.3
        if norm(action_obj.department) == norm(gt.get("department", "")):
            step_reward += 0.3
        
        # Apply schema penalty if any
        step_reward += penalty
        
        self.episode_reward += step_reward
        self.steps += 1
        
        # Termination logic (single email per episode in this version for simplicity/alignment)
        # Or you can iterate through database. The example implies a single email selected on reset.
        terminated = True 
        truncated = False
        
        observation = self._get_observation()
        info = self.state()
        
        return observation, round(step_reward, 2), terminated, truncated, info

if __name__ == "__main__":
    env = EmailTriageEnv(target_difficulty="easy")
    obs, info = env.reset(seed=42)
    print(f"Reset Obs: {obs}")
    print(f"Info: {info}")
    
    action = {"category": "Inquiry", "priority": "Low", "department": "Support"}
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step Reward: {reward}, Terminated: {terminated}")