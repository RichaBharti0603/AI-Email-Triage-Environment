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
    OpenEnv Compliant Email Triage Environment using Gymnasium.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, target_difficulty: str = "easy"):
        super().__init__()
        self.target_difficulty = target_difficulty
        self.dataset_path = f"datasets/{target_difficulty}.json"
        self.dataset = self._load_dataset()
        
        # Internal placeholders
        self.current_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.dataset)
        self.previous_action: str = "None"
        
        # Valid label sets for penalty logic
        self.valid_labels = {
            "category": {"Spam", "Inquiry", "Complaint", "Request"},
            "priority": {"Low", "Medium", "High", "Urgent"},
            "department": {"Sales", "Support", "HR", "Finance", "Tech"}
        }

        # Define spaces (though OpenEnv uses Pydantic/YAML, Gymnasium requires these)
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
        # gymnasium seeding
        super().reset(seed=seed)
        
        self.current_idx = 0
        self.steps_taken = 0
        self.previous_action = "None"
        
        obs = self._get_obs()
        info = self._get_info()
        return obs.dict(), info

    def _get_obs(self) -> Observation:
        if self.current_idx >= len(self.dataset):
            return Observation(
                subject="End of Dataset",
                body="End of task reached.",
                sender="system@openenv.ai",
                urgency_hint="N/A",
                intent_hint="N/A"
            )
        
        email = self.dataset[self.current_idx]
        return Observation(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            urgency_hint="Pending..." if self.target_difficulty != "easy" else "Clear",
            intent_hint="Detecting..."
        )

    def _get_info(self) -> Dict[str, Any]:
        email = self.dataset[self.current_idx] if self.current_idx < len(self.dataset) else None
        return {
            "ground_truth": email["ground_truth"] if email else None,
            "previous_action": self.previous_action,
            "steps_taken": self.steps_taken
        }

    def state(self) -> Dict[str, Any]:
        return {
            "current_idx": self.current_idx,
            "steps_taken": self.steps_taken,
            "target_difficulty": self.target_difficulty,
            "previous_action": self.previous_action,
            "is_done": self.steps_taken >= self.max_steps
        }

    def step(self, action: Union[Dict[str, Any], Action]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # Seeding check or logic could go here if needed per step
        
        # Parse action
        if isinstance(action, dict):
            try:
                action_obj = Action(**action)
            except Exception:
                # Penalty for invalid schema
                self.previous_action = str(action)
                return self._get_obs().dict(), -0.1, False, False, self._get_info()
        else:
            action_obj = action

        self.previous_action = json.dumps(action_obj.dict())
        
        if self.current_idx >= len(self.dataset):
            return self._get_obs().dict(), 0.0, True, False, self._get_info()

        gt = self.dataset[self.current_idx]["ground_truth"]
        
        # Grading logic with validity checks
        score = 0.0
        
        # Normalize and validate
        cat = action_obj.category.strip().title()
        prio = action_obj.priority.strip().title()
        dept = action_obj.department.strip().title()

        # Check for invalid labels (optional penalty)
        if cat not in self.valid_labels["category"] or \
           prio not in self.valid_labels["priority"] or \
           dept not in self.valid_labels["department"]:
            score -= 0.05 # Smaller penalty for out-of-bounds but valid schema
        
        # Accuracy checks
        if cat.lower() == gt["category"].lower():
            score += 0.4
        if prio.lower() == gt["priority"].lower():
            score += 0.3
        if dept.lower() == gt["department"].lower():
            score += 0.3

        self.current_idx += 1
        self.steps_taken += 1
        
        terminated = self.steps_taken >= self.max_steps
        truncated = False
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs.dict(), round(score, 2), terminated, truncated, info

if __name__ == "__main__":
    env = EmailTriageEnv(target_difficulty="easy")
    # Test seeding
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert obs1 == obs2, "Deterministic seeding failed!"
    print(f"Reset Obs (Seed 42): {obs1}")
    
    action = {"category": "Inquiry", "priority": "Low", "department": "Support"}
    obs, reward, terminated, _, info = env.step(action)
    print(f"Step Reward: {reward}, Terminated: {terminated}")