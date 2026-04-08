import json
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# OpenEnv Spec Models
class Observation(BaseModel):
    subject: str = Field(..., description="Subject of the email")
    body: str = Field(..., description="Body of the email")
    sender: str = Field(..., description="Sender of the email")
    urgency_hint: str = Field(..., description="A hint about the email urgency based on body analysis")
    intent_hint: str = Field(..., description="A hint about the email intent based on body analysis")

class Action(BaseModel):
    category: str = Field(..., description="Spam, Inquiry, Complaint, Request")
    priority: str = Field(..., description="Low, Medium, High, Urgent")
    department: str = Field(..., description="Sales, Support, HR, Finance, Tech")

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]

class EmailTriageEnv:
    """
    OpenEnv Compliant Email Triage Environment.
    """
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.dataset_path = f"datasets/{difficulty}.json"
        self.dataset = self._load_dataset()
        self.current_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.dataset)

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            return []
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def reset(self) -> Observation:
        self.current_idx = 0
        self.steps_taken = 0
        return self._get_obs()

    def _get_obs(self) -> Observation:
        if self.current_idx >= len(self.dataset):
            return Observation(subject="", body="", sender="", urgency_hint="", intent_hint="")
        
        email = self.dataset[self.current_idx]
        return Observation(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            urgency_hint="Analysis pending..." if self.difficulty != "easy" else "Clear",
            intent_hint="Primary intent detection..."
        )

    def state(self) -> Dict[str, Any]:
        return {
            "current_idx": self.current_idx,
            "steps_taken": self.steps_taken,
            "difficulty": self.difficulty,
            "ground_truth": self.dataset[self.current_idx]["ground_truth"] if self.current_idx < len(self.dataset) else None
        }

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.current_idx >= len(self.dataset):
            return self._get_obs(), Reward(value=0.0, breakdown={}), True, {}

        gt = self.dataset[self.current_idx]["ground_truth"]
        
        # Internal dense reward (for training feedback)
        score = 0.0
        breakdown = {}
        
        if action.category == gt["category"]:
            score += 0.4
            breakdown["category"] = 0.4
        if action.priority == gt["priority"]:
            score += 0.3
            breakdown["priority"] = 0.3
        if action.department == gt["department"]:
            score += 0.3
            breakdown["department"] = 0.3

        reward = Reward(value=score, breakdown=breakdown)
        
        self.current_idx += 1
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
        
        obs = self._get_obs()
        info = {"ground_truth": gt}
        
        return obs, reward, done, info