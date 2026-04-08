import json
import os
import random
from typing import Dict, Any, Optional, List, Tuple
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
    def __init__(self, target_difficulty: str = "easy"):
        self.target_difficulty = target_difficulty
        self.dataset_path = f"datasets/{target_difficulty}.json"
        self.dataset = self._load_dataset()
        self.current_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.dataset)
        self.previous_action: str = "None"
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            # Fallback for validation if dataset not found
            return [{
                "subject": "System Test",
                "body": "This is a placeholder for validation.",
                "sender": "system@openenv.ai",
                "ground_truth": {"category": "Inquiry", "priority": "Low", "department": "Support"}
            }]
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
        
        self.current_idx = 0
        self.steps_taken = 0
        self.previous_action = "None"
        
        obs = self._get_obs()
        info = self._get_info()
        return obs.dict(), info

    def _get_obs(self) -> Observation:
        if self.current_idx >= len(self.dataset):
            return Observation(subject="", body="", sender="", urgency_hint="", intent_hint="")
        
        email = self.dataset[self.current_idx]
        return Observation(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            urgency_hint="Analysis pending..." if self.target_difficulty != "easy" else "Clear",
            intent_hint="Primary intent detection..."
        )

    def _get_info(self) -> Dict[str, Any]:
        email = self.dataset[self.current_idx] if self.current_idx < len(self.dataset) else None
        return {
            "ground_truth": email["ground_truth"] if email else None,
            "previous_action": self.previous_action,
            "steps_taken": self.steps_taken,
            "total_steps": self.max_steps
        }

    def state(self) -> Dict[str, Any]:
        """Returns the structured internal state."""
        return {
            "current_idx": self.current_idx,
            "steps_taken": self.steps_taken,
            "target_difficulty": self.target_difficulty,
            "previous_action": self.previous_action,
            "is_done": self.steps_taken >= self.max_steps
        }

    def step(self, action: Action) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.previous_action = str(action.dict())
        
        if self.current_idx >= len(self.dataset):
            return self._get_obs().dict(), 0.0, True, False, self._get_info()

        gt = self.dataset[self.current_idx]["ground_truth"]
        
        # Grading logic (mirrored in graders.py)
        score = 0.0
        breakdown = {}
        
        # Normalize for comparison
        def norm(v): return str(v).strip().lower()

        if norm(action.category) == norm(gt["category"]):
            score += 0.4
            breakdown["category"] = 0.4
        if norm(action.priority) == norm(gt["priority"]):
            score += 0.3
            breakdown["priority"] = 0.3
        if norm(action.department) == norm(gt["department"]):
            score += 0.3
            breakdown["department"] = 0.3

        self.current_idx += 1
        self.steps_taken += 1
        
        terminated = self.steps_taken >= self.max_steps
        truncated = False
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs.dict(), round(score, 2), terminated, truncated, info

if __name__ == "__main__":
    # Internal validation
    env = EmailTriageEnv(target_difficulty="easy")
    obs, info = env.reset()
    print(f"Reset Obs: {obs}")
    
    action = Action(category="Inquiry", priority="Low", department="Support")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step Reward: {reward}, Terminated: {terminated}")
    print(f"State: {env.state()}")