import os
import json
import time
from typing import Dict, Any, List
from environment import EmailTriageEnv, Action
from graders import grade_task

# Diagnostic Heuristic Agent (Phase 1 Baseline)
def policy(observation: Dict[str, Any]) -> Action:
    body = observation.get("body", "").lower()
    
    # Defaults
    category = "Inquiry"
    priority = "Medium"
    department = "Support"
    
    # Decision logic based on keywords
    if any(k in body for k in ["refund", "invoice", "billing", "cost", "payment"]):
        department = "Finance"
    elif any(k in body for k in ["error", "reset", "bug", "tech", "login"]):
        department = "Tech"
    elif any(k in body for k in ["buy", "order", "sales", "demo"]):
        department = "Sales"
        
    if any(k in body for k in ["urgent", "immediately", "asap", "emergency"]):
        priority = "Urgent"
    elif any(k in body for k in ["suggestion", "maybe later"]):
        priority = "Low"
        
    if any(k in body for k in ["cancel", "disappoint", "worst", "unhappy"]):
        category = "Complaint"
    elif any(k in body for k in ["spam", "winner", "free", "prize"]):
        category = "Spam"
        priority = "Low"
        
    return Action(category=category, priority=priority, department=department)

def run_evaluation(difficulty: str, episodes: int = 1):
    env = EmailTriageEnv(target_difficulty=difficulty)
    
    # Strict OpenEnv Logging: [START]
    print(f"[START]")
    print(f"Task: {difficulty}")
    
    total_score = 0.0
    total_steps = 0
    
    for ep in range(1, episodes + 1):
        # Reset before every episode
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Agent selects action
            action = policy(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Grade step
            gt = info.get("ground_truth")
            score = grade_task(action.dict(), gt, difficulty) if gt else 0.0
            
            # Strict OpenEnv Logging: [STEP]
            print(f"[STEP]")
            print(f"Episode: {ep}")
            print(f"Action: {json.dumps(action.dict())}")
            print(f"Reward: {reward}")
            print(f"Score: {score}")
            
            total_score += score
            total_steps += 1
            obs = next_obs

    # Strict OpenEnv Logging: [END]
    avg_score = total_score / total_steps if total_steps > 0 else 0.0
    print(f"[END]")
    print(f"Average_Score: {round(avg_score, 2)}")
    print(f"Success_Rate: {1.0 if avg_score >= 0.7 else 0.0}")
    print("-" * 20)

if __name__ == "__main__":
    # Run all tasks for a comprehensive verification
    for task in ["easy", "medium", "hard"]:
        try:
            run_evaluation(task)
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
