import os
import json
import time
from typing import Dict, Any, List
from environment import EmailTriageEnv, Action
from graders import grade_task

# Heuristic Agent Logic (Fallback if OpenAI Key not provided or for baseline testing)
def diagnostic_heuristic(observation: Dict[str, Any]) -> Action:
    body = observation.get("body", "").lower()
    subject = observation.get("subject", "").lower()
    
    # Default
    category = "Inquiry"
    priority = "Medium"
    department = "Support"
    
    # Keyword analysis
    if any(k in body for k in ["refund", "invoice", "price", "billing", "cost", "payment"]):
        department = "Finance"
    elif any(k in body for k in ["error", "reset", "bug", "technical", "tech", "download", "login"]):
        department = "Tech"
    elif any(k in body for k in ["buy", "order", "enterprise", "sales", "purchase", "demo"]):
        department = "Sales"
    elif any(k in body for k in ["hiring", "job", "career", "resume", "interview"]):
        department = "HR"
        
    if any(k in body for k in ["urgent", "immediately", "asap", "emergency", "broken"]):
        priority = "Urgent"
    elif any(k in body for k in ["feature request", "suggestion"]):
        priority = "Low"
        
    if any(k in body for k in ["cancel", "un happy", "disappoint", "frustrated", "worst"]):
        category = "Complaint"
    elif any(k in body for k in ["spam", "winner", "free", "lottery", "prize", "congratulations"]):
        category = "Spam"
        priority = "Low"
        
    return Action(category=category, priority=priority, department=department)

def run_evaluation(difficulty: str, num_episodes: int = 1):
    env = EmailTriageEnv(target_difficulty=difficulty)
    
    print(f"[START]")
    print(f"Task: {difficulty}")
    
    total_score = 0.0
    total_steps = 0
    
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Run heuristic agent
            action = diagnostic_heuristic(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Official Grade
            gt = info.get("ground_truth")
            score = grade_task(action.dict(), gt, difficulty) if gt else 0.0
            
            print(f"[STEP]")
            print(f"Episode: {ep}")
            print(f"Action: {json.dumps(action.dict())}")
            print(f"Reward: {reward}")
            print(f"Score: {score}")
            
            total_score += score
            total_steps += 1
            obs = next_obs

    avg_score = total_score / total_steps if total_steps > 0 else 0.0
    
    print(f"[END]")
    print(f"Average_Score: {round(avg_score, 2)}")
    print(f"Success_Rate: {1.0 if avg_score > 0.7 else 0.0}")
    print("-" * 20)

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        try:
            run_evaluation(task)
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
