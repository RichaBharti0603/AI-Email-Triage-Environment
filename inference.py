import os
import json
import argparse
import time
from typing import Dict, Any, List
from environment import EmailTriageEnv, Action
from graders import grade_task

# Diagnostic Heuristic Agent
def policy(observation: Dict[str, Any]) -> Action:
    body = observation.get("body", "").lower()
    cat, prio, dept = "Inquiry", "Medium", "Support"
    
    if any(k in body for k in ["refund", "billing", "invoice", "cost"]):
        dept = "Finance"
    elif any(k in body for k in ["error", "bug", "tech", "login", "broken"]):
        dept = "Tech"
    elif any(k in body for k in ["demo", "sales", "enterprise", "plan"]):
        dept = "Sales"
        
    if any(k in body for k in ["urgent", "immediately", "asap", "fire"]):
        prio = "Urgent"
    elif any(k in body for k in ["suggestion", "future"]):
        prio = "Low"
        
    if any(k in body for k in ["cancel", "hate", "unhappy", "terrible"]):
        cat = "Complaint"
    elif any(k in body for k in ["spam", "free", "winner", "prize"]):
        cat = "Spam"
        prio = "Low"
        
    return Action(category=cat, priority=prio, department=dept)

def run_evaluation(difficulty: str, episodes: int):
    # Root placement of environment guaranteed in Docker
    env = EmailTriageEnv(target_difficulty=difficulty)
    
    # Strict OpenEnv Logging: [START]
    print(f"[START]")
    print(f"Difficulty: {difficulty}")
    print(f"Episodes: {episodes}")
    
    total_score = 0.0
    total_steps = 0
    
    for ep in range(1, episodes + 1):
        # Reset with deterministic seed for reproducibility
        obs, info = env.reset(seed=42)
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select action
            action = policy(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Grade step
            gt = info.get("ground_truth")
            score = grade_task(action.dict(), gt, difficulty) if gt else 0.0
            
            # Strict OpenEnv Logging: [STEP]
            print(f"[STEP] Episode:{ep} Reward:{reward} Score:{score}")
            
            total_score += score
            total_steps += 1

    # Strict OpenEnv Logging: [END]
    avg_score = total_score / total_steps if total_steps > 0 else 0.0
    print(f"[END]")
    print(f"Average_Score: {round(avg_score, 2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Email Triage Inference")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    
    run_evaluation(args.difficulty, args.episodes)
