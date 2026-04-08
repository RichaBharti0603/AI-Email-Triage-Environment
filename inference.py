import os
import json
import time
from typing import Dict, Any
from environment import EmailTriageEnv, Action
from graders import grade_task

# Heuristic Agent Logic
def heuristic_agent(observation) -> Action:
    body = observation.body.lower()
    subject = observation.subject.lower()
    
    # Default
    category = "Inquiry"
    priority = "Medium"
    department = "Support"
    
    # Heuristics
    if any(k in body for k in ["refund", "invoice", "price", "billing", "cost"]):
        department = "Finance"
    elif any(k in body for k in ["error", "reset", "bug", "technical", "tech", "download", "portal"]):
        department = "Tech"
    elif any(k in body for k in ["buy", "order", "enterprise", "pro plan", "sales", "purchase"]):
        department = "Sales"
        
    if any(k in body for k in ["urgent", "immediately", "asap", "fix this", "cancel"]):
        priority = "Urgent"
        category = "Complaint"
    elif any(k in body for k in ["spam", "winner", "free", "cruise"]):
        category = "Spam"
        priority = "Low"
        
    return Action(category=category, priority=priority, department=department)

def run_evaluation(difficulty: str, num_episodes: int):
    env = EmailTriageEnv(difficulty=difficulty)
    
    print(f"[START]")
    print(f"task: {difficulty}")
    print(f"episodes: {num_episodes}")
    
    total_reward = 0.0
    total_score = 0.0
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        
        while not done:
            # Baseline Heuristic Agent
            action = heuristic_agent(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Use deterministic grader for official score
            score = grade_task(action.dict(), info["ground_truth"], difficulty)
            
            print(f"[STEP]")
            print(f"episode: {ep}")
            print(f"action: {action.dict()}")
            print(f"reward: {reward.value}")
            print(f"score: {score}")
            
            total_reward += reward.value
            total_score += score
            obs = next_obs

    avg_reward = total_reward / num_episodes
    avg_score = total_score / num_episodes
    success_rate = 1.0 if avg_score > 0.7 else 0.0 # Example metric
    
    print(f"[END]")
    print(f"avg_reward: {avg_reward}")
    print(f"avg_score: {avg_score}")
    print(f"success_rate: {success_rate}")
    print("-" * 20)
    
    return avg_reward, avg_score

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_evaluation(task, 2) # Running 2 episodes per task for demo
