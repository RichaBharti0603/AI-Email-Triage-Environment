import os
import json
import time
from typing import Dict, Any, List
from environment import EmailTriageEnv, Action
from graders import grade_task

# Diagnostic Policy (Baseline Heuristic)
def policy(observation: Dict[str, Any]) -> Action:
    """Heuristic rule-based agent as fallback for baseline evaluation."""
    body = observation.get("body", "").lower()
    
    cat, prio, dept = "Inquiry", "Medium", "Support"
    
    # Department routing
    if any(k in body for k in ["refund", "billing", "invoice", "cost"]):
        dept = "Finance"
    elif any(k in body for k in ["error", "bug", "tech", "login", "broken"]):
        dept = "Tech"
    elif any(k in body for k in ["demo", "sales", "enterprise", "plan"]):
        dept = "Sales"
    elif any(k in body for k in ["hiring", "apply", "job", "hr"]):
        dept = "HR"
        
    # Priority scaling
    if any(k in body for k in ["urgent", "immediately", "asap", "fire"]):
        prio = "Urgent"
    elif any(k in body for k in ["suggestion", "future"]):
        prio = "Low"
        
    # Category detection
    if any(k in body for k in ["cancel", "hate", "unhappy", "terrible"]):
        cat = "Complaint"
    elif any(k in body for k in ["spam", "free", "winner", "prize"]):
        cat = "Spam"
        prio = "Low"
        
    return Action(category=cat, priority=prio, department=dept)

def run_evaluation(difficulty: str, seed: int = 42):
    env = EmailTriageEnv(target_difficulty=difficulty)
    
    # Strict OpenEnv Logging: [START]
    print(f"[START]")
    print(f"Task: {difficulty}")
    
    # Reset with deterministic seed for reproducible Phase 1 checks
    obs, info = env.reset(seed=seed)
    
    total_score = 0.0
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    episodes_run = 0
    max_episodes = 1 # OpenEnv standard often evaluates one sequence per task
    
    while episodes_run < max_episodes:
        # Step through the dataset
        while not (terminated or truncated):
            # Select action via heuristic policy
            action = policy(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Grade via official grader
            gt = info.get("ground_truth")
            score = grade_task(action.dict(), gt, difficulty) if gt else 0.0
            
            # Strict OpenEnv Logging: [STEP]
            print(f"[STEP]")
            print(f"Action: {json.dumps(action.dict())}")
            print(f"Reward: {reward}")
            print(f"Score: {score}")
            
            total_score += score
            total_reward += reward
            steps += 1
            
        episodes_run += 1
        if episodes_run < max_episodes:
            obs, info = env.reset() # No seed on subsequent resets of the same batch

    # Strict OpenEnv Logging: [END]
    avg_score = total_score / steps if steps > 0 else 0.0
    print(f"[END]")
    print(f"Average_Score: {round(avg_score, 2)}")
    print(f"Average_Reward: {round(total_reward / steps, 2) if steps > 0 else 0.0}")
    print("-" * 20)

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        try:
            run_evaluation(task)
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            import traceback
            traceback.print_exc()
