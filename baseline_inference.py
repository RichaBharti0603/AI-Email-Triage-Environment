import os
import json
import random
import re
import logging
from typing import Dict, Any
from openai import OpenAI
from environment import EmailTriageEnv
from grader import evaluate

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BaselineInference")

def run_baseline_evaluation():
    """
    Production-grade baseline inference script.
    Evaluates model performance on the updated Email Triage tasks.
    """
    # 1. Determinism
    random.seed(42)
    
    # 2. Environment Setup
    env = EmailTriageEnv()
    emails = env.email_database[:50] 
    num_episodes = len(emails)
    
    # 3. API Setup
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set.")
        return

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )

    # 4. Evaluation Loop
    easy_total = 0.0
    medium_total = 0.0
    hard_total = 0.0
    total_reward = 0.0
    
    results_history = []

    logger.info(f"Starting Baseline Evaluation for {num_episodes} episodes...")

    for i, truth in enumerate(emails):
        email_text = truth['text']

        try:
            # 5. Model Inference
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an email triage assistant. Output ONLY a JSON object with keys: category, priority, department."},
                    {"role": "user", "content": email_text}
                ],
                temperature=0.0
            )
            raw_output = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Episode {i+1}: API Error - {e}")
            raw_output = "{}"

        # 6. Step Environment (Parsing is handled internally in env.step now)
        _, reward, _, _, info = env.step(raw_output)
        
        pred_action = info['predicted']
        reward_breakdown = info['reward_breakdown']
        
        # 7. Grading (Scores for Easy, Medium, Hard)
        scores = evaluate(pred_action, truth)
        
        easy_total += scores["easy"]
        medium_total += scores["medium"]
        hard_total += scores["hard"]
        total_reward += reward

        results_history.append({
            "episode": i + 1,
            "text": email_text,
            "truth": truth,
            "predicted": pred_action,
            "reward": reward,
            "scores": scores,
            "breakdown": reward_breakdown
        })

        logger.info(f"Episode {i+1}/{num_episodes}: Reward={reward:.2f}, Scores={scores}")

    # 8. Final Metrics
    if num_episodes > 0:
        metrics = {
            "avg_easy": easy_total / num_episodes,
            "avg_medium": medium_total / num_episodes,
            "avg_hard": hard_total / num_episodes,
            "avg_reward": total_reward / num_episodes,
            "total_episodes": num_episodes
        }
    else:
        metrics = {"avg_easy": 0, "avg_medium": 0, "avg_hard": 0, "avg_reward": 0, "total_episodes": 0}

    # 9. Output & Persistence
    print("\n=== Baseline Evaluation Results ===")
    print(json.dumps(metrics, indent=4))

    # Print sample comparison for Requirement 7
    if results_history:
        sample = results_history[0]
        print("\n--- Sample Comparison (Episode 1) ---")
        print(f"Text:  {sample['text']}")
        print(f"Pred:  {sample['predicted']}")
        print(f"Truth: {sample['truth']}")
        print(f"Scores: {sample['scores']}")

    with open("baseline_results.json", "w") as f:
        json.dump({"metrics": metrics, "history": results_history}, f, indent=4)
        logger.info("Full results saved to baseline_results.json")

if __name__ == "__main__":
    run_baseline_evaluation()
