import os
import json
import random
import re
from typing import Dict, Any
from openai import OpenAI
from environment import EmailTriageEnv
from grader import evaluate

def normalize_action(action: Dict[str, Any]) -> Dict[str, str]:
    """Normalize model output to Title Case for comparison."""
    normalized = {}
    for key in ["category", "priority", "department"]:
        val = action.get(key)
        if isinstance(val, str):
            normalized[key] = val.strip().capitalize()
        else:
            normalized[key] = "Normal" if key == "category" else ("Medium" if key == "priority" else "HR")
    return normalized

def extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from model output using regex."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}

def run_baseline_evaluation():
    """
    Production-grade baseline inference script for AI Email Triage.
    Evaluates model performance on a fixed slice of the dataset.
    """
    # 1. Determinism
    random.seed(42)
    
    # 2. Environment & Data Setup
    env = EmailTriageEnv()
    emails = env.email_database[:50] # Use a fixed slice of 50
    num_episodes = len(emails)
    
    # 3. API Setup
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        return

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )

    # 4. Evaluation Loop
    easy_total = 0.0
    medium_total = 0.0
    hard_total = 0.0
    
    results_history = []
    sample_entry = None

    print(f"Starting Baseline Evaluation for {num_episodes} episodes...")

    for i, email_data in enumerate(emails):
        email_text = email_data['text']
        truth = {
            "category": email_data['category'],
            "priority": email_data['priority'],
            "department": email_data['department']
        }

        try:
            # 5. Model Inference
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini", # Or any specified model
                messages=[
                    {"role": "system", "content": "You are an email triage assistant. Output ONLY a JSON object with keys: category, priority, department."},
                    {"role": "user", "content": email_text}
                ],
                temperature=0.0
            )
            raw_output = response.choices[0].message.content
        except Exception as e:
            print(f"Episode {i+1}: API Error - {e}")
            raw_output = "{}"

        # 6. Parsing & Normalization
        pred_action_raw = extract_json(raw_output)
        pred_action = normalize_action(pred_action_raw)

        # 7. Environment Interaction
        # Note: We manually set the email in the env if needed, or simply step
        # Since env.reset() is random, we'll manually feed the action and truth
        # for strict correspondence in this benchmark script.
        _, reward, _, _, info = env.step(pred_action)
        
        # 8. Grading
        scores = evaluate(pred_action, truth)
        
        easy_total += scores["easy"]
        medium_total += scores["medium"]
        hard_total += scores["hard"]

        results_history.append({
            "episode": i + 1,
            "text": email_text,
            "truth": truth,
            "predicted": pred_action,
            "scores": scores
        })

        if i == 0:
            sample_entry = results_history[-1]

        print(f"Episode {i+1}/{num_episodes}: Easy={scores['easy']}, Medium={scores['medium']}, Hard={scores['hard']}")

    # 9. Final Metrics
    if num_episodes > 0:
        avg_easy = easy_total / num_episodes
        avg_medium = medium_total / num_episodes
        avg_hard = hard_total / num_episodes
    else:
        avg_easy = avg_medium = avg_hard = 0.0

    # 10. Output Results
    print("\n=== Baseline Evaluation Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Easy:   {avg_easy:.2f}")
    print(f"Medium: {avg_medium:.2f}")
    print(f"Hard:   {avg_hard:.2f}")

    if sample_entry:
        print("\n--- Sample Prediction vs Truth ---")
        print(f"Email: {sample_entry['text']}")
        print(f"Pred:  {sample_entry['predicted']}")
        print(f"Truth: {sample_entry['truth']}")

    # 11. Persistence
    final_output = {
        "metrics": {
            "avg_easy": avg_easy,
            "avg_medium": avg_medium,
            "avg_hard": avg_hard,
            "total_episodes": num_episodes
        },
        "history": results_history
    }
    
    with open("baseline_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
        print("\nFull results saved to baseline_results.json")

if __name__ == "__main__":
    run_baseline_evaluation()
