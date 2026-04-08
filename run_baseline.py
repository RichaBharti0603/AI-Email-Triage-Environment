import argparse
import logging
from environment import EmailTriageEnv
from graders import grade_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BaselineEvaluator")

def run_evaluation(episodes: int, difficulty: str):
    """Run baseline agent across N episodes and report metrics."""
    env = EmailTriageEnv(config={'difficulty': difficulty, 'max_steps': 1})
    
    total_reward = 0.0
    total_score = 0.0
    success_count = 0
    
    logger.info(f"Starting baseline evaluation: {episodes} episodes, Difficulty: {difficulty}")

    for i in range(episodes):
        obs, _ = env.reset()
        
        # Simple Logic-based Baseline (Rules for priority/department)
        text = obs['email_text'].lower()
        prediction = {
            "category": "Inquiry",
            "priority": "Medium",
            "department": "Support"
        }
        
        if "help" in text or "fix" in text:
            prediction["category"] = "Request"
            prediction["priority"] = "Urgent"
            prediction["department"] = "Tech"
        elif "buy" in text or "plan" in text:
            prediction["department"] = "Sales"
        elif "refund" in text or "overcharg" in text or "bill" in text:
            prediction["department"] = "Finance"
            prediction["category"] = "Complaint"
            
        _, reward, _, _, info = env.step(prediction)
        
        score = grade_task(prediction, info['ground_truth'], difficulty)
        
        total_reward += reward
        total_score += score
        if score > 0.8: success_count += 1
        
        logger.info(f"Episode {i+1}: Reward: {reward:.2f}, Score: {score:.2f}")

    avg_reward = total_reward / episodes
    avg_score = total_score / episodes
    success_rate = (success_count / episodes) * 100

    print("\n" + "="*30)
    print(f"Results: {difficulty.upper()}")
    print("="*30)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Score:  {avg_score:.2f}")
    print(f"Success Rate:   {success_rate:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", type=str, choices=['easy', 'medium', 'hard'], default='medium')
    args = parser.parse_args()
    run_evaluation(args.episodes, args.difficulty)
