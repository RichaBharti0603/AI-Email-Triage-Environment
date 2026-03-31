from env.email_env import EmailEnv
from agents.llm_agent import llm_agent
from evaluation.grader import evaluate


def run_episode():
    env = EmailEnv()

    state = env.reset()

    predictions = []
    ground_truths = []

    done = False
    step_count = 0
    total_reward = 0

    while not done:
        step_count += 1

        # Agent takes action
        action = llm_agent(state)

        # Environment step
        next_state, reward, done, info = env.step(action)

        # Store results
        predictions.append(action)
        ground_truths.append(info["ground_truth"])

        total_reward += reward

        # 🔍 Debug Logging (IMPORTANT)
        print(f"\nStep {step_count}")
        print("Email:", state["email_text"][:100])
        print("Prediction:", action)
        print("Ground Truth:", info["ground_truth"])
        print("Reward:", reward)
        print("-" * 60)

        state = next_state

    return predictions, ground_truths, total_reward


def main():
    print("🚀 Starting Email RL Environment with LLM Agent...\n")

    predictions, ground_truths, total_reward = run_episode()

    # 📊 Final Evaluation
    results = evaluate(predictions, ground_truths)

    print("\n📊 FINAL RESULTS")
    print("=" * 60)
    print("Final Score:", results["final_score"])
    print("Total Reward:", total_reward)
    print("Total Steps:", len(predictions))
    print("=" * 60)


if __name__ == "__main__":
    main()