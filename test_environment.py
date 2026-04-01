# test_enhanced.py
from environment_fixed import EmailTriageEnv
from agent_improved import ImprovedRuleBasedAgent
def run_enhanced_test():
    """Run test with improved agent and proper metrics"""
    print("=" * 70)
    print("📧 ENHANCED EMAIL TRIAGE SYSTEM TEST")
    print("=" * 70)
    env = EmailTriageEnv()
    agent = ImprovedRuleBasedAgent()
    observation, info = env.reset()
    print(f"\n📊 Environment Info:")
    print(f"   Total Emails: {info['total_emails']}")
    print(f"   Categories: urgent, normal, spam")
    print(f"   Agent Type: Enhanced Rule-Based Classifier")
    print("\n" + "-" * 70)
    print("📨 PROCESSING EMAILS")
    print("-" * 70)
    step_count = 0
    correct = 0
    total_reward = 0
    category_stats = {
        'urgent': {'total': 0, 'correct': 0},
        'normal': {'total': 0, 'correct': 0},
        'spam': {'total': 0, 'correct': 0}
    }
    while True:
        action = agent.predict(observation)
        predicted = ['URGENT', 'NORMAL', 'SPAM'][action]
        next_obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        total_reward += reward
        ground_truth = info['ground_truth'].upper()
        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1
        category = ground_truth.lower()
        category_stats[category]['total'] += 1
        if is_correct:
            category_stats[category]['correct'] += 1
        print(f"\n📧 Email #{step_count}")
        print(f"   Content: {observation['email_text'][:70]}...")
        print(f"   🤖 Agent: {predicted}")
        print(f"   ✅ Actual: {ground_truth}")
        print(f"   {'✓' if is_correct else '✗'} {'CORRECT' if is_correct else 'WRONG'}")
        print(f"   💰 Reward: {reward:+d}")
        observation = next_obs
        if done:
            break
    print("\n" + "=" * 70)
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 70)
    accuracy = (correct / step_count) * 100 if step_count > 0 else 0
    print(f"\n🎯 Overall Performance:")
    print(f"   Total Emails: {step_count}")
    print(f"   Correct Predictions: {correct}/{step_count}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Total Reward: {total_reward}")
    print(f"   Average Reward: {total_reward/step_count:.2f}")
    print(f"\n📈 Performance by Category:")
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            cat_acc = (stats['correct'] / stats['total']) * 100
            print(f"   {category.upper()}: {stats['correct']}/{stats['total']} ({cat_acc:.1f}%)")
    print(f"\n🔍 Detailed Analysis:")
    print("-" * 70)
    env.reset()
    observation, _ = env.reset()
    done = False
    misclassified = []
    while not done:
        action = agent.predict(observation)
        predicted = ['urgent', 'normal', 'spam'][action]
        _, _, done, _, info = env.step(action)
        if predicted != info['ground_truth']:
            misclassified.append({
                'email': observation['email_text'][:50],
                'predicted': predicted,
                'actual': info['ground_truth']
            })
    if misclassified:
        print("\n❌ Misclassified Emails:")
        for i, m in enumerate(misclassified[:5], 1):
            print(f"   {i}. {m['email']}...")
            print(f"      Predicted: {m['predicted']} | Actual: {m['actual']}")
    print(f"\n💡 Recommendations for Improvement:")
    if category_stats['urgent']['accuracy'] < 80:
        print("   • Improve urgent email detection - add more urgent keywords")
    if category_stats['spam']['accuracy'] < 80:
        print("   • Enhance spam detection - look for more spam patterns")
    if accuracy < 70:
        print("   • Consider adding ML-based classification")
    print("\n" + "=" * 70)
    return accuracy
if __name__ == "__main__":
    run_enhanced_test()