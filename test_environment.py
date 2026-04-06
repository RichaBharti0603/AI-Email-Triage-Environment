# test_environment.py
from environment import EmailTriageEnv
from models import get_triage_model
from grader import EmailGrader

def run_inference_test():
    """Run test using the DistilBART TriageModel in the RL Environment"""
    print("=" * 70)
    print("📧 DISTILBART EMAIL TRIAGE INFERENCE TEST")
    print("=" * 70)
    
    env = EmailTriageEnv()
    model = get_triage_model()
    grader = EmailGrader()
    
    observation, info = env.reset()
    print(f"\n📊 Environment Info:")
    print(f"   Total Emails in Batch: {info['total_emails']}")
    print("\n" + "-" * 70)
    print("📨 PROCESSING EMAILS WITH LLM")
    print("-" * 70)
    
    step_count = 0
    total_reward = 0
    
    while True:
        # 1. Get prediction from LLM
        email_text = observation['email_text']
        prediction = model.predict(email_text)
        
        # 2. Execute step in environment
        next_obs, reward, done, truncated, info = env.step(prediction)
        
        # 3. Grade the result
        grade_res = grader.grade(prediction, info['ground_truth'])
        
        step_count += 1
        total_reward += reward
        
        is_correct = grade_res['overall_correct']
        
        print(f"\n📧 Email #{step_count}")
        print(f"   Content: {email_text[:70]}...")
        print(f"   🤖 Predicted: {prediction}")
        print(f"   ✅ Actual:    {info['ground_truth']}")
        print(f"   {'✓' if is_correct else '✗'} {'MATCH' if is_correct else 'MISMATCH'}")
        print(f"   💰 Reward: {reward:+d}")
        
        observation = next_obs
        if done:
            break
            
    print("\n" + "=" * 70)
    print("📊 FINAL PERFORMANCE REPORT (INFERENCE)")
    print("=" * 70)
    
    final_stats = grader.llm_scoring()
    print(f"\n🎯 Overall Accuracy (All fields matching): {final_stats['overall_accuracy']*100:.1f}%")
    print(f"   Total Reward Accumulated: {total_reward}")
    
    print("\n📈 Field Metrics:")
    for field, m in final_stats['field_metrics'].items():
        print(f"   {field.upper():12}: Accuracy {m['accuracy']*100:5.1f}% | F1 {m['f1_score']:.2f}")
    
    print(f"\n💡 LLM Feedback: {final_stats['llm_feedback']}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_inference_test()