import os
import yaml
from environment import EmailTriageEnv
from schemas import TriageAction, EmailObservation

def test_lightweight():
    print("=" * 70)
    print("🔍 LIGHTWEIGHT ENV VERIFICATION (NO MODEL)")
    print("=" * 70)
    
    env = EmailTriageEnv()

    # 1. Test Reset
    print("\n[1/3] Reset & Observation...")
    obs, info = env.reset()
    try:
        valid_obs = EmailObservation(**obs)
        print(f"   ✅ Obs: {valid_obs.dict()}")
    except Exception as e:
        print(f"   ❌ Obs Schema Error: {e}")

    # 2. Test Step & Reward [0, 1]
    print("\n[2/3] Step & Reward [0.5, 0.3, 0.2] weights...")
    # Perfect match action
    perfect_action = {
        'category': env.current_email['category'],
        'priority': env.current_email['priority'],
        'department': env.current_email['department']
    }
    _, reward, done, _, _ = env.step(perfect_action)
    print(f"   💰 Perfect Match Reward: {reward}")
    if reward == 1.0 and done:
        print(f"   ✅ Perfect reward and single-step enforcement verified.")
    else:
        print(f"   ❌ Logic error: reward={reward}, done={done}")

    # 3. Test Backward Compatibility
    print("\n[3/3] Legacy Integer Action...")
    env.reset()
    # 0 = Urgent
    _, rew, _, _, info = env.step(0)
    print(f"   ✅ Legacy (0) mapped to: {info['predicted']['category']}")
    print(f"   💰 Legacy Reward (Category match only): {rew}")

    print("\n" + "=" * 70)
    print("✨ LIGHTWEIGHT VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_lightweight()
