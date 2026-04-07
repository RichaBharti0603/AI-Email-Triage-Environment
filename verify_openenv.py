import os
import yaml
from environment import EmailTriageEnv
from models import get_triage_model
from schemas import TriageAction, EmailObservation

def test_openenv_compliance():
    print("=" * 70)
    print("🔍 OPENENV COMPLIANCE VERIFICATION")
    print("=" * 70)
    
    # 1. Check openenv.yaml
    print("\n[1/5] Checking openenv.yaml...")
    if os.path.exists("openenv.yaml"):
        with open("openenv.yaml", 'r') as f:
            config = yaml.safe_load(f)
            print(f"   ✅ Found openenv.yaml: {config.get('name')} v{config.get('version')}")
            print(f"   ✅ Reward range: {config['reward']['minimum']} to {config['reward']['maximum']}")
    else:
        print("   ❌ openenv.yaml NOT FOUND")

    env = EmailTriageEnv()
    model = get_triage_model()

    # 2. Test Reset
    print("\n[2/5] Testing Reset & Observation Schema...")
    obs, info = env.reset()
    try:
        # Validate with Pydantic
        valid_obs = EmailObservation(**obs)
        print(f"   ✅ Observation matches schema: {valid_obs.dict()}")
    except Exception as e:
        print(f"   ❌ Observation SHCEMA MISMATCH: {e}")

    # 3. Test Step & Reward Range
    print("\n[3/5] Testing Step & Reward Normalization [0, 1]...")
    # Get a real prediction
    prediction = model.predict(obs['email_text'])
    print(f"   🤖 Agent Prediction: {prediction}")
    
    next_obs, reward, done, truncated, info = env.step(prediction)
    print(f"   💰 Reward received: {reward}")
    
    if 0.0 <= reward <= 1.0:
        print(f"   ✅ Reward is correctly normalized to [0, 1]")
    else:
        print(f"   ❌ Reward OUT OF RANGE: {reward}")
        
    if done:
        print(f"   ✅ Episode terminated after 1 step (max_steps=1)")
    else:
        print(f"   ❌ Episode NOT terminated (max_steps=1 enforcement failed)")

    # 4. Test Backward Compatibility (Legacy Agent)
    print("\n[4/5] Testing Backward Compatibility (Legacy Integer Action)...")
    env.reset()
    # Legacy agents send an integer (e.g. 0 for Urgent)
    legacy_action = 0 
    _, legacy_reward, _, _, legacy_info = env.step(legacy_action)
    print(f"   ✅ Legacy Action (int 0) processed.")
    print(f"   💰 Legacy Reward: {legacy_reward}")
    print(f"   🤖 Internal mapping: {legacy_info['predicted'].get('category')}")

    # 5. Check Label Capitalization
    print("\n[5/5] Verifying Label Capitalization...")
    labels = legacy_info['predicted']
    for key, val in labels.items():
        if val[0].isupper():
            print(f"   ✅ {key}: {val} (Standardized)")
        else:
            print(f"   ❌ {key}: {val} (Lowercase mismatch)")

    print("\n" + "=" * 70)
    print("✨ VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_openenv_compliance()
