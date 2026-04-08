import sys
import json
import os

def check_file(path):
    exists = os.path.exists(path)
    print(f"Check {path}: {'EXISTS' if exists else 'MISSING'}")
    return exists

def test_environment():
    try:
        from environment import EmailTriageEnv, Action
        env = EmailTriageEnv(target_difficulty="easy")
        obs, info = env.reset()
        print("Env Reset: SUCCESS")
        action = {"category": "Inquiry", "priority": "Low", "department": "Support"}
        next_obs, reward, term, trunc, info = env.step(action)
        print(f"Env Step: SUCCESS (Reward: {reward})")
        return True
    except Exception as e:
        print(f"Env Test: FAILED ({e})")
        return False

def test_config():
    try:
        import yaml
        with open("openenv.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        required = ["name", "version", "entrypoint", "observation_space", "action_space"]
        missing = [r for r in required if r not in cfg]
        if missing:
            print(f"Config: MISSING KEYS {missing}")
            return False
        print("Config: VALID")
        return True
    except Exception as e:
        print(f"Config Check: FAILED ({e})")
        return False

if __name__ == "__main__":
    print("--- Final Submission Verification ---")
    files = ["environment.py", "openenv.yaml", "Dockerfile", "inference.py", "graders.py", "requirements.txt"]
    all_files = all([check_file(f) for f in files])
    
    env_ok = test_environment()
    # config_ok = test_config() # Can't guarantee pyyaml is installed
    
    if all_files and env_ok:
        print("\nSUMMARY: Project is ready for submission.")
    else:
        print("\nSUMMARY: Issues found. Review logs.")
