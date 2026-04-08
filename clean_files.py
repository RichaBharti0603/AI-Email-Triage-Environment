import os

files_to_delete = [
    "agent.py", "app.py", "baseline_inference.py", "grader.py", 
    "run_baseline.py", "run_checks.py", "tasks.py", 
    "test_environment.py", "test_reward.py", "train_dqn.py", 
    "verify_env_no_model.py", "verify_grader.py", "verify_openenv.py", 
    "verify_fix.py", "schemas.py"
]

for f in files_to_delete:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")
    else:
        print(f"Not found: {f}")
