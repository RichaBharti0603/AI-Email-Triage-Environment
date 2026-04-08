import requests
import time
import subprocess
import os

def test_final_compliance():
    # Start the server in a subprocess
    proc = subprocess.Popen(["uvicorn", "fastapi_app:app", "--host", "127.0.0.1", "--port", "7860"], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    time.sleep(5) # Wait for server to start
    
    try:
        # 1. Test Root /
        print("Testing GET / ...")
        r_root = requests.get("http://127.0.0.1:7860/")
        print(f"Root Status: {r_root.status_code}")
        assert r_root.status_code == 200
        assert r_root.json()["status"] == "ok"

        # 2. Test /reset with EMPTY BODY
        # Note: requests.post with no json/data sends empty body
        print("Testing POST /reset with EMPTY BODY...")
        r_reset = requests.post("http://127.0.0.1:7860/reset")
        print(f"Reset Status: {r_reset.status_code}")
        print(f"Reset Response: {r_reset.text}")
        assert r_reset.status_code == 200
        assert "observation" in r_reset.json()

        # 3. Test /step with RAW DICT
        print("Testing POST /step...")
        action = {"category": "Inquiry", "priority": "Low", "department": "Support"}
        r_step = requests.post("http://127.0.0.1:7860/step", json=action)
        print(f"Step Status: {r_step.status_code}")
        assert r_step.status_code == 200
        
        print("\n✅ ALL FINAL COMPLIANCE CHECKS PASSED!")
        
    except Exception as e:
        print(f"\n❌ COMPLIANCE CHECK FAILED: {e}")
    finally:
        # Terminate the server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()

if __name__ == "__main__":
    test_final_compliance()
