import requests
import time
import subprocess
import os
import signal

def test_api():
    # Start the server in a subprocess
    proc = subprocess.Popen(["uvicorn", "fastapi_app:app", "--host", "127.0.0.1", "--port", "7860"], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    time.sleep(5) # Wait for server to start
    
    try:
        # Test /reset POST
        print("Testing POST /reset...")
        resp = requests.post("http://127.0.0.1:7860/reset", json={"seed": 42})
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        
        assert resp.status_code == 200
        assert "observation" in resp.json()
        print("API Test Passed!")
        
    except Exception as e:
        print(f"API Test Failed: {e}")
    finally:
        # Terminate the server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()

if __name__ == "__main__":
    test_api()
