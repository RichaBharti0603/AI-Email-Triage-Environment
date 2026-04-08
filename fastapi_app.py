from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any, Optional
import uvicorn
import json
from environment import EmailTriageEnv

app = FastAPI(title="OpenEnv Email Triage API")

# Singleton environment instance
env = EmailTriageEnv(target_difficulty="easy")

@app.get("/")
async def root():
    """Root health check for validator probes."""
    return {"status": "ok", "env": "EmailTriageEnv"}

@app.post("/reset")
async def reset(payload: Optional[Dict[str, Any]] = Body(None)):
    """
    Reset endpoint that accepts empty POST body, 
    as required by the OpenEnv validator.
    """
    # Extract seed if present in payload, else None
    seed = payload.get("seed") if payload else None
    obs, info = env.reset(seed=seed)
    return {"observation": obs, "info": info}

@app.post("/step")
async def step(action: Dict[str, Any]):
    """
    Step endpoint that accepts raw dict actions to avoid 
    strict Pydantic validation failures during validation.
    """
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        return {
            "observation": obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def get_state():
    return env.state()

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
