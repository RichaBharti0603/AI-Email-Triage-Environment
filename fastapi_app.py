from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uvicorn
import json
from environment import EmailTriageEnv, Observation, Action

app = FastAPI(title="OpenEnv Email Triage API")

# Singleton environment instance
# In a real production scenario, you might want to handle multiple sessions
env = EmailTriageEnv(target_difficulty="easy")

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(..., description="Action to take in the environment")

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

@app.post("/reset", response_model=Dict[str, Any])
async def reset(request: Optional[ResetRequest] = None):
    seed = request.seed if request else None
    obs, info = env.reset(seed=seed)
    return {"observation": obs, "info": info}

@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    try:
        obs, reward, terminated, truncated, info = env.step(request.action)
        return StepResponse(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def get_state():
    return env.state()

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "EmailTriageEnv"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
