from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
from engine import PhysicsEngine

app = FastAPI(title="AstroDynamics Physics Engine", version="0.1.0")

# Global Engine Instance
engine = None

class Vector3(BaseModel):
    x: float
    y: float
    z: float

class StateVector(BaseModel):
    position: Vector3
    velocity: Vector3
    time: str  # ISO format

class PropagateRequest(BaseModel):
    state: StateVector
    dt: float  # Seconds

@app.on_event("startup")
async def startup_event():
    global engine
    print("Initializing Physics Engine...")
    engine = PhysicsEngine()
    print("Physics Engine Ready.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "system": "AstroDynamics Physics Engine"}

@app.post("/ephemeris")
async def get_ephemeris(body: str, time: str):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    try:
        pos, vel = engine.get_body_state(body, time)
        return {
            "body": body,
            "time": time,
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "velocity": {"x": vel[0], "y": vel[1], "z": vel[2]}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/propagate")
async def propagate_state(request: PropagateRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Convert input to list [rx, ry, rz, vx, vy, vz]
        y0 = [
            request.state.position.x,
            request.state.position.y,
            request.state.position.z,
            request.state.velocity.x,
            request.state.velocity.y,
            request.state.velocity.z
        ]
        
        # Use interpolated propagation for performance
        # Default cache step 300s (5 mins) is good for tactical. 
        # For longer durations, we might want to adjust, but 300s is safe.
        final_state = engine.propagate_interpolated(y0, request.state.time, request.dt, cache_step=300)
        
        # Calculate new time string (simple addition for now, ideally engine returns it)
        # For MVP, we just return the state. The client knows t + dt.
        
        return {
            "status": "success",
            "dt": request.dt,
            "final_state": {
                "position": {"x": final_state[0], "y": final_state[1], "z": final_state[2]},
                "velocity": {"x": final_state[3], "y": final_state[4], "z": final_state[5]}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
