"""
app/main.py — Strict State Machine Backend (OpenEnv-Compliant)
FastAPI-based REST backend with thread-safe state management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import threading
import time

from app.env import DisasterEnv

# ─── Schemas ──────────────────────────────────────────────────────────────────

class InitSchema(BaseModel):
    cpu: float = Field(default=85.0, ge=0.0, le=100.0, description="Starting CPU usage %")
    battery: float = Field(default=30.0, ge=0.0, le=100.0, description="Starting battery level %")
    incident_description: Optional[str] = Field(
        default="System under high load", description="Text description of the incident"
    )

class StepSchema(BaseModel):
    action: str = Field(..., description="Action to apply: optimize_cpu | close_apps | throttle_gpu | hibernate_idle")

class ExplainSchema(BaseModel):
    action: str
    state_before: dict
    state_after: dict

# ─── App & State ──────────────────────────────────────────────────────────────

app = FastAPI(title="RL Thermal Manager API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared mutable state — protected by a lock
_lock = threading.Lock()
_env: DisasterEnv = DisasterEnv()
_episode_step: int = 0
_total_reward: float = 0.0
_history: list = []          # list of step dicts
_initialized: bool = False
_init_params: dict = {}

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/reset")
def reset(init: Optional[InitSchema] = None):
    """
    Reset the environment to a (possibly user-defined) initial disaster state.
    Accepts an optional InitSchema to configure starting conditions.
    """
    global _episode_step, _total_reward, _history, _initialized, _init_params

    params = init or InitSchema()

    with _lock:
        obs = _env.reset(
            cpu=params.cpu,
            battery=params.battery,
        )
        _episode_step = 0
        _total_reward = 0.0
        _history = []
        _initialized = True
        _init_params = {
            "cpu": params.cpu,
            "battery": params.battery,
            "incident_description": params.incident_description,
        }

    return {
        "status": "reset",
        "observation": obs,
        "init_params": _init_params,
    }


@app.post("/step")
def step(body: StepSchema):
    """
    Advance the environment by one step using the given action.
    Returns the new observation, reward, done flag, and info.
    """
    global _episode_step, _total_reward, _history, _initialized

    if not _initialized:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    with _lock:
        obs, reward, done, info = _env.step(body.action)
        _episode_step += 1
        _total_reward += reward

        record = {
            "step": _episode_step,
            "action": body.action,
            "observation": obs,
            "reward": round(reward, 4),
            "cumulative_reward": round(_total_reward, 4),
            "done": done,
            "info": info,
        }
        _history.append(record)

    return record


@app.get("/state")
def get_state():
    """Return the current internal state without advancing it."""
    if not _initialized:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    with _lock:
        obs = _env.get_observation()
        return {
            "observation": obs,
            "step": _episode_step,
            "total_reward": round(_total_reward, 4),
            "done": _env.is_done(),
            "init_params": _init_params,
        }


@app.get("/history")
def get_history():
    """Return the full episode history of steps."""
    with _lock:
        return {
            "episode_step": _episode_step,
            "total_reward": round(_total_reward, 4),
            "history": list(_history),
        }


@app.get("/actions")
def get_actions():
    """Return the list of valid actions."""
    return {"actions": list(_env.action_space)}


@app.post("/explain")
def explain(body: ExplainSchema):
    """
    Return a plain-language rationale for why a given action was chosen
    based on before/after state delta.
    """
    s0 = body.state_before
    s1 = body.state_after
    action = body.action

    delta_cpu = s0.get("cpu", 0) - s1.get("cpu", 0)
    delta_bat = s1.get("battery", 0) - s0.get("battery", 0)

    rationale_map = {
        "optimize_cpu": f"CPU scheduler was rebalanced. CPU dropped by {delta_cpu:.1f}%. "
                        "This action is most effective when CPU > 80%.",
        "close_apps":   f"Background applications were terminated. CPU reduced by {delta_cpu:.1f}%. "
                        "Side-effect: slight battery recovery of {delta_bat:.1f}%.",
        "throttle_gpu": f"GPU clock rate was throttled. Thermal load decreased, CPU relief: {delta_cpu:.1f}%. "
                        "Battery improved by {delta_bat:.1f}%.",
        "hibernate_idle": f"Idle processes were hibernated. CPU freed: {delta_cpu:.1f}%. "
                          "Battery conserved: {delta_bat:.1f}%.",
    }

    rationale = rationale_map.get(
        action,
        f"Action '{action}' applied. CPU Δ={delta_cpu:.1f}%, Battery Δ={delta_bat:.1f}%."
    )

    severity = "CRITICAL" if s0.get("cpu", 0) > 90 else ("HIGH" if s0.get("cpu", 0) > 75 else "MODERATE")

    return {
        "action": action,
        "rationale": rationale,
        "severity_at_decision": severity,
        "delta": {
            "cpu": round(delta_cpu, 2),
            "battery": round(delta_bat, 2),
        },
    }
