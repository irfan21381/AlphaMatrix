"""
app/main.py — AlphaMatrix FastAPI Backend
Thread-safe state machine. Runs on port 8000.
All endpoints return OpenEnv-compliant JSON.
"""

import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.env import AlphaMatrixEnv, TASKS, ACTIONS

# ── Schemas ───────────────────────────────────────────────────────────────────

class ResetSchema(BaseModel):
    task: str = Field(default="thermal_throttling",
                      description=f"One of: {TASKS}")

class StepSchema(BaseModel):
    action: str

class ExplainSchema(BaseModel):
    task:        str
    action:      str
    state_before: dict
    state_after:  dict

# ── Shared state ──────────────────────────────────────────────────────────────

_lock          = threading.Lock()
_env           = AlphaMatrixEnv()
_step_count    = 0
_total_reward  = 0.0
_history: list = []
_initialized   = False
_current_task  = "thermal_throttling"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AlphaMatrix OpenEnv API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS, "actions": ACTIONS}


@app.post("/reset")
def reset(body: Optional[ResetSchema] = None):
    global _step_count, _total_reward, _history, _initialized, _current_task
    task = (body.task if body else "thermal_throttling")
    if task not in TASKS:
        raise HTTPException(400, f"Unknown task '{task}'. Valid: {TASKS}")

    with _lock:
        obs           = _env.reset(task=task)
        _step_count   = 0
        _total_reward = 0.0
        _history      = []
        _initialized  = True
        _current_task = task

    return {"status": "reset", "task": task, "observation": obs}


@app.post("/step")
def step(body: StepSchema):
    global _step_count, _total_reward, _history
    if not _initialized:
        raise HTTPException(400, "Call /reset first.")

    with _lock:
        result        = _env.step(body.action)   # OpenEnv dict
        _step_count  += 1
        _total_reward += result["reward"]
        record = {
            "step":              _step_count,
            "action":            body.action,
            "observation":       result["observation"],
            "reward":            result["reward"],
            "cumulative_reward": round(_total_reward, 6),
            "done":              result["done"],
            "info":              result["info"],
        }
        _history.append(record)

    return record


@app.get("/state")
def state():
    if not _initialized:
        raise HTTPException(400, "Call /reset first.")
    with _lock:
        return {
            "task":         _current_task,
            "observation":  _env.get_observation(),
            "step":         _step_count,
            "total_reward": round(_total_reward, 6),
            "done":         _env.is_done(),
            "actions":      ACTIONS[_current_task],
        }


@app.get("/history")
def history():
    with _lock:
        return {
            "task":         _current_task,
            "step_count":   _step_count,
            "total_reward": round(_total_reward, 6),
            "history":      list(_history),
        }


@app.post("/explain")
def explain(body: ExplainSchema):
    s0, s1 = body.state_before, body.state_after

    # Compute deltas for any numeric key present in both states
    deltas = {
        k: round(s0[k] - s1.get(k, s0[k]), 3)
        for k in s0 if isinstance(s0[k], (int, float))
    }
    improved_keys = [k for k, d in deltas.items() if d > 0]

    rationale = (
        f"Action `{body.action}` on task `{body.task}`. "
        f"Improvements: {improved_keys if improved_keys else 'none this step'}. "
        f"Deltas: {deltas}."
    )
    return {
        "task":      body.task,
        "action":    body.action,
        "rationale": rationale,
        "deltas":    deltas,
    }