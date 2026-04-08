"""
app/main.py — AlphaMatrix FastAPI Backend (FINAL)
Thread-safe state machine + LiteLLM integration.
Runs on port 8000.
"""

import threading
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.env import AlphaMatrixEnv, TASKS, ACTIONS

# 🔥 LiteLLM import
from litellm import completion

# ── ENV CONFIG ────────────────────────────────────────────────────────────────

LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "https://your-litellm-proxy-url")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "your-proxy-key")
MODEL_NAME       = os.getenv("LITELLM_MODEL", "gpt-4o-mini")

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

app = FastAPI(title="AlphaMatrix OpenEnv API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
        result        = _env.step(body.action)
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


# 🔥 MAIN FIX — LLM INTEGRATION
@app.post("/explain")
def explain(body: ExplainSchema):
    s0, s1 = body.state_before, body.state_after

    # Compute deltas
    deltas = {
        k: round(s0[k] - s1.get(k, s0[k]), 3)
        for k in s0 if isinstance(s0[k], (int, float))
    }
    improved_keys = [k for k, d in deltas.items() if d > 0]

    # Prompt for LLM
    prompt = f"""
    You are an AI system explaining environment transitions.

    Task: {body.task}
    Action: {body.action}

    State Before:
    {s0}

    State After:
    {s1}

    Improvements: {improved_keys}
    Deltas: {deltas}

    Explain clearly what happened and why this action matters.
    """

    # 🔥 LiteLLM call (MANDATORY FOR HACKATHON)
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            api_base=LITELLM_API_BASE,
            api_key=LITELLM_API_KEY
        )

        ai_text = response["choices"][0]["message"]["content"]

    except Exception as e:
        # fallback (never break API)
        ai_text = (
            f"LLM unavailable. Action `{body.action}` caused changes: "
            f"{deltas}. Error: {str(e)}"
        )

    return {
        "task":      body.task,
        "action":    body.action,
        "rationale": ai_text,
        "deltas":    deltas,
    }


# 🔥 OPTIONAL TEST ENDPOINT (GOOD FOR VALIDATION)
@app.get("/llm-test")
def llm_test():
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Test connection"}],
            api_base=LITELLM_API_BASE,
            api_key=LITELLM_API_KEY
        )

        return {"status": "success", "response": response}

    except Exception as e:
        return {"status": "error", "message": str(e)}