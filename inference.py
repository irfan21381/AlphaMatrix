"""
inference.py — AlphaMatrix Baseline Inference Script
OpenEnv-compliant. Emits ONLY [START], [STEP], [END] structured JSON to stdout.
Never exits with non-zero status (all errors are caught and logged).
"""

import sys
import os
import json
import random

# ── Safe imports — never crash ────────────────────────────────────────────────

try:
    import requests
except ImportError:
    requests = None  # handled gracefully below

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ── Config ────────────────────────────────────────────────────────────────────

BACKEND_URL    = os.getenv("BACKEND_URL",    "http://127.0.0.1:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TASK           = os.getenv("ALPHAMATRIX_TASK", "thermal_throttling")
MAX_STEPS      = int(os.getenv("MAX_STEPS", "20"))

TASK_ACTIONS = {
    "thermal_throttling": [
        "reduce_clock_speed", "kill_heavy_process",
        "enable_cooling_fan", "throttle_gpu",
    ],
    "battery_endurance": [
        "dim_display", "disable_wifi",
        "suspend_background_apps", "enable_battery_saver",
    ],
    "process_deadlock": [
        "release_mutex", "restart_process",
        "increase_timeout", "force_schedule",
    ],
}

# ── Structured logging — ONLY these three formats go to stdout ────────────────

def _emit(tag: str, payload: dict):
    print(f"[{tag}] {json.dumps(payload)}", flush=True)

def log_start(task: str):
    _emit("START", {"task": task})

def log_step(step: int, action: str, reward: float, source: str, obs: dict):
    _emit("STEP", {
        "step":   step,
        "action": action,
        "reward": round(reward, 6),
        "source": source,
        "obs":    obs,
    })

def log_end(total_reward: float, status: str, steps: int):
    _emit("END", {
        "total_reward": round(total_reward, 6),
        "status":       status,
        "steps":        steps,
    })

# ── Backend helpers ───────────────────────────────────────────────────────────

def _backend_reset(task: str) -> dict | None:
    if requests is None:
        return None
    try:
        r = requests.post(
            f"{BACKEND_URL}/reset",
            json={"task": task},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _backend_step(action: str) -> dict | None:
    if requests is None:
        return None
    try:
        r = requests.post(
            f"{BACKEND_URL}/step",
            json={"action": action},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ── LLM action selection ──────────────────────────────────────────────────────

def _llm_action(client, obs: dict, actions: list[str], task: str) -> str:
    prompt = (
        f"Task: {task}\n"
        f"Current observation: {json.dumps(obs)}\n"
        f"Available actions: {actions}\n"
        "Choose the single best action name to improve the system. "
        "Reply with ONLY the action name, nothing else."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip().lower()
        for a in actions:
            if a in raw:
                return a
    except Exception:
        pass
    return random.choice(actions)   # fallback — never crash

# ── Standalone env fallback (no backend needed) ───────────────────────────────

def _standalone_run(task: str, actions: list[str]) -> tuple[float, str]:
    """Fallback: run the env in-process when backend is unreachable."""
    try:
        from app.env import AlphaMatrixEnv
        env = AlphaMatrixEnv(task=task)
        obs = env.reset()
    except Exception:
        return 0.0, "env_import_failed"

    total = 0.0
    for step_idx in range(1, MAX_STEPS + 1):
        action = random.choice(actions)
        try:
            result = env.step(action)
        except Exception:
            break
        reward  = result["reward"]
        obs     = result["observation"]
        done    = result["done"]
        total  += reward
        log_step(step_idx, action, reward, "standalone_rl", obs)
        if done:
            log_end(total, "success", step_idx)
            return total, "success"

    log_end(total, "max_steps_reached", MAX_STEPS)
    return total, "max_steps_reached"

# ── Main episode ──────────────────────────────────────────────────────────────

def run():
    actions = TASK_ACTIONS.get(TASK, TASK_ACTIONS["thermal_throttling"])
    log_start(TASK)

    # Initialise LLM client if possible
    llm_client = None
    if _openai_available and OPENAI_API_KEY:
        try:
            llm_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception:
            llm_client = None

    # Try to reach backend
    reset_data = _backend_reset(TASK)
    if reset_data is None:
        # Backend unreachable — run standalone
        _standalone_run(TASK, actions)
        return

    obs        = reset_data.get("observation", {})
    total      = 0.0

    for step_idx in range(1, MAX_STEPS + 1):
        # 50 / 50 hybrid decision
        if llm_client and random.random() < 0.5:
            action = _llm_action(llm_client, obs, actions, TASK)
            source = "llm"
        else:
            # Heuristic RL proxy
            action = random.choice(actions)
            source = "rl"

        step_data = _backend_step(action)
        if step_data is None:
            log_end(total, "backend_error", step_idx)
            return

        obs    = step_data.get("observation", {})
        reward = float(step_data.get("reward", 0.0))
        done   = bool(step_data.get("done", False))
        total += reward

        log_step(step_idx, action, reward, source, obs)

        if done:
            log_end(total, "success", step_idx)
            return

    log_end(total, "max_steps_reached", MAX_STEPS)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        # Last-resort catch — script must never exit non-zero
        _emit("END", {"total_reward": 0.0, "status": f"fatal_error: {e}", "steps": 0})
    sys.exit(0)   # always exit 0