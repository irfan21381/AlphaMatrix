"""
inference.py — AlphaMatrix ADVANCED HYBRID ENGINE
✔ RL + LLM fusion (confidence-based)
✔ Multi-step planning
✔ Learning updates
✔ OpenEnv compliant logs
"""

import sys
import os
import json
import random

from app.agent import QLearningAgent

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


BACKEND_URL    = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TASK           = os.getenv("ALPHAMATRIX_TASK", "thermal_throttling")
MAX_STEPS      = int(os.getenv("MAX_STEPS", "20"))

TASK_ACTIONS = {
    "thermal_throttling": [
        "reduce_clock_speed", "kill_heavy_process",
        "enable_cooling_fan", "throttle_gpu",
    ]
}

# ── Logging ─────────────────────────────────────────

def _emit(tag: str, payload: dict):
    print(f"[{tag}] {json.dumps(payload)}", flush=True)

def log_start(task: str):
    _emit("START", {"task": task})

def log_step(step: int, action: str, reward: float, source: str, obs: dict):
    _emit("STEP", {
        "step": step,
        "action": action,
        "reward": round(reward, 6),
        "source": source,
        "obs": obs,
    })

def log_end(total_reward: float, status: str, steps: int):
    _emit("END", {
        "total_reward": round(total_reward, 6),
        "status": status,
        "steps": steps,
    })

# ── Backend ─────────────────────────────────────────

def _backend_reset(task: str):
    try:
        r = requests.post(f"{BACKEND_URL}/reset", json={"task": task}, timeout=5)
        return r.json()
    except:
        return None

def _backend_step(action: str):
    try:
        r = requests.post(f"{BACKEND_URL}/step", json={"action": action}, timeout=5)
        return r.json()
    except:
        return None

# ── LLM with confidence ─────────────────────────────

def _llm_action(client, obs, actions, task):
    prompt = f"""
    Task: {task}
    Observation: {obs}
    Actions: {actions}
    Return JSON: {{"action": "...", "confidence": 0.0-1.0}}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=50,
        )
        data = json.loads(resp.choices[0].message.content)
        return data["action"], float(data["confidence"])
    except:
        return random.choice(actions), 0.5

# ── Multi-step planner (NEW) ────────────────────────

def _plan_actions(agent, obs, llm_client, actions):
    """
    Generate 2-step plan using RL + LLM
    """
    plan = []

    for _ in range(2):
        rl_probs = agent.get_q_confidence(obs)

        if llm_client:
            llm_action, llm_conf = _llm_action(llm_client, obs, actions, TASK)
        else:
            llm_action, llm_conf = random.choice(actions), 0.5

        scores = {}
        for a in actions:
            scores[a] = 0.7 * rl_probs.get(a, 0) + 0.3 * (llm_conf if a == llm_action else 0)

        best = max(scores, key=scores.get)
        plan.append(best)

    return plan

# ── MAIN ───────────────────────────────────────────

def run():
    actions = TASK_ACTIONS[TASK]
    log_start(TASK)

    agent = QLearningAgent()

    llm_client = OpenAI(api_key=OPENAI_API_KEY) if _openai_available and OPENAI_API_KEY else None

    reset_data = _backend_reset(TASK)
    if reset_data is None:
        log_end(0.0, "backend_error", 0)
        return

    obs = reset_data.get("observation", {})
    total = 0.0

    for step_idx in range(1, MAX_STEPS + 1):

        # 🔥 Multi-step plan
        plan = _plan_actions(agent, obs, llm_client, actions)
        action = plan[0]

        step_data = _backend_step(action)
        if step_data is None:
            log_end(total, "backend_error", step_idx)
            return

        next_obs = step_data.get("observation", {})
        reward = float(step_data.get("reward", 0.0))
        done = step_data.get("done", False)

        total += reward

        # 🔥 RL learning
        agent.update(obs, action, reward, next_obs, done)
        agent.save()

        log_step(step_idx, action, reward, "hybrid_planner", next_obs)

        obs = next_obs

        if done:
            log_end(total, "success", step_idx)
            return

    log_end(total, "max_steps_reached", MAX_STEPS)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        _emit("END", {"total_reward": 0.0, "status": f"fatal_error: {e}", "steps": 0})
    sys.exit(0)