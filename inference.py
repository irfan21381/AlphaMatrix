import sys
import os
import json
import random
from typing import List

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
MODEL_NAME     = os.getenv("MODEL_NAME", "hybrid-agent")
API_KEY        = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

TASK           = os.getenv("ALPHAMATRIX_TASK", "thermal_throttling")
BENCHMARK      = "alphamatrix"
MAX_STEPS      = int(os.getenv("MAX_STEPS", "20"))

# TASK_ACTIONS = {
#     "thermal_throttling": [
#         "reduce_clock_speed",
#         "kill_heavy_process",
#         "enable_cooling_fan",
#         "throttle_gpu",
#     ]
# }

TASK_ACTIONS = {
    "thermal_throttling": [
        "optimize_cpu",
        "close_apps",
        "throttle_gpu",
        "hibernate_idle",
    ]
}


# ── LOGGING (STRICT FORMAT) ─────────────────────────

def log_start():
    print(f"[START] task={TASK} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── BACKEND ─────────────────────────────────────────

def _backend_reset():
    try:
        r = requests.post(f"{BACKEND_URL}/reset", json={}, timeout=5)
        return r.json()
    except:
        return None

def _backend_step(action):
    try:
        r = requests.post(f"{BACKEND_URL}/step", json={"action": action}, timeout=5)
        return r.json()
    except:
        return None


# ── LLM ─────────────────────────────────────────────

def _llm_action(client, obs, actions):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": str(obs)}],
            max_tokens=20,
        )
        return random.choice(actions), 0.5
    except:
        return random.choice(actions), 0.5


# ── MAIN ───────────────────────────────────────────

def run():
    actions = TASK_ACTIONS[TASK]
    agent = QLearningAgent()

    client = OpenAI(base_url=os.getenv("API_BASE_URL"), api_key=API_KEY) if _openai_available and API_KEY else None

    rewards = []
    total = 0.0
    steps = 0

    log_start()

    reset_data = _backend_reset()
    if reset_data is None:
        log_end(False, 0, 0.0, [])
        return

    obs = reset_data.get("observation", {})

    for step in range(1, MAX_STEPS + 1):

        # hybrid decision
        rl_probs = agent.get_q_confidence(obs)

        if client:
            llm_action, llm_conf = _llm_action(client, obs, actions)
        else:
            llm_action, llm_conf = random.choice(actions), 0.5

        scores = {}
        for a in actions:
            scores[a] = 0.7 * rl_probs.get(a, 0) + 0.3 * (llm_conf if a == llm_action else 0)

        action = max(scores, key=scores.get)

        step_data = _backend_step(action)
        if step_data is None:
            log_step(step, action, 0.0, False, "backend_error")
            break

        next_obs = step_data.get("observation", {})
        reward = float(step_data.get("reward", 0.0))
        done = step_data.get("done", False)

        rewards.append(reward)
        total += reward
        steps = step

        agent.update(obs, action, reward, next_obs, done)

        log_step(step, action, reward, done, None)

        obs = next_obs

        if done:
            break

    score = min(max(total / MAX_STEPS, 0.0), 1.0)
    success = score > 0.1

    log_end(success, steps, score, rewards)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log_end(False, 0, 0.0, [])
    sys.exit(0)