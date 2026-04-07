import sys
import os
from typing import List

from app.agent import QLearningAgent
from app.env import ACTIONS, ThermalEnv, TASK

MODEL_NAME = os.getenv("MODEL_NAME", "qlearning-agent")
BENCHMARK = "alphamatrix"
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))

# TASK_ACTIONS = {
#     "thermal_throttling": [
#         "reduce_clock_speed",
#         "kill_heavy_process",
#         "enable_cooling_fan",
#         "throttle_gpu",
#     ]
# }

TASK_ACTIONS = {TASK: list(ACTIONS)}


# ── LOGGING (STRICT FORMAT) ─────────────────────────

def log_start():
    print(f"[START] task={TASK} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def _env_rollout():
    env = ThermalEnv()
    obs = env.reset(cpu=90.0, battery=20.0)
    return env, obs


# ── MAIN ───────────────────────────────────────────

def run():
    actions = TASK_ACTIONS[TASK]
    agent = QLearningAgent()
    env, obs = _env_rollout()

    rewards = []
    total = 0.0
    steps = 0

    log_start()

    for step in range(1, MAX_STEPS + 1):

        action = agent.act(obs)
        res = env.step(action)
        next_obs = res.observation
        reward = float(res.reward)
        done = bool(res.done)

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