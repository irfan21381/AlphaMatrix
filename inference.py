import json
import os
import sys
import time
from typing import Dict, List

from app.agent import QLearningAgent
from app.env import ACTIONS, TASK, ThermalEnv

MODEL_NAME = os.getenv("MODEL_NAME", "qlearning-agent")
BENCHMARK = os.getenv("BENCHMARK", "alphamatrix")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


def reset(env: ThermalEnv, cpu: float = 90.0, battery: float = 20.0) -> Dict:
    """
    OpenEnv-compatible reset payload:
    {
      "observation": {...},
      "reward": 0,
      "done": false,
      "info": {}
    }
    """
    return env.reset_openenv(cpu=cpu, battery=battery)


def step(env: ThermalEnv, action: str) -> Dict:
    """
    OpenEnv-style step payload:
    {
      "observation": {...},
      "reward": <float>,
      "done": <bool>,
      "info": {...}
    }
    """
    return env.step_openenv(action)


def _emit(tag: str, payload: Dict) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=True)}", flush=True)


def _running_on_hf_spaces() -> bool:
    # HuggingFace Spaces sets SPACE_ID for running apps.
    return bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")


def run() -> None:
    env = ThermalEnv(max_steps=MAX_STEPS)
    agent = QLearningAgent()

    _emit(
        "START",
        {
            "task": TASK,
            "env": BENCHMARK,
            "model": MODEL_NAME,
            "actions": list(ACTIONS),
            "max_steps": MAX_STEPS,
        },
    )

    r0 = reset(env, cpu=90.0, battery=20.0)
    obs = dict(r0["observation"])

    rewards: List[float] = []
    steps = 0
    total = 0.0
    success = False

    for i in range(1, MAX_STEPS + 1):
        action, _conf = agent.act_with_confidence(obs)
        out = step(env, action)

        reward = float(out.get("reward", 0.0))
        done = bool(out.get("done", False))
        next_obs = dict(out.get("observation", {}))

        agent.update(obs, action, reward, next_obs, done)
        rewards.append(reward)
        total += reward
        steps = i

        _emit(
            "STEP",
            {
                "step": i,
                "action": action,
                "reward": reward,
                "observation": next_obs,
                "done": done,
            },
        )

        obs = next_obs
        if done:
            success = True
            break

    score = float(total)
    _emit(
        "END",
        {
            "success": bool(success),
            "steps": int(steps),
            "score": score,
            "rewards": rewards,
        },
    )

    # Spaces expects a long-running process; a clean exit is shown as "Runtime error".
    if _running_on_hf_spaces():
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    try:
        run()
        sys.exit(0)
    except Exception as e:
        _emit("END", {"success": False, "steps": 0, "score": 0.0, "rewards": [], "error": str(e)})
        sys.exit(0)