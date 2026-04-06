"""
inference.py — Hybrid Decision Engine (RL + LLM)
50/50 split: half the time uses Q-learning policy, half uses LLM completion.
Strict stdout logging in the required format.
"""

import os
import sys
import json
import time
import random
import argparse
import requests
from openai import OpenAI

from app.env import DisasterEnv
from app.agent import QLearningAgent

# ─── Config ───────────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TASK_NAME = "Thermal Management"

VALID_ACTIONS = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]

# ─── Logging ──────────────────────────────────────────────────────────────────

def log_start(task: str):
    print(f'[START] {json.dumps({"task": task})}', flush=True)

def log_step(step: int, action: str, reward: float, source: str, cpu: float, battery: float):
    print(
        f'[STEP] {json.dumps({"step": step, "action": action, "reward": round(reward, 4), "source": source, "cpu": cpu, "battery": battery})}',
        flush=True,
    )

def log_end(total_reward: float, status: str, steps: int):
    print(
        f'[END] {json.dumps({"total_reward": round(total_reward, 4), "status": status, "steps": steps})}',
        flush=True,
    )

def log_error(msg: str):
    print(f'[ERROR] {json.dumps({"error": msg})}', flush=True)

# ─── LLM Decision ─────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: dict, history: list) -> str:
    """
    Ask the LLM to choose an action given the current thermal state.
    Returns one of the VALID_ACTIONS strings.
    """
    history_text = ""
    if history:
        last = history[-3:]  # last 3 steps for context
        history_text = "\nRecent history:\n" + "\n".join(
            f"  step {h['step']}: {h['action']} → reward {h['reward']}" for h in last
        )

    prompt = f"""You are an AI thermal management agent controlling a compute device.

Current state:
  CPU usage: {obs['cpu']:.1f}%
  Battery level: {obs['battery']:.1f}%
{history_text}

Goal: reduce CPU below 60% and keep battery above 50%.

Choose exactly ONE action from this list:
  - optimize_cpu    : reschedule kernel threads (moderate CPU drop)
  - close_apps      : kill background apps (strong CPU drop, frees battery)
  - throttle_gpu    : lower GPU clocks (thermal relief, good battery)
  - hibernate_idle  : hibernate idle processes (modest CPU, best battery)

Respond with ONLY the action name, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Fuzzy match
        for action in VALID_ACTIONS:
            if action in raw:
                return action
        # Fallback: pick action with highest expected impact based on state
        return _heuristic_fallback(obs)
    except Exception as e:
        log_error(f"LLM call failed: {e}")
        return _heuristic_fallback(obs)


def _heuristic_fallback(obs: dict) -> str:
    """Rule-based fallback when LLM or RL is unavailable."""
    cpu = obs.get("cpu", 80)
    bat = obs.get("battery", 40)
    if cpu > 85:
        return "close_apps"
    elif bat < 30:
        return "throttle_gpu"
    elif cpu > 70:
        return "optimize_cpu"
    else:
        return "hibernate_idle"

# ─── Backend helpers ───────────────────────────────────────────────────────────

def backend_reset(cpu: float = 90.0, battery: float = 25.0) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/reset",
        json={"cpu": cpu, "battery": battery, "incident_description": "Inference script started"},
    )
    r.raise_for_status()
    return r.json()

def backend_step(action: str) -> dict:
    r = requests.post(f"{BACKEND_URL}/step", json={"action": action})
    r.raise_for_status()
    return r.json()

# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_episode(
    start_cpu: float = 90.0,
    start_battery: float = 25.0,
    use_llm: bool = True,
):
    """
    Run a single episode of the Hybrid Decision Engine.

    Decision logic:
        if random() < 0.5  →  RL agent (Q-learning policy)
        else               →  LLM (OpenAI chat completion)
    """
    log_start(TASK_NAME)

    # Initialise subsystems
    rl_agent = QLearningAgent()
    llm_client = OpenAI(api_key=OPENAI_API_KEY) if (use_llm and OPENAI_API_KEY) else None

    # Reset backend
    try:
        reset_data = backend_reset(start_cpu, start_battery)
        obs = reset_data["observation"]
    except Exception as e:
        log_error(f"Backend /reset failed: {e}. Running in standalone mode.")
        env = DisasterEnv()
        obs = env.reset(cpu=start_cpu, battery=start_battery)
        env_standalone = env
    else:
        env_standalone = None

    total_reward = 0.0
    step_history = []

    for step_idx in range(1, MAX_STEPS + 1):
        # ── 50/50 decision ────────────────────────────────────────────────────
        if random.random() < 0.5:
            action = rl_agent.get_action(obs)
            source = "rl"
        else:
            if llm_client:
                action = get_llm_action(llm_client, obs, step_history)
                source = "llm"
            else:
                action = rl_agent.get_action(obs)
                source = "rl_fallback"

        # ── Execute action ────────────────────────────────────────────────────
        if env_standalone:
            next_obs, reward, done, info = env_standalone.step(action)
        else:
            try:
                step_data  = backend_step(action)
                next_obs   = step_data["observation"]
                reward     = step_data["reward"]
                done       = step_data["done"]
            except Exception as e:
                log_error(f"Backend /step failed at step {step_idx}: {e}")
                break

        # ── Update RL agent ───────────────────────────────────────────────────
        rl_agent.update(obs, action, reward, next_obs, done)

        total_reward += reward
        log_step(step_idx, action, reward, source, next_obs["cpu"], next_obs["battery"])

        step_history.append({
            "step": step_idx,
            "action": action,
            "reward": round(reward, 4),
            "source": source,
        })

        obs = next_obs

        if done:
            log_end(total_reward, "success", step_idx)
            rl_agent.save()
            return total_reward, "success"

        time.sleep(0.05)  # small yield to avoid hammering backend

    # Episode ended without reaching goal
    log_end(total_reward, "max_steps_reached", MAX_STEPS)
    rl_agent.save()
    return total_reward, "max_steps_reached"


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RL+LLM Thermal Manager")
    parser.add_argument("--cpu",     type=float, default=92.0, help="Starting CPU %")
    parser.add_argument("--battery", type=float, default=20.0, help="Starting battery %")
    parser.add_argument("--no-llm",  action="store_true",      help="Disable LLM, use RL only")
    args = parser.parse_args()

    total, status = run_episode(
        start_cpu=args.cpu,
        start_battery=args.battery,
        use_llm=not args.no_llm,
    )
    sys.exit(0 if status == "success" else 1)
