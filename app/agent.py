"""
app/agent.py — Simple Tabular Q-Learning Agent
Used by inference.py as the "policy-based" half of the 50/50 hybrid engine.
"""

import random
import json
import os
from typing import Dict, List

ACTIONS = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]
QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.json")


def _discretize(obs: Dict[str, float]) -> str:
    """
    Map a continuous observation to a discrete bucket string.
    CPU: 0-40 → L, 40-70 → M, 70-85 → H, 85-100 → C (critical)
    Battery: 0-20 → L, 20-50 → M, 50-100 → H
    """
    cpu = obs.get("cpu", 50.0)
    bat = obs.get("battery", 50.0)

    if cpu < 40:
        c = "L"
    elif cpu < 70:
        c = "M"
    elif cpu < 85:
        c = "H"
    else:
        c = "C"

    if bat < 20:
        b = "L"
    elif bat < 50:
        b = "M"
    else:
        b = "H"

    return f"cpu_{c}_bat_{b}"


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        self.alpha   = alpha    # learning rate
        self.gamma   = gamma    # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q: Dict[str, Dict[str, float]] = {}
        self._load()

    # ── Q-table helpers ───────────────────────────────────────────────────────

    def _get_q(self, state: str, action: str) -> float:
        return self.q.get(state, {}).get(action, 0.0)

    def _set_q(self, state: str, action: str, value: float):
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in ACTIONS}
        self.q[state][action] = value

    # ── Policy ────────────────────────────────────────────────────────────────

    def get_action(self, obs: Dict[str, float]) -> str:
        """ε-greedy policy."""
        state = _discretize(obs)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_vals = {a: self._get_q(state, a) for a in ACTIONS}
        return max(q_vals, key=q_vals.get)

    def update(
        self,
        obs: Dict[str, float],
        action: str,
        reward: float,
        next_obs: Dict[str, float],
        done: bool,
    ):
        """Single Q-learning update step."""
        state      = _discretize(obs)
        next_state = _discretize(next_obs)

        current_q  = self._get_q(state, action)
        max_next_q = max(self._get_q(next_state, a) for a in ACTIONS) if not done else 0.0
        target     = reward + self.gamma * max_next_q
        new_q      = current_q + self.alpha * (target - current_q)

        self._set_q(state, action, new_q)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        try:
            with open(QTABLE_PATH, "w") as f:
                json.dump(self.q, f)
        except Exception as e:
            print(f"[AGENT] Could not save Q-table: {e}")

    def _load(self):
        if os.path.exists(QTABLE_PATH):
            try:
                with open(QTABLE_PATH) as f:
                    self.q = json.load(f)
                print(f"[AGENT] Q-table loaded ({len(self.q)} states)")
            except Exception:
                self.q = {}
