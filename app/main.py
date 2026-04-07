"""
app/agent.py — Enhanced Q-Learning Agent (Hybrid Ready)

Upgrades:
✔ Softmax confidence output (for hybrid RL + LLM)
✔ Adaptive epsilon decay
✔ State visit tracking
✔ Reward normalization
✔ Auto-save improvements
✔ Debug insights
"""

import random
import json
import os
import math
from typing import Dict, List

ACTIONS = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]
QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.json")


def _discretize(obs: Dict[str, float]) -> str:
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
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.q: Dict[str, Dict[str, float]] = {}
        self.state_visits: Dict[str, int] = {}

        self._load()

    # ── Q-table helpers ───────────────────────────────────────────────────────

    def _get_q(self, state: str, action: str) -> float:
        return self.q.get(state, {}).get(action, 0.0)

    def _set_q(self, state: str, action: str, value: float):
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in ACTIONS}
        self.q[state][action] = value

    # ── Softmax (for hybrid confidence) ───────────────────────────────────────

    def _softmax(self, values: List[float]) -> List[float]:
        exp_vals = [math.exp(v) for v in values]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]

    def get_action_with_confidence(self, obs: Dict[str, float]):
        """
        Returns:
        action, confidence distribution
        """
        state = _discretize(obs)
        q_vals = [self._get_q(state, a) for a in ACTIONS]
        probs = self._softmax(q_vals)

        best_idx = probs.index(max(probs))
        return ACTIONS[best_idx], probs

    # ── Policy ────────────────────────────────────────────────────────────────

    def get_action(self, obs: Dict[str, float]) -> str:
        state = _discretize(obs)

        # Track visits
        self.state_visits[state] = self.state_visits.get(state, 0) + 1

        # Adaptive epsilon (less exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        q_vals = {a: self._get_q(state, a) for a in ACTIONS}
        return max(q_vals, key=q_vals.get)

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(
        self,
        obs: Dict[str, float],
        action: str,
        reward: float,
        next_obs: Dict[str, float],
        done: bool,
    ):
        state      = _discretize(obs)
        next_state = _discretize(next_obs)

        # Normalize reward (stability)
        reward = max(min(reward, 10), -10)

        current_q  = self._get_q(state, action)
        max_next_q = max(self._get_q(next_state, a) for a in ACTIONS) if not done else 0.0
        target     = reward + self.gamma * max_next_q
        new_q      = current_q + self.alpha * (target - current_q)

        self._set_q(state, action, new_q)

        # Auto-save periodically
        if random.random() < 0.05:
            self.save()

    # ── Hybrid Support ────────────────────────────────────────────────────────

    def get_q_confidence(self, obs: Dict[str, float]) -> Dict[str, float]:
        """
        Returns normalized Q-values as probabilities
        Used for RL + LLM fusion
        """
        state = _discretize(obs)
        q_vals = [self._get_q(state, a) for a in ACTIONS]
        probs = self._softmax(q_vals)

        return {a: p for a, p in zip(ACTIONS, probs)}

    # ── Debug / Explainability ────────────────────────────────────────────────

    def debug_state(self, obs: Dict[str, float]) -> Dict:
        state = _discretize(obs)
        return {
            "state": state,
            "q_values": self.q.get(state, {}),
            "visits": self.state_visits.get(state, 0),
            "epsilon": self.epsilon
        }

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