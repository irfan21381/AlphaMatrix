"""
app/agent.py — Lightweight Q-learning agent for the thermal task.

No external APIs. Uses a small discrete state space and a persistent Q-table.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.env import ACTIONS

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.json")


def _discretize(obs: Dict[str, float]) -> str:
    cpu = float(obs.get("cpu", 50.0))
    bat = float(obs.get("battery", 50.0))

    # CPU buckets
    if cpu < 40:
        c = "L"
    elif cpu < 60:
        c = "M"
    elif cpu < 80:
        c = "H"
    else:
        c = "C"

    # Battery buckets
    if bat < 20:
        b = "L"
    elif bat < 50:
        b = "M"
    else:
        b = "H"

    return f"cpu_{c}_bat_{b}"


def _softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    t = max(1e-6, float(temperature))
    m = max(values) if values else 0.0
    exps = [math.exp((v - m) / t) for v in values]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


@dataclass
class AgentDebug:
    state: str
    q_values: Dict[str, float]
    epsilon: float
    visits: int


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.12,
        gamma: float = 0.92,
        epsilon: float = 0.35,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.q: Dict[str, Dict[str, float]] = {}
        self.visits: Dict[str, int] = {}
        self._load()

    def _ensure_state(self, s: str) -> None:
        if s not in self.q:
            self.q[s] = {a: 0.0 for a in ACTIONS}

    def get_q(self, obs: Dict[str, float]) -> Dict[str, float]:
        s = _discretize(obs)
        self._ensure_state(s)
        return dict(self.q[s])

    def act(self, obs: Dict[str, float]) -> str:
        s = _discretize(obs)
        self._ensure_state(s)

        self.visits[s] = self.visits.get(s, 0) + 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        qvals = self.q[s]
        best = max(qvals, key=qvals.get)
        return best

    def act_with_confidence(self, obs: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        s = _discretize(obs)
        self._ensure_state(s)
        qvals = [self.q[s][a] for a in ACTIONS]
        probs = _softmax(qvals, temperature=1.0)
        conf = {a: float(p) for a, p in zip(ACTIONS, probs)}
        action = max(conf, key=conf.get)
        return action, conf

    def update(
        self,
        obs: Dict[str, float],
        action: str,
        reward: float,
        next_obs: Dict[str, float],
        done: bool,
    ) -> None:
        if action not in ACTIONS:
            return

        s = _discretize(obs)
        ns = _discretize(next_obs)
        self._ensure_state(s)
        self._ensure_state(ns)

        r = float(max(0.0, min(1.0, reward)))
        cur = self.q[s][action]
        nxt = 0.0 if done else max(self.q[ns].values())
        target = r + self.gamma * nxt
        self.q[s][action] = cur + self.alpha * (target - cur)

    def debug(self, obs: Dict[str, float]) -> AgentDebug:
        s = _discretize(obs)
        self._ensure_state(s)
        return AgentDebug(
            state=s,
            q_values=dict(self.q[s]),
            epsilon=float(self.epsilon),
            visits=int(self.visits.get(s, 0)),
        )

    def save(self) -> None:
        try:
            with open(QTABLE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.q, f)
        except Exception:
            pass

    def _load(self) -> None:
        if not os.path.exists(QTABLE_PATH):
            return
        try:
            with open(QTABLE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.q = data
        except Exception:
            self.q = {}