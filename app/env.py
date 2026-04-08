"""
app/env.py — Thermal throttling environment (FINAL FIXED)
Compatible with main.py (AlphaMatrixEnv + TASKS added)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

# ✅ REQUIRED FOR MAIN.PY
TASKS = ["thermal_throttling"]

ACTIONS = {
    "thermal_throttling": [
        "optimize_cpu",
        "close_apps",
        "throttle_gpu",
        "hibernate_idle"
    ]
}

CPU_MIN = 5.0


@dataclass
class StepResult:
    observation: Dict[str, float]
    reward: float
    done: bool
    info: Dict


class ThermalEnv:
    def __init__(
        self,
        cpu_safe: float = 60.0,
        battery_safe: float = 50.0,
        max_steps: int = 50,
        seed: int | None = None,
    ):
        self.cpu_safe = float(cpu_safe)
        self.battery_safe = float(battery_safe)
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)

        self._cpu = 90.0
        self._battery = 20.0
        self._step = 0

    def reset(self, cpu: float = 90.0, battery: float = 20.0):
        self._cpu = float(max(CPU_MIN, min(100.0, cpu)))
        self._battery = float(max(0.0, min(100.0, battery)))
        self._step = 0
        return self.get_observation()

    def get_observation(self):
        return {
            "cpu": round(self._cpu, 3),
            "battery": round(self._battery, 3)
        }

    def is_done(self):
        return (self._cpu < 20.0) or (self._step >= self.max_steps)

    def step(self, action: str):
        if action not in ACTIONS["thermal_throttling"]:
            raise ValueError(f"Invalid action: {action}")

        before = self.get_observation()
        cpu0, bat0 = self._cpu, self._battery

        # Background drift
        self._cpu = min(100.0, max(CPU_MIN, self._cpu + self._rng.uniform(-0.5, 3.5)))
        self._battery = min(100.0, max(0.0, self._battery - self._rng.uniform(0.4, 1.6)))

        # Actions
        if action == "optimize_cpu":
            self._cpu -= self._rng.uniform(6.0, 14.0)
            self._battery += self._rng.uniform(-0.5, 2.0)
        elif action == "close_apps":
            self._cpu -= self._rng.uniform(10.0, 20.0)
            self._battery += self._rng.uniform(0.5, 4.5)
        elif action == "throttle_gpu":
            self._cpu -= self._rng.uniform(5.0, 12.0)
            self._battery += self._rng.uniform(2.0, 7.0)
        else:  # hibernate_idle
            self._cpu -= self._rng.uniform(3.0, 9.0)
            self._battery += self._rng.uniform(4.0, 10.0)

        # Clamp
        self._cpu = min(100.0, max(CPU_MIN, self._cpu))
        self._battery = min(100.0, max(0.0, self._battery))
        self._step += 1

        after = self.get_observation()

        # Reward
        d_cpu = cpu0 - self._cpu
        d_bat = self._battery - bat0

        step_penalty = 0.1
        over_throttle_penalty = 0.0

        if self._cpu < 10.0:
            over_throttle_penalty = (10.0 - self._cpu) * 0.2

        reward = (d_cpu * 0.5) + (d_bat * 0.3) - step_penalty - over_throttle_penalty

        return {
            "observation": after,
            "reward": round(float(reward), 6),
            "done": self.is_done(),
            "info": {
                "task": "thermal_throttling",
                "action": action,
                "step": self._step,
            },
        }


# ✅ WRAPPER REQUIRED BY main.py
class AlphaMatrixEnv:
    def __init__(self):
        self.env = ThermalEnv()

    def reset(self, task="thermal_throttling"):
        return self.env.reset()

    def step(self, action: str):
        return self.env.step(action)

    def get_observation(self):
        return self.env.get_observation()

    def is_done(self):
        return self.env.is_done()