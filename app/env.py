"""
app/env.py — Thermal throttling environment (OpenEnv-style)

Single task with a single, consistent action space:
["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

TASK = "thermal_throttling"
ACTIONS = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]


@dataclass
class StepResult:
    observation: Dict[str, float]
    reward: float
    done: bool
    info: Dict


class ThermalEnv:
    """
    State:
      - cpu:     0..100  (higher is worse)
      - battery: 0..100  (higher is better)

    Done when CPU is cool enough AND battery is healthy enough,
    or max steps reached.
    """

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

    def reset(self, cpu: float = 90.0, battery: float = 20.0) -> Dict[str, float]:
        self._cpu = float(max(0.0, min(100.0, cpu)))
        self._battery = float(max(0.0, min(100.0, battery)))
        self._step = 0
        return self.observation()

    def observation(self) -> Dict[str, float]:
        return {"cpu": round(self._cpu, 3), "battery": round(self._battery, 3)}

    def is_done(self) -> bool:
        return (
            (self._cpu <= self.cpu_safe and self._battery >= self.battery_safe)
            or self._step >= self.max_steps
        )

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Must be one of {ACTIONS}.")

        before = self.observation()
        cpu0, bat0 = self._cpu, self._battery

        # Background drift: load + small battery drain each step
        self._cpu = min(100.0, max(0.0, self._cpu + self._rng.uniform(-1.0, 2.5)))
        self._battery = min(100.0, max(0.0, self._battery - self._rng.uniform(0.2, 1.2)))

        # Action effects (stochastic)
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

        self._cpu = min(100.0, max(0.0, self._cpu))
        self._battery = min(100.0, max(0.0, self._battery))
        self._step += 1

        after = self.observation()

        # Reward: encourage CPU down and battery up, scaled to 0..1
        d_cpu = max(0.0, cpu0 - self._cpu)
        d_bat = max(0.0, self._battery - bat0)
        # Typical best-case ~ (20 cpu + 10 bat) in one step
        raw = (d_cpu / 20.0) * 0.7 + (d_bat / 10.0) * 0.3
        reward = float(max(0.0, min(1.0, raw)))

        done = self.is_done()
        info = {
            "task": TASK,
            "action": action,
            "step": self._step,
            "delta_cpu": round(before["cpu"] - after["cpu"], 3),
            "delta_battery": round(after["battery"] - before["battery"], 3),
            "goal": {"cpu_safe": self.cpu_safe, "battery_safe": self.battery_safe},
        }

        return StepResult(observation=after, reward=round(reward, 6), done=done, info=info)


def explain_action(action: str, state_before: Dict[str, float], state_after: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    dc = float(state_before.get("cpu", 0.0) - state_after.get("cpu", 0.0))
    db = float(state_after.get("battery", 0.0) - state_before.get("battery", 0.0))
    deltas = {"cpu": round(dc, 3), "battery": round(db, 3)}

    if action == "optimize_cpu":
        msg = "Rebalanced CPU scheduling and reduced background contention to cool the CPU with minimal battery impact."
    elif action == "close_apps":
        msg = "Terminated high-load background apps to quickly cut CPU usage while improving battery stability."
    elif action == "throttle_gpu":
        msg = "Throttled GPU to reduce thermal pressure and improve efficiency, trading peak graphics throughput for stability."
    else:
        msg = "Hibernated idle processes to steadily reduce heat and recover battery by minimizing unnecessary work."

    return msg, deltas
