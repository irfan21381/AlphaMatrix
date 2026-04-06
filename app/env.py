"""
app/env.py — RL Thermal Management Environment
Stochastic state transitions, normalized reward in [0, 1].
"""

import random
from typing import Tuple, Dict, Any, List

# ─── Constants ────────────────────────────────────────────────────────────────

CPU_THRESHOLD_SAFE   = 60.0   # CPU% considered "safe"
BAT_THRESHOLD_SAFE   = 50.0   # Battery% considered "safe"
MAX_CPU              = 100.0
MIN_CPU              = 0.0
MAX_BAT              = 100.0
MIN_BAT              = 0.0

# Maximum theoretical improvement per step (used to normalise reward)
MAX_CPU_IMPROVEMENT  = 20.0   # best-case CPU drop per action
MAX_BAT_IMPROVEMENT  = 10.0   # best-case battery gain per action
MAX_POSSIBLE_IMPROVEMENT = MAX_CPU_IMPROVEMENT + MAX_BAT_IMPROVEMENT  # = 30.0

# Action space
ACTION_SPACE: List[str] = [
    "optimize_cpu",
    "close_apps",
    "throttle_gpu",
    "hibernate_idle",
]


# ─── Environment ──────────────────────────────────────────────────────────────

class DisasterEnv:
    """
    A stochastic single-agent environment modelling a thermal disaster
    on a compute device.

    Observation space  : {"cpu": float, "battery": float}
    Action space       : {optimize_cpu, close_apps, throttle_gpu, hibernate_idle}
    Reward             : normalised ΔCpu + ΔBattery ∈ [0.0, 1.0]
    Terminal condition : cpu ≤ CPU_THRESHOLD_SAFE AND battery ≥ BAT_THRESHOLD_SAFE
    """

    def __init__(self):
        self.action_space = ACTION_SPACE
        self._cpu: float = 85.0
        self._battery: float = 30.0
        self._step_count: int = 0
        self._max_steps: int = 50   # hard episode limit

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, cpu: float = 85.0, battery: float = 30.0) -> Dict[str, float]:
        """Initialise (or re-initialise) the environment."""
        self._cpu = float(cpu)
        self._battery = float(battery)
        self._step_count = 0
        return self.get_observation()

    def step(self, action: str) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Apply *action* and return (observation, reward, done, info).

        All transitions are stochastic (uniform noise) so the agent must
        generalise rather than memorise a fixed mapping.
        """
        if action not in ACTION_SPACE:
            raise ValueError(f"Unknown action '{action}'. Valid: {ACTION_SPACE}")

        cpu_before  = self._cpu
        bat_before  = self._battery

        # ── Stochastic transitions ────────────────────────────────────────────
        if action == "optimize_cpu":
            # Reschedule kernel threads — moderate CPU relief, no battery effect
            delta_cpu = random.uniform(8.0, 18.0)
            delta_bat = random.uniform(-1.0, 2.0)   # tiny battery side-effect

        elif action == "close_apps":
            # Kill background apps — strong CPU relief per spec [10, 20]
            delta_cpu = random.uniform(10.0, 20.0)
            delta_bat = random.uniform(1.0, 5.0)    # apps freed battery too

        elif action == "throttle_gpu":
            # Lower GPU clocks — reduces thermal load on CPU
            delta_cpu = random.uniform(5.0, 12.0)
            delta_bat = random.uniform(3.0, 8.0)    # GPU is expensive

        elif action == "hibernate_idle":
            # Hibernate idle processes — modest CPU, good battery
            delta_cpu = random.uniform(3.0, 10.0)
            delta_bat = random.uniform(5.0, 10.0)

        # Clamp values to valid range
        self._cpu     = max(MIN_CPU, min(MAX_CPU,  self._cpu - delta_cpu))
        self._battery = max(MIN_BAT, min(MAX_BAT,  self._battery + delta_bat))

        self._step_count += 1

        # ── Reward ────────────────────────────────────────────────────────────
        actual_cpu_improvement = max(0.0, cpu_before - self._cpu)
        actual_bat_improvement = max(0.0, self._battery - bat_before)
        reward = self._compute_reward(actual_cpu_improvement, actual_bat_improvement)

        # ── Done ──────────────────────────────────────────────────────────────
        done = self.is_done()

        obs = self.get_observation()
        info = {
            "delta_cpu":  round(cpu_before - self._cpu, 3),
            "delta_bat":  round(self._battery - bat_before, 3),
            "step_count": self._step_count,
        }

        return obs, reward, done, info

    def get_observation(self) -> Dict[str, float]:
        return {
            "cpu":     round(self._cpu, 2),
            "battery": round(self._battery, 2),
        }

    def is_done(self) -> bool:
        return (
            (self._cpu <= CPU_THRESHOLD_SAFE and self._battery >= BAT_THRESHOLD_SAFE)
            or self._step_count >= self._max_steps
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_reward(delta_cpu: float, delta_bat: float) -> float:
        """
        Normalise reward to [0.0, 1.0].

        Formula:
            Reward = (ΔCPU + ΔBattery) / MaxPossibleImprovement

        A perfect step that drops CPU by 20 and recovers battery by 10
        yields reward = 1.0.  A step with no improvement yields 0.0.
        """
        raw = delta_cpu + delta_bat
        normalised = raw / MAX_POSSIBLE_IMPROVEMENT
        # Clamp to [0, 1] — should never exceed 1 by design, but be safe
        return round(max(0.0, min(1.0, normalised)), 4)
