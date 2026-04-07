"""
app/env.py — AlphaMatrix RL Environment
OpenEnv-compliant. Three tasks, stochastic transitions, rewards in (0.0, 1.0).
"""

import random
from typing import Any

# ── Task registry ─────────────────────────────────────────────────────────────

TASKS = ["thermal_throttling", "battery_endurance", "process_deadlock"]

# ── Action spaces per task ────────────────────────────────────────────────────

ACTIONS = {
    "thermal_throttling": [
        "reduce_clock_speed",
        "kill_heavy_process",
        "enable_cooling_fan",
        "throttle_gpu",
    ],
    "battery_endurance": [
        "dim_display",
        "disable_wifi",
        "suspend_background_apps",
        "enable_battery_saver",
    ],
    "process_deadlock": [
        "release_mutex",
        "restart_process",
        "increase_timeout",
        "force_schedule",
    ],
}

# ── Per-task terminal thresholds ───────────────────────────────────────────────

THRESHOLDS = {
    "thermal_throttling": {"cpu_temp": 70.0,  "cpu_load": 60.0},   # goal: both below
    "battery_endurance":  {"battery_drain_rate": 5.0},              # goal: drain ≤ 5 %/h
    "process_deadlock":   {"deadlock_score": 10.0},                 # goal: score ≤ 10
}

MAX_STEPS = 50


class AlphaMatrixEnv:
    """
    Three-task RL environment for AlphaMatrix.

    step() return contract (OpenEnv):
        {
            "observation": dict[str, float],
            "reward":      float  ∈ (0.0, 1.0),
            "done":        bool,
            "info":        dict,
        }
    """

    def __init__(self, task: str = "thermal_throttling"):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {TASKS}.")
        self.task          = task
        self.action_space  = ACTIONS[task]
        self._step_count   = 0
        self._state: dict  = {}
        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task: str | None = None) -> dict:
        if task is not None:
            if task not in TASKS:
                raise ValueError(f"Unknown task '{task}'.")
            self.task         = task
            self.action_space = ACTIONS[task]
        self._step_count = 0
        self._reset_state()
        return self._observation()

    def step(self, action: str) -> dict:
        """OpenEnv-compliant step. Always returns reward ∈ (0.0, 1.0)."""
        if action not in self.action_space:
            raise ValueError(f"Invalid action '{action}' for task '{self.task}'.")

        prev_state = dict(self._state)
        self._apply_action(action)
        self._step_count += 1

        reward = self._compute_reward(prev_state)
        done   = self._is_done()

        return {
            "observation": self._observation(),
            "reward":      reward,
            "done":        done,
            "info": {
                "task":       self.task,
                "action":     action,
                "step_count": self._step_count,
                "prev_state": prev_state,
            },
        }

    def get_observation(self) -> dict:
        return self._observation()

    def is_done(self) -> bool:
        return self._is_done()

    # ── Task: Thermal Throttling ───────────────────────────────────────────────

    def _init_thermal(self):
        self._state = {
            "cpu_temp": random.uniform(85.0, 100.0),
            "cpu_load": random.uniform(75.0, 100.0),
            "gpu_temp": random.uniform(70.0, 95.0),
        }

    def _apply_thermal(self, action: str):
        s = self._state
        noise = random.uniform(0.85, 1.15)          # ±15 % stochastic noise

        if action == "reduce_clock_speed":
            s["cpu_temp"] -= 8.0  * noise
            s["cpu_load"] -= 12.0 * noise

        elif action == "kill_heavy_process":
            s["cpu_load"] -= 18.0 * noise
            s["cpu_temp"] -= 4.0  * noise

        elif action == "enable_cooling_fan":
            s["cpu_temp"] -= 10.0 * noise
            s["gpu_temp"] -= 8.0  * noise

        elif action == "throttle_gpu":
            s["gpu_temp"] -= 12.0 * noise
            s["cpu_temp"] -= 2.0  * noise

        # Random environment drift (heat continues to build slightly)
        s["cpu_temp"] += random.uniform(0.0, 1.5)
        s["cpu_load"] += random.uniform(0.0, 2.0)

        # Clamp
        s["cpu_temp"] = max(40.0, min(110.0, s["cpu_temp"]))
        s["cpu_load"] = max(0.0,  min(100.0, s["cpu_load"]))
        s["gpu_temp"] = max(30.0, min(110.0, s["gpu_temp"]))

    def _reward_thermal(self, prev: dict) -> float:
        """
        Reward = how much closer we got to the safe zone.
        Normalised to (0.0, 1.0) via min-max scaling.
        """
        # Improvement = reduction in cpu_temp + reduction in cpu_load
        d_temp = prev["cpu_temp"] - self._state["cpu_temp"]
        d_load = prev["cpu_load"] - self._state["cpu_load"]
        raw    = d_temp + d_load                     # range ≈ [-4, 36]
        # min-max: worst=-4, best=36 → range=40
        scaled = (raw + 4.0) / 40.0
        return float(min(0.999, max(0.001, scaled)))

    def _done_thermal(self) -> bool:
        t = THRESHOLDS["thermal_throttling"]
        return (
            self._state["cpu_temp"] <= t["cpu_temp"]
            and self._state["cpu_load"] <= t["cpu_load"]
        ) or self._step_count >= MAX_STEPS

    # ── Task: Battery Endurance ───────────────────────────────────────────────

    def _init_battery(self):
        self._state = {
            "battery_level":      random.uniform(20.0, 50.0),
            "battery_drain_rate": random.uniform(20.0, 40.0),  # % per hour
            "screen_brightness":  random.uniform(70.0, 100.0),
            "active_radios":      random.uniform(3.0, 5.0),    # number of active radios
        }

    def _apply_battery(self, action: str):
        s     = self._state
        noise = random.uniform(0.85, 1.15)

        if action == "dim_display":
            s["battery_drain_rate"]  -= 4.0 * noise
            s["screen_brightness"]   -= 30.0 * noise

        elif action == "disable_wifi":
            s["battery_drain_rate"]  -= 5.0 * noise
            s["active_radios"]       -= 1.0 * noise

        elif action == "suspend_background_apps":
            s["battery_drain_rate"]  -= 7.0 * noise

        elif action == "enable_battery_saver":
            s["battery_drain_rate"]  -= 10.0 * noise
            s["screen_brightness"]   -= 15.0 * noise

        # Battery drains each step
        s["battery_level"]      -= s["battery_drain_rate"] / 60.0   # 1-min tick
        s["battery_drain_rate"] += random.uniform(-1.0, 1.5)        # drift

        s["battery_level"]      = max(0.0,  min(100.0, s["battery_level"]))
        s["battery_drain_rate"] = max(1.0,  min(60.0,  s["battery_drain_rate"]))
        s["screen_brightness"]  = max(10.0, min(100.0, s["screen_brightness"]))
        s["active_radios"]      = max(0.0,  min(5.0,   s["active_radios"]))

    def _reward_battery(self, prev: dict) -> float:
        d_drain = prev["battery_drain_rate"] - self._state["battery_drain_rate"]
        # range ≈ [-2, 11]  →  shift by 2, divide by 13
        scaled = (d_drain + 2.0) / 13.0
        return float(min(0.999, max(0.001, scaled)))

    def _done_battery(self) -> bool:
        t = THRESHOLDS["battery_endurance"]
        return (
            self._state["battery_drain_rate"] <= t["battery_drain_rate"]
            or self._state["battery_level"] <= 0.0
        ) or self._step_count >= MAX_STEPS

    # ── Task: Process Deadlock ────────────────────────────────────────────────

    def _init_deadlock(self):
        self._state = {
            "deadlock_score":    random.uniform(60.0, 100.0),  # lower = better
            "blocked_processes": random.uniform(5.0, 15.0),
            "mutex_contention":  random.uniform(50.0, 100.0),
            "cpu_wait_time":     random.uniform(200.0, 500.0), # ms
        }

    def _apply_deadlock(self, action: str):
        s     = self._state
        noise = random.uniform(0.85, 1.15)

        if action == "release_mutex":
            s["mutex_contention"]  -= 20.0 * noise
            s["deadlock_score"]    -= 15.0 * noise
            s["blocked_processes"] -= 2.0  * noise

        elif action == "restart_process":
            s["blocked_processes"] -= 4.0  * noise
            s["deadlock_score"]    -= 10.0 * noise
            s["cpu_wait_time"]     -= 50.0 * noise

        elif action == "increase_timeout":
            s["cpu_wait_time"]     -= 80.0 * noise
            s["deadlock_score"]    -= 5.0  * noise

        elif action == "force_schedule":
            s["deadlock_score"]    -= 20.0 * noise
            s["blocked_processes"] -= 3.0  * noise
            s["cpu_wait_time"]     -= 30.0 * noise

        # Stochastic environment: new deadlocks may form
        s["deadlock_score"]    += random.uniform(0.0, 5.0)
        s["blocked_processes"] += random.uniform(0.0, 1.0)

        s["deadlock_score"]    = max(0.0,   min(100.0, s["deadlock_score"]))
        s["blocked_processes"] = max(0.0,   min(20.0,  s["blocked_processes"]))
        s["mutex_contention"]  = max(0.0,   min(100.0, s["mutex_contention"]))
        s["cpu_wait_time"]     = max(0.0,   min(600.0, s["cpu_wait_time"]))

    def _reward_deadlock(self, prev: dict) -> float:
        d_score = prev["deadlock_score"] - self._state["deadlock_score"]
        # range ≈ [-5, 25]  →  shift by 5, divide by 30
        scaled = (d_score + 5.0) / 30.0
        return float(min(0.999, max(0.001, scaled)))

    def _done_deadlock(self) -> bool:
        t = THRESHOLDS["process_deadlock"]
        return (
            self._state["deadlock_score"] <= t["deadlock_score"]
        ) or self._step_count >= MAX_STEPS

    # ── Dispatch helpers ──────────────────────────────────────────────────────

    def _reset_state(self):
        dispatch = {
            "thermal_throttling": self._init_thermal,
            "battery_endurance":  self._init_battery,
            "process_deadlock":   self._init_deadlock,
        }
        dispatch[self.task]()

    def _apply_action(self, action: str):
        dispatch = {
            "thermal_throttling": self._apply_thermal,
            "battery_endurance":  self._apply_battery,
            "process_deadlock":   self._apply_deadlock,
        }
        dispatch[self.task](action)

    def _compute_reward(self, prev: dict) -> float:
        dispatch = {
            "thermal_throttling": self._reward_thermal,
            "battery_endurance":  self._reward_battery,
            "process_deadlock":   self._reward_deadlock,
        }
        return dispatch[self.task](prev)

    def _is_done(self) -> bool:
        dispatch = {
            "thermal_throttling": self._done_thermal,
            "battery_endurance":  self._done_battery,
            "process_deadlock":   self._done_deadlock,
        }
        return dispatch[self.task]()

    def _observation(self) -> dict:
        return {k: round(float(v), 4) for k, v in self._state.items()}