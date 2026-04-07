import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


_LATEST = {"start": None, "end": None, "steps": 0}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = (
            "<html><head><title>OpenEnv Runner</title></head><body>"
            "<h2>OpenEnv runner is alive</h2>"
            "<p>This Space runs <code>python inference.py</code> for hackathon validation.</p>"
            f"<pre>{json.dumps(_LATEST, indent=2)}</pre>"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


def _start_spaces_server() -> None:
    port = int(os.getenv("PORT", "7860"))
    httpd = HTTPServer(("0.0.0.0", port), _Handler)
    httpd.serve_forever()


def run() -> None:
    env = ThermalEnv(max_steps=MAX_STEPS)
    agent = QLearningAgent()

    start_payload = {
        "task": TASK,
        "env": BENCHMARK,
        "model": MODEL_NAME,
        "actions": list(ACTIONS),
        "max_steps": MAX_STEPS,
    }
    _LATEST["start"] = start_payload
    _LATEST["end"] = None
    _LATEST["steps"] = 0

    # Spaces requires a listening web port to leave "Starting..."
    if _running_on_hf_spaces():
        t = threading.Thread(target=_start_spaces_server, daemon=True)
        t.start()

    _emit(
        "START",
        start_payload,
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
        _LATEST["steps"] = i

        obs = next_obs
        if done:
            success = True
            break

    score = float(total)
    end_payload = {
        "success": bool(success),
        "steps": int(steps),
        "score": score,
        "rewards": rewards,
    }
    _LATEST["end"] = end_payload
    _emit(
        "END",
        end_payload,
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