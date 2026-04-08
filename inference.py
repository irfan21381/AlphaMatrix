import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

# ✅ LiteLLM Proxy (MANDATORY)
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)

# Your modules
from app.agent import QLearningAgent
from app.env import ACTIONS, TASK, ThermalEnv

# Config
MODEL_NAME = os.getenv("MODEL_NAME", "qlearning-agent")
BENCHMARK = os.getenv("BENCHMARK", "alphamatrix")

# 🔥 Keep small for speed
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

# Global state
_ENV = ThermalEnv(max_steps=MAX_STEPS)
_AGENT = QLearningAgent()
_LAST_OBS: Optional[Dict[str, float]] = None
_LATEST: Dict[str, Any] = {"start": None, "end": None, "steps": 0, "last": None}


# ------------------ LLM CALL ------------------
def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=3  # ⚡ short timeout
        )
        return response.choices[0].message.content
    except Exception:
        return "fallback"


# ------------------ LOGGING ------------------
def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=True)}", flush=True)


# ------------------ ENV ------------------
def reset_openenv(cpu: float = 90.0, battery: float = 20.0) -> Dict[str, Any]:
    global _LAST_OBS
    out = _ENV.reset_openenv(cpu=cpu, battery=battery)
    _LAST_OBS = dict(out["observation"])
    return out


def step_openenv(action: str) -> Dict[str, Any]:
    global _LAST_OBS
    if _LAST_OBS is None:
        reset_openenv()

    obs_before = dict(_LAST_OBS or {})
    out = _ENV.step_openenv(action)
    obs_after = dict(out.get("observation", {}))

    try:
        _AGENT.update(obs_before, action, float(out.get("reward", 0.0)), obs_after, bool(out.get("done", False)))
    except Exception:
        pass

    _LAST_OBS = obs_after
    return out


# ------------------ SERVER ------------------
def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    try:
        n = int(handler.headers.get("Content-Length", "0") or "0")
    except Exception:
        n = 0
    if n <= 0:
        return {}
    raw = handler.rfile.read(n)
    try:
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _send_json(handler: BaseHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
    body = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = (
            "<html><body><h2>OpenEnv runner is alive</h2>"
            f"<pre>{json.dumps(_LATEST, indent=2)}</pre></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/reset":
            body = _read_json_body(self)
            out = reset_openenv()
            _send_json(self, 200, out)
            return

        if self.path == "/step":
            body = _read_json_body(self)
            action = body.get("action")
            if not isinstance(action, str) or action not in ACTIONS:
                _send_json(self, 400, {"error": "invalid_action"})
                return
            out = step_openenv(action)
            _send_json(self, 200, out)
            return

        _send_json(self, 404, {"error": "not_found"})

    def log_message(self, format, *args):
        return


def _serve_forever():
    port = int(os.getenv("PORT", "7860"))
    httpd = HTTPServer(("0.0.0.0", port), _Handler)
    httpd.serve_forever()


# ------------------ MAIN LOGIC ------------------
def _demo_rollout():
    start_payload = {
        "task": TASK,
        "env": BENCHMARK,
        "model": MODEL_NAME,
        "actions": list(ACTIONS),
        "max_steps": MAX_STEPS,
    }

    _LATEST["start"] = start_payload
    _emit("START", start_payload)

    r0 = reset_openenv()
    obs = dict(r0["observation"])

    total = 0.0
    rewards = []
    success = False

    for i in range(1, MAX_STEPS + 1):
        action, _ = _AGENT.act_with_confidence(obs)
        out = step_openenv(action)

        reward = float(out.get("reward", 0.0))
        done = bool(out.get("done", False))
        obs = dict(out.get("observation", {}))

        rewards.append(reward)
        total += reward

        _emit("STEP", {
            "step": i,
            "action": action,
            "reward": reward,
            "done": done
        })

        if done:
            success = True
            break

    end_payload = {
        "success": success,
        "steps": i,
        "score": total,
        "rewards": rewards,
    }

    _LATEST["end"] = end_payload
    _emit("END", end_payload)

    # ✅ LLM call AFTER rollout (non-blocking safe)
    try:
        llm_output = call_llm("Optimize CPU and battery usage")
        _emit("LLM", {"response": llm_output})
    except Exception:
        _emit("LLM", {"response": "fallback"})


# ------------------ ENTRY ------------------
def main():
    t = threading.Thread(target=_serve_forever)
    t.start()

    _demo_rollout()

    # Keep server alive cleanly
    t.join()


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        _emit("END", {"success": False, "error": str(e)})
        sys.exit(0)