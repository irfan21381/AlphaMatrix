# import json
# import os
# import sys
# import time
# import threading
# from http.server import BaseHTTPRequestHandler, HTTPServer
# from typing import Any, Dict, List, Optional

# from app.agent import QLearningAgent
# from app.env import ACTIONS, TASK, ThermalEnv

# MODEL_NAME = os.getenv("MODEL_NAME", "qlearning-agent")
# BENCHMARK = os.getenv("BENCHMARK", "alphamatrix")
# MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))

# # Shared in-process OpenEnv state (no FastAPI, no external APIs)
# _ENV = ThermalEnv(max_steps=MAX_STEPS)
# _AGENT = QLearningAgent()
# _LAST_OBS: Optional[Dict[str, float]] = None

# _LATEST: Dict[str, Any] = {"start": None, "end": None, "steps": 0, "last": None}


# def _emit(tag: str, payload: Dict[str, Any]) -> None:
#     print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=True)}", flush=True)


# def reset_openenv(cpu: float = 90.0, battery: float = 20.0) -> Dict[str, Any]:
#     """
#     OpenEnv Reset payload:
#     {
#       "observation": {...},
#       "reward": 0,
#       "done": false,
#       "info": {}
#     }
#     """
#     global _LAST_OBS
#     out = _ENV.reset_openenv(cpu=cpu, battery=battery)
#     _LAST_OBS = dict(out["observation"])
#     return out


# def step_openenv(action: str) -> Dict[str, Any]:
#     global _LAST_OBS
#     if _LAST_OBS is None:
#         reset_openenv()

#     obs_before = dict(_LAST_OBS or {})
#     out = _ENV.step_openenv(action)
#     obs_after = dict(out.get("observation", {}))

#     # Optional online learning (not required by OpenEnv, but safe)
#     try:
#         _AGENT.update(obs_before, action, float(out.get("reward", 0.0)), obs_after, bool(out.get("done", False)))
#     except Exception:
#         pass

#     _LAST_OBS = obs_after
#     return out


# def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
#     try:
#         n = int(handler.headers.get("Content-Length", "0") or "0")
#     except Exception:
#         n = 0
#     if n <= 0:
#         return {}
#     raw = handler.rfile.read(n)
#     try:
#         obj = json.loads(raw.decode("utf-8"))
#         return obj if isinstance(obj, dict) else {}
#     except Exception:
#         return {}


# def _send_json(handler: BaseHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
#     body = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
#     handler.send_response(status)
#     handler.send_header("Content-Type", "application/json; charset=utf-8")
#     handler.send_header("Content-Length", str(len(body)))
#     handler.end_headers()
#     handler.wfile.write(body)


# class _Handler(BaseHTTPRequestHandler):
#     def do_GET(self):  # noqa: N802
#         # Spaces / proxies may hit paths with query params like "/?logs=container".
#         # For any GET, serve a small health page; OpenEnv checks use POST for API calls.
#         body = (
#             "<html><head><title>OpenEnv Runner</title></head><body>"
#             "<h2>OpenEnv runner is alive</h2>"
#             "<p>Endpoints: <code>POST /reset</code>, <code>POST /step</code></p>"
#             f"<pre>{json.dumps(_LATEST, indent=2)}</pre>"
#             "</body></html>"
#         ).encode("utf-8")
#         self.send_response(200)
#         self.send_header("Content-Type", "text/html; charset=utf-8")
#         self.send_header("Content-Length", str(len(body)))
#         self.end_headers()
#         self.wfile.write(body)
#         return

#     def do_POST(self):  # noqa: N802
#         if self.path == "/reset":
#             body = _read_json_body(self)
#             cpu = float(body.get("cpu", 90.0))
#             battery = float(body.get("battery", 20.0))
#             out = reset_openenv(cpu=cpu, battery=battery)
#             _LATEST["last"] = {"reset": out}
#             _send_json(self, 200, out)
#             return

#         if self.path == "/step":
#             body = _read_json_body(self)
#             action = body.get("action")
#             if not isinstance(action, str) or action not in ACTIONS:
#                 _send_json(self, 400, {"error": "invalid_action", "actions": list(ACTIONS)})
#                 return
#             out = step_openenv(action)
#             _LATEST["last"] = {"step": out}
#             _send_json(self, 200, out)
#             return

#         _send_json(self, 404, {"error": "not_found"})

#     def log_message(self, format, *args):  # noqa: A002
#         return


# def _serve_forever() -> None:
#     port = int(os.getenv("PORT", "7860"))
#     httpd = HTTPServer(("0.0.0.0", port), _Handler)
#     httpd.serve_forever()


# def _demo_rollout() -> None:
#     start_payload = {
#         "task": TASK,
#         "env": BENCHMARK,
#         "model": MODEL_NAME,
#         "actions": list(ACTIONS),
#         "max_steps": MAX_STEPS,
#     }
#     _LATEST["start"] = start_payload
#     _LATEST["end"] = None
#     _LATEST["steps"] = 0
#     _emit("START", start_payload)

#     r0 = reset_openenv(cpu=90.0, battery=20.0)
#     obs = dict(r0["observation"])

#     rewards: List[float] = []
#     total = 0.0
#     steps = 0
#     success = False

#     for i in range(1, MAX_STEPS + 1):
#         action, _conf = _AGENT.act_with_confidence(obs)
#         out = step_openenv(action)
#         next_obs = dict(out.get("observation", {}))
#         reward = float(out.get("reward", 0.0))
#         done = bool(out.get("done", False))

#         rewards.append(reward)
#         total += reward
#         steps = i

#         _emit(
#             "STEP",
#             {
#                 "step": i,
#                 "action": action,
#                 "reward": reward,
#                 "observation": next_obs,
#                 "done": done,
#             },
#         )
#         _LATEST["steps"] = i

#         obs = next_obs
#         if done:
#             success = True
#             break

#     end_payload = {
#         "success": bool(success),
#         "steps": int(steps),
#         "score": float(total),
#         "rewards": rewards,
#     }
#     _LATEST["end"] = end_payload
#     _emit("END", end_payload)


# def main() -> None:
#     # Start OpenEnv HTTP API for POST /reset validation
#     t = threading.Thread(target=_serve_forever, daemon=True)
#     t.start()

#     # Emit one rollout worth of logs for inference.py validation
#     _demo_rollout()

#     # Keep container alive
#     while True:
#         time.sleep(3600)


# if __name__ == "__main__":
#     try:
#         main()
#         sys.exit(0)
#     except Exception as e:
#         _emit("END", {"success": False, "steps": 0, "score": 0.0, "rewards": [], "error": str(e)})
#         sys.exit(0)
# import json
# import os
# import sys
# import time
# import threading
# from http.server import BaseHTTPRequestHandler, HTTPServer
# from typing import Dict, List

# from app.agent import QLearningAgent
# from app.env import ACTIONS, TASK, ThermalEnv

# MODEL_NAME = os.getenv("MODEL_NAME", "qlearning-agent")
# BENCHMARK = os.getenv("BENCHMARK", "alphamatrix")
# MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


# def reset(env: ThermalEnv, cpu: float = 90.0, battery: float = 20.0) -> Dict:
#     """
#     OpenEnv-compatible reset payload:
#     {
#       "observation": {...},
#       "reward": 0,
#       "done": false,
#       "info": {}
#     }
#     """
#     return env.reset_openenv(cpu=cpu, battery=battery)


# def step(env: ThermalEnv, action: str) -> Dict:
#     """
#     OpenEnv-style step payload:
#     {
#       "observation": {...},
#       "reward": <float>,
#       "done": <bool>,
#       "info": {...}
#     }
#     """
#     return env.step_openenv(action)


# def _emit(tag: str, payload: Dict) -> None:
#     print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=True)}", flush=True)


# def _running_on_hf_spaces() -> bool:
#     # HuggingFace Spaces sets SPACE_ID for running apps.
#     return bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")


# _LATEST = {"start": None, "end": None, "steps": 0}


# class _Handler(BaseHTTPRequestHandler):
#     def do_GET(self):  # noqa: N802
#         body = (
#             "<html><head><title>OpenEnv Runner</title></head><body>"
#             "<h2>OpenEnv runner is alive</h2>"
#             "<p>This Space runs <code>python inference.py</code> for hackathon validation.</p>"
#             f"<pre>{json.dumps(_LATEST, indent=2)}</pre>"
#             "</body></html>"
#         ).encode("utf-8")
#         self.send_response(200)
#         self.send_header("Content-Type", "text/html; charset=utf-8")
#         self.send_header("Content-Length", str(len(body)))
#         self.end_headers()
#         self.wfile.write(body)

#     def log_message(self, format, *args):  # noqa: A002
#         return


# def _start_spaces_server() -> None:
#     port = int(os.getenv("PORT", "7860"))
#     httpd = HTTPServer(("0.0.0.0", port), _Handler)
#     httpd.serve_forever()


# def run() -> None:
#     env = ThermalEnv(max_steps=MAX_STEPS)
#     agent = QLearningAgent()

#     start_payload = {
#         "task": TASK,
#         "env": BENCHMARK,
#         "model": MODEL_NAME,
#         "actions": list(ACTIONS),
#         "max_steps": MAX_STEPS,
#     }
#     _LATEST["start"] = start_payload
#     _LATEST["end"] = None
#     _LATEST["steps"] = 0

#     # Spaces requires a listening web port to leave "Starting..."
#     if _running_on_hf_spaces():
#         t = threading.Thread(target=_start_spaces_server, daemon=True)
#         t.start()

#     _emit(
#         "START",
#         start_payload,
#     )

#     r0 = reset(env, cpu=90.0, battery=20.0)
#     obs = dict(r0["observation"])

#     rewards: List[float] = []
#     steps = 0
#     total = 0.0
#     success = False

#     for i in range(1, MAX_STEPS + 1):
#         action, _conf = agent.act_with_confidence(obs)
#         out = step(env, action)

#         reward = float(out.get("reward", 0.0))
#         done = bool(out.get("done", False))
#         next_obs = dict(out.get("observation", {}))

#         agent.update(obs, action, reward, next_obs, done)
#         rewards.append(reward)
#         total += reward
#         steps = i

#         _emit(
#             "STEP",
#             {
#                 "step": i,
#                 "action": action,
#                 "reward": reward,
#                 "observation": next_obs,
#                 "done": done,
#             },
#         )
#         _LATEST["steps"] = i

#         obs = next_obs
#         if done:
#             success = True
#             break

#     score = float(total)
#     end_payload = {
#         "success": bool(success),
#         "steps": int(steps),
#         "score": score,
#         "rewards": rewards,
#     }
#     _LATEST["end"] = end_payload
#     _emit(
#         "END",
#         end_payload,
#     )

#     # Spaces expects a long-running process; a clean exit is shown as "Runtime error".
#     if _running_on_hf_spaces():
#         while True:
#             time.sleep(3600)


# if __name__ == "__main__":
#     try:
#         run()
#         sys.exit(0)
#     except Exception as e:
#         _emit("END", {"success": False, "steps": 0, "score": 0.0, "rewards": [], "error": str(e)})
#         sys.exit(0)

import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

# Phase 2 Recovery: Prevent crash if dependency fails
try:
    import requests
except ImportError:
    print(f'[END] {json.dumps({"success": False, "error": "requests module missing", "score": 0.0})}')
    sys.exit(0)

from app.agent import QLearningAgent
from app.env import ACTIONS, TASK, ThermalEnv

# Environment Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "qlearning-agent")
BENCHMARK = os.getenv("BENCHMARK", "alphamatrix")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))

# Shared in-process OpenEnv state
_ENV = ThermalEnv(max_steps=MAX_STEPS)
_AGENT = QLearningAgent()
_LAST_OBS: Optional[Dict[str, float]] = None
_LATEST: Dict[str, Any] = {"start": None, "end": None, "steps": 0, "last": None}

def _emit(tag: str, payload: Dict[str, Any]) -> None:
    """Standardized OpenEnv Logging format."""
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=True)}", flush=True)

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

def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    try:
        n = int(handler.headers.get("Content-Length", "0") or "0")
    except Exception:
        n = 0
    if n <= 0: return {}
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
            "<html><head><title>OpenEnv Runner</title></head><body>"
            "<h2>OpenEnv runner is alive</h2>"
            "<p>Endpoints: <code>POST /reset</code>, <code>POST /step</code></p>"
            f"<pre>{json.dumps(_LATEST, indent=2)}</pre>"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/reset":
            body = _read_json_body(self)
            cpu = float(body.get("cpu", 90.0))
            battery = float(body.get("battery", 20.0))
            out = reset_openenv(cpu=cpu, battery=battery)
            _LATEST["last"] = {"reset": out}
            _send_json(self, 200, out)
            return

        if self.path == "/step":
            body = _read_json_body(self)
            action = body.get("action")
            if not isinstance(action, str) or action not in ACTIONS:
                _send_json(self, 400, {"error": "invalid_action", "actions": list(ACTIONS)})
                return
            out = step_openenv(action)
            _LATEST["last"] = {"step": out}
            _send_json(self, 200, out)
            return
        _send_json(self, 404, {"error": "not_found"})

    def log_message(self, format, *args): return

def _serve_forever() -> None:
    port = int(os.getenv("PORT", "7860")) # Use 8000 for internal backend spec
    httpd = HTTPServer(("0.0.0.0", port), _Handler)
    httpd.serve_forever()

def _demo_rollout() -> None:
    start_payload = {
        "task": TASK,
        "env": BENCHMARK,
        "model": MODEL_NAME,
        "actions": list(ACTIONS),
        "max_steps": MAX_STEPS,
    }
    _LATEST["start"] = start_payload
    _emit("START", start_payload)

    r0 = reset_openenv(cpu=90.0, battery=20.0)
    obs = dict(r0["observation"])

    rewards: List[float] = []
    total = 0.0
    steps = 0
    success = False

    for i in range(1, MAX_STEPS + 1):
        action, _conf = _AGENT.act_with_confidence(obs)
        out = step_openenv(action)
        next_obs = dict(out.get("observation", {}))
        reward = float(out.get("reward", 0.0))
        done = bool(out.get("done", False))

        rewards.append(reward)
        total += reward
        steps = i

        _emit("STEP", {
            "step": i,
            "action": action,
            "reward": reward,
            "observation": next_obs,
            "done": done,
        })
        _LATEST["steps"] = i
        obs = next_obs
        if done:
            success = True
            break

    end_payload = {
        "success": bool(success),
        "steps": int(steps),
        "score": float(total),
        "rewards": rewards,
    }
    _LATEST["end"] = end_payload
    _emit("END", end_payload)

#def main() -> None:
    # Start OpenEnv API for POST /reset validation on port 8000
   # t = threading.Thread(target=_serve_forever, daemon=True)
    #
    
    # Run a single rollout to generate logs for the grader
   # _demo_rollout()

    # Keep container alive for agentic evaluation
   # while True:
       # time.sleep(3600)


def main() -> None:
    # Start OpenEnv API
    t = threading.Thread(target=_serve_forever)
    t.start()
    
    # Run rollout (fast execution)
    _demo_rollout()

    # Keep server alive properly (IMPORTANT)
    t.join()

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        _emit("END", {"success": False, "score": 0.0, "error": str(e)})
        sys.exit(0)