"""
app.py — RL-LLM Thermal Manager  (HuggingFace Spaces, Python 3.13)
FastAPI backend runs in a daemon thread on port 8000.
Streamlit frontend runs on port 7860 (HF required port).
No version-pinned packages — uses whatever HF installs.
"""

import os
import time
import random
import threading
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import streamlit as st
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — RL ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

ACTIONS: List[str] = [
    "optimize_cpu",
    "close_apps",
    "throttle_gpu",
    "hibernate_idle",
]
CPU_SAFE     = 60.0
BAT_SAFE     = 50.0
MAX_IMPROVE  = 30.0   # max CPU drop (20) + max battery gain (10)


class DisasterEnv:
    def __init__(self):
        self._cpu  = 85.0
        self._bat  = 30.0
        self._step = 0

    def reset(self, cpu: float = 85.0, battery: float = 30.0) -> dict:
        self._cpu, self._bat, self._step = float(cpu), float(battery), 0
        st.session_state.learning_curve = []
        st.session_state.confidence_hist = []
        return self._obs()

    def step(self, action: str):
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        c0, b0 = self._cpu, self._bat
        if action == "optimize_cpu":
            dc, db = random.uniform(8, 18),  random.uniform(-1, 2)
        elif action == "close_apps":
            dc, db = random.uniform(10, 20), random.uniform(1, 5)
        elif action == "throttle_gpu":
            dc, db = random.uniform(5, 12),  random.uniform(3, 8)
        else:  # hibernate_idle
            dc, db = random.uniform(3, 10),  random.uniform(5, 10)

        self._cpu  = max(0.0, min(100.0, self._cpu - dc))
        self._bat  = max(0.0, min(100.0, self._bat + db))
        self._step += 1

        reward = round(
            min(1.0, (max(0, c0 - self._cpu) + max(0, self._bat - b0)) / MAX_IMPROVE), 4
        )
        done = (self._cpu <= CPU_SAFE and self._bat >= BAT_SAFE) or self._step >= 50
        info = {
            "delta_cpu": round(c0 - self._cpu, 3),
            "delta_bat": round(self._bat - b0, 3),
            "step":      self._step,
        }
        return self._obs(), reward, done, info

    def _obs(self):
        return {"cpu": round(self._cpu, 2), "battery": round(self._bat, 2)}

    def is_done(self):
        return self._cpu <= CPU_SAFE and self._bat >= BAT_SAFE


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — FASTAPI BACKEND  (daemon thread on port 8000)
# ═══════════════════════════════════════════════════════════════════════════════

_lock         = threading.Lock()
_env          = DisasterEnv()
_episode_step = 0
_total_reward = 0.0
_history: list = []
_initialized  = False
_init_params: dict = {}


class InitSchema(BaseModel):
    cpu:                  float = Field(default=85.0, ge=0, le=100)
    battery:              float = Field(default=30.0, ge=0, le=100)
    incident_description: Optional[str] = "System under high load"


class StepSchema(BaseModel):
    action: str


class ExplainSchema(BaseModel):
    action:       str
    state_before: dict
    state_after:  dict



app = FastAPI(title="RL Thermal Manager")
api = app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def api_reset(init: Optional[InitSchema] = None):
    global _episode_step, _total_reward, _history, _initialized, _init_params
    p = init or InitSchema()
    with _lock:
        obs           = _env.reset(cpu=p.cpu, battery=p.battery)
        _episode_step = 0
        _total_reward = 0.0
        _history      = []
        _initialized  = True
        _init_params  = {
            "cpu": p.cpu, "battery": p.battery,
            "incident_description": p.incident_description,
        }
    return {"status": "reset", "observation": obs, "init_params": _init_params}


@app.post("/step")
def api_step(body: StepSchema):
    global _episode_step, _total_reward, _history
    if not _initialized:
        raise HTTPException(400, "Call /reset first.")
    with _lock:
        obs, reward, done, info = _env.step(body.action)
        _episode_step += 1
        _total_reward += reward
        rec = {
            "step":              _episode_step,
            "action":            body.action,
            "observation":       obs,
            "reward":            round(reward, 4),
            "cumulative_reward": round(_total_reward, 4),
            "done":              done,
            "info":              info,
        }
        _history.append(rec)
    return rec


@app.get("/state")
def api_state():
    if not _initialized:
        raise HTTPException(400, "Call /reset first.")
    with _lock:
        return {
            "observation":  _env._obs(),
            "step":         _episode_step,
            "total_reward": round(_total_reward, 4),
            "done":         _env.is_done(),
            "init_params":  _init_params,
        }


@app.post("/explain")
def api_explain(body: ExplainSchema):
    s0, s1 = body.state_before, body.state_after
    dc  = s0.get("cpu", 0) - s1.get("cpu", 0)
    db  = s1.get("battery", 0) - s0.get("battery", 0)
    msgs = {
        "optimize_cpu":   f"CPU scheduler rebalanced. CPU dropped {dc:.1f}%.",
        "close_apps":     f"Background apps killed. CPU ↓{dc:.1f}%, Battery ↑{db:.1f}%.",
        "throttle_gpu":   f"GPU clocks throttled. CPU ↓{dc:.1f}%, Battery ↑{db:.1f}%.",
        "hibernate_idle": f"Idle procs hibernated. CPU ↓{dc:.1f}%, Battery ↑{db:.1f}%.",
    }
    sev = "CRITICAL" if s0.get("cpu", 0) > 90 else ("HIGH" if s0.get("cpu", 0) > 75 else "MODERATE")
    return {
        "action":               body.action,
        "rationale":            msgs.get(body.action, f"Applied {body.action}."),
        "severity_at_decision": sev,
        "delta":                {"cpu": round(dc, 2), "battery": round(db, 2)},
    }


# def _run_backend():
#     uvicorn.run(api, host="0.0.0.0", port=8000, log_level="error")


# # ── Start backend thread exactly once ────────────────────────────────────────
# if "backend_started" not in st.session_state:
#     t = threading.Thread(target=_run_backend, daemon=True)
#     t.start()
#     st.session_state["backend_started"] = True
#     time.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — STREAMLIT FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════

# BACKEND  = "http://127.0.0.1:8000"
INTERVAL = 0.8

ACTION_COLORS = {
    "optimize_cpu":   "#00d4ff",
    "close_apps":     "#ff6b6b",
    "throttle_gpu":   "#ffd166",
    "hibernate_idle": "#06d6a0",
}
DARK = "#0d1117"
GRID = "#21262d"
TXT  = "#c9d1d9"
ACC  = "#58a6ff"
RED  = "#ff6b6b"
GRN  = "#06d6a0"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ RL Thermal Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
html,body,[class*="css"]{
    font-family:'JetBrains Mono',monospace!important;
    background:#0d1117!important;color:#c9d1d9!important;
}
.stApp{background:#0d1117!important;}
[data-testid="metric-container"]{
    background:#161b22;border:1px solid #30363d;
    border-radius:8px;padding:12px;
}
[data-testid="metric-container"] label{
    color:#8b949e!important;font-size:11px!important;
    text-transform:uppercase;letter-spacing:.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    color:#58a6ff!important;font-size:22px!important;font-weight:700!important;
}
.stButton>button{
    background:linear-gradient(135deg,#1f6feb,#388bfd)!important;
    color:#fff!important;border:none!important;border-radius:6px!important;
    font-weight:600!important;padding:.45rem 1.2rem;
}
[data-testid="stSidebar"]{
    background:#161b22!important;border-right:1px solid #21262d!important;
}
.stTextArea textarea,.stTextInput input{
    background:#161b22!important;color:#c9d1d9!important;
    border:1px solid #30363d!important;border-radius:6px!important;
}
hr{border-color:#21262d!important;}
</style>
""", unsafe_allow_html=True)

# ── session state defaults ────────────────────────────────────────────────────
for _k, _v in dict(
    cpu_hist=[], bat_hist=[], rew_hist=[], act_hist=[], steps=[],
    total_reward=0.0, cur_cpu=None, cur_bat=None, initialized=False,
    rationale="_Reset the environment, then run a step to see AI rationale._",
    status="Awaiting reset…", log=[],
).items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── backend helpers ───────────────────────────────────────────────────────────

def _post(path, payload=None):
    try:
        url = path if path.startswith("http") else f"{path}"
        r = requests.post(url, json=payload or {}, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _get(path):
    try:
        url = path if path.startswith("http") else f"{path}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def _record(step, cpu, bat, rew, act):
    st.session_state.steps.append(step)
    st.session_state.cpu_hist.append(cpu)
    st.session_state.bat_hist.append(bat)
    st.session_state.rew_hist.append(rew)
    st.session_state.act_hist.append(act)
    st.session_state.total_reward = round(st.session_state.total_reward + rew, 4)
    # add below your existing code inside _record()

    if "learning_curve" not in st.session_state:
        st.session_state.learning_curve = []
    st.session_state.learning_curve.append(st.session_state.total_reward)

    if "confidence_hist" not in st.session_state:
        st.session_state.confidence_hist = []

    # simple confidence (temporary safe version)
    st.session_state.confidence_hist.append([0.25, 0.25, 0.25, 0.25])


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.append(f"[{ts}] {msg}")
    st.session_state.log = st.session_state.log[-100:]


def _explain(action, before, after) -> str:
    data, err = _post("/explain", {
        "action": action, "state_before": before, "state_after": after,
    })
    if err:
        return f"⚠ Explain error: {err}"
    return (
        f"### 🤖 AI Rationale\n\n"
        f"**Action:** `{data['action']}`  \n"
        f"**Severity:** `{data['severity_at_decision']}`\n\n"
        f"{data['rationale']}\n\n"
        f"**Δ CPU:** `{data['delta']['cpu']:+.2f}%`  |  "
        f"**Δ Battery:** `{data['delta']['battery']:+.2f}%`"
    )

# ── plots ─────────────────────────────────────────────────────────────────────

def _cpu_plot():
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=DARK)
    ax.set_facecolor(DARK)
    steps = st.session_state.steps
    cpus  = st.session_state.cpu_hist
    bats  = st.session_state.bat_hist
    acts  = st.session_state.act_hist

    if steps:
        ax.plot(steps, cpus, color=RED, lw=2, label="CPU %",     zorder=3)
        ax.plot(steps, bats, color=GRN, lw=2, ls="--", label="Battery %", zorder=3)
        ax.axhspan(80, 100, alpha=0.07, color=RED)
        ax.axhline(60, color=ACC, lw=0.8, ls=":", alpha=0.6)
        for s, c, a in zip(steps, cpus, acts):
            ax.scatter(s, c, color=ACTION_COLORS.get(a, "#fff"), s=45, zorder=5, edgecolors="none")
        ax.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=TXT, fontsize=8, loc="upper right")
        ax.set_xlim(max(0, len(steps) - 25), max(1, len(steps)))
    else:
        ax.text(.5, .5, "No data yet — reset and run a step",
                ha="center", va="center", color=TXT, fontsize=9, transform=ax.transAxes)

    ax.set_ylim(0, 105)
    ax.set_xlabel("Step", color=TXT, fontsize=9)
    ax.set_ylabel("%", color=TXT, fontsize=9)
    ax.set_title("CPU Stability & Battery Level", color=TXT, fontsize=11, pad=8)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return fig


def _reward_plot():
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=DARK)
    ax.set_facecolor(DARK)
    steps = st.session_state.steps
    rews  = st.session_state.rew_hist

    if steps:
        cols = ["#06d6a0" if r >= .6 else ("#ffd166" if r >= .3 else "#ff6b6b") for r in rews]
        ax.bar(steps, rews, color=cols, width=0.7, zorder=3)
        cum = np.cumsum(rews)
        ax2 = ax.twinx()
        ax2.plot(steps, cum, color=ACC, lw=1.5, ls="--", zorder=4)
        ax2.set_ylabel("Cumulative", color=ACC, fontsize=8)
        ax2.tick_params(colors=ACC, labelsize=7)
        ax2.set_facecolor(DARK)
        for sp in ax2.spines.values():
            sp.set_edgecolor(GRID)
        ax.set_xlim(max(0, len(steps) - 25), max(1, len(steps)))
    else:
        ax.text(.5, .5, "No reward data yet",
                ha="center", va="center", color=TXT, fontsize=9, transform=ax.transAxes)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Step", color=TXT, fontsize=9)
    ax.set_ylabel("Reward (norm.)", color=TXT, fontsize=9)
    ax.set_title("Reward Gradient", color=TXT, fontsize=11, pad=8)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return fig

# ── autopilot helper ──────────────────────────────────────────────────────────

def _autopilot_action(obs: dict):
    if random.random() < 0.5:
        cpu, bat = obs["cpu"], obs["battery"]
        if cpu > 85:    act = "close_apps"
        elif bat < 25:  act = "throttle_gpu"
        elif cpu > 70:  act = "optimize_cpu"
        else:           act = "hibernate_idle"
        return act, "RL"
    w = [0.25, 0.35 if obs["cpu"] > 80 else 0.15, 0.20, 0.20]
    return random.choices(ACTIONS, weights=w, k=1)[0], "LLM"

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.markdown("---")
    cpu_init = st.slider("Starting CPU %",     0.0, 100.0, 90.0, 0.5)
    bat_init = st.slider("Starting Battery %", 0.0, 100.0, 20.0, 0.5)
    incident = st.text_area("Incident Description",
                             value="Thermal spike after GPU benchmark", height=80)

    if st.button("🔄 Reset Environment", use_container_width=True):
        data, err = _post("/reset", {
            "cpu": cpu_init, "battery": bat_init,
            "incident_description": incident,
        })
        if err:
            st.error(f"Reset failed: {err}")
        else:
            for k in ["cpu_hist","bat_hist","rew_hist","act_hist","steps","log"]:
                st.session_state[k] = []
            st.session_state.total_reward = 0.0
            obs = data["observation"]
            st.session_state.cur_cpu     = obs["cpu"]
            st.session_state.cur_bat     = obs["battery"]
            st.session_state.initialized = True
            st.session_state.status      = f"✅ Reset — CPU={obs['cpu']}%  Battery={obs['battery']}%"
            st.session_state.rationale   = "### 🔄 Reset\n\nEnvironment ready. Run a step."
            _log(f"RESET cpu={obs['cpu']} bat={obs['battery']}")
            st.rerun()

    st.markdown("---")
    st.markdown("## 🎮 Manual Step")
    action_sel = st.selectbox("Action", ACTIONS)

    if st.button("▶ Execute Step", use_container_width=True):
        if not st.session_state.initialized:
            st.warning("Please reset the environment first.")
        else:
            sd, err = _get("/state")
            if err:
                st.error(err)
            else:
                obs_before = sd["observation"]
                step_no    = sd["step"] + 1
                rd, err    = _post("/step", {"action": action_sel})
                if err:
                    st.error(err)
                else:
                    o = rd["observation"]
                    _record(step_no, o["cpu"], o["battery"], rd["reward"], action_sel)
                    st.session_state.cur_cpu   = o["cpu"]
                    st.session_state.cur_bat   = o["battery"]
                    st.session_state.rationale = _explain(action_sel, obs_before, o)
                    tag = " 🎯 Goal!" if rd["done"] else ""
                    st.session_state.status = (
                        f"[manual] step={step_no}  {action_sel}  "
                        f"reward={rd['reward']:.4f}{tag}"
                    )
                    _log(f"STEP {step_no} {action_sel} r={rd['reward']:.4f} "
                         f"cpu={o['cpu']} bat={o['battery']}")
                    st.rerun()

    st.markdown("---")
    st.markdown("## 🚀 Autopilot")
    n_steps  = st.slider("Steps", 5, 50, 20, 1)
    run_auto = st.button("▶▶ Run Autopilot", use_container_width=True)

# ── main area ─────────────────────────────────────────────────────────────────

st.markdown("# ⚡ RL Thermal Manager — Live Debugger")
st.markdown("**Hybrid RL + LLM Decision Engine**  ·  Real-time State Machine Inspector")
st.markdown("---")

st.info(st.session_state.status)

c1, c2, c3, c4 = st.columns(4)
with c1:
    v = f"{st.session_state.cur_cpu:.1f}%" if st.session_state.cur_cpu is not None else "—"
    st.metric("🌡 CPU", v)
with c2:
    v = f"{st.session_state.cur_bat:.1f}%" if st.session_state.cur_bat is not None else "—"
    st.metric("🔋 Battery", v)
with c3:
    st.metric("📍 Steps", len(st.session_state.steps))
with c4:
    st.metric("🏆 Total Reward", f"{st.session_state.total_reward:.4f}")

st.markdown("---")
st.markdown("### 📈 Live Graphs")
gc1, gc2 = st.columns(2)
with gc1:
    st.pyplot(_cpu_plot(),    use_container_width=True)
with gc2:
    st.pyplot(_reward_plot(), use_container_width=True)

st.markdown("---")
st.markdown("### 💡 AI Rationale")
st.markdown(st.session_state.rationale)

with st.expander("📋 Step Log", expanded=False):
    if st.session_state.log:
        st.code("\n".join(reversed(st.session_state.log[-50:])), language="text")
    else:
        st.caption("No steps yet.")

# ── autopilot loop ────────────────────────────────────────────────────────────

if run_auto:
    if not st.session_state.initialized:
        st.warning("Please reset the environment first.")
    else:
        bar = st.empty()
        bar.info("🚀 Autopilot starting…")
        for i in range(int(n_steps)):
            sd, err = _get("/state")
            if err:
                bar.error(f"Aborted: {err}")
                break
            obs      = sd["observation"]
            ob0      = dict(obs)
            sno      = sd["step"] + 1
            act, src = _autopilot_action(obs)
            rd, err  = _post("/step", {"action": act})
            if err:
                bar.error(f"Step failed: {err}")
                break
            o = rd["observation"]
            _record(sno, o["cpu"], o["battery"], rd["reward"], act)
            st.session_state.cur_cpu   = o["cpu"]
            st.session_state.cur_bat   = o["battery"]
            st.session_state.rationale = f"**[{src}]** " + _explain(act, ob0, o)
            tag = " 🎯 Goal!" if rd["done"] else ""
            st.session_state.status = (
                f"[{src}] step={sno}  {act}  "
                f"reward={rd['reward']:.4f}  "
                f"cpu={o['cpu']:.1f}%  bat={o['battery']:.1f}%{tag}"
            )
            _log(f"AUTO {sno} [{src}] {act} r={rd['reward']:.4f} "
                 f"cpu={o['cpu']} bat={o['battery']}")
            bar.info(f"🚀 Step {i+1}/{n_steps} — `{act}` [{src}]  reward={rd['reward']:.4f}{tag}")
            time.sleep(INTERVAL)
            if rd["done"]:
                bar.success(
                    f"🎯 Goal reached in {sno} steps!  "
                    f"Total reward = {st.session_state.total_reward:.4f}"
                )
                break
        else:
            bar.success(
                f"✅ Autopilot finished {n_steps} steps.  "
                f"Total reward = {st.session_state.total_reward:.4f}"
            )
        st.rerun()


# # ===================== 🔥 NEW IMPORTS =====================
# from app.agent import QLearningAgent

# # ===================== 🔥 HYBRID AGENT =====================
# class HybridAgent:
#     def __init__(self):
#         self.rl_agent = QLearningAgent()
#         self.memory = []
#         self.max_memory = 10

#     def remember(self, state):
#         self.memory.append(state)
#         if len(self.memory) > self.max_memory:
#             self.memory.pop(0)

#     def plan(self, obs):
#         rl_probs = self.rl_agent.get_q_confidence(obs)

#         # Simulated LLM (can replace with OpenAI)
#         llm_action = random.choice(ACTIONS)
#         llm_conf = 0.6

#         scores = {}
#         for a in ACTIONS:
#             scores[a] = 0.7 * rl_probs.get(a, 0) + 0.3 * (llm_conf if a == llm_action else 0)

#         return max(scores, key=scores.get), scores


# # ===================== 🔥 SESSION INIT =====================
# if "hybrid_agent" not in st.session_state:
#     st.session_state.hybrid_agent = HybridAgent()

# if "confidence_hist" not in st.session_state:
#     st.session_state.confidence_hist = []

# if "learning_curve" not in st.session_state:
#     st.session_state.learning_curve = []


# # ===================== 🔥 UPDATED RECORD =====================
# # def _record(step, cpu, bat, rew, act):
# #     st.session_state.steps.append(step)
# #     st.session_state.cpu_hist.append(cpu)
# #     st.session_state.bat_hist.append(bat)
# #     st.session_state.rew_hist.append(rew)
# #     st.session_state.act_hist.append(act)
# #     st.session_state.total_reward += rew

# #     # 🔥 learning curve
# #     st.session_state.learning_curve.append(st.session_state.total_reward)

# #     # 🔥 confidence tracking
# #     state = {"cpu": cpu, "battery": bat}
# #     _, scores = st.session_state.hybrid_agent.plan(state)
# #     st.session_state.confidence_hist.append(list(scores.values()))

# def _record(step, cpu, bat, rew, act):
#     st.session_state.steps.append(step)
#     st.session_state.cpu_hist.append(cpu)
#     st.session_state.bat_hist.append(bat)
#     st.session_state.rew_hist.append(rew)
#     st.session_state.act_hist.append(act)

#     # ✅ FIX: cumulative reward
#     st.session_state.total_reward = round(st.session_state.total_reward + rew, 4)

#     # ✅ FIX: learning curve
#     if "learning_curve" not in st.session_state:
#         st.session_state.learning_curve = []
#     st.session_state.learning_curve.append(st.session_state.total_reward)

#     # ✅ FIX: confidence tracking
#     if "confidence_hist" not in st.session_state:
#         st.session_state.confidence_hist = []

#     state = {"cpu": cpu, "battery": bat}

#     try:
#         _, scores = st.session_state.hybrid_agent.plan(state)
#         st.session_state.confidence_hist.append(list(scores.values()))
#     except:
#         # fallback to avoid crash
#         st.session_state.confidence_hist.append([0.25, 0.25, 0.25, 0.25])


# # ===================== 🔥 CONFIDENCE GRAPH =====================
# def _confidence_plot():
#     fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=DARK)
#     ax.set_facecolor(DARK)

#     data = st.session_state.confidence_hist
#     if data:
#         data = np.array(data)
#         for i, a in enumerate(ACTIONS):
#             ax.plot(data[:, i], label=a)

#         ax.legend()
#         ax.set_title("Hybrid Confidence (RL + LLM)", color=TXT)
#     else:
#         ax.text(.5, .5, "No confidence data", ha="center", color=TXT)

#     return fig


# # ===================== 🔥 LEARNING CURVE =====================
# def _learning_plot():
#     fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=DARK)
#     ax.set_facecolor(DARK)

#     data = st.session_state.learning_curve
#     if data:
#         ax.plot(data, linewidth=2)
#         ax.set_title("Learning Curve", color=TXT)
#     else:
#         ax.text(.5, .5, "No learning data", ha="center", color=TXT)

#     return fig


# # ===================== 🔥 UPDATED AUTOPILOT =====================
# def _autopilot_action(obs: dict):
#     agent = st.session_state.hybrid_agent
#     agent.remember(obs)
#     action, scores = agent.plan(obs)
#     return action, "HYBRID_AI"


# # ===================== 🔥 GRAPH UI UPDATE =====================
# gc1, gc2 = st.columns(2)
# gc3, gc4 = st.columns(2)

# with gc1:
#     st.pyplot(_cpu_plot(), use_container_width=True)

# with gc2:
#     st.pyplot(_reward_plot(), use_container_width=True)

# with gc3:
#     st.pyplot(_confidence_plot(), use_container_width=True)

# with gc4:
#     st.pyplot(_learning_plot(), use_container_width=True)


# # ===================== 🔥 AUTOPILOT LOOP UPDATE =====================
# if run_auto:
#     if not st.session_state.initialized:
#         st.warning("Please reset first.")
#     else:
#         for i in range(int(n_steps)):
#             sd, _ = _get("/state")
#             obs = sd["observation"]
#             ob0 = dict(obs)

#             act, src = _autopilot_action(obs)
#             rd, _ = _post("/step", {"action": act})

#             o = rd["observation"]

#             # 🔥 RL LEARNING
#             st.session_state.hybrid_agent.rl_agent.update(ob0, act, rd["reward"], o, rd["done"])
#             st.session_state.hybrid_agent.rl_agent.save()

#             _record(sd["step"]+1, o["cpu"], o["battery"], rd["reward"], act)

#             if rd["done"]:
#                 break