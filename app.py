"""
app.py — Unified RL Thermal Manager for HuggingFace Spaces
Runs FastAPI backend in a background thread + Streamlit frontend in main thread.
Single file, single port (7860), zero external dependencies issues.
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import time
import json
import random
import threading
from typing import Optional, Tuple, Dict, Any, List

# ── third-party ───────────────────────────────────────────────────────────────
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

# ═════════════════════════════════════════════════════════════════════════════
# 1.  RL ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

CPU_SAFE   = 60.0
BAT_SAFE   = 50.0
MAX_IMPROVE = 30.0   # 20 CPU + 10 Battery

ACTIONS: List[str] = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]

class DisasterEnv:
    def __init__(self):
        self._cpu = 85.0
        self._bat = 30.0
        self._step = 0

    def reset(self, cpu=85.0, battery=30.0):
        self._cpu, self._bat, self._step = float(cpu), float(battery), 0
        return self._obs()

    def step(self, action: str):
        if action not in ACTIONS:
            raise ValueError(f"Bad action: {action}")
        c0, b0 = self._cpu, self._bat
        if action == "optimize_cpu":
            dc, db = random.uniform(8, 18),  random.uniform(-1, 2)
        elif action == "close_apps":
            dc, db = random.uniform(10, 20), random.uniform(1, 5)
        elif action == "throttle_gpu":
            dc, db = random.uniform(5, 12),  random.uniform(3, 8)
        else:  # hibernate_idle
            dc, db = random.uniform(3, 10),  random.uniform(5, 10)

        self._cpu = max(0.0, min(100.0, self._cpu - dc))
        self._bat = max(0.0, min(100.0, self._bat + db))
        self._step += 1

        imp_cpu = max(0.0, c0 - self._cpu)
        imp_bat = max(0.0, self._bat - b0)
        reward  = round(min(1.0, (imp_cpu + imp_bat) / MAX_IMPROVE), 4)
        done    = (self._cpu <= CPU_SAFE and self._bat >= BAT_SAFE) or self._step >= 50
        info    = {"delta_cpu": round(c0 - self._cpu, 3),
                   "delta_bat": round(self._bat - b0, 3),
                   "step": self._step}
        return self._obs(), reward, done, info

    def _obs(self):
        return {"cpu": round(self._cpu, 2), "battery": round(self._bat, 2)}

    def is_done(self):
        return self._cpu <= CPU_SAFE and self._bat >= BAT_SAFE

# ═════════════════════════════════════════════════════════════════════════════
# 2.  FASTAPI BACKEND  (runs in background thread on port 8000)
# ═════════════════════════════════════════════════════════════════════════════

# ── shared state ──────────────────────────────────────────────────────────────
_lock          = threading.Lock()
_env           = DisasterEnv()
_episode_step  = 0
_total_reward  = 0.0
_history: list = []
_initialized   = False
_init_params: dict = {}

# ── pydantic schemas ──────────────────────────────────────────────────────────

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

# ── app ───────────────────────────────────────────────────────────────────────

api = FastAPI(title="RL Thermal Manager API")
api.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/reset")
def reset(init: Optional[InitSchema] = None):
    global _episode_step, _total_reward, _history, _initialized, _init_params
    p = init or InitSchema()
    with _lock:
        obs = _env.reset(cpu=p.cpu, battery=p.battery)
        _episode_step = 0; _total_reward = 0.0; _history = []
        _initialized  = True
        _init_params  = {"cpu": p.cpu, "battery": p.battery,
                         "incident_description": p.incident_description}
    return {"status": "reset", "observation": obs, "init_params": _init_params}

@api.post("/step")
def do_step(body: StepSchema):
    global _episode_step, _total_reward, _history
    if not _initialized:
        raise HTTPException(400, "Call /reset first.")
    with _lock:
        obs, reward, done, info = _env.step(body.action)
        _episode_step += 1
        _total_reward += reward
        rec = {"step": _episode_step, "action": body.action,
               "observation": obs, "reward": round(reward, 4),
               "cumulative_reward": round(_total_reward, 4),
               "done": done, "info": info}
        _history.append(rec)
    return rec

@api.get("/state")
def get_state():
    if not _initialized:
        raise HTTPException(400, "Call /reset first.")
    with _lock:
        return {"observation": _env._obs(), "step": _episode_step,
                "total_reward": round(_total_reward, 4),
                "done": _env.is_done(), "init_params": _init_params}

@api.get("/history")
def get_history():
    with _lock:
        return {"episode_step": _episode_step,
                "total_reward": round(_total_reward, 4),
                "history": list(_history)}

@api.get("/actions")
def get_actions():
    return {"actions": ACTIONS}

@api.post("/explain")
def explain(body: ExplainSchema):
    s0, s1 = body.state_before, body.state_after
    dc = s0.get("cpu", 0) - s1.get("cpu", 0)
    db = s1.get("battery", 0) - s0.get("battery", 0)
    msgs = {
        "optimize_cpu":   f"CPU scheduler rebalanced. CPU dropped {dc:.1f}%.",
        "close_apps":     f"Background apps terminated. CPU down {dc:.1f}%, battery +{db:.1f}%.",
        "throttle_gpu":   f"GPU clocks throttled. Thermal relief: CPU -{dc:.1f}%, battery +{db:.1f}%.",
        "hibernate_idle": f"Idle processes hibernated. CPU -{dc:.1f}%, battery conserved +{db:.1f}%.",
    }
    sev = "CRITICAL" if s0.get("cpu",0) > 90 else ("HIGH" if s0.get("cpu",0) > 75 else "MODERATE")
    return {"action": body.action,
            "rationale": msgs.get(body.action, f"Applied {body.action}."),
            "severity_at_decision": sev,
            "delta": {"cpu": round(dc,2), "battery": round(db,2)}}

def _start_backend():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="error")

# ── launch backend thread once ────────────────────────────────────────────────
_backend_thread = threading.Thread(target=_start_backend, daemon=True)
_backend_thread.start()
time.sleep(1.5)   # give uvicorn a moment to bind

# ═════════════════════════════════════════════════════════════════════════════
# 3.  STREAMLIT FRONTEND
# ═════════════════════════════════════════════════════════════════════════════

BACKEND = "http://localhost:8000"
INTERVAL = 0.8

ACTION_COLORS = {
    "optimize_cpu":   "#00d4ff",
    "close_apps":     "#ff6b6b",
    "throttle_gpu":   "#ffd166",
    "hibernate_idle": "#06d6a0",
}

DARK_BG  = "#0d1117"
GRID_CLR = "#21262d"
TEXT_CLR = "#c9d1d9"
ACCENT   = "#58a6ff"
RED_CLR  = "#ff6b6b"
GRN_CLR  = "#06d6a0"

# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⚡ RL Thermal Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
.stApp { background-color: #0d1117 !important; }
[data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:12px;
}
[data-testid="metric-container"] label {
    color:#8b949e !important; font-size:11px !important;
    text-transform:uppercase; letter-spacing:.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#58a6ff !important; font-size:22px !important; font-weight:700 !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1f6feb,#388bfd) !important;
    color:#fff !important; border:none !important; border-radius:6px !important;
    font-family:'JetBrains Mono',monospace !important; font-weight:600 !important;
    padding:.45rem 1.2rem;
}
[data-testid="stSidebar"] {
    background:#161b22 !important; border-right:1px solid #21262d !important;
}
.stTextArea textarea, .stTextInput input {
    background:#161b22 !important; color:#c9d1d9 !important;
    border:1px solid #30363d !important; border-radius:6px !important;
    font-family:'JetBrains Mono',monospace !important;
}
.stRadio label, .stSelectbox label { color:#c9d1d9 !important; }
hr { border-color:#21262d !important; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = dict(
        cpu_hist=[], bat_hist=[], rew_hist=[], act_hist=[], steps=[],
        total_reward=0.0, cur_cpu=None, cur_bat=None,
        initialized=False,
        rationale="_Reset the environment and run a step to see AI rationale._",
        status="Awaiting reset…", log=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── helpers ───────────────────────────────────────────────────────────────────

def _post(path, payload=None):
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload or {}, timeout=5)
        r.raise_for_status(); return r.json(), None
    except Exception as e:
        return None, str(e)

def _get(path):
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=5)
        r.raise_for_status(); return r.json(), None
    except Exception as e:
        return None, str(e)

def _record(step, cpu, bat, rew, act):
    st.session_state.steps.append(step)
    st.session_state.cpu_hist.append(cpu)
    st.session_state.bat_hist.append(bat)
    st.session_state.rew_hist.append(rew)
    st.session_state.act_hist.append(act)
    st.session_state.total_reward += rew

def _log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.append(f"[{ts}] {msg}")
    st.session_state.log = st.session_state.log[-100:]

def _explain(action, before, after):
    data, err = _post("/explain", {"action": action,
                                    "state_before": before,
                                    "state_after": after})
    if err:
        return f"⚠ Explain error: {err}"
    return (f"### 🤖 AI Rationale\n\n**Action:** `{data['action']}`  \n"
            f"**Severity:** `{data['severity_at_decision']}`\n\n"
            f"{data['rationale']}\n\n"
            f"**Δ CPU:** `{data['delta']['cpu']:+.2f}%`  |  "
            f"**Δ Battery:** `{data['delta']['battery']:+.2f}%`")

# ── plots ─────────────────────────────────────────────────────────────────────

def _cpu_plot():
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    steps = st.session_state.steps
    cpus  = st.session_state.cpu_hist
    bats  = st.session_state.bat_hist
    acts  = st.session_state.act_hist

    if steps:
        ax.plot(steps, cpus, color=RED_CLR, lw=2, label="CPU %",     zorder=3)
        ax.plot(steps, bats, color=GRN_CLR, lw=2, ls="--",
                label="Battery %", zorder=3)
        ax.axhspan(80, 100, alpha=0.07, color=RED_CLR)
        ax.axhline(60,  color=ACCENT, lw=0.8, ls=":", alpha=0.6)
        for s, c, a in zip(steps, cpus, acts):
            ax.scatter(s, c, color=ACTION_COLORS.get(a,"#fff"),
                       s=45, zorder=5, edgecolors="none")
        ax.legend(facecolor="#161b22", edgecolor=GRID_CLR,
                  labelcolor=TEXT_CLR, fontsize=8, loc="upper right")
        w = 25
        ax.set_xlim(max(0, len(steps)-w), max(1, len(steps)))
    else:
        ax.text(.5,.5,"No data yet — reset and run a step",
                ha="center",va="center",color=TEXT_CLR,fontsize=9,
                transform=ax.transAxes)

    ax.set_ylim(0, 105)
    ax.set_xlabel("Step", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("%",    color=TEXT_CLR, fontsize=9)
    ax.set_title("CPU Stability & Battery Level", color=TEXT_CLR, fontsize=11, pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return fig


def _reward_plot():
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    steps = st.session_state.steps
    rews  = st.session_state.rew_hist

    if steps:
        cols = ["#06d6a0" if r>=.6 else ("#ffd166" if r>=.3 else "#ff6b6b")
                for r in rews]
        ax.bar(steps, rews, color=cols, width=0.7, zorder=3)
        cum = np.cumsum(rews)
        ax2 = ax.twinx()
        ax2.plot(steps, cum, color=ACCENT, lw=1.5, ls="--", zorder=4)
        ax2.set_ylabel("Cumulative", color=ACCENT, fontsize=8)
        ax2.tick_params(colors=ACCENT, labelsize=7)
        ax2.set_facecolor(DARK_BG)
        for sp in ax2.spines.values(): sp.set_edgecolor(GRID_CLR)
        w = 25
        ax.set_xlim(max(0, len(steps)-w), max(1, len(steps)))
    else:
        ax.text(.5,.5,"No reward data yet",
                ha="center",va="center",color=TEXT_CLR,fontsize=9,
                transform=ax.transAxes)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Step", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("Reward (norm.)", color=TEXT_CLR, fontsize=9)
    ax.set_title("Reward Gradient", color=TEXT_CLR, fontsize=11, pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return fig

# ── autopilot action ──────────────────────────────────────────────────────────

def _autopilot_action(obs):
    if random.random() < 0.5:
        cpu, bat = obs["cpu"], obs["battery"]
        if cpu > 85:       act = "close_apps"
        elif bat < 25:     act = "throttle_gpu"
        elif cpu > 70:     act = "optimize_cpu"
        else:              act = "hibernate_idle"
        return act, "RL"
    w = [0.25, 0.35 if obs["cpu"]>80 else 0.15, 0.20, 0.20]
    return random.choices(ACTIONS, weights=w, k=1)[0], "LLM"

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.markdown("---")

    cpu_init = st.slider("Starting CPU %",     0.0, 100.0, 90.0, 0.5)
    bat_init = st.slider("Starting Battery %", 0.0, 100.0, 20.0, 0.5)
    incident = st.text_area("Incident Description",
                             value="Thermal spike after GPU benchmark",
                             height=80)

    if st.button("🔄 Reset Environment", use_container_width=True):
        data, err = _post("/reset", {"cpu": cpu_init, "battery": bat_init,
                                      "incident_description": incident})
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
            st.warning("Reset first.")
        else:
            sd, err = _get("/state")
            if err:
                st.error(err)
            else:
                obs_before = sd["observation"]
                step_no    = sd["step"] + 1
                rd, err = _post("/step", {"action": action_sel})
                if err:
                    st.error(err)
                else:
                    o = rd["observation"]
                    _record(step_no, o["cpu"], o["battery"], rd["reward"], action_sel)
                    st.session_state.cur_cpu   = o["cpu"]
                    st.session_state.cur_bat   = o["battery"]
                    st.session_state.rationale = _explain(action_sel, obs_before, o)
                    tag = " 🎯 Goal reached!" if rd["done"] else ""
                    st.session_state.status = (
                        f"[manual] step={step_no}  {action_sel}  "
                        f"r={rd['reward']:.4f}{tag}")
                    _log(f"STEP {step_no} {action_sel} r={rd['reward']:.4f} "
                         f"cpu={o['cpu']} bat={o['battery']}")
                    st.rerun()

    st.markdown("---")
    st.markdown("## 🚀 Autopilot")
    n_steps     = st.slider("Steps", 5, 50, 20, 1)
    run_auto    = st.button("▶▶ Run Autopilot", use_container_width=True)

# ── main area ─────────────────────────────────────────────────────────────────

st.markdown("# ⚡ RL Thermal Manager — Live Debugger")
st.markdown("**Hybrid RL + LLM Decision Engine**  ·  Real-time State Machine Inspector")
st.markdown("---")

st.info(st.session_state.status)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🌡 CPU",         f"{st.session_state.cur_cpu:.1f}%"
              if st.session_state.cur_cpu is not None else "—")
with c2:
    st.metric("🔋 Battery",     f"{st.session_state.cur_bat:.1f}%"
              if st.session_state.cur_bat is not None else "—")
with c3:
    st.metric("📍 Steps",       len(st.session_state.steps))
with c4:
    st.metric("🏆 Total Reward",f"{st.session_state.total_reward:.4f}")

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
        st.warning("Reset first.")
    else:
        bar = st.empty()
        bar.info("🚀 Autopilot starting…")
        for i in range(int(n_steps)):
            sd, err = _get("/state")
            if err:
                bar.error(f"Aborted: {err}"); break
            obs = sd["observation"]
            ob0 = dict(obs)
            sno = sd["step"] + 1
            act, src = _autopilot_action(obs)
            rd, err  = _post("/step", {"action": act})
            if err:
                bar.error(f"Step failed: {err}"); break
            o = rd["observation"]
            _record(sno, o["cpu"], o["battery"], rd["reward"], act)
            st.session_state.cur_cpu   = o["cpu"]
            st.session_state.cur_bat   = o["battery"]
            st.session_state.rationale = f"**[{src}]** " + _explain(act, ob0, o)
            tag = " 🎯 Goal!" if rd["done"] else ""
            st.session_state.status = (
                f"[{src}] step={sno}  {act}  r={rd['reward']:.4f}  "
                f"cpu={o['cpu']:.1f}%  bat={o['battery']:.1f}%{tag}")
            _log(f"AUTO {sno} [{src}] {act} r={rd['reward']:.4f} "
                 f"cpu={o['cpu']} bat={o['battery']}")
            bar.info(f"🚀 Step {i+1}/{n_steps} — `{act}` [{src}]  "
                     f"reward={rd['reward']:.4f}{tag}")
            time.sleep(INTERVAL)
            if rd["done"]:
                bar.success(f"🎯 Goal reached in {sno} steps!  "
                            f"Total reward={st.session_state.total_reward:.4f}")
                break
        else:
            bar.success(f"✅ Autopilot done.  "
                        f"Total reward={st.session_state.total_reward:.4f}")
        st.rerun()