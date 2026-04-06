"""
frontend/app.py — Premium Interactive Live Debugger Dashboard
Streamlit-based UI (replaces Gradio to avoid huggingface_hub/Python 3.9 conflicts).

Run with:
    streamlit run frontend/app.py
"""

import time
import json
import random
import threading
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────

BACKEND_URL        = "http://localhost:8000"
AUTOPILOT_INTERVAL = 0.8

ACTIONS = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]

ACTION_COLORS = {
    "optimize_cpu":   "#00d4ff",
    "close_apps":     "#ff6b6b",
    "throttle_gpu":   "#ffd166",
    "hibernate_idle": "#06d6a0",
}

DARK_BG   = "#0d1117"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#c9d1d9"
ACCENT    = "#58a6ff"
RED_CLR   = "#ff6b6b"
GREEN_CLR = "#06d6a0"

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⚡ RL Thermal Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
.stApp { background-color: #0d1117 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    padding: 0.45rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}

/* Sliders */
.stSlider > div > div > div { background: #30363d !important; }

/* Text inputs */
.stTextArea textarea, .stTextInput input {
    background: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Radio */
.stRadio label { color: #c9d1d9 !important; }

/* Info/success/error boxes */
.stAlert { border-radius: 6px !important; font-family: 'JetBrains Mono', monospace !important; }

/* Divider */
hr { border-color: #21262d !important; }

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
}

/* Code / markdown */
code { background: #161b22 !important; color: #79c0ff !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "cpu_history":     [],
        "bat_history":     [],
        "reward_history":  [],
        "action_history":  [],
        "step_labels":     [],
        "total_reward":    0.0,
        "current_cpu":     None,
        "current_bat":     None,
        "initialized":     False,
        "rationale":       "_Run a step or start autopilot to see AI rationale._",
        "status":          "Awaiting reset…",
        "log":             [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ─── Backend helpers ──────────────────────────────────────────────────────────

def _post(path: str, payload: dict = None):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=payload or {}, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def _get(path: str):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def _record(step, cpu, battery, reward, action):
    st.session_state.cpu_history.append(cpu)
    st.session_state.bat_history.append(battery)
    st.session_state.reward_history.append(reward)
    st.session_state.action_history.append(action)
    st.session_state.step_labels.append(step)
    st.session_state.total_reward += reward

def _add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.append(f"[{ts}] {msg}")
    if len(st.session_state.log) > 100:
        st.session_state.log = st.session_state.log[-100:]

# ─── Fetch explain ────────────────────────────────────────────────────────────

def _fetch_explain(action: str, before: dict, after: dict) -> str:
    data, err = _post("/explain", {
        "action": action,
        "state_before": before,
        "state_after": after,
    })
    if err:
        return f"⚠ Could not reach `/explain`: {err}"
    return (
        f"### 🤖 AI Rationale\n\n"
        f"**Action:** `{data['action']}`  \n"
        f"**Severity at decision:** `{data['severity_at_decision']}`\n\n"
        f"{data['rationale']}\n\n"
        f"**Δ CPU:** `{data['delta']['cpu']:+.2f}%`  |  "
        f"**Δ Battery:** `{data['delta']['battery']:+.2f}%`"
    )

# ─── Plotting ─────────────────────────────────────────────────────────────────

def _make_cpu_plot():
    fig, ax = plt.subplots(figsize=(7, 3.0), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    steps   = st.session_state.step_labels
    cpus    = st.session_state.cpu_history
    bats    = st.session_state.bat_history
    actions = st.session_state.action_history

    if not steps:
        ax.text(0.5, 0.5, "No data yet — reset and run a step",
                ha="center", va="center", color=TEXT_CLR, fontsize=10,
                transform=ax.transAxes)
    else:
        ax.plot(steps, cpus, color=RED_CLR, linewidth=2, label="CPU %", zorder=3)
        ax.plot(steps, bats, color=GREEN_CLR, linewidth=2,
                linestyle="--", label="Battery %", zorder=3)
        ax.axhspan(80, 100, alpha=0.08, color=RED_CLR)
        ax.axhline(60, color=ACCENT, linewidth=0.8, linestyle=":", alpha=0.6)

        for s, c_val, act in zip(steps, cpus, actions):
            ax.scatter(s, c_val, color=ACTION_COLORS.get(act, "#fff"),
                       s=45, zorder=5, edgecolors="none")

    window = 25
    if steps:
        ax.set_xlim(max(0, len(steps) - window), max(1, len(steps)))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Step", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("%", color=TEXT_CLR, fontsize=9)
    ax.set_title("CPU Stability & Battery Level", color=TEXT_CLR, fontsize=11, pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#161b22", edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, fontsize=8, loc="upper right")
    fig.tight_layout(pad=0.8)
    return fig


def _make_reward_plot():
    fig, ax = plt.subplots(figsize=(7, 3.0), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    steps   = st.session_state.step_labels
    rewards = st.session_state.reward_history

    if not steps:
        ax.text(0.5, 0.5, "No reward data yet",
                ha="center", va="center", color=TEXT_CLR, fontsize=10,
                transform=ax.transAxes)
    else:
        colours = [
            "#06d6a0" if r >= 0.6 else ("#ffd166" if r >= 0.3 else "#ff6b6b")
            for r in rewards
        ]
        ax.bar(steps, rewards, color=colours, width=0.7, zorder=3)

        cumulative = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(steps, cumulative, color=ACCENT, linewidth=1.5,
                 linestyle="--", label="Cumulative", zorder=4)
        ax2.set_ylabel("Cumulative", color=ACCENT, fontsize=8)
        ax2.tick_params(colors=ACCENT, labelsize=7)
        ax2.set_facecolor(DARK_BG)
        for sp in ax2.spines.values():
            sp.set_edgecolor(GRID_CLR)

    window = 25
    if steps:
        ax.set_xlim(max(0, len(steps) - window), max(1, len(steps)))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Step", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("Reward (norm.)", color=TEXT_CLR, fontsize=9)
    ax.set_title("Reward Gradient", color=TEXT_CLR, fontsize=11, pad=8)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    return fig

# ─── Heuristic autopilot action ───────────────────────────────────────────────

def _pick_autopilot_action(obs: dict) -> tuple[str, str]:
    """50/50 split: RL heuristic vs weighted-random LLM simulation."""
    if random.random() < 0.5:
        cpu, bat = obs["cpu"], obs["battery"]
        if cpu > 85:
            action = "close_apps"
        elif bat < 25:
            action = "throttle_gpu"
        elif cpu > 70:
            action = "optimize_cpu"
        else:
            action = "hibernate_idle"
        return action, "RL"
    else:
        cpu = obs.get("cpu", 80)
        weights = [0.25, 0.35 if cpu > 80 else 0.15, 0.20, 0.20]
        action = random.choices(ACTIONS, weights=weights, k=1)[0]
        return action, "LLM"

# ─── Sidebar: Configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.markdown("---")

    cpu_init = st.slider("Starting CPU %",     0.0, 100.0, 90.0, 0.5)
    bat_init = st.slider("Starting Battery %", 0.0, 100.0, 20.0, 0.5)

    incident = st.text_area(
        "Incident Description",
        value="Thermal spike after GPU benchmark",
        height=80,
    )

    if st.button("🔄 Reset Environment", use_container_width=True):
        data, err = _post("/reset", {
            "cpu": cpu_init,
            "battery": bat_init,
            "incident_description": incident,
        })
        if err:
            st.error(f"Reset failed: {err}")
        else:
            # Clear history
            for k in ["cpu_history","bat_history","reward_history",
                      "action_history","step_labels","log"]:
                st.session_state[k] = []
            st.session_state.total_reward = 0.0
            obs = data["observation"]
            st.session_state.current_cpu = obs["cpu"]
            st.session_state.current_bat = obs["battery"]
            st.session_state.initialized = True
            st.session_state.status = f"✅ Reset complete — CPU={obs['cpu']}%  Battery={obs['battery']}%"
            st.session_state.rationale = "### 🔄 Reset\n\nEnvironment initialised. Choose an action to begin."
            _add_log(f"RESET  cpu={obs['cpu']}  bat={obs['battery']}")
            st.rerun()

    st.markdown("---")
    st.markdown("## 🎮 Manual Step")
    action_choice = st.selectbox("Action", ACTIONS)

    if st.button("▶ Execute Step", use_container_width=True):
        if not st.session_state.initialized:
            st.warning("Reset the environment first.")
        else:
            state_data, err = _get("/state")
            if err:
                st.error(err)
            else:
                obs_before = state_data["observation"]
                step_no    = state_data["step"] + 1
                step_data, err = _post("/step", {"action": action_choice})
                if err:
                    st.error(err)
                else:
                    obs = step_data["observation"]
                    _record(step_no, obs["cpu"], obs["battery"],
                            step_data["reward"], action_choice)
                    st.session_state.current_cpu = obs["cpu"]
                    st.session_state.current_bat = obs["battery"]
                    st.session_state.rationale   = _fetch_explain(action_choice, obs_before, obs)
                    done_tag = " 🎯 Goal reached!" if step_data["done"] else ""
                    st.session_state.status = (
                        f"[manual] step={step_no}  action={action_choice}  "
                        f"reward={step_data['reward']:.4f}{done_tag}"
                    )
                    _add_log(f"STEP {step_no}  {action_choice}  r={step_data['reward']:.4f}  "
                             f"cpu={obs['cpu']}  bat={obs['battery']}")
                    st.rerun()

    st.markdown("---")
    st.markdown("## 🚀 Autopilot")
    n_steps = st.slider("Steps", 5, 50, 20, 1)

    run_autopilot = st.button("▶▶ Run Autopilot", use_container_width=True)

# ─── Main area ────────────────────────────────────────────────────────────────

st.markdown("""
# ⚡ RL Thermal Manager — Live Debugger
**Hybrid RL + LLM Decision Engine**  ·  Real-time State Machine Inspector
""")
st.markdown("---")

# ── Status bar ────────────────────────────────────────────────────────────────
st.info(st.session_state.status)

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    cpu_val = f"{st.session_state.current_cpu:.1f}%" if st.session_state.current_cpu is not None else "—"
    st.metric("🌡 Current CPU", cpu_val)
with col2:
    bat_val = f"{st.session_state.current_bat:.1f}%" if st.session_state.current_bat is not None else "—"
    st.metric("🔋 Current Battery", bat_val)
with col3:
    steps_done = len(st.session_state.step_labels)
    st.metric("📍 Steps Taken", steps_done)
with col4:
    st.metric("🏆 Total Reward", f"{st.session_state.total_reward:.4f}")

st.markdown("---")

# ── Live graphs ───────────────────────────────────────────────────────────────
st.markdown("### 📈 Live Graphs")
graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    st.pyplot(_make_cpu_plot(), use_container_width=True)
with graph_col2:
    st.pyplot(_make_reward_plot(), use_container_width=True)

st.markdown("---")

# ── AI Rationale ──────────────────────────────────────────────────────────────
st.markdown("### 💡 AI Rationale")
st.markdown(st.session_state.rationale)

st.markdown("---")

# ── Step log ──────────────────────────────────────────────────────────────────
with st.expander("📋 Step Log", expanded=False):
    if st.session_state.log:
        log_text = "\n".join(reversed(st.session_state.log[-50:]))
        st.code(log_text, language="text")
    else:
        st.caption("No steps yet.")

# ─── Autopilot (generator loop with st.empty) ─────────────────────────────────

if run_autopilot:
    if not st.session_state.initialized:
        st.warning("Reset the environment first.")
    else:
        autopilot_status = st.empty()
        autopilot_status.info("🚀 Autopilot starting…")

        for i in range(int(n_steps)):
            state_data, err = _get("/state")
            if err:
                autopilot_status.error(f"Autopilot aborted: {err}")
                break

            obs = state_data["observation"]
            obs_before = dict(obs)
            step_no = state_data["step"] + 1

            action, source = _pick_autopilot_action(obs)

            step_data, err = _post("/step", {"action": action})
            if err:
                autopilot_status.error(f"Step failed: {err}")
                break

            obs_after = step_data["observation"]
            _record(step_no, obs_after["cpu"], obs_after["battery"],
                    step_data["reward"], action)
            st.session_state.current_cpu = obs_after["cpu"]
            st.session_state.current_bat = obs_after["battery"]
            st.session_state.rationale   = f"**[{source}]** " + _fetch_explain(
                action, obs_before, obs_after
            )
            done_tag = " 🎯 Goal reached!" if step_data["done"] else ""
            st.session_state.status = (
                f"[{source}] step={step_no}  action={action}  "
                f"reward={step_data['reward']:.4f}  "
                f"cpu={obs_after['cpu']:.1f}%  bat={obs_after['battery']:.1f}%{done_tag}"
            )
            _add_log(
                f"AUTO {step_no}  [{source}]  {action}  "
                f"r={step_data['reward']:.4f}  "
                f"cpu={obs_after['cpu']}  bat={obs_after['battery']}"
            )

            autopilot_status.info(
                f"🚀 Autopilot running… step {i+1}/{n_steps} — "
                f"`{action}` [{source}] → reward={step_data['reward']:.4f}{done_tag}"
            )

            # Yield control / update every AUTOPILOT_INTERVAL seconds
            time.sleep(AUTOPILOT_INTERVAL)

            if step_data["done"]:
                autopilot_status.success(f"🎯 Goal reached in {step_no} steps! "
                                         f"Total reward = {st.session_state.total_reward:.4f}")
                break
        else:
            autopilot_status.success(
                f"✅ Autopilot finished {n_steps} steps. "
                f"Total reward = {st.session_state.total_reward:.4f}"
            )

        st.rerun()