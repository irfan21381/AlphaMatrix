"""
app.py — RL LLM Thermal Manager (Streamlit-only unified app)

Single-process design for HuggingFace Spaces:
- Streamlit UI
- Environment + RL agent run in-process (no FastAPI/uvicorn)
"""

from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.agent import QLearningAgent
from app.env import ACTIONS, TASK, ThermalEnv, explain_action


def _init_state() -> None:
    defaults = {
        "env": ThermalEnv(),
        "agent": QLearningAgent(),
        "initialized": False,
        "incident": "Thermal spike after GPU benchmark",
        "status": "Ready. Click Reset to initialize the environment.",
        "rationale": "Reset the environment, then execute a step to see the AI rationale.",
        "history": [],  # list[dict]
        "total_reward": 0.0,
        "autopilot_running": False,
        "last_obs": {"cpu": None, "battery": None},
        "last_action": None,
        "last_reward": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _theme() -> None:
    st.set_page_config(
        page_title="RL LLM Thermal Manager",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 20% 0%, #0b1630 0%, #070b14 50%, #06070d 100%); color: #e6edf3; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.03) !important; border-right: 1px solid rgba(255,255,255,0.08) !important; }
[data-testid="metric-container"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 12px; }
.stButton>button { border-radius: 10px; font-weight: 700; padding: .55rem 1rem; }
div[data-baseweb="select"] > div { background: rgba(255,255,255,0.04); border-radius: 10px; }
textarea, input { background: rgba(255,255,255,0.04) !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _log_event(kind: str, payload: Dict) -> None:
    st.session_state.history.append(payload)
    st.session_state.history = st.session_state.history[-400:]
    if kind == "step":
        st.session_state.total_reward = round(st.session_state.total_reward + float(payload["reward"]), 6)


def _reset_env(cpu: float, battery: float, incident: str) -> None:
    obs = st.session_state.env.reset(cpu=cpu, battery=battery)
    st.session_state.initialized = True
    st.session_state.incident = incident
    st.session_state.status = f"Reset OK. CPU={obs['cpu']:.1f}% · Battery={obs['battery']:.1f}%"
    st.session_state.rationale = "Environment ready. Execute a step (manual or autopilot)."
    st.session_state.history = []
    st.session_state.total_reward = 0.0
    st.session_state.last_obs = obs
    st.session_state.last_action = None
    st.session_state.last_reward = None


def _do_step(action: str, source: str) -> None:
    if not st.session_state.initialized:
        st.session_state.status = "Reset required before stepping."
        return

    before = st.session_state.env.observation()
    res = st.session_state.env.step(action)
    after = res.observation

    # RL learning update
    st.session_state.agent.update(before, action, res.reward, after, res.done)
    st.session_state.agent.save()

    msg, deltas = explain_action(action, before, after)
    severity = "CRITICAL" if before["cpu"] is not None and before["cpu"] >= 90 else ("HIGH" if before["cpu"] >= 75 else "MODERATE")

    st.session_state.rationale = (
        f"### AI Rationale\n"
        f"- **Source**: `{source}`\n"
        f"- **Action**: `{action}`\n"
        f"- **Severity**: `{severity}`\n"
        f"- **Why**: {msg}\n\n"
        f"**Δ CPU**: `{deltas['cpu']:+.2f}%` · **Δ Battery**: `{deltas['battery']:+.2f}%`"
    )

    step_no = (len([h for h in st.session_state.history if h.get('kind') == 'step']) + 1)
    payload = {
        "kind": "step",
        "ts": time.strftime("%H:%M:%S"),
        "task": TASK,
        "step": step_no,
        "action": action,
        "source": source,
        "observation": after,
        "reward": float(res.reward),
        "done": bool(res.done),
        "info": dict(res.info),
    }
    _log_event("step", payload)

    st.session_state.last_obs = after
    st.session_state.last_action = action
    st.session_state.last_reward = float(res.reward)

    tag = " · GOAL!" if res.done else ""
    st.session_state.status = f"{source} step {step_no}: {action} · reward={res.reward:.4f}{tag}"


def _history_df() -> pd.DataFrame:
    rows: List[Dict] = []
    for h in st.session_state.history:
        if h.get("kind") != "step":
            continue
        obs = h.get("observation") or {}
        rows.append(
            {
                "step": h.get("step"),
                "action": h.get("action"),
                "source": h.get("source"),
                "reward": h.get("reward"),
                "cpu": obs.get("cpu"),
                "battery": obs.get("battery"),
                "done": h.get("done"),
                "ts": h.get("ts"),
            }
        )
    return pd.DataFrame(rows)


_theme()
_init_state()

with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Single unified Streamlit runtime (no backend server).")
    st.divider()

    cpu_init = st.slider("Starting CPU (%)", 0.0, 100.0, 90.0, 0.5)
    bat_init = st.slider("Starting Battery (%)", 0.0, 100.0, 20.0, 0.5)
    incident = st.text_area("Incident context", value=st.session_state.incident, height=90)

    c_reset, c_step = st.columns(2)
    with c_reset:
        if st.button("Reset", use_container_width=True):
            _reset_env(cpu=cpu_init, battery=bat_init, incident=incident)
            st.rerun()
    with c_step:
        action_sel = st.selectbox("Action", ACTIONS, index=0)
        if st.button("Execute Step", use_container_width=True, disabled=not st.session_state.initialized):
            _do_step(action_sel, source="MANUAL")
            st.rerun()

    st.divider()
    st.markdown("## Autopilot")
    auto_steps = st.slider("Steps to run", 5, 50, 20, 1)
    if st.button("Run Autopilot", use_container_width=True, disabled=not st.session_state.initialized):
        st.session_state.autopilot_running = True

    st.divider()
    st.markdown("## Agent (RL)")
    if st.session_state.initialized:
        dbg = st.session_state.agent.debug(st.session_state.env.observation())
        st.caption(f"State: `{dbg.state}` · ε={dbg.epsilon:.3f} · visits={dbg.visits}")


st.markdown("## ⚡ RL LLM Thermal Manager — Hybrid AI System")
st.caption("Hackathon-ready single app: environment + RL agent + stable Streamlit dashboard.")

st.info(st.session_state.status)

obs_now = st.session_state.env.observation() if st.session_state.initialized else {"cpu": None, "battery": None}

m1, m2, m3, m4 = st.columns(4)
m1.metric("CPU (%)", "—" if obs_now["cpu"] is None else f"{obs_now['cpu']:.1f}")
m2.metric("Battery (%)", "—" if obs_now["battery"] is None else f"{obs_now['battery']:.1f}")
m3.metric("Steps", str(len(_history_df())))
m4.metric("Total Reward", f"{st.session_state.total_reward:.4f}")

df = _history_df()

g1, g2, g3 = st.columns([1.2, 1.0, 1.0])
with g1:
    st.markdown("### CPU vs Battery")
    if df.empty:
        st.caption("Reset, then take a step to populate charts.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["step"], y=df["cpu"], mode="lines+markers", name="CPU", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=df["step"], y=df["battery"], mode="lines+markers", name="Battery", line=dict(width=3, dash="dash")))
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

with g2:
    st.markdown("### Reward Trend")
    if df.empty:
        st.caption("No reward yet.")
    else:
        fig = px.bar(df, x="step", y="reward", color="action", height=320, template="plotly_dark")
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

with g3:
    st.markdown("### Learning Curve")
    if df.empty:
        st.caption("No learning data yet.")
    else:
        df2 = df.copy()
        df2["cumulative_reward"] = df2["reward"].cumsum()
        fig = px.line(df2, x="step", y="cumulative_reward", height=320, template="plotly_dark")
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### AI Rationale")
st.markdown(st.session_state.rationale)

with st.expander("Event Log", expanded=False):
    if df.empty:
        st.caption("No events yet.")
    else:
        st.dataframe(df.tail(50), use_container_width=True, hide_index=True)


if st.session_state.autopilot_running:
    st.session_state.autopilot_running = False

    progress = st.progress(0)
    status = st.empty()

    for i in range(int(auto_steps)):
        obs = st.session_state.env.observation()
        action, conf = st.session_state.agent.act_with_confidence(obs)
        _do_step(action, source="AUTOPILOT")
        progress.progress(int(((i + 1) / int(auto_steps)) * 100))
        status.caption(f"Autopilot {i+1}/{auto_steps} · action={action} · confidence={max(conf.values()):.2f}")
        if st.session_state.env.is_done():
            break

    st.rerun()


