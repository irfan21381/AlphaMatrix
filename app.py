"""
frontend/app.py — Premium Interactive Live Debugger Dashboard
Gradio-based UI with live graphs, AI rationale, and autopilot generator.
"""

import time
import json
import random
import threading
import requests
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# ─── Config ───────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000"
AUTOPILOT_INTERVAL = 0.8   # seconds between autopilot steps

ACTIONS = ["optimize_cpu", "close_apps", "throttle_gpu", "hibernate_idle"]

ACTION_COLORS = {
    "optimize_cpu":   "#00d4ff",
    "close_apps":     "#ff6b6b",
    "throttle_gpu":   "#ffd166",
    "hibernate_idle": "#06d6a0",
}

# ─── Session state (module-level, protected by lock) ──────────────────────────

_lock = threading.Lock()
_cpu_history:     list = []
_bat_history:     list = []
_reward_history:  list = []
_action_history:  list = []
_step_labels:     list = []

def _clear_history():
    global _cpu_history, _bat_history, _reward_history, _action_history, _step_labels
    _cpu_history    = []
    _bat_history    = []
    _reward_history = []
    _action_history = []
    _step_labels    = []

def _record(step, cpu, battery, reward, action):
    _cpu_history.append(cpu)
    _bat_history.append(battery)
    _reward_history.append(reward)
    _action_history.append(action)
    _step_labels.append(step)

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

# ─── Plotting ─────────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#c9d1d9"
ACCENT    = "#58a6ff"
RED_CLR   = "#ff6b6b"
GREEN_CLR = "#06d6a0"

def _make_cpu_stability_plot() -> Figure:
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    with _lock:
        steps   = list(_step_labels)
        cpus    = list(_cpu_history)
        bats    = list(_bat_history)
        actions = list(_action_history)

    if not steps:
        ax.text(0.5, 0.5, "No data yet — run a step or start autopilot",
                ha="center", va="center", color=TEXT_CLR, fontsize=10,
                transform=ax.transAxes)
    else:
        # CPU line
        ax.plot(steps, cpus, color=RED_CLR, linewidth=2, label="CPU %", zorder=3)
        ax.plot(steps, bats, color=GREEN_CLR, linewidth=2, linestyle="--",
                label="Battery %", zorder=3)

        # Shade danger zone
        ax.axhspan(80, 100, alpha=0.08, color=RED_CLR)
        ax.axhline(60, color=ACCENT, linewidth=0.8, linestyle=":", alpha=0.6)

        # Action markers
        color_map = {a: ACTION_COLORS[a] for a in ACTIONS}
        for i, (s, c_val, act) in enumerate(zip(steps, cpus, actions)):
            ax.scatter(s, c_val, color=color_map.get(act, "#ffffff"),
                       s=50, zorder=5, edgecolors="none")

    ax.set_xlim(left=max(0, len(steps) - 25) if steps else 0,
                right=max(1, len(steps)) if steps else 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Step", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("%", color=TEXT_CLR, fontsize=9)
    ax.set_title("CPU Stability & Battery Level", color=TEXT_CLR, fontsize=11, pad=10)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(facecolor="#161b22", edgecolor=GRID_CLR,
                       labelcolor=TEXT_CLR, fontsize=8, loc="upper right")
    fig.tight_layout(pad=1.0)
    return fig


def _make_reward_gradient_plot() -> Figure:
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    with _lock:
        steps   = list(_step_labels)
        rewards = list(_reward_history)

    if not steps:
        ax.text(0.5, 0.5, "No reward data yet",
                ha="center", va="center", color=TEXT_CLR, fontsize=10,
                transform=ax.transAxes)
    else:
        # Bar chart coloured by reward value
        colours = [
            "#06d6a0" if r >= 0.6 else ("#ffd166" if r >= 0.3 else "#ff6b6b")
            for r in rewards
        ]
        ax.bar(steps, rewards, color=colours, width=0.7, zorder=3)

        # Cumulative line on secondary axis
        cumulative = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(steps, cumulative, color=ACCENT, linewidth=1.5,
                 linestyle="--", label="Cumulative", zorder=4)
        ax2.set_ylabel("Cumulative Reward", color=ACCENT, fontsize=8)
        ax2.tick_params(colors=ACCENT, labelsize=7)
        ax2.set_facecolor(DARK_BG)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_CLR)

    ax.set_xlim(left=max(0, len(steps) - 25) if steps else 0,
                right=max(1, len(steps)) if steps else 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Step", color=TEXT_CLR, fontsize=9)
    ax.set_ylabel("Reward (normalised)", color=TEXT_CLR, fontsize=9)
    ax.set_title("Reward Gradient", color=TEXT_CLR, fontsize=11, pad=10)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.0)
    return fig

# ─── Action helpers ───────────────────────────────────────────────────────────

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

# ─── UI Callbacks ─────────────────────────────────────────────────────────────

def do_reset(cpu_slider: float, bat_slider: float, incident_text: str):
    _clear_history()
    data, err = _post("/reset", {
        "cpu": cpu_slider,
        "battery": bat_slider,
        "incident_description": incident_text or "Manual reset from dashboard",
    })
    if err:
        return (
            f"❌ Reset failed: {err}",
            f"CPU: {cpu_slider:.0f}%",
            f"Battery: {bat_slider:.0f}%",
            _make_cpu_stability_plot(),
            _make_reward_gradient_plot(),
            "Reset failed — is the backend running?",
        )
    obs = data["observation"]
    return (
        f"✅ Environment reset. CPU={obs['cpu']}%  Battery={obs['battery']}%",
        f"CPU: {obs['cpu']:.1f}%",
        f"Battery: {obs['battery']:.1f}%",
        _make_cpu_stability_plot(),
        _make_reward_gradient_plot(),
        "### 🔄 Reset\n\nEnvironment initialised. Choose an action to begin.",
    )


def do_manual_step(action: str):
    state_before, err = _get("/state")
    if err:
        return (
            f"❌ {err}", "—", "—",
            _make_cpu_stability_plot(), _make_reward_gradient_plot(),
            f"⚠ Could not get state: {err}",
        )

    obs_before = state_before["observation"]
    step_no    = state_before["step"] + 1

    data, err = _post("/step", {"action": action})
    if err:
        return (
            f"❌ Step failed: {err}", "—", "—",
            _make_cpu_stability_plot(), _make_reward_gradient_plot(),
            f"⚠ {err}",
        )

    obs = data["observation"]
    with _lock:
        _record(step_no, obs["cpu"], obs["battery"], data["reward"], action)

    rationale = _fetch_explain(action, obs_before, obs)

    status = "✅ Goal reached!" if data["done"] else f"Step {step_no} complete."

    return (
        f"{status}  reward={data['reward']:.4f}  cumulative={data['cumulative_reward']:.4f}",
        f"CPU: {obs['cpu']:.1f}%",
        f"Battery: {obs['battery']:.1f}%",
        _make_cpu_stability_plot(),
        _make_reward_gradient_plot(),
        rationale,
    )


def autopilot_generator(n_steps: int):
    """
    Python generator that yields UI updates every AUTOPILOT_INTERVAL seconds.
    This runs inside a Gradio generator endpoint so the UI stays responsive.
    """
    state_data, err = _get("/state")
    if err:
        yield (
            f"❌ Cannot start autopilot: {err}", "—", "—",
            _make_cpu_stability_plot(), _make_reward_gradient_plot(),
            f"⚠ Backend not reachable: {err}",
        )
        return

    current_step = state_data["step"]

    for i in range(int(n_steps)):
        # Simple heuristic autopilot: pick action based on current state
        state_data, err = _get("/state")
        if err:
            break
        obs = state_data["observation"]

        # Mimic 50/50 hybrid logic (RL heuristic + random)
        if random.random() < 0.5:
            # "RL" branch — heuristic policy
            if obs["cpu"] > 85:
                action = "close_apps"
            elif obs["battery"] < 25:
                action = "throttle_gpu"
            elif obs["cpu"] > 70:
                action = "optimize_cpu"
            else:
                action = "hibernate_idle"
            source = "RL"
        else:
            # "LLM" branch — random weighted by urgency
            weights = [0.25, 0.35 if obs["cpu"] > 80 else 0.15, 0.20, 0.20]
            action  = random.choices(ACTIONS, weights=weights, k=1)[0]
            source  = "LLM"

        obs_before = dict(obs)
        current_step += 1

        step_data, err = _post("/step", {"action": action})
        if err:
            break

        obs_after = step_data["observation"]
        with _lock:
            _record(current_step, obs_after["cpu"], obs_after["battery"],
                    step_data["reward"], action)

        rationale = _fetch_explain(action, obs_before, obs_after)
        rationale = f"**[{source}]** " + rationale

        done_msg = " 🎯 **Goal reached!**" if step_data["done"] else ""
        status = (
            f"[{source}] Step {current_step} — `{action}` → "
            f"reward={step_data['reward']:.4f}  "
            f"cumulative={step_data['cumulative_reward']:.4f}{done_msg}"
        )

        yield (
            status,
            f"CPU: {obs_after['cpu']:.1f}%",
            f"Battery: {obs_after['battery']:.1f}%",
            _make_cpu_stability_plot(),
            _make_reward_gradient_plot(),
            rationale,
        )

        if step_data["done"]:
            return

        time.sleep(AUTOPILOT_INTERVAL)

# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
/* ── Dark terminal aesthetic ── */
body, .gradio-container {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

.gr-button { border-radius: 4px !important; font-weight: 600 !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important; border: none !important;
}
.gr-button-secondary {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
.gr-button-stop {
    background: #da3633 !important; color: #fff !important;
}

.gr-box, .gr-panel, .gr-form {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}

label, .gr-label { color: #8b949e !important; font-size: 11px !important; letter-spacing: 0.06em; text-transform: uppercase; }

.status-bar {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #58a6ff;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 14px;
    text-align: center;
}
"""

def build_ui():
    with gr.Blocks(
        title="⚡ RL Thermal Manager — Live Debugger",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("JetBrains Mono"),
        ),
        css=CSS,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown(
            """
# ⚡ RL Thermal Manager — Live Debugger
**Hybrid RL + LLM Decision Engine**  |  Real-time State Machine Inspector
---
"""
        )

        # ── Row 1: State Input + Metrics ───────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🔧 Initial State Configuration")
                cpu_slider = gr.Slider(
                    minimum=0, maximum=100, value=90,
                    label="Starting CPU Usage (%)", step=0.5,
                )
                bat_slider = gr.Slider(
                    minimum=0, maximum=100, value=20,
                    label="Starting Battery Level (%)", step=0.5,
                )
                incident_box = gr.Textbox(
                    label="Incident Description",
                    placeholder="e.g. CPU spike after kernel update, thermal runaway on GPU…",
                    lines=2,
                )
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Live Metrics")
                cpu_metric  = gr.Textbox(label="Current CPU",     value="—", interactive=False)
                bat_metric  = gr.Textbox(label="Current Battery", value="—", interactive=False)

        # ── Row 2: Status Bar ──────────────────────────────────────────────────
        status_box = gr.Textbox(
            label="System Status", value="Awaiting reset…",
            interactive=False, lines=2,
        )

        # ── Row 3: Action Panel ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🎮 Manual Action")
                action_radio = gr.Radio(
                    choices=ACTIONS,
                    value=ACTIONS[0],
                    label="Select Action",
                )
                step_btn = gr.Button("▶ Execute Step", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### 🤖 Autopilot")
                autopilot_steps = gr.Slider(
                    minimum=5, maximum=50, value=20, step=1,
                    label="Number of Autopilot Steps",
                )
                autopilot_btn = gr.Button("🚀 Run Autopilot", variant="primary")

        # ── Row 4: Live Graphs ─────────────────────────────────────────────────
        gr.Markdown("### 📈 Live Graphs")
        with gr.Row():
            cpu_plot    = gr.Plot(label="CPU Stability & Battery Level")
            reward_plot = gr.Plot(label="Reward Gradient")

        # ── Row 5: AI Rationale ────────────────────────────────────────────────
        gr.Markdown("### 💡 AI Rationale")
        rationale_box = gr.Markdown(
            value="_Run a step to see why the AI chose an action._",
        )

        # ── Wiring ─────────────────────────────────────────────────────────────
        _outputs = [status_box, cpu_metric, bat_metric, cpu_plot, reward_plot, rationale_box]

        reset_btn.click(
            fn=do_reset,
            inputs=[cpu_slider, bat_slider, incident_box],
            outputs=_outputs,
        )

        step_btn.click(
            fn=do_manual_step,
            inputs=[action_radio],
            outputs=_outputs,
        )

        # Autopilot uses a generator — streams updates every 0.8 s
        autopilot_btn.click(
            fn=autopilot_generator,
            inputs=[autopilot_steps],
            outputs=_outputs,
        )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
