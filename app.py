"""
app.py — AlphaMatrix Gradio Dashboard
Runs on port 7860. Talks to FastAPI backend on port 8000.
"""

import time
import random
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import requests
    _requests_ok = True
except ImportError:
    _requests_ok = False

try:
    import gradio as gr
    _gradio_ok = True
except ImportError:
    _gradio_ok = False
    raise RuntimeError("gradio is required: pip install gradio")

BACKEND   = "http://127.0.0.1:8000"
INTERVAL  = 0.8

TASKS = ["thermal_throttling", "battery_endurance", "process_deadlock"]

TASK_ACTIONS = {
    "thermal_throttling": [
        "reduce_clock_speed", "kill_heavy_process",
        "enable_cooling_fan", "throttle_gpu",
    ],
    "battery_endurance": [
        "dim_display", "disable_wifi",
        "suspend_background_apps", "enable_battery_saver",
    ],
    "process_deadlock": [
        "release_mutex", "restart_process",
        "increase_timeout", "force_schedule",
    ],
}

DARK  = "#0d1117"
GRID  = "#21262d"
TXT   = "#c9d1d9"
ACC   = "#58a6ff"
RED   = "#ff6b6b"
GRN   = "#06d6a0"

# ── Backend helpers ───────────────────────────────────────────────────────────

def _post(path, payload=None):
    if not _requests_ok:
        return None, "requests not installed"
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload or {}, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def _get(path):
    if not _requests_ok:
        return None, "requests not installed"
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _make_plots(step_hist, reward_hist, obs_hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=DARK)

    for ax in (ax1, ax2):
        ax.set_facecolor(DARK)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    # ── Left: observation values over time ─────────────────────────────────
    if obs_hist:
        keys   = list(obs_hist[0].keys())
        colors = [RED, GRN, ACC, "#ffd166", "#c77dff"]
        for i, k in enumerate(keys):
            vals = [o.get(k, 0) for o in obs_hist]
            ax1.plot(step_hist, vals, label=k,
                     color=colors[i % len(colors)], lw=1.6)
        if keys:
            ax1.legend(facecolor="#161b22", edgecolor=GRID,
                       labelcolor=TXT, fontsize=7, loc="upper right")
    else:
        ax1.text(.5, .5, "No data yet", ha="center", va="center",
                 color=TXT, fontsize=9, transform=ax1.transAxes)

    ax1.set_xlabel("Step", color=TXT, fontsize=9)
    ax1.set_ylabel("Value", color=TXT, fontsize=9)
    ax1.set_title("Observation Metrics", color=TXT, fontsize=11, pad=8)

    # ── Right: reward gradient ─────────────────────────────────────────────
    if reward_hist:
        cols = ["#06d6a0" if r >= .6 else ("#ffd166" if r >= .3 else "#ff6b6b")
                for r in reward_hist]
        ax2.bar(step_hist, reward_hist, color=cols, width=0.7, zorder=3)
        cum = np.cumsum(reward_hist)
        ax3 = ax2.twinx()
        ax3.plot(step_hist, cum, color=ACC, lw=1.5, ls="--")
        ax3.set_ylabel("Cumulative", color=ACC, fontsize=8)
        ax3.tick_params(colors=ACC, labelsize=7)
        ax3.set_facecolor(DARK)
        for sp in ax3.spines.values():
            sp.set_edgecolor(GRID)
    else:
        ax2.text(.5, .5, "No reward data yet", ha="center", va="center",
                 color=TXT, fontsize=9, transform=ax2.transAxes)

    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Step", color=TXT, fontsize=9)
    ax2.set_ylabel("Reward (0–1)", color=TXT, fontsize=9)
    ax2.set_title("Reward Gradient", color=TXT, fontsize=11, pad=8)

    fig.tight_layout(pad=1.0)
    return fig

# ── Callback functions ────────────────────────────────────────────────────────

def do_reset(task):
    data, err = _post("/reset", {"task": task})
    if err:
        return (
            f"❌ Reset failed: {err}",
            gr.update(choices=TASK_ACTIONS.get(task, []),
                      value=TASK_ACTIONS.get(task, [""])[0]),
            _make_plots([], [], []),
            "Reset failed.",
            [], [], [],
        )
    obs = data.get("observation", {})
    actions = TASK_ACTIONS.get(task, [])
    return (
        f"✅ Reset — Task: `{task}` | Obs: {obs}",
        gr.update(choices=actions, value=actions[0]),
        _make_plots([], [], []),
        f"**Task:** `{task}`\n\n**Initial observation:**\n```json\n{json.dumps(obs, indent=2)}\n```",
        [], [], [],
    )


def do_step(action, step_hist, reward_hist, obs_hist):
    sd, err = _get("/state")
    if err:
        return f"❌ {err}", _make_plots(step_hist, reward_hist, obs_hist), \
               "State fetch failed.", step_hist, reward_hist, obs_hist

    obs_before = sd.get("observation", {})
    step_no    = sd.get("step", 0) + 1

    rd, err = _post("/step", {"action": action})
    if err:
        return f"❌ {err}", _make_plots(step_hist, reward_hist, obs_hist), \
               "Step failed.", step_hist, reward_hist, obs_hist

    obs    = rd.get("observation", {})
    reward = rd.get("reward", 0.0)
    done   = rd.get("done", False)

    step_hist   = step_hist   + [step_no]
    reward_hist = reward_hist + [reward]
    obs_hist    = obs_hist    + [obs]

    # Explain
    xd, _  = _post("/explain", {
        "task": sd.get("task", ""),
        "action": action,
        "state_before": obs_before,
        "state_after": obs,
    })
    rationale = xd.get("rationale", "") if xd else ""

    tag = " 🎯 **Done!**" if done else ""
    status = (
        f"Step {step_no} — `{action}` → reward=**{reward:.4f}**  "
        f"cumulative=**{rd.get('cumulative_reward', 0):.4f}**{tag}"
    )
    return (
        status,
        _make_plots(step_hist, reward_hist, obs_hist),
        f"**Rationale:** {rationale}\n\n**Obs:** {obs}",
        step_hist, reward_hist, obs_hist,
    )


def do_autopilot(task, n_steps, step_hist, reward_hist, obs_hist):
    """Generator — yields UI updates every INTERVAL seconds."""
    actions = TASK_ACTIONS.get(task, [])
    for i in range(int(n_steps)):
        sd, err = _get("/state")
        if err:
            yield (f"❌ {err}", _make_plots(step_hist, reward_hist, obs_hist),
                   f"Error: {err}", step_hist, reward_hist, obs_hist)
            return

        obs_before = sd.get("observation", {})
        step_no    = sd.get("step", 0) + 1
        action     = random.choice(actions)

        rd, err = _post("/step", {"action": action})
        if err:
            yield (f"❌ {err}", _make_plots(step_hist, reward_hist, obs_hist),
                   f"Error: {err}", step_hist, reward_hist, obs_hist)
            return

        obs     = rd.get("observation", {})
        reward  = rd.get("reward", 0.0)
        done    = rd.get("done", False)

        step_hist   = step_hist   + [step_no]
        reward_hist = reward_hist + [reward]
        obs_hist    = obs_hist    + [obs]

        xd, _ = _post("/explain", {
            "task": task, "action": action,
            "state_before": obs_before, "state_after": obs,
        })
        rationale = xd.get("rationale", "") if xd else ""

        tag = " 🎯 Done!" if done else ""
        status = (
            f"[Auto] Step {step_no}/{n_steps} — `{action}` → "
            f"reward={reward:.4f}{tag}"
        )

        yield (
            status,
            _make_plots(step_hist, reward_hist, obs_hist),
            f"**Rationale:** {rationale}",
            step_hist, reward_hist, obs_hist,
        )

        if done:
            return
        time.sleep(INTERVAL)

# ── Build Gradio UI ───────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0d1117 !important; color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important; }
.gr-button-primary { background: linear-gradient(135deg,#1f6feb,#388bfd) !important;
    color: #fff !important; border: none !important; border-radius: 6px !important; }
.gr-button-secondary { background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important; }
.gr-box, .gr-panel { background: #161b22 !important; border: 1px solid #21262d !important; }
"""

def build_ui():
    with gr.Blocks(title="⚡ AlphaMatrix", css=CSS) as demo:

        gr.Markdown("# ⚡ AlphaMatrix — RL System Optimizer\n"
                    "**Three Tasks · Stochastic Env · OpenEnv Compliant**")

        # ── Hidden state stores ───────────────────────────────────────────────
        step_store   = gr.State([])
        reward_store = gr.State([])
        obs_store    = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                task_dd    = gr.Dropdown(choices=TASKS, value=TASKS[0], label="Task")
                reset_btn  = gr.Button("🔄 Reset", variant="primary")

                gr.Markdown("### 🎮 Manual Step")
                action_dd  = gr.Dropdown(
                    choices=TASK_ACTIONS[TASKS[0]],
                    value=TASK_ACTIONS[TASKS[0]][0],
                    label="Action",
                )
                step_btn   = gr.Button("▶ Execute Step", variant="secondary")

                gr.Markdown("### 🚀 Autopilot")
                n_steps_sl = gr.Slider(5, 50, value=20, step=1, label="Steps")
                auto_btn   = gr.Button("▶▶ Run Autopilot", variant="primary")

            with gr.Column(scale=3):
                status_box    = gr.Markdown("_Reset to begin._")
                plot_out      = gr.Plot(label="Live Graphs")
                rationale_box = gr.Markdown("_AI rationale appears here._")

        _out = [status_box, plot_out, rationale_box,
                step_store, reward_store, obs_store]

        reset_btn.click(
            fn=do_reset,
            inputs=[task_dd],
            outputs=[status_box, action_dd, plot_out,
                     rationale_box, step_store, reward_store, obs_store],
        )

        step_btn.click(
            fn=do_step,
            inputs=[action_dd, step_store, reward_store, obs_store],
            outputs=_out,
        )

        auto_btn.click(
            fn=do_autopilot,
            inputs=[task_dd, n_steps_sl, step_store, reward_store, obs_store],
            outputs=_out,
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)