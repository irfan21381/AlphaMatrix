# ⚡ RL Thermal Manager

A production-grade Hybrid RL + LLM thermal management system with a live debugger dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Gradio)                     │
│  frontend/app.py  — Live Debugger Dashboard              │
│  • Sliders (CPU/Battery init)  • Incident text input     │
│  • CPU Stability graph         • Reward Gradient graph   │
│  • AI Rationale (Markdown)     • Autopilot generator     │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP REST
┌────────────────────▼────────────────────────────────────┐
│              Backend (FastAPI)  app/main.py              │
│  Strict State Machine + threading.Lock()                 │
│  Endpoints: /reset  /step  /state  /history              │
│             /actions  /explain  /health                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              RL Environment  app/env.py                  │
│  Stochastic transitions (random.uniform)                 │
│  Reward = (ΔCPU + ΔBattery) / 30.0  ∈ [0, 1]           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         Hybrid Decision Engine  inference.py             │
│  50% → RL (Q-learning)   50% → LLM (OpenAI)             │
│  Strict stdout logging: [START] [STEP] [END]             │
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the dashboard
```bash
python frontend/app.py
# Open http://localhost:7860
```

### 4. Run inference script (optional — standalone)
```bash
# RL only (no OpenAI key needed)
python inference.py --cpu 95 --battery 15 --no-llm

# Full hybrid (requires OPENAI_API_KEY env var)
export OPENAI_API_KEY=sk-...
python inference.py --cpu 92 --battery 20
```

---

## API Reference

| Method | Endpoint   | Body               | Returns                            |
|--------|------------|--------------------|------------------------------------|
| POST   | /reset     | `InitSchema?`      | observation, init_params           |
| POST   | /step      | `{action: str}`    | obs, reward, done, info            |
| GET    | /state     | —                  | observation, step, total_reward    |
| GET    | /history   | —                  | full episode history               |
| GET    | /actions   | —                  | list of valid actions              |
| POST   | /explain   | `ExplainSchema`    | rationale, severity, deltas        |
| GET    | /health    | —                  | {status: ok}                       |

### InitSchema
```json
{
  "cpu": 90.0,
  "battery": 25.0,
  "incident_description": "Thermal spike after benchmark"
}
```

---

## Inference Log Format

```
[START] {"task": "Thermal Management"}
[STEP]  {"step": 1, "action": "close_apps", "reward": 0.8, "source": "rl", "cpu": 74.3, "battery": 27.1}
[STEP]  {"step": 2, "action": "throttle_gpu", "reward": 0.61, "source": "llm", "cpu": 65.1, "battery": 32.4}
[END]   {"total_reward": 4.5, "status": "success", "steps": 8}
```

---

## Reward Formula

```
Reward = (ΔCPU_reduction + ΔBattery_recovery) / 30.0

Where:
  ΔCPU_reduction  = max(0, cpu_before - cpu_after)
  ΔBattery_recovery = max(0, battery_after - battery_before)
  MaxPossibleImprovement = 20 (max CPU drop) + 10 (max battery gain) = 30
```

Reward is always in **[0.0, 1.0]**.

---

## Actions & Stochastic Transitions

| Action          | CPU Reduction        | Battery Change    |
|-----------------|----------------------|-------------------|
| `optimize_cpu`  | uniform(8, 18)%      | uniform(-1, 2)%   |
| `close_apps`    | uniform(10, 20)%     | uniform(1, 5)%    |
| `throttle_gpu`  | uniform(5, 12)%      | uniform(3, 8)%    |
| `hibernate_idle`| uniform(3, 10)%      | uniform(5, 10)%   |

---

## Thread Safety

All backend state mutations are wrapped in `threading.Lock()`:
```python
with _lock:
    obs = _env.step(action)
    _total_reward += reward
```

This ensures that simultaneous calls from the dashboard and inference script never corrupt state.
