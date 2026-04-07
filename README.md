---
title: RL LLM Thermal Manager
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# ⚡ RL-LLM Hybrid Thermal Manager

A production-ready hybrid AI system that combines **Reinforcement Learning** (Q-Learning) and an **LLM surrogate** to manage system thermal and battery conditions in real time.

## Architecture

```
┌─────────────────────────────────────────────┐
│              app.py  (port 7860)            │
│                                             │
│  ┌──────────────┐   ┌────────────────────┐  │
│  │  FastAPI     │   │  Streamlit UI      │  │
│  │  (direct     │◄──│  (graphs, autopilot│  │
│  │  function    │   │   manual control,  │  │
│  │  calls)      │   │   metrics, logs)   │  │
│  └──────┬───────┘   └────────────────────┘  │
│         │                                   │
│  ┌──────▼───────────────────────────────┐   │
│  │  DisasterEnv  (RL Environment)       │   │
│  │  QLearningAgent + HybridAgent        │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Current state |
| GET | `/history` | Episode history |
| POST | `/explain` | AI rationale |

## Actions

- `optimize_cpu` — Rebalance CPU scheduler
- `close_apps` — Kill background processes
- `throttle_gpu` — Reduce GPU clock speeds
- `hibernate_idle` — Hibernate idle processes

## Goal

Reach **CPU ≤ 60%** and **Battery ≥ 50%** within 50 steps.

## Deployment

Single `app.py` runs on HuggingFace Spaces port 7860. No threading, no localhost backend — FastAPI routes are called directly as Python functions from the Streamlit UI.