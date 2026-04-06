#!/usr/bin/env bash
# start.sh — Start backend + frontend locally (no Docker)
# Usage: bash start.sh

set -e

echo "=== Installing dependencies ==="
pip install --no-cache-dir "huggingface_hub==0.20.3"
pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.25.0" \
    "streamlit>=1.32.0" \
    "matplotlib>=3.8.0" \
    "numpy>=1.26.0" \
    "requests>=2.31.0"

echo "=== Starting FastAPI backend on :8000 ==="
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2

echo "=== Starting Streamlit frontend on :8501 ==="
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true &
FRONTEND_PID=$!

echo ""
echo "✅ Backend  → http://localhost:8000"
echo "✅ Frontend → http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait