FROM python:3.9-slim

WORKDIR /code

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ── Critical: install in the correct order to avoid huggingface_hub conflicts ──
# 1. Install huggingface_hub first at a known-good version
RUN pip install --no-cache-dir "huggingface_hub==0.20.3"

# 2. Install remaining deps (streamlit will NOT pull in gradio or override hf_hub)
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.25.0" \
    "streamlit>=1.32.0" \
    "matplotlib>=3.8.0" \
    "numpy>=1.26.0" \
    "requests>=2.31.0"

COPY . .

# Expose both backend (8000) and frontend (8501) ports
EXPOSE 8000 8501

# Default: start FastAPI backend
# Override CMD to start the Streamlit frontend on port 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]