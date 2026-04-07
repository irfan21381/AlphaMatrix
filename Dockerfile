FROM python:3.11-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "inference.py"]


# FROM python:3.11-slim

# # 1. Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     HOME=/code

# WORKDIR /code

# # 2. Install system dependencies (needed for health checks)
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# # 3. Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # 4. Copy all project files
# COPY . .

# # 5. Create the server directory if it doesn't exist (to fix the multi-mode error)
# # This ensures the validator sees the path it expects
# RUN mkdir -p server && \
#     echo "from app.main import app as main_app; app = main_app" > server/app.py && \
#     touch server/__init__.py

# # 6. Expose ports (8000 for Backend, 7860 for Gradio Frontend)
# EXPOSE 8000
# EXPOSE 7860

# # 7. Start Command: 
# # Runs FastAPI backend on port 8000 in background (&)
# # Then runs the Gradio app on port 7860
# CMD uvicorn server.app:app --host 0.0.0.0 --port 8000 & python app.py