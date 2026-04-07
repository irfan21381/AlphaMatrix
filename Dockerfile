FROM python:3.9-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir "huggingface_hub==0.20.3"

RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.25.0" \
    "matplotlib>=3.8.0" \
    "numpy>=1.26.0" \
    "requests>=2.31.0"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]