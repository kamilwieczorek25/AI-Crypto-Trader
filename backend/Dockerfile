FROM python:3.12-slim

WORKDIR /app

# System deps (needed for some ccxt/pandas native builds + PostgreSQL client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps — install PyTorch CPU-only first to avoid pulling the CUDA variant
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source (also bind-mounted in dev, so this is for production image)
COPY app/ ./app/

# Ensure /data directory exists for SQLite
RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
