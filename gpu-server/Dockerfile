FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model at build time
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY server.py .

EXPOSE 9090
ENV GPU_SERVER_PORT=9090
ENV DATA_DIR=/app/data

CMD ["python3", "server.py"]
