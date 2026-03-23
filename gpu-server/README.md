# GPU Inference Server

Standalone FastAPI server that runs the upgraded ML models:
- **Transformer** (multi-head attention, 10 features, 30-candle window)
- **Dueling Double DQN** with prioritized experience replay
- **Semantic sentiment** via `sentence-transformers/all-MiniLM-L6-v2`
- **Ensemble endpoint** combining all models into a single signal

Works on any machine. Falls back to CPU if no GPU is available, but CUDA is strongly recommended.

---

## Quick Start

```bash
cd gpu-server
pip install -r requirements.txt
python server.py
```

Server starts on `http://0.0.0.0:9090`. Check health at `http://localhost:9090/health`.

---

## Installing PyTorch with CUDA

By default, `pip install torch` installs the **CPU-only** build. To use your GPU you need the CUDA variant.

### 1. Check your NVIDIA driver version

```bash
nvidia-smi
```

Note the **Driver Version** and **CUDA Version** in the top-right of the output.

### 2. Install the matching PyTorch CUDA build

```bash
# Uninstall CPU-only torch first
pip uninstall torch torchvision torchaudio -y

# Pick the right command for your driver:
```

| Driver version | CUDA toolkit | Install command |
|---|---|---|
| >= 550 | 12.4 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| >= 530 | 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| >= 520 | 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

### 3. Verify CUDA is available

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A', '| PyTorch CUDA:', torch.version.cuda)"
```

Expected output:
```
CUDA: True | GPU: NVIDIA GeForce RTX XXXX | PyTorch CUDA: 12.4
```

If it still shows `False`, your PyTorch build doesn't match your driver — try a different `cu` index from the table above.

### 4. Restart the server

```bash
python server.py
```

Startup log should show:
```
GPU Server starting on device: cuda
GPU: NVIDIA GeForce RTX XXXX (VRAM: X.X GB)
```

---

## Connecting to the Bot

On the machine running the bot, add to `.env`:

```env
GPU_SERVER_URL=http://<gpu-machine-ip>:9090
```

Use `http://localhost:9090` if running on the same machine.

The bot's `/api/health` endpoint and dashboard GPU card will show the connection status.

---

## Authentication (optional)

Set the same token on both machines to require bearer auth:

**GPU machine** — set env var before starting:
```bash
export GPU_SERVER_TOKEN=my-secret-token
python server.py
```

**Bot machine** — add to `.env`:
```env
GPU_SERVER_TOKEN=my-secret-token
```

The `/health` endpoint is always accessible without auth.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status, GPU info, loaded models |
| POST | `/train/lstm` | Train Transformer + LSTM ensemble |
| POST | `/predict/lstm` | Predict with Transformer/LSTM (60/40 weighted) |
| POST | `/train/rl` | Train Dueling Double DQN |
| POST | `/predict/rl` | Get RL Q-values |
| POST | `/sentiment` | Score headlines with sentence-transformer |
| POST | `/predict/ensemble` | Combined signal from all models |
