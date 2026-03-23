"""GPU Inference Server — runs on Windows/Linux machine with NVIDIA GPU.

Quality-focused upgrades over CPU versions:
1. Transformer model (multi-head attention) replaces basic LSTM
2. Dueling Double DQN with prioritized replay replaces vanilla DQN
3. Sentence-transformer news sentiment (semantic understanding vs keywords)
4. Ensemble endpoint combining all models for higher-confidence signals

Usage:
    pip install torch fastapi uvicorn numpy sentence-transformers
    # For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124
    python server.py
"""

import asyncio
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu-server")

# ── Device setup ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("GPU Server starting on device: %s", DEVICE)
if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    _vram = getattr(_props, 'total_memory', None) or getattr(_props, 'total_mem', 0)
    logger.info("GPU: %s (VRAM: %.1f GB)", torch.cuda.get_device_name(0), _vram / 1e9)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN = 30          # longer look-back for transformer (was 20)
N_FEATURES = 10       # richer features (was 6)
N_CLASSES = 3
LOOKAHEAD = 3
LABEL_THRESHOLD = 0.012
ACTIONS = {0: "SELL", 1: "HOLD", 2: "BUY"}

STATE_SIZE = 14        # expanded RL state (was 12)
ACTION_SIZE = 3

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRANSFORMER MODEL — replaces basic LSTM
#    Multi-head self-attention captures complex temporal patterns that
#    recurrent layers miss. Positional encoding preserves ordering.
# ═══════════════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    """Transformer encoder for time-series classification.

    Why better than LSTM:
    - Multi-head attention sees ALL positions simultaneously (no vanishing gradient)
    - Learns which past candles matter most for the current prediction
    - Layer norm + residual connections = more stable training
    """
    def __init__(self, n_features=N_FEATURES, d_model=128, nhead=4,
                 num_layers=3, dim_ff=256, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=200)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)           # → (batch, seq, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]                  # last time-step output
        return self.head(x)


# Also keep lightweight LSTM for ensemble diversity
class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, N_CLASSES))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DUELING DOUBLE DQN — replaces vanilla DQN
#    - Dueling: separates state value V(s) from advantage A(s,a)
#      → learns which states are good regardless of action
#    - Double: uses policy net to SELECT action, target net to EVALUATE
#      → eliminates Q-value overestimation bias
#    - Prioritized replay: learns more from surprising transitions
# ═══════════════════════════════════════════════════════════════════════════════


class DuelingDQN(nn.Module):
    """Dueling architecture: V(s) + A(s,a) - mean(A)"""
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # Advantage stream
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)               # (batch, 1)
        a = self.advantage(f)            # (batch, action_size)
        return v + a - a.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    """Prioritized experience replay — learns more from surprising transitions."""
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        probs = self.priorities[:n]
        probs = probs / probs.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        batch = [self.buffer[i] for i in indices]
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(td) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SENTENCE-TRANSFORMER SENTIMENT — replaces keyword counting
#    Uses a pre-trained model that actually UNDERSTANDS language.
#    "SEC sues Binance" → strong negative even though none of the
#    44 keywords match. Keyword approach would score this as neutral.
# ═══════════════════════════════════════════════════════════════════════════════

_sentiment_model = None
_sentiment_tokenizer = None


def _load_sentiment_model():
    global _sentiment_model, _sentiment_tokenizer
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"  # 80MB, fast, good quality
        logger.info("Loading sentence-transformer: %s …", model_name)
        _sentiment_model = SentenceTransformer(model_name, device=str(DEVICE))
        logger.info("Sentence-transformer loaded on %s", DEVICE)
    except Exception as e:
        logger.warning("Sentence-transformer load failed: %s (sentiment will use fallback)", e)


# Reference embeddings for sentiment anchoring
_POS_ANCHORS = [
    "bullish momentum breakout rally surge new all-time high",
    "institutional adoption partnership major upgrade launch",
    "strong buying pressure accumulation growth profit",
]
_NEG_ANCHORS = [
    "bearish crash dump plunge severe decline liquidation",
    "hack exploit breach vulnerability security incident",
    "SEC lawsuit ban regulatory crackdown fraud scam",
]
_pos_embeddings = None
_neg_embeddings = None


def _init_sentiment_anchors():
    global _pos_embeddings, _neg_embeddings
    if _sentiment_model is None:
        return
    _pos_embeddings = _sentiment_model.encode(_POS_ANCHORS, convert_to_numpy=True).mean(axis=0)
    _neg_embeddings = _sentiment_model.encode(_NEG_ANCHORS, convert_to_numpy=True).mean(axis=0)


def _semantic_sentiment(texts: list[str]) -> list[float]:
    """Score texts on [-1, 1] using cosine similarity to sentiment anchors."""
    if _sentiment_model is None or _pos_embeddings is None:
        return [0.0] * len(texts)
    embeddings = _sentiment_model.encode(texts, convert_to_numpy=True, batch_size=32)
    scores = []
    for emb in embeddings:
        pos_sim = float(np.dot(emb, _pos_embeddings) / (np.linalg.norm(emb) * np.linalg.norm(_pos_embeddings) + 1e-10))
        neg_sim = float(np.dot(emb, _neg_embeddings) / (np.linalg.norm(emb) * np.linalg.norm(_neg_embeddings) + 1e-10))
        score = np.clip(pos_sim - neg_sim, -1.0, 1.0)
        scores.append(round(float(score), 4))
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED FEATURES — 10 features instead of 6
# Added: OBV trend, VWAP deviation, Stochastic %K, high/low range
# ═══════════════════════════════════════════════════════════════════════════════

def _rsi(c, p=14):
    out = np.full(len(c), 50.0)
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    l_ = np.where(d < 0, -d, 0.0)
    if len(g) < p:
        return out
    ag, al = g[:p].mean(), l_[:p].mean()
    out[p] = 100 - 100 / (1 + ag / (al + 1e-10))
    for i in range(p, len(d)):
        ag = (ag * (p - 1) + g[i]) / p
        al = (al * (p - 1) + l_[i]) / p
        out[i + 1] = 100 - 100 / (1 + ag / (al + 1e-10))
    return out


def _ema(arr, p):
    out = np.zeros(len(arr))
    if len(arr) < p:
        return out
    out[p - 1] = arr[:p].mean()
    k = 2 / (p + 1)
    for i in range(p, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _macd_hist(c, fast=12, slow=26, sig=9):
    macd = _ema(c, fast) - _ema(c, slow)
    return macd - _ema(macd, sig)


def _bb_pct(c, p=20, m=2.0):
    out = np.full(len(c), 0.5)
    for i in range(p - 1, len(c)):
        w = c[i - p + 1: i + 1]
        std = w.std()
        if std > 0:
            mn = w.mean()
            out[i] = (c[i] - (mn - m * std)) / (2 * m * std + 1e-10)
    return out


def _atr(h, l, c, p=14):
    out = np.zeros(len(c))
    for i in range(1, len(c)):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        out[i] = out[i - 1] + (tr - out[i - 1]) / min(i, p) if i > 0 else tr
    return out


def _stochastic_k(h, l, c, p=14):
    """Stochastic %K — where is close relative to high-low range."""
    out = np.full(len(c), 50.0)
    for i in range(p - 1, len(c)):
        hh = h[i - p + 1: i + 1].max()
        ll = l[i - p + 1: i + 1].min()
        r = hh - ll
        out[i] = ((c[i] - ll) / (r + 1e-10)) * 100 if r > 0 else 50.0
    return out


def _obv_trend(c, v, p=20):
    """OBV slope normalised — detects volume-confirmed trends."""
    obv = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            obv[i] = obv[i - 1] - v[i]
        else:
            obv[i] = obv[i - 1]
    # Slope of OBV over p periods
    out = np.zeros(len(c))
    for i in range(p, len(c)):
        slope = (obv[i] - obv[i - p]) / (abs(obv[i - p]) + 1e-10)
        out[i] = slope
    return out


def _vwap_dev(c, v, p=20):
    """VWAP deviation — how far price is from volume-weighted fair value."""
    out = np.zeros(len(c))
    for i in range(p - 1, len(c)):
        cv = c[i - p + 1: i + 1] * v[i - p + 1: i + 1]
        vol_sum = v[i - p + 1: i + 1].sum()
        vwap = cv.sum() / (vol_sum + 1e-10) if vol_sum > 0 else c[i]
        out[i] = (c[i] - vwap) / (vwap + 1e-10)
    return out


def build_features(candles):
    """10 features (up from 6): adds stochastic %K, OBV trend, VWAP deviation, high-low range."""
    if len(candles) < SEQ_LEN + LOOKAHEAD + 20:
        return None
    closes = np.array([c[4] for c in candles], dtype=np.float64)
    highs = np.array([c[2] for c in candles], dtype=np.float64)
    lows = np.array([c[3] for c in candles], dtype=np.float64)
    volumes = np.array([c[5] for c in candles], dtype=np.float64)
    n = len(closes)

    ret = np.zeros(n)
    ret[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)
    vol_ma = np.convolve(volumes, np.ones(20) / 20, mode="same")
    vol_r = np.where(vol_ma > 0, volumes / vol_ma, 1.0)
    rsi = _rsi(closes, 14) / 100.0
    macd_n = np.where(closes > 0, _macd_hist(closes) / closes, 0.0) * 100
    bb = _bb_pct(closes, 20, 2.0)
    atr_p = _atr(highs, lows, closes, 14) / (closes + 1e-10) * 100
    stoch = _stochastic_k(highs, lows, closes, 14) / 100.0
    obv_t = _obv_trend(closes, volumes, 20)
    vwap_d = _vwap_dev(closes, volumes, 20)
    hl_range = (highs - lows) / (closes + 1e-10)

    feat = np.stack([
        np.clip(ret, -0.10, 0.10),
        np.clip(vol_r, 0.00, 5.00) / 5.0,
        np.clip(rsi, 0.00, 1.00),
        np.clip(macd_n, -5.00, 5.00) / 5.0,
        np.clip(bb, -0.50, 1.50),
        np.clip(atr_p, 0.00, 10.00) / 10.0,
        np.clip(stoch, 0.00, 1.00),
        np.clip(obv_t, -1.00, 1.00),
        np.clip(vwap_d, -0.05, 0.05) * 10,
        np.clip(hl_range, 0.00, 0.10) * 10,
    ], axis=1).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def build_labels(candles, threshold=LABEL_THRESHOLD):
    closes = np.array([c[4] for c in candles], dtype=np.float64)
    labels = np.ones(len(closes), dtype=np.int64)
    for i in range(len(closes) - LOOKAHEAD):
        future_return = (closes[i + LOOKAHEAD] - closes[i]) / (closes[i] + 1e-10)
        if future_return > threshold:
            labels[i] = 2
        elif future_return < -threshold:
            labels[i] = 0
    return labels


# ── Global model state ────────────────────────────────────────────────────────
transformer_model: TransformerPredictor | None = None
lstm_model: LSTMNet | None = None
transformer_trained = False
lstm_trained = False

rl_policy: DuelingDQN | None = None
rl_target: DuelingDQN | None = None
rl_buffer: PrioritizedReplayBuffer | None = None
rl_trained = False


def _load_models():
    global transformer_model, transformer_trained, lstm_model, lstm_trained
    # Transformer
    path = DATA_DIR / "transformer_model.pt"
    if path.exists():
        try:
            m = TransformerPredictor().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.eval()
            transformer_model = m
            transformer_trained = True
            logger.info("Transformer loaded from %s", path)
        except Exception as e:
            logger.warning("Transformer load failed: %s", e)
    # LSTM (ensemble member)
    path = DATA_DIR / "lstm_model.pt"
    if path.exists():
        try:
            m = LSTMNet().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.eval()
            lstm_model = m
            lstm_trained = True
            logger.info("LSTM loaded from %s", path)
        except Exception as e:
            logger.warning("LSTM load failed: %s", e)


def _load_rl():
    global rl_policy, rl_target, rl_buffer, rl_trained
    rl_buffer = PrioritizedReplayBuffer()
    path = DATA_DIR / "rl_agent.pt"
    if path.exists():
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
            rl_policy = DuelingDQN().to(DEVICE)
            rl_target = DuelingDQN().to(DEVICE)
            rl_policy.load_state_dict(ckpt["policy"])
            rl_target.load_state_dict(ckpt["target"])
            rl_trained = True
            logger.info("Dueling DQN loaded from %s", path)
        except Exception as e:
            logger.warning("RL load failed: %s", e)


# ── API schemas ───────────────────────────────────────────────────────────────
class TrainLSTMRequest(BaseModel):
    candles: dict[str, list]
    epochs: int = 60


class PredictLSTMRequest(BaseModel):
    candles: list


class TrainRLRequest(BaseModel):
    candles: dict[str, list]
    gradient_steps: int = 500


class PredictRLRequest(BaseModel):
    state: list[float]


class SentimentRequest(BaseModel):
    texts: list[str]


class EnsembleRequest(BaseModel):
    candles: list
    state: list[float]
    headlines: list[str] = []


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="GPU Inference Server v2")

_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:9000,http://127.0.0.1:9000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "authorization"],
)

# Shared authentication token (set GPU_SERVER_TOKEN env var on both machines)
_GPU_TOKEN = os.environ.get("GPU_SERVER_TOKEN", "")


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Verify bearer token on all non-health endpoints when GPU_SERVER_TOKEN is set."""
    if _GPU_TOKEN and request.url.path != "/health":
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {_GPU_TOKEN}":
            from starlette.responses import JSONResponse
            return JSONResponse(status_code=403, content={"detail": "Invalid or missing GPU server token"})
    return await call_next(request)


@app.on_event("startup")
async def startup():
    _load_models()
    _load_rl()
    _load_sentiment_model()
    _init_sentiment_anchors()
    # Start continuous background training loop
    asyncio.create_task(_background_training_loop())


@app.get("/health")
def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    vram_gb = (getattr(_p, 'total_memory', None) or getattr(_p, 'total_mem', 0)) / 1e9 if _p else 0
    return {
        "status": "ok",
        "device": str(DEVICE),
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "transformer_trained": transformer_trained,
        "lstm_trained": lstm_trained,
        "rl_trained": rl_trained,
        "sentiment_loaded": _sentiment_model is not None,
        "bg_training_active": _bg_training_active,
        "bg_training_cycles": _bg_epoch_count,
        "stored_symbols": len(_stored_candles),
    }


# ── TRAIN: Transformer + LSTM ensemble ────────────────────────────────────────
@app.post("/train/lstm")
async def train_lstm(req: TrainLSTMRequest):
    """Trains BOTH the Transformer (primary) and LSTM (ensemble member)."""
    global transformer_model, transformer_trained, lstm_model, lstm_trained
    # Store candles for continuous background retraining
    _stored_candles.update(req.candles)

    def _train():
        global transformer_model, transformer_trained, lstm_model, lstm_trained
        X_all, y_all = [], []
        for sym, candles in req.candles.items():
            feat = build_features(candles)
            if feat is None:
                continue
            labels = build_labels(candles)
            n = len(feat) - SEQ_LEN - LOOKAHEAD
            for i in range(max(n, 0)):
                X_all.append(feat[i: i + SEQ_LEN])
                y_all.append(int(labels[i + SEQ_LEN - 1]))

        if len(X_all) < 128:
            return {"status": "skipped", "reason": f"only {len(X_all)} samples"}

        X = torch.tensor(np.array(X_all), dtype=torch.float32)
        y = torch.tensor(y_all, dtype=torch.long)
        counts = torch.bincount(y, minlength=N_CLASSES).float()
        weights = (counts.sum() / (N_CLASSES * counts.clamp(min=1))).to(DEVICE)

        t0 = time.time()

        # --- Train Transformer (primary model) ---
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True, drop_last=True)
        trf = TransformerPredictor().to(DEVICE)
        opt = optim.AdamW(trf.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)
        crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

        trf.train()
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(req.epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(trf(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(trf.parameters(), 1.0)
                opt.step()
                total += loss.item()
            sched.step()
            avg_loss = total / len(loader)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        trf.eval()
        transformer_model = trf
        transformer_trained = True
        torch.save(trf.state_dict(), DATA_DIR / "transformer_model.pt")

        # --- Train LSTM (ensemble member, fewer epochs) ---
        lstm = LSTMNet().to(DEVICE)
        opt2 = optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
        crit2 = nn.CrossEntropyLoss(weight=weights)
        lstm.train()
        for epoch in range(min(30, req.epochs)):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt2.zero_grad()
                loss = crit2(lstm(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
                opt2.step()
        lstm.eval()
        lstm_model = lstm
        lstm_trained = True
        torch.save(lstm.state_dict(), DATA_DIR / "lstm_model.pt")

        elapsed = time.time() - t0
        return {
            "status": "ok",
            "samples": len(X_all),
            "features": N_FEATURES,
            "seq_len": SEQ_LEN,
            "transformer_loss": round(best_loss, 4),
            "elapsed_seconds": round(elapsed, 2),
            "device": str(DEVICE),
            "models_trained": ["transformer", "lstm"],
        }

    return await asyncio.to_thread(_train)


# ── PREDICT: Ensemble (Transformer + LSTM average) ───────────────────────────
@app.post("/predict/lstm")
def predict_lstm(req: PredictLSTMRequest):
    neutral = {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33,
               "signal": "HOLD", "confidence": 0.34, "status": "untrained"}

    feat = build_features(req.candles)
    if feat is None or len(feat) < SEQ_LEN:
        return {**neutral, "status": "insufficient_data"}

    seq = feat[-SEQ_LEN:]
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    probs_list = []

    # Transformer prediction
    if transformer_trained and transformer_model is not None:
        with torch.no_grad():
            p = torch.softmax(transformer_model(tensor), dim=1).cpu().numpy()[0]
            probs_list.append(p * 0.6)  # 60% weight to transformer

    # LSTM prediction
    if lstm_trained and lstm_model is not None:
        with torch.no_grad():
            p = torch.softmax(lstm_model(tensor), dim=1).cpu().numpy()[0]
            probs_list.append(p * 0.4)  # 40% weight to LSTM

    if not probs_list:
        return neutral

    # Weighted ensemble average
    probs = sum(probs_list)
    probs = probs / probs.sum()  # re-normalise

    top_idx = int(np.argmax(probs))
    return {
        "BUY": float(probs[2]),
        "HOLD": float(probs[1]),
        "SELL": float(probs[0]),
        "signal": ACTIONS[top_idx],
        "confidence": float(probs[top_idx]),
        "status": "ok",
        "ensemble": True,
    }


# ── TRAIN RL: Dueling Double DQN ─────────────────────────────────────────────
@app.post("/train/rl")
async def train_rl(req: TrainRLRequest):
    global rl_policy, rl_target, rl_buffer, rl_trained
    _stored_candles.update(req.candles)

    def _train():
        global rl_policy, rl_target, rl_buffer, rl_trained
        policy = DuelingDQN().to(DEVICE)
        target = DuelingDQN().to(DEVICE)
        target.load_state_dict(policy.state_dict())
        opt = optim.Adam(policy.parameters(), lr=5e-5)
        buf = PrioritizedReplayBuffer(capacity=20000)

        # Build experience from candles
        for sym, candles in req.candles.items():
            feat = build_features(candles)
            if feat is None or len(feat) < SEQ_LEN + 10:
                continue
            closes = np.array([c[4] for c in candles], dtype=np.float64)
            for i in range(SEQ_LEN, len(feat) - 1):
                state = feat[i - 6:i].flatten()[:STATE_SIZE]
                if len(state) < STATE_SIZE:
                    state = np.pad(state, (0, STATE_SIZE - len(state)))
                next_state = feat[i - 5:i + 1].flatten()[:STATE_SIZE]
                if len(next_state) < STATE_SIZE:
                    next_state = np.pad(next_state, (0, STATE_SIZE - len(next_state)))
                ret = (closes[i + 1] - closes[i]) / (closes[i] + 1e-10)
                action = 2 if ret > 0.005 else (0 if ret < -0.005 else 1)
                reward = ret * 100
                buf.push(state, action, reward, next_state, 0.0)

        if len(buf) < 128:
            return {"status": "skipped", "reason": f"only {len(buf)} experiences"}

        t0 = time.time()
        gamma = 0.99

        for step in range(min(req.gradient_steps, len(buf))):
            beta = min(1.0, 0.4 + step * (1.0 - 0.4) / req.gradient_steps)
            s, a, r, s2, d, weights, indices = buf.sample(64, beta=beta)
            s, a, r, s2, d, weights = (t.to(DEVICE) for t in [s, a, r, s2, d, weights])

            # Double DQN: policy selects action, target evaluates
            q = policy(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_actions = policy(s2).argmax(1)
                q_next = target(s2).gather(1, next_actions.unsqueeze(1)).squeeze()
                q_target = r + gamma * q_next * (1 - d)

            td_errors = (q - q_target).detach().cpu().numpy()
            buf.update_priorities(indices, td_errors)

            loss = (weights.to(DEVICE) * F.smooth_l1_loss(q, q_target, reduction='none')).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            if step % 100 == 0:
                target.load_state_dict(policy.state_dict())

        elapsed = time.time() - t0
        rl_policy = policy
        rl_target = target
        rl_buffer = buf
        rl_trained = True

        torch.save({"policy": policy.state_dict(), "target": target.state_dict()},
                    DATA_DIR / "rl_agent.pt")

        return {
            "status": "ok",
            "gradient_steps": min(req.gradient_steps, len(buf)),
            "experiences": len(buf),
            "elapsed_seconds": round(elapsed, 2),
            "device": str(DEVICE),
            "architecture": "dueling_double_dqn",
        }

    return await asyncio.to_thread(_train)


# ── PREDICT RL ────────────────────────────────────────────────────────────────
@app.post("/predict/rl")
def predict_rl(req: PredictRLRequest):
    if not rl_trained or rl_policy is None:
        return {"q_values": {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}, "trained": False}

    state = req.state
    # Pad to STATE_SIZE if caller sends old 12-dim state
    if len(state) < STATE_SIZE:
        state = state + [0.0] * (STATE_SIZE - len(state))

    tensor = torch.FloatTensor(state[:STATE_SIZE]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q = rl_policy(tensor).cpu().numpy()[0]

    action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return {
        "q_values": {action_map[i]: round(float(q[i]), 4) for i in range(ACTION_SIZE)},
        "action": action_map[int(np.argmax(q))],
        "trained": True,
        "architecture": "dueling_double_dqn",
    }


# ── SENTIMENT: Semantic scoring ───────────────────────────────────────────────
@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    """Score news headlines using sentence-transformer embeddings."""
    if not req.texts:
        return {"scores": [], "model": None}
    scores = _semantic_sentiment(req.texts)
    return {
        "scores": scores,
        "avg_sentiment": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "model": "all-MiniLM-L6-v2" if _sentiment_model is not None else "fallback",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONTINUOUS BACKGROUND TRAINING — keeps the GPU warm
#    Between bot cycles the GPU sits idle. This loop continuously retrains
#    on stored candle data with data augmentation (noise, time shifts),
#    improving model generalisation and keeping the GPU busy.
# ═══════════════════════════════════════════════════════════════════════════════

_stored_candles: dict[str, list] = {}   # cached training data from last /train call
_bg_training_active = False
_bg_epoch_count = 0


def _augment_features(X_batch: torch.Tensor) -> torch.Tensor:
    """Add slight noise + time-warp for training diversity on GPU."""
    noise = torch.randn_like(X_batch) * 0.02
    # Random magnitude scaling per sample: 0.95 – 1.05
    scale = 0.95 + torch.rand(X_batch.size(0), 1, 1, device=X_batch.device) * 0.10
    return (X_batch + noise) * scale


async def _background_training_loop():
    """Runs forever in background — retrains Transformer + DQN on augmented data."""
    global _bg_training_active, _bg_epoch_count
    global transformer_model, transformer_trained, rl_policy, rl_target, rl_trained

    _bg_training_active = True
    logger.info("Background training loop started")

    while True:
        await asyncio.sleep(60)  # wait between cycles

        if not _stored_candles or len(_stored_candles) < 2:
            continue

        try:
            # --- Rebuild dataset from stored candles ---
            X_all, y_all = [], []
            for sym, candles in _stored_candles.items():
                feat = build_features(candles)
                if feat is None:
                    continue
                labels = build_labels(candles)
                n = len(feat) - SEQ_LEN - LOOKAHEAD
                for i in range(max(n, 0)):
                    X_all.append(feat[i: i + SEQ_LEN])
                    y_all.append(int(labels[i + SEQ_LEN - 1]))

            if len(X_all) < 128:
                continue

            X = torch.tensor(np.array(X_all), dtype=torch.float32)
            y = torch.tensor(y_all, dtype=torch.long)

            def _bg_train():
                global transformer_model, transformer_trained, _bg_epoch_count
                global rl_policy, rl_target, rl_trained

                loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True, drop_last=True)
                counts = torch.bincount(y, minlength=N_CLASSES).float()
                weights = (counts.sum() / (N_CLASSES * counts.clamp(min=1))).to(DEVICE)

                # --- Transformer fine-tune with augmentation (10 epochs) ---
                if transformer_model is not None:
                    trf = transformer_model
                    trf.train()
                    opt = optim.AdamW(trf.parameters(), lr=1e-4, weight_decay=1e-3)
                    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
                    for _ in range(10):
                        for xb, yb in loader:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            xb = _augment_features(xb)  # data augmentation
                            opt.zero_grad()
                            loss = crit(trf(xb), yb)
                            loss.backward()
                            nn.utils.clip_grad_norm_(trf.parameters(), 1.0)
                            opt.step()
                    trf.eval()
                    torch.save(trf.state_dict(), DATA_DIR / "transformer_model.pt")

                # --- RL continued training from replay buffer ---
                if rl_policy is not None and rl_buffer is not None and len(rl_buffer) >= 128:
                    policy = rl_policy
                    target = rl_target
                    opt = optim.Adam(policy.parameters(), lr=1e-5)
                    policy.train()
                    for step in range(200):
                        beta = min(1.0, 0.4 + step * 0.003)
                        s, a, r, s2, d, w, idx = rl_buffer.sample(64, beta=beta)
                        s, a, r, s2, d, w = (t.to(DEVICE) for t in [s, a, r, s2, d, w])
                        q = policy(s).gather(1, a.unsqueeze(1)).squeeze()
                        with torch.no_grad():
                            next_a = policy(s2).argmax(1)
                            q_next = target(s2).gather(1, next_a.unsqueeze(1)).squeeze()
                            q_targ = r + 0.99 * q_next * (1 - d)
                        td = (q - q_targ).detach().cpu().numpy()
                        rl_buffer.update_priorities(idx, td)
                        loss = (w.to(DEVICE) * F.smooth_l1_loss(q, q_targ, reduction='none')).mean()
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        opt.step()
                        if step % 50 == 0:
                            target.load_state_dict(policy.state_dict())
                    policy.eval()
                    torch.save({"policy": policy.state_dict(), "target": target.state_dict()},
                               DATA_DIR / "rl_agent.pt")

                _bg_epoch_count += 1
                logger.info("Background training cycle #%d complete (%d samples)", _bg_epoch_count, len(X_all))

            await asyncio.to_thread(_bg_train)

        except Exception as e:
            logger.warning("Background training error (non-fatal): %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO PRICE SIMULATION — GPU-parallel risk estimation
#    Runs thousands of price paths simultaneously on GPU to estimate
#    probability of hitting stop-loss vs take-profit.
# ═══════════════════════════════════════════════════════════════════════════════

class MonteCarloRequest(BaseModel):
    candles: list           # recent OHLCV for volatility estimation
    entry_price: float
    stop_loss_pct: float    # e.g. 3.0 for -3%
    take_profit_pct: float  # e.g. 6.0 for +6%
    hours_ahead: int = 24
    simulations: int = 10000
#    Transformer + LSTM + RL + Sentiment → single recommendation
#    Agreement across diverse models = much higher conviction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/ensemble")
def predict_ensemble(req: EnsembleRequest):
    """Combine Transformer, LSTM, RL, and sentiment for a unified signal."""
    result = {
        "transformer": None,
        "lstm": None,
        "rl": None,
        "sentiment": 0.0,
        "ensemble_signal": "HOLD",
        "ensemble_confidence": 0.0,
        "agreement_score": 0.0,
    }

    feat = build_features(req.candles)

    # ── Price models ──
    price_probs = []
    if feat is not None and len(feat) >= SEQ_LEN:
        seq = feat[-SEQ_LEN:]
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        if transformer_trained and transformer_model is not None:
            with torch.no_grad():
                p = torch.softmax(transformer_model(tensor), dim=1).cpu().numpy()[0]
            result["transformer"] = {"SELL": float(p[0]), "HOLD": float(p[1]), "BUY": float(p[2])}
            price_probs.append(("transformer", p, 0.45))

        if lstm_trained and lstm_model is not None:
            with torch.no_grad():
                p = torch.softmax(lstm_model(tensor), dim=1).cpu().numpy()[0]
            result["lstm"] = {"SELL": float(p[0]), "HOLD": float(p[1]), "BUY": float(p[2])}
            price_probs.append(("lstm", p, 0.25))

    # ── RL model ──
    if rl_trained and rl_policy is not None and req.state:
        state = req.state
        if len(state) < STATE_SIZE:
            state = state + [0.0] * (STATE_SIZE - len(state))
        tensor = torch.FloatTensor(state[:STATE_SIZE]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = rl_policy(tensor).cpu().numpy()[0]
        rl_probs = np.exp(q) / np.exp(q).sum()  # softmax over Q-values
        result["rl"] = {"SELL": float(rl_probs[0]), "HOLD": float(rl_probs[1]), "BUY": float(rl_probs[2])}
        price_probs.append(("rl", rl_probs, 0.15))

    # ── Sentiment ──
    if req.headlines:
        scores = _semantic_sentiment(req.headlines)
        avg_sent = sum(scores) / len(scores) if scores else 0.0
        result["sentiment"] = round(avg_sent, 4)
        # Convert sentiment to direction probability
        sent_probs = np.array([
            max(0, -avg_sent) * 0.5 + 0.15,   # SELL
            0.4 - abs(avg_sent) * 0.2,          # HOLD
            max(0, avg_sent) * 0.5 + 0.15,     # BUY
        ])
        sent_probs = sent_probs / sent_probs.sum()
        price_probs.append(("sentiment", sent_probs, 0.15))

    if not price_probs:
        return result

    # ── Weighted ensemble ──
    total_weight = sum(w for _, _, w in price_probs)
    ensemble = np.zeros(3)
    for name, probs, weight in price_probs:
        ensemble += probs * (weight / total_weight)
    ensemble = ensemble / ensemble.sum()

    top_idx = int(np.argmax(ensemble))
    result["ensemble_signal"] = ACTIONS[top_idx]
    result["ensemble_confidence"] = round(float(ensemble[top_idx]), 4)

    # Agreement: how many models agree on the top signal
    votes = []
    for name, probs, _ in price_probs:
        votes.append(int(np.argmax(probs)))
    agreement = votes.count(top_idx) / len(votes)
    result["agreement_score"] = round(agreement, 2)

    return result


# ── Monte Carlo simulation endpoint ──────────────────────────────────────────
@app.post("/simulate/montecarlo")
def monte_carlo(req: MonteCarloRequest):
    """GPU-parallel Monte Carlo: estimate SL/TP hit probabilities."""
    closes = np.array([c[4] for c in req.candles], dtype=np.float64)
    if len(closes) < 20:
        return {"error": "Need at least 20 candles"}

    # Estimate hourly return distribution from recent data
    returns = np.diff(np.log(closes))
    mu = float(returns.mean())
    sigma = float(returns.std())
    if sigma < 1e-10:
        sigma = 0.001

    # Run simulations on GPU
    def _simulate():
        n_steps = req.hours_ahead
        n_sims = req.simulations
        sl_level = req.entry_price * (1 - req.stop_loss_pct / 100)
        tp_level = req.entry_price * (1 + req.take_profit_pct / 100)

        # Generate all random walks on GPU at once
        rand = torch.randn(n_sims, n_steps, device=DEVICE) * sigma + mu
        log_prices = torch.cumsum(rand, dim=1)
        prices = req.entry_price * torch.exp(log_prices)

        # Check SL/TP hits
        hit_sl = (prices <= sl_level).any(dim=1)
        hit_tp = (prices >= tp_level).any(dim=1)

        # First-hit analysis: which gets hit first?
        sl_first_step = torch.where(prices <= sl_level, torch.arange(n_steps, device=DEVICE).unsqueeze(0).expand_as(prices), torch.tensor(n_steps + 1, device=DEVICE)).min(dim=1).values
        tp_first_step = torch.where(prices >= tp_level, torch.arange(n_steps, device=DEVICE).unsqueeze(0).expand_as(prices), torch.tensor(n_steps + 1, device=DEVICE)).min(dim=1).values

        tp_wins = (tp_first_step < sl_first_step).sum().item()
        sl_wins = (sl_first_step < tp_first_step).sum().item()
        neither = n_sims - tp_wins - sl_wins

        final_prices = prices[:, -1]
        expected_return = ((final_prices / req.entry_price) - 1).mean().item() * 100
        max_drawdown = ((prices.min(dim=1).values / req.entry_price) - 1).mean().item() * 100

        return {
            "simulations": n_sims,
            "hours_ahead": n_steps,
            "tp_probability": round(tp_wins / n_sims * 100, 1),
            "sl_probability": round(sl_wins / n_sims * 100, 1),
            "neutral_probability": round(neither / n_sims * 100, 1),
            "expected_return_pct": round(expected_return, 3),
            "avg_max_drawdown_pct": round(max_drawdown, 3),
            "reward_risk_ratio": round((tp_wins / max(sl_wins, 1)), 2),
            "edge": round((tp_wins * req.take_profit_pct - sl_wins * req.stop_loss_pct) / n_sims, 3),
            "volatility_hourly": round(sigma * 100, 4),
        }

    return _simulate()


if __name__ == "__main__":
    port = int(os.environ.get("GPU_SERVER_PORT", "9090"))
    logger.info("Starting GPU server on port %d (device=%s)", port, DEVICE)
    uvicorn.run(app, host="0.0.0.0", port=port)
