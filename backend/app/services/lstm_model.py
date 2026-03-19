"""LSTM predictor — sequential price direction prediction (SELL / HOLD / BUY)."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("torch not installed — LSTM predictor disabled (install torch to enable)")

logger = logging.getLogger(__name__)

# ── Hyperparams ───────────────────────────────────────────────────────────────
SEQ_LEN         = 20      # look-back window (candles)
N_FEATURES      = 6       # features per candle
N_CLASSES       = 3       # SELL=0, HOLD=1, BUY=2
LOOKAHEAD       = 3       # candles ahead to measure future return
LABEL_THRESHOLD = 0.012   # 1.2% move = signal
MODEL_PATH      = Path(os.environ.get("DATA_DIR", "/data")) / "lstm_model.pt"
ACTIONS         = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ── Model ─────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(N_FEATURES, 128, num_layers=2,
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, N_CLASSES),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])
else:
    _LSTMNet = None  # type: ignore[assignment,misc]


# ── Feature / label helpers ───────────────────────────────────────────────────
def build_features(candles: list) -> Optional[np.ndarray]:
    """Convert raw [[t,o,h,l,c,v],...] → (N, N_FEATURES) float32 array."""
    if len(candles) < SEQ_LEN + LOOKAHEAD + 20:
        return None

    closes  = np.array([c[4] for c in candles], dtype=np.float64)
    highs   = np.array([c[2] for c in candles], dtype=np.float64)
    lows    = np.array([c[3] for c in candles], dtype=np.float64)
    volumes = np.array([c[5] for c in candles], dtype=np.float64)
    n = len(closes)

    # 1. Close return
    ret = np.zeros(n)
    ret[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)

    # 2. Volume vs 20-bar MA
    vol_ma = np.convolve(volumes, np.ones(20) / 20, mode="same")
    vol_r  = np.where(vol_ma > 0, volumes / vol_ma, 1.0)

    # 3. RSI-14 normalised
    rsi = _rsi(closes, 14) / 100.0

    # 4. MACD histogram / price × 100
    macd_h = _macd_hist(closes, 12, 26, 9)
    macd_n = np.where(closes > 0, macd_h / closes, 0.0) * 100

    # 5. Bollinger Band %B
    bb = _bb_pct(closes, 20, 2.0)

    # 6. ATR / price × 100
    atr   = _atr(highs, lows, closes, 14)
    atr_p = atr / (closes + 1e-10) * 100

    feat = np.stack([
        np.clip(ret,   -0.10,  0.10),
        np.clip(vol_r,  0.00,  5.00) / 5.0,
        np.clip(rsi,    0.00,  1.00),
        np.clip(macd_n,-5.00,  5.00) / 5.0,
        np.clip(bb,    -0.50,  1.50),
        np.clip(atr_p,  0.00, 10.00) / 10.0,
    ], axis=1).astype(np.float32)

    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _build_labels(candles: list, threshold: float = LABEL_THRESHOLD) -> np.ndarray:
    """Label each candle based on the return over the next LOOKAHEAD candles.

    The label at index i represents the action that SHOULD have been taken
    at candle i, based on what happened next.  We assign the label to index i
    (not i+LOOKAHEAD) so that it lines up with the input sequence ending at i.
    The training loop already accounts for this: it uses label[i + SEQ_LEN - 1]
    for the sequence ending at that position.

    Only candles with enough future data get a directional label; the tail
    candles keep the default HOLD to avoid look-ahead into non-existent data.
    """
    closes = np.array([c[4] for c in candles], dtype=np.float64)
    labels = np.ones(len(closes), dtype=np.int64)   # default HOLD
    for i in range(len(closes) - LOOKAHEAD):
        future_return = (closes[i + LOOKAHEAD] - closes[i]) / (closes[i] + 1e-10)
        if future_return > threshold:
            labels[i] = 2   # BUY — price went up
        elif future_return < -threshold:
            labels[i] = 0   # SELL — price went down
    return labels


# ── Service class ─────────────────────────────────────────────────────────────
class LSTMPredictor:
    """Trains on historical candle data; predicts direction probabilities."""

    def __init__(self) -> None:
        if not TORCH_AVAILABLE:
            self.device = None
            self.model  = None
            self._trained = False
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[_LSTMNet] = None
        self._trained = False
        self._load_from_disk()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load_from_disk(self) -> None:
        if not TORCH_AVAILABLE or not MODEL_PATH.exists():
            return
        try:
            m = _LSTMNet().to(self.device)
            m.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            )
            m.eval()
            self.model = m
            self._trained = True
            logger.info("LSTM model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("LSTM load failed: %s", exc)

    def _save_to_disk(self) -> None:
        if not TORCH_AVAILABLE:
            return
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), MODEL_PATH)
            logger.info("LSTM model saved → %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("LSTM save failed: %s", exc)

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, all_candles: dict[str, list]) -> bool:
        """Blocking — call train_async() to avoid blocking the event loop."""
        if not TORCH_AVAILABLE:
            return False
        logger.info("LSTM: preparing training data from %d symbols …", len(all_candles))
        X_all, y_all = [], []

        for sym, candles in all_candles.items():
            feat = build_features(candles)
            if feat is None:
                continue
            labels = _build_labels(candles)
            # Only use samples where the label is not in the LOOKAHEAD tail
            # (tail labels default to HOLD and would add noise)
            n = len(feat) - SEQ_LEN - LOOKAHEAD
            for i in range(max(n, 0)):
                X_all.append(feat[i: i + SEQ_LEN])
                y_all.append(int(labels[i + SEQ_LEN - 1]))

        if len(X_all) < 128:
            logger.warning("LSTM: only %d samples — skipping", len(X_all))
            return False

        X = torch.tensor(np.array(X_all), dtype=torch.float32)
        y = torch.tensor(y_all, dtype=torch.long)

        # Balanced class weights
        counts  = torch.bincount(y, minlength=N_CLASSES).float()
        weights = (counts.sum() / (N_CLASSES * counts.clamp(min=1))).to(self.device)

        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True, drop_last=True)
        model  = _LSTMNet().to(self.device)
        opt    = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        crit   = nn.CrossEntropyLoss(weight=weights)

        model.train()
        best_loss = float("inf")
        patience_counter = 0
        patience = 8  # early stopping patience
        for epoch in range(40):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            sched.step()
            avg_loss = total / len(loader)
            if (epoch + 1) % 10 == 0:
                logger.info("LSTM epoch %d/40 — loss=%.4f", epoch + 1, avg_loss)
            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("LSTM early stopping at epoch %d (loss=%.4f)", epoch + 1, avg_loss)
                    break

        model.eval()
        self.model   = model
        self._trained = True
        self._save_to_disk()
        logger.info("LSTM training complete (%d samples)", len(X_all))
        return True

    async def train_async(self, all_candles: dict[str, list]) -> bool:
        return await asyncio.to_thread(self.train, all_candles)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, candles: list) -> dict:
        """Return {'BUY': p, 'HOLD': p, 'SELL': p, 'signal': str, 'confidence': float}"""
        neutral = {
            "BUY": 0.33, "HOLD": 0.34, "SELL": 0.33,
            "signal": "HOLD", "confidence": 0.34, "status": "untrained",
        }
        if not TORCH_AVAILABLE or not self._trained or self.model is None:
            return neutral

        feat = build_features(candles)
        if feat is None or len(feat) < SEQ_LEN:
            return {**neutral, "status": "insufficient_data"}

        seq    = feat[-SEQ_LEN:]
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        return {
            "BUY":        float(probs[2]),
            "HOLD":       float(probs[1]),
            "SELL":       float(probs[0]),
            "signal":     ACTIONS[top_idx],
            "confidence": float(probs[top_idx]),
            "status":     "ok",
        }

    @property
    def is_trained(self) -> bool:
        return self._trained


# ── Pure-numpy indicator helpers ──────────────────────────────────────────────
def _rsi(c: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(c), 50.0)
    d   = np.diff(c)
    g   = np.where(d > 0, d,  0.0)
    l_  = np.where(d < 0, -d, 0.0)
    if len(g) < p:
        return out
    ag, al = g[:p].mean(), l_[:p].mean()
    out[p] = 100 - 100 / (1 + ag / (al + 1e-10))
    for i in range(p, len(d)):
        ag = (ag * (p - 1) + g[i])  / p
        al = (al * (p - 1) + l_[i]) / p
        out[i + 1] = 100 - 100 / (1 + ag / (al + 1e-10))
    return out


def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.zeros(len(arr))
    if len(arr) < p:
        return out
    out[p - 1] = arr[:p].mean()
    k = 2 / (p + 1)
    for i in range(p, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _macd_hist(c: np.ndarray, fast=12, slow=26, sig=9) -> np.ndarray:
    macd = _ema(c, fast) - _ema(c, slow)
    return macd - _ema(macd, sig)


def _bb_pct(c: np.ndarray, p=20, m=2.0) -> np.ndarray:
    out = np.full(len(c), 0.5)
    for i in range(p - 1, len(c)):
        w   = c[i - p + 1: i + 1]
        std = w.std()
        if std > 0:
            mn     = w.mean()
            out[i] = (c[i] - (mn - m * std)) / (2 * m * std + 1e-10)
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p=14) -> np.ndarray:
    out = np.zeros(len(c))
    for i in range(1, len(c)):
        tr     = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        out[i] = (out[i - 1] * (p - 1) + tr) / p if i >= p else tr
    return out


lstm_predictor = LSTMPredictor()
