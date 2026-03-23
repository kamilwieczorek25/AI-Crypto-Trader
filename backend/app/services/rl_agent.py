"""DQN reinforcement-learning agent for trading recommendations.

Learns from live paper-trading outcomes via experience replay.
Pre-trains on historical data at startup; improves every cycle.
"""

import asyncio
import logging
import os
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)
if not TORCH_AVAILABLE:
    logger.warning("torch not installed — RL agent disabled (install torch to enable)")

AGENT_PATH  = Path(os.environ.get("DATA_DIR", "/data")) / "rl_agent.pt"
STATE_SIZE  = 14        # matches GPU server DuelingDQN input size
ACTION_SIZE = 3         # 0=HOLD, 1=BUY, 2=SELL
ACTION_MAP  = {0: "HOLD", 1: "BUY", 2: "SELL"}
ACTION_INV  = {"HOLD": 0, "BUY": 1, "SELL": 2}


# ── Neural network ────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class _DQN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_SIZE, 128), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, ACTION_SIZE),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    _DQN = None  # type: ignore[assignment,misc]


# ── Experience replay buffer ──────────────────────────────────────────────────
class _ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done) -> None:
        self._buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self._buf, n)
        s, a, r, s2, d = zip(*batch)
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for RL sampling")
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── Agent ─────────────────────────────────────────────────────────────────────
class RLTradingAgent:
    """
    Online DQN that learns from actual paper-trading outcomes.

    Each cycle:
      1. build_state()   → 12-dim observation vector
      2. recommend()     → suggested action + Q-values (given to Claude as signal)
      3. record_and_learn() → after the cycle, provide new portfolio value;
                              agent stores transition and updates weights.
    """

    def __init__(self) -> None:
        self.epsilon  = 0.9
        self._steps   = 0
        self._trained = False
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int]        = None
        self._prev_value: Optional[float]       = None

        if not TORCH_AVAILABLE:
            self.device = None
            self.policy = None
            self.target = None
            self.opt    = None
            self.buffer = _ReplayBuffer()
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = _DQN().to(self.device)
        self.target = _DQN().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt    = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.buffer = _ReplayBuffer()
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not TORCH_AVAILABLE or not AGENT_PATH.exists():
            return
        try:
            ckpt = torch.load(AGENT_PATH, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(ckpt["policy"])
            self.target.load_state_dict(ckpt["target"])
            self.epsilon  = ckpt.get("epsilon", 0.3)
            self._steps   = ckpt.get("steps",   0)
            self._trained = True
            logger.info("RL agent loaded (steps=%d ε=%.3f)", self._steps, self.epsilon)
        except Exception as exc:
            logger.warning("RL load failed: %s", exc)

    def _save(self) -> None:
        if not TORCH_AVAILABLE:
            return
        try:
            AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy":  self.policy.state_dict(),
                "target":  self.target.state_dict(),
                "epsilon": self.epsilon,
                "steps":   self._steps,
            }, AGENT_PATH)
        except Exception as exc:
            logger.warning("RL save failed: %s", exc)

    # ── state construction ────────────────────────────────────────────────────
    def build_state(
        self,
        ind: dict,
        portfolio: dict,
        lstm: dict,
        has_position: bool,
    ) -> np.ndarray:
        """Build a normalised 14-dim state vector.

        Dims 0-11 are the original 12 features.
        Dims 12-13 are new: OBV trend and BB squeeze — both always present in
        the indicators dict and meaningful for momentum/breakout detection.
        The 14-dim size matches the GPU server's DuelingDQN input so remote
        predictions use all features rather than zero-padding the last two.
        """
        total = max(portfolio.get("total_value_usdt", 10_000), 1.0)
        close = ind.get("close", 1.0) or 1.0
        return np.array([
            # ── Original 12 dims ────────────────────────────────────────
            np.clip(ind.get("rsi14", 50) / 100,  0,  1),
            np.clip(ind.get("macd_hist", 0) / (close * 0.02 + 1e-10), -1, 1),
            np.clip(ind.get("bb_pct_b", 0.5),    0,  1),
            np.clip(ind.get("volume_ratio", 1) / 5, 0, 1),
            np.clip(
                (ind.get("close", 0) - ind.get("ema20", ind.get("close", 0)))
                / (close + 1e-6) * 50, -1, 1
            ),
            np.clip(ind.get("atr", 0) / (close + 1e-6) * 50, 0, 1),
            lstm.get("BUY",  0.33),
            lstm.get("SELL", 0.33),
            1.0 if has_position else 0.0,
            np.clip(portfolio.get("cash_usdt", 5000) / total, 0, 1),
            np.clip(portfolio.get("total_pnl_pct", 0) / 20, -1, 1),
            float(np.clip(ind.get("trend", 0), -1, 1)) * 0.5 + 0.5,
            # ── New dims 12-13 (match GPU server STATE_SIZE=14) ─────────
            # OBV trend: -1=falling, 0=flat, +1=rising → normalised to [0, 1]
            float(np.clip(ind.get("obv_trend", 0), -1, 1)) * 0.5 + 0.5,
            # BB squeeze: 1.0 = squeeze active (breakout likely), 0.0 = no squeeze
            float(np.clip(ind.get("bb_squeeze", 0), 0, 1)),
        ], dtype=np.float32)

    # ── recommendation ────────────────────────────────────────────────────────
    async def recommend_remote(self, state: np.ndarray) -> dict | None:
        """Try GPU server for RL prediction; returns None if unavailable."""
        from app.services import gpu_client
        if not gpu_client.is_enabled():
            return None
        result = await gpu_client.predict_rl(state.tolist())
        if result and result.get("trained"):
            return {
                "action":   result["action"],
                "q_values": result["q_values"],
                "epsilon":  round(self.epsilon, 3),
                "trained":  True,
                "steps":    self._steps,
            }
        return None

    def recommend(self, state: np.ndarray) -> dict:
        """Return action recommendation with Q-values for Claude's context."""
        action_idx = random.randint(0, ACTION_SIZE - 1)
        q = np.zeros(ACTION_SIZE, dtype=np.float32)

        if TORCH_AVAILABLE and self.policy is not None:
            with torch.no_grad():
                q = self.policy(
                    torch.FloatTensor(state).unsqueeze(0).to(self.device)
                ).cpu().numpy()[0]
            if self._trained and random.random() >= self.epsilon:
                action_idx = int(np.argmax(q))

        return {
            "action":   ACTION_MAP[action_idx],
            "q_values": {ACTION_MAP[i]: round(float(q[i]), 4) for i in range(ACTION_SIZE)},
            "epsilon":  round(self.epsilon, 3),
            "trained":  self._trained,
            "steps":    self._steps,
        }

    # ── online learning ───────────────────────────────────────────────────────
    def observe_cycle_end(self, next_state: np.ndarray, portfolio_value: float) -> None:
        """Call AFTER trade execution to attribute reward to the correct action.

        Previously this was called at cycle START, which attributed the reward
        to the wrong action (off-by-one).  Now called at cycle END so reward
        is computed immediately after the action that caused it.
        """
        if self._prev_state is not None and self._prev_action is not None:
            reward = (portfolio_value - self._prev_value) / (self._prev_value + 1e-10)
            if self._prev_action in (1, 2):
                reward -= 0.0003  # transaction cost penalty
            self.buffer.push(self._prev_state, self._prev_action, reward, next_state, 0.0)
            if TORCH_AVAILABLE:
                self._train_step()
                self.epsilon = max(0.10, self.epsilon * 0.999)
                self._steps += 1
                if self._steps % 50 == 0:
                    self.target.load_state_dict(self.policy.state_dict())
                    self._save()
                    self._trained = True

        self._prev_value = portfolio_value

    def record_action(self, state: np.ndarray, action: str) -> None:
        """Record the action taken this cycle for immediate reward computation."""
        self._prev_state  = state
        self._prev_action = ACTION_INV.get(action, 0)

    def set_cycle_baseline(self, portfolio_value: float) -> None:
        """Set the portfolio value baseline at the start of a cycle."""
        self._prev_value = portfolio_value

    def _train_step(self) -> Optional[float]:
        if not TORCH_AVAILABLE or self.policy is None or len(self.buffer) < 64:
            return None
        s, a, r, s2, done = (t.to(self.device) for t in self.buffer.sample(64))
        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            q_tgt  = r + 0.95 * q_next * (1 - done)
        loss = F.smooth_l1_loss(q_pred, q_tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    # ── offline pre-training ──────────────────────────────────────────────────
    def pretrain_on_history(self, all_candles: dict[str, list]) -> None:
        """
        Warm-start the DQN by simulating simple buy/hold/sell decisions on
        historical candle data and learning from the resulting P&L.
        """
        if not TORCH_AVAILABLE:
            return
        logger.info("RL pre-training on %d symbols …", len(all_candles))
        from app.services.lstm_model import build_features

        for sym, candles in all_candles.items():
            feat = build_features(candles)
            if feat is None or len(feat) < 40:
                continue
            closes = np.array([c[4] for c in candles])
            cash, position_qty, position_price = 10_000.0, 0.0, 0.0
            portfolio_value = 10_000.0
            prev_portfolio_value = 10_000.0

            for i in range(20, len(feat) - 5):
                ind_proxy = {
                    "rsi14":         float(feat[i, 2] * 100),
                    "macd_hist":     float(feat[i, 3] * closes[i] * 0.05),
                    "bb_pct_b":      float(feat[i, 4]),
                    "volume_ratio":  float(feat[i, 1] * 5),
                    "close":         float(closes[i]),
                    "ema20":         float(closes[i]),
                    "atr":           float(feat[i, 5] * closes[i] * 0.1),
                    "trend":         float(feat[i, 0]) * 5,
                }
                lstm_dummy = {
                    "BUY":  float(feat[i, 0] > 0) * 0.6 + 0.2,
                    "SELL": float(feat[i, 0] < 0) * 0.6 + 0.2,
                }
                pf = {"total_value_usdt": portfolio_value, "cash_usdt": cash, "total_pnl_pct": 0.0}
                state = self.build_state(ind_proxy, pf, lstm_dummy, position_qty > 0)

                # Pick action using current policy (epsilon-greedy) rather than
                # supervised future-return labels (which cause look-ahead bias)
                if TORCH_AVAILABLE and self.policy is not None and random.random() > 0.5:
                    with torch.no_grad():
                        q = self.policy(
                            torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        ).cpu().numpy()[0]
                    action_idx = int(np.argmax(q))
                else:
                    action_idx = random.randint(0, ACTION_SIZE - 1)

                price = closes[i]
                if action_idx == 1 and cash > 100:       # BUY
                    qty = cash * 0.3 / price
                    position_qty  += qty
                    position_price = price
                    cash          -= qty * price
                elif action_idx == 2 and position_qty > 0:  # SELL
                    cash          += position_qty * price
                    position_qty   = 0.0

                next_price      = closes[min(i + 1, len(closes) - 1)]
                portfolio_value = cash + position_qty * next_price
                # Reward = actual portfolio return from this step
                reward = (portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-10)
                if action_idx in (1, 2):
                    reward -= 0.0003  # transaction cost
                prev_portfolio_value = portfolio_value

                if i + 1 < len(feat):
                    ni = ind_proxy.copy(); ni["close"] = float(next_price)
                    npf = {"total_value_usdt": portfolio_value, "cash_usdt": cash, "total_pnl_pct": 0.0}
                    next_state = self.build_state(ni, npf, lstm_dummy, position_qty > 0)
                else:
                    next_state = state

                self.buffer.push(state, action_idx, reward, next_state, 0.0)

        trained_steps = sum(1 for _ in range(300) if self._train_step() is not None)
        if trained_steps > 0 and TORCH_AVAILABLE and self.policy is not None:
            self.target.load_state_dict(self.policy.state_dict())
            self.epsilon  = 0.4
            self._trained = True
            self._save()
            logger.info("RL pre-training done (%d gradient steps)", trained_steps)

    async def pretrain_async(self, all_candles: dict[str, list]) -> None:
        # Try GPU server first
        from app.services import gpu_client
        if gpu_client.is_enabled():
            result = await gpu_client.train_rl(all_candles)
            if result and result.get("status") == "ok":
                logger.info("RL pre-trained on GPU server (%s)", result)
                self._trained = True
                return
            logger.info("GPU RL train unavailable, falling back to local CPU")
        await asyncio.to_thread(self.pretrain_on_history, all_candles)

    @property
    def is_trained(self) -> bool:
        return self._trained


rl_agent = RLTradingAgent()
