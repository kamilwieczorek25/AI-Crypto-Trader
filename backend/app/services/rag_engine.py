"""RAG engine — TF-IDF retrieval over news, past decisions, and market insights.

Improvements over v1:
- SQLite persistence: index survives server restarts (loaded at startup)
- Outcome-weighted ranking: profitable trade signals surface higher in results
- Temporal decay: recent documents score higher via exponential half-life
- Longer reasoning capture (800 chars vs 250) for richer Claude context
- More news headlines per symbol (5 vs 3)
- Batched index rebuilds (every 5 adds vs every single add)
- Hybrid scoring: TF-IDF cosine similarity × outcome_weight × recency_weight
"""

import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# DB stored alongside the trader SQLite DB
_DB_PATH = Path(os.environ.get("DATA_DIR", "/data")) / "rag_store.db"

# Recency half-life: a document loses half its recency weight every 24 h
_RECENCY_HALF_LIFE_SECONDS = 86_400.0

# Max characters of Claude reasoning to capture per decision (was 250)
REASONING_MAX_CHARS = 800

# Max news headlines to index per symbol per call (was 3)
NEWS_MAX_HEADLINES = 5

# Rebuild TF-IDF matrix after this many adds (batched, not on every write)
_REBUILD_EVERY_N = 5


class RAGEngine:
    """
    In-memory TF-IDF document retrieval with SQLite persistence and hybrid ranking.

    Documents are ingested from:
    - Recent news headlines (per symbol)
    - Completed Claude decisions with their P&L outcome
    - Free-form market insights

    Before each Claude call, query() returns the k most relevant snippets,
    giving the LLM a "memory" of what worked and what didn't.

    Ranking blends three signals:
      1. TF-IDF cosine similarity   — relevance to the current query
      2. Outcome weight             — profitable past trades surface higher
      3. Recency weight             — newer documents ranked over stale ones
    """

    def __init__(self, max_docs: int = 1000) -> None:
        # Each doc is a dict: {text: str, outcome_weight: float, ts: float}
        self._docs: list[dict] = []
        self._max_docs = max_docs

        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._tfidf: Optional[np.ndarray] = None
        self._dirty = True

        # Batched rebuild counter — only rebuild TF-IDF every N adds
        self._add_counter = 0

        # Open / create the persistence DB and warm-load documents
        self._db: Optional[sqlite3.Connection] = self._init_db()
        self._load_from_db()

    # ── persistence ───────────────────────────────────────────────────────────

    def _init_db(self) -> Optional[sqlite3.Connection]:
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_docs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    text            TEXT    NOT NULL,
                    outcome_weight  REAL    NOT NULL DEFAULT 1.0,
                    ts              REAL    NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON rag_docs (ts DESC)")
            conn.commit()
            logger.info("RAG: persistence DB opened at %s", _DB_PATH)
            return conn
        except Exception as exc:
            logger.warning("RAG: could not open persistence DB (%s) — running in-memory only", exc)
            return None

    def _load_from_db(self) -> None:
        """Warm-load the most recent max_docs documents from SQLite on startup."""
        if self._db is None:
            return
        try:
            rows = self._db.execute(
                "SELECT text, outcome_weight, ts FROM rag_docs "
                "ORDER BY ts DESC LIMIT ?",
                (self._max_docs,),
            ).fetchall()
            # Reverse to restore chronological order (oldest first)
            self._docs = [
                {"text": r[0], "outcome_weight": r[1], "ts": r[2]}
                for r in reversed(rows)
            ]
            if self._docs:
                self._dirty = True
            logger.info("RAG: loaded %d documents from persistent store", len(self._docs))
        except Exception as exc:
            logger.warning("RAG: DB load failed: %s", exc)

    def _persist(self, text: str, outcome_weight: float, ts: float) -> None:
        """Write a single document to the DB and trim to max_docs."""
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT INTO rag_docs (text, outcome_weight, ts) VALUES (?, ?, ?)",
                (text, outcome_weight, ts),
            )
            # Keep the DB bounded to max_docs rows
            self._db.execute(
                "DELETE FROM rag_docs WHERE id NOT IN "
                "(SELECT id FROM rag_docs ORDER BY ts DESC LIMIT ?)",
                (self._max_docs,),
            )
            self._db.commit()
        except Exception as exc:
            logger.warning("RAG: DB persist failed: %s", exc)

    # ── document ingestion ────────────────────────────────────────────────────

    def add_news(self, symbol: str, headlines: list[str], sentiment: float) -> None:
        """Index news articles for a symbol with sentiment label."""
        label = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        for h in headlines[:NEWS_MAX_HEADLINES]:
            self._add(f"{symbol} market news [{label} sentiment]: {h}", outcome_weight=1.0)

    def add_decision_outcome(
        self,
        symbol: str,
        action: str,
        reasoning: str,
        pnl_pct: Optional[float],
        signals: list[str],
    ) -> None:
        """Index a completed trade with its P&L outcome.

        Outcome weighting:
          - Profitable trades (pnl_pct > 0) get weight > 1.0 → surface higher
          - Losing trades still indexed but down-weighted → surface less
          - +10% P&L → weight 2.0, -10% P&L → weight 0.3, 0% → weight 1.0
          - Unresolved HOLDs get weight 0.8 (slight down-weight)
        """
        if pnl_pct is not None:
            outcome = f"PROFIT +{pnl_pct:.2f}%" if pnl_pct > 0 else f"LOSS {pnl_pct:.2f}%"
            outcome_weight = float(max(0.2, 1.0 + pnl_pct / 10.0))
        else:
            outcome = "no P&L (HOLD or still open)"
            outcome_weight = 0.8

        signals_str = "; ".join(signals[:5]) if signals else "—"
        doc = (
            f"Historical trade: {action} {symbol} → {outcome}. "
            f"Signals at entry: {signals_str}. "
            f"Reasoning: {reasoning[:REASONING_MAX_CHARS]}"
        )
        self._add(doc, outcome_weight=outcome_weight)

    def add_insight(self, text: str) -> None:
        """Index any free-form market insight or indicator summary."""
        self._add(text, outcome_weight=1.0)

    def _add(self, text: str, outcome_weight: float = 1.0) -> None:
        ts = time.time()
        entry = {"text": text, "outcome_weight": outcome_weight, "ts": ts}

        self._docs.append(entry)
        if len(self._docs) > self._max_docs:
            self._docs = self._docs[-self._max_docs:]

        self._persist(text, outcome_weight, ts)

        # Batched rebuild: mark dirty every N adds, not on every single write
        self._add_counter += 1
        if self._add_counter >= _REBUILD_EVERY_N:
            self._dirty = True
            self._add_counter = 0

    # ── retrieval ─────────────────────────────────────────────────────────────

    def query(self, query: str, k: int = 5) -> list[str]:
        """Return the k most relevant documents using hybrid scoring.

        Score = TF-IDF_cosine_similarity × outcome_weight × recency_weight

        This means:
          - A perfectly keyword-matching document from yesterday with +15% P&L
            scores much higher than the same text from 2 weeks ago with a loss.
          - Irrelevant documents (cosine ≈ 0) are not boosted regardless of
            how profitable or recent they are.
        """
        if not self._docs:
            return []
        if self._dirty:
            self._rebuild()
        if self._tfidf is None:
            return [d["text"] for d in self._docs[-k:]]

        q_vec = self._vectorize([query])
        if q_vec is None:
            return [d["text"] for d in self._docs[-k:]]

        # 1. TF-IDF cosine similarity scores
        tfidf_scores = (self._tfidf @ q_vec[0]).ravel()

        # 2. Recency weights: exponential decay — halves every RECENCY_HALF_LIFE
        now = time.time()
        recency_weights = np.array(
            [2.0 ** (-(now - d["ts"]) / _RECENCY_HALF_LIFE_SECONDS) for d in self._docs],
            dtype=np.float32,
        )

        # 3. Outcome weights from stored metadata
        outcome_weights = np.array(
            [d["outcome_weight"] for d in self._docs],
            dtype=np.float32,
        )

        # Hybrid score: only blend when there is actual TF-IDF signal
        # (avoids surfacing irrelevant docs just because they're recent/profitable)
        hybrid = tfidf_scores * outcome_weights * recency_weights

        top_k = min(k, len(self._docs))
        idx = np.argpartition(hybrid, -top_k)[-top_k:]
        idx = idx[np.argsort(hybrid[idx])[::-1]]
        return [self._docs[i]["text"] for i in idx if hybrid[i] > 1e-8]

    # ── TF-IDF internals ──────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        tokens = list(words)
        # Bigrams for better phrase matching
        tokens += [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        return tokens

    def _rebuild(self) -> None:
        try:
            vocab: dict[str, int] = {}
            tf_rows = []

            for doc in self._docs:
                tokens = self._tokenize(doc["text"])
                tf: dict[str, float] = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                total = max(sum(tf.values()), 1)
                for t in list(tf):
                    tf[t] /= total
                    if t not in vocab:
                        vocab[t] = len(vocab)
                tf_rows.append(tf)

            V, N = len(vocab), len(self._docs)
            if V == 0 or N == 0:
                return

            tf_mat = np.zeros((N, V), dtype=np.float32)
            for i, tf in enumerate(tf_rows):
                for t, v in tf.items():
                    tf_mat[i, vocab[t]] = v

            df = (tf_mat > 0).sum(axis=0).astype(np.float32) + 1
            idf = np.log((N + 1) / df).astype(np.float32)
            tfidf = tf_mat * idf

            norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)

            self._tfidf = tfidf / norms
            self._vocab = vocab
            self._idf = idf
            self._dirty = False
        except Exception as exc:
            logger.warning("RAG index rebuild failed: %s", exc)

    def _vectorize(self, texts: list[str]) -> Optional[np.ndarray]:
        if not self._vocab or self._idf is None:
            return None
        V = len(self._vocab)
        mat = np.zeros((len(texts), V), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf: dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            total = max(sum(tf.values()), 1)
            for t, v in tf.items():
                if t in self._vocab:
                    mat[i, self._vocab[t]] = (v / total) * self._idf[self._vocab[t]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return mat / norms

    @property
    def doc_count(self) -> int:
        return len(self._docs)


rag_engine = RAGEngine()
