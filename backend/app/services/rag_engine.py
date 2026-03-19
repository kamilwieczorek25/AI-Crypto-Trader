"""RAG engine — TF-IDF retrieval over news, past decisions, and market insights.

Pure-numpy/scikit-free implementation (no extra deps beyond what's already required).
Indexes documents as they arrive; retrieves relevant context before each Claude call.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    In-memory TF-IDF document retrieval.

    Documents are ingested from:
    - Recent news headlines (per symbol)
    - Completed Claude decisions with their P&L outcome
    - Free-form market insights

    Before each Claude call, query() returns the k most relevant snippets,
    giving the LLM a "memory" of what worked and what didn't.
    """

    def __init__(self, max_docs: int = 600) -> None:
        self._docs: list[str]        = []
        self._max_docs               = max_docs
        self._vocab: dict[str, int]  = {}
        self._idf:   Optional[np.ndarray] = None
        self._tfidf: Optional[np.ndarray] = None
        self._dirty  = True

    # ── document ingestion ────────────────────────────────────────────────────
    def add_news(self, symbol: str, headlines: list[str], sentiment: float) -> None:
        """Index news articles for a symbol with sentiment label."""
        label = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        for h in headlines[:3]:
            self._add(f"{symbol} market news [{label} sentiment]: {h}")

    def add_decision_outcome(
        self,
        symbol:    str,
        action:    str,
        reasoning: str,
        pnl_pct:   Optional[float],
        signals:   list[str],
    ) -> None:
        """Index a completed trade with its P&L outcome so Claude learns from history."""
        if pnl_pct is not None:
            outcome = f"PROFIT +{pnl_pct:.2f}%" if pnl_pct > 0 else f"LOSS {pnl_pct:.2f}%"
        else:
            outcome = "no P&L (HOLD or still open)"

        signals_str = "; ".join(signals[:3]) if signals else "—"
        doc = (
            f"Historical trade: {action} {symbol} → {outcome}. "
            f"Signals at entry: {signals_str}. "
            f"Reasoning: {reasoning[:250]}"
        )
        self._add(doc)

    def add_insight(self, text: str) -> None:
        """Index any free-form market insight or indicator summary."""
        self._add(text)

    def _add(self, doc: str) -> None:
        self._docs.append(doc)
        if len(self._docs) > self._max_docs:
            self._docs = self._docs[-self._max_docs:]
        self._dirty = True

    # ── retrieval ─────────────────────────────────────────────────────────────
    def query(self, query: str, k: int = 5) -> list[str]:
        """Return the k most semantically relevant documents."""
        if not self._docs:
            return []
        if self._dirty:
            self._rebuild()
        if self._tfidf is None:
            return self._docs[-k:]

        q_vec = self._vectorize([query])
        if q_vec is None:
            return self._docs[-k:]

        scores  = (self._tfidf @ q_vec[0]).ravel()
        top_k   = min(k, len(self._docs))
        idx     = np.argpartition(scores, -top_k)[-top_k:]
        idx     = idx[np.argsort(scores[idx])[::-1]]
        return [self._docs[i] for i in idx if scores[i] > 1e-6]

    # ── TF-IDF internals ──────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words  = re.sub(r"[^\w\s]", " ", text.lower()).split()
        tokens = list(words)
        tokens += [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return tokens

    def _rebuild(self) -> None:
        try:
            vocab: dict[str, int] = {}
            tf_rows = []

            for doc in self._docs:
                tokens = self._tokenize(doc)
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

            df  = (tf_mat > 0).sum(axis=0).astype(np.float32) + 1
            idf = np.log((N + 1) / df).astype(np.float32)
            tfidf = tf_mat * idf

            norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)

            self._tfidf = tfidf / norms
            self._vocab = vocab
            self._idf   = idf
            self._dirty = False
        except Exception as exc:
            logger.warning("RAG index rebuild failed: %s", exc)

    def _vectorize(self, texts: list[str]) -> Optional[np.ndarray]:
        if not self._vocab or self._idf is None:
            return None
        V   = len(self._vocab)
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
