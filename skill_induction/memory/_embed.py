"""Embedding-based retrieval helper, shared by all three memories."""

from __future__ import annotations

import math
import os
import re
from typing import Sequence

# Cap BLAS/OMP threads before importing numpy/sklearn/torch: on a loaded node OpenBLAS otherwise spawns one thread per core and dies with "pthread_create failed", taking retrieval down with it.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(_v, "1" if _v != "TOKENIZERS_PARALLELISM"
                          else "false")


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Run the embedder on CPU by default: it is tiny (all-MiniLM-L6-v2, ~90MB) and encoding is milliseconds, while default CUDA placement competes with a co-resident vLLM server for VRAM and hits CUDA OOM. Override with SKILL_INDUCTION_EMBED_DEVICE=cuda if a GPU is free.
            _dev = os.environ.get("SKILL_INDUCTION_EMBED_DEVICE", "cpu")
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=_dev)
        except Exception:  # pragma: no cover
            _MODEL = "lexical"
    return _MODEL


def topk_indices(query: str, corpus: Sequence[str], k: int) -> list[int]:
    """Return the indices of the top-k most relevant items in `corpus`.

    If `k >= len(corpus)`, returns every index in relevance order.
    """
    if not corpus:
        return []
    k = min(k, len(corpus))
    m = _model()
    if m == "lexical":
        return _lexical_topk(query, corpus, k)
    q_emb = m.encode([query], normalize_embeddings=True)
    c_emb = m.encode(list(corpus), normalize_embeddings=True)
    # cosine similarity = dot product when both are normalized
    sims = (q_emb @ c_emb.T).flatten().tolist()
    order = sorted(range(len(corpus)), key=lambda i: sims[i], reverse=True)
    return order[:k]


def cosine_sim(a: str, b: str) -> float:
    """Cosine similarity in [-1, 1] between two short strings.

    Uses the same sentence-transformer the memories already pay to load, so call sites get embedding semantics without re-paying.
    """
    m = _model()
    if m == "lexical":
        # token-Jaccard as a cheap stand-in
        ta, tb = set(_tokenize(a)), set(_tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
    emb = m.encode([a, b], normalize_embeddings=True)
    return float((emb[0] * emb[1]).sum())


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def _lexical_topk(query: str, corpus: Sequence[str], k: int) -> list[int]:
    """Trivial fallback: token-overlap with light IDF weighting."""
    docs = [_tokenize(c) for c in corpus]
    df: dict[str, int] = {}
    for d in docs:
        for tok in set(d):
            df[tok] = df.get(tok, 0) + 1
    n = len(docs) + 1
    qtoks = set(_tokenize(query))
    scores = []
    for d in docs:
        score = 0.0
        for tok in qtoks:
            if tok in d:
                idf = math.log(n / (df.get(tok, 0) + 1))
                score += idf
        scores.append(score)
    return sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:k]
