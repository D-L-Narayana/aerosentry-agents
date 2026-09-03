"""Hybrid vector store: dense embeddings (hashing-trick by default, OpenAI-compatible optional) +
BM25 sparse retrieval, fused with Reciprocal Rank Fusion. Pure numpy; persists to disk."""
import hashlib
import json
import math
import os
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HashEmbedder:
    """Deterministic 512-d unigram+bigram feature-hashing embedder. No model download, fully offline."""

    name, dim = "hash-512", 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for g in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


class OpenAIEmbedder:
    """OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, etc.)."""

    def __init__(self, model: str = "text-embedding-3-small"):
        import httpx

        from ..config import settings

        self.name, self.dim = model, 0
        self.client, self.url = httpx.Client(timeout=60), settings.resolved_base_url + "/embeddings"
        self.headers = {"Authorization": f"Bearer {settings.api_key}"}

    def embed(self, texts):
        r = self.client.post(self.url, json={"model": self.name, "input": texts}, headers=self.headers)
        r.raise_for_status()
        vecs = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


class HybridStore:
    def __init__(self, embedder=None, k1: float = 1.5, b: float = 0.75):
        self.embedder = embedder or HashEmbedder()
        self.k1, self.b = k1, b
        self.chunks: list[dict] = []
        self.vecs = np.zeros((0, 1), dtype=np.float32)
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.avgdl = 1.0

    def add(self, chunks: list[dict]) -> None:
        new_vecs = self.embedder.embed([c["text"] for c in chunks])
        self.vecs = new_vecs if len(self.chunks) == 0 else np.vstack([self.vecs, new_vecs])
        for c in chunks:
            toks = tokenize(c["text"])
            tf = Counter(toks)
            self.tf.append(tf)
            self.df.update(tf.keys())
            self.chunks.append(c)
        self.avgdl = sum(sum(t.values()) for t in self.tf) / max(1, len(self.tf))

    def dense_search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.vecs @ q
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def bm25_search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        n, out = len(self.chunks), []
        qt = tokenize(query)
        for i, tf in enumerate(self.tf):
            dl, s = sum(tf.values()), 0.0
            for t in qt:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                out.append((i, s))
        return sorted(out, key=lambda x: -x[1])[:k]

    def hybrid_search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[dict]:
        fused: dict[int, float] = {}
        for ranked in (self.dense_search(query, k * 2), self.bm25_search(query, k * 2)):
            for rank, (i, _) in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [{**self.chunks[i], "score": round(s, 5)} for i, s in top]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vecs.npy"), self.vecs)
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"embedder": self.embedder.name, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: str, embedder=None) -> "HybridStore":
        with open(os.path.join(path, "chunks.json"), encoding="utf-8") as f:
            meta = json.load(f)
        store = cls(embedder)
        store.add(meta["chunks"])  # rebuilds sparse stats; dense vecs recomputed deterministically
        return store
