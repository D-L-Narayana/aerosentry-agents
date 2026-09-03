"""Ingest operator manuals (markdown) and flight logs (JSONL) into the hybrid store with
paragraph-aware chunking and overlap, so agents are grounded in domain-specific knowledge."""
import glob
import json
import os

from ..config import settings
from .store import HashEmbedder, HybridStore, OpenAIEmbedder


def chunk_text(text: str, size: int = 420, overlap: int = 60) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] + "\n" + p).strip() if overlap and buf else p
            while len(buf) > size:
                chunks.append(buf[:size])
                buf = buf[size - overlap:]
    if buf:
        chunks.append(buf)
    return chunks


def load_documents(data_dir: str | None = None) -> list[dict]:
    data_dir = data_dir or settings.data_dir
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "manuals", "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title = text.splitlines()[0].lstrip("# ").strip() if text else os.path.basename(path)
        for i, ch in enumerate(chunk_text(text)):
            docs.append({"id": f"{os.path.basename(path)}#{i}", "text": ch, "source": os.path.basename(path), "title": title, "kind": "manual"})
    log_path = os.path.join(data_dir, "logs", "incidents.jsonl")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                text = f"[{rec['ts']}] {rec['site_id']} {rec['drone_id']} {rec['event']}: {rec['detail']} Resolution: {rec['resolution']}"
                docs.append({"id": f"incidents#{i}", "text": text, "source": "incidents.jsonl", "title": "Incident log", "kind": "log"})
    return docs


def build_store(data_dir: str | None = None) -> HybridStore:
    embedder = OpenAIEmbedder() if settings.embed_provider == "openai" else HashEmbedder()
    store = HybridStore(embedder)
    store.add(load_documents(data_dir))
    return store


if __name__ == "__main__":
    s = build_store()
    out = os.path.join(settings.data_dir, "index")
    s.save(out)
    print(f"indexed {len(s.chunks)} chunks -> {out}")
