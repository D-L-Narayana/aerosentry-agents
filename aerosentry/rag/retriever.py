from functools import lru_cache

from ..observability import trace_event
from .ingest import build_store


@lru_cache(maxsize=1)
def get_store():
    return build_store()


def retrieve(query: str, k: int = 4) -> list[dict]:
    hits = get_store().hybrid_search(query, k=k)
    trace_event("rag.retrieve", {"query": query[:120], "k": k, "sources": [h["id"] for h in hits]})
    return hits


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
