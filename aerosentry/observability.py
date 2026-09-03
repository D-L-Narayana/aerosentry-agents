"""Structured JSONL tracing for every LLM call, tool call, routing decision and safety verdict.

LangSmith: LangGraph emits full graph traces automatically when
LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set (see README). The local
JSONL log below is the zero-dependency fallback and is what the eval harness reads.
"""
import contextvars
import json
import os
import time
import uuid

from .config import settings

_run_id = contextvars.ContextVar("aerosentry_run_id", default="")


def start_run(name: str) -> str:
    rid = uuid.uuid4().hex[:12]
    _run_id.set(rid)
    trace_event("run.start", {"name": name})
    return rid


def current_run() -> str:
    return _run_id.get()


def trace_event(kind: str, payload: dict) -> None:
    rec = {"ts": round(time.time(), 3), "run_id": _run_id.get(), "event": kind, **payload}
    os.makedirs(os.path.dirname(settings.trace_path) or ".", exist_ok=True)
    with open(settings.trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def read_traces(run_id: str | None = None, limit: int = 500) -> list[dict]:
    if not os.path.exists(settings.trace_path):
        return []
    out = []
    with open(settings.trace_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id is None or rec.get("run_id") == run_id:
                out.append(rec)
    return out[-limit:]


def langsmith_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and bool(os.getenv("LANGCHAIN_API_KEY"))
