"""Persistent memory (SQLite): durable site facts + episodic summaries of past runs, so agents
carry context across threads instead of starting cold every time."""
import os
import sqlite3
import time

from .config import settings


def _conn():
    os.makedirs(os.path.join(settings.data_dir, "state"), exist_ok=True)
    c = sqlite3.connect(os.path.join(settings.data_dir, "state", "memory.db"))
    c.execute("CREATE TABLE IF NOT EXISTS facts(site_id TEXT, key TEXT, value TEXT, ts REAL, PRIMARY KEY(site_id,key))")
    c.execute("CREATE TABLE IF NOT EXISTS episodes(thread_id TEXT, route TEXT, summary TEXT, ts REAL)")
    return c


def remember_fact(site_id: str, key: str, value: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO facts VALUES(?,?,?,?)", (site_id, key, value, time.time()))


def remember_episode(thread_id: str, route: str, summary: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO episodes VALUES(?,?,?,?)", (thread_id, route, summary[:600], time.time()))


def recall(site_id: str, limit: int = 5) -> list[str]:
    with _conn() as c:
        facts = [f"{k}: {v}" for k, v in c.execute("SELECT key,value FROM facts WHERE site_id=? ORDER BY ts DESC LIMIT ?", (site_id, limit))]
        eps = [f"past run ({r}): {s}" for r, s in c.execute("SELECT route,summary FROM episodes ORDER BY ts DESC LIMIT ?", (limit,))]
    return facts + eps
