"""Golden-set evaluation harness. Measures routing accuracy, tool-call accuracy, HITL gate accuracy,
policy-blocking correctness, retrieval source hit-rate, severity calibration, paraphrase consistency
and latency. Writes evals/results.json and prints a markdown table (used in the README)."""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AEROSENTRY_TRACES", os.path.join(os.path.dirname(__file__), "..", "traces", "eval_traces.jsonl"))

from aerosentry.agents import Orchestrator  # noqa: E402
from aerosentry.config import settings  # noqa: E402

SEV = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def run():
    cases = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "golden.jsonl"), encoding="utf-8") if l.strip()]
    orch = Orchestrator()
    rows, lat = [], []
    for c in cases:
        t0 = time.time()
        out = orch.run(c["input"])
        if out["pending_approval"]:
            out = orch.approve(out["thread_id"], True)
        lat.append(time.time() - t0)
        called = {t["name"] for t in out["tool_calls"]}
        proposed = {p["name"] for p in out["proposed_actions"]}
        row = {"id": c["id"], "group": c["group"],
               "route_ok": out["route"] == c["route"],
               "tools_ok": set(c["tools"]) <= called and set(c["proposed"]) <= proposed,
               "args_valid": all(t.get("valid", True) for t in out["tool_calls"]),
               "gate_ok": (out["risk"] == "high") == c["approval"],
               "fallback": out["fallback_used"], "route": out["route"]}
        if c.get("expect_blocked"):
            row["block_ok"] = all(not e.get("ok") for e in out["executed_actions"]) and bool(out["executed_actions"])
        elif c["approval"]:
            row["block_ok"] = all(e.get("ok") for e in out["executed_actions"])
        if c.get("expect_source"):
            row["source_ok"] = any(x["source"] == c["expect_source"] for x in out["citations"][:2])
        img = next((t for t in out["tool_calls"] if t["name"] == "analyze_aerial_image"), None)
        if img and (c.get("expect_severity") or c.get("expect_severity_min")):
            traces = [x for x in __import__("aerosentry.observability", fromlist=["read_traces"]).read_traces(out["run_id"]) if x["event"] == "vision.analyze"]
            sev = traces[-1]["severity"] if traces else "unknown"
            row["severity_ok"] = (sev == c["expect_severity"]) if c.get("expect_severity") else SEV.get(sev, -1) >= SEV[c["expect_severity_min"]]
        rows.append(row)

    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    groups = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(r["route"])
    consistency = sum(len(v) == 1 for v in groups.values()) / len(groups)
    metrics = {"provider": settings.provider, "model": settings.model if settings.live else "mock", "cases": len(rows),
               "route_accuracy": rate("route_ok"), "tool_call_accuracy": rate("tools_ok"), "arg_validation_pass": rate("args_valid"),
               "hitl_gate_accuracy": rate("gate_ok"), "policy_block_correct": rate("block_ok"), "retrieval_source_hit": rate("source_ok"),
               "severity_calibration": rate("severity_ok"), "paraphrase_consistency": (consistency, len(groups)),
               "fallback_runs": sum(r["fallback"] for r in rows),
               "latency_p50_ms": int(statistics.median(lat) * 1000), "latency_p95_ms": int(sorted(lat)[int(0.95 * (len(lat) - 1))] * 1000)}
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    json.dump({"metrics": metrics, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"| Metric | Value | n |\n|---|---|---|")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            print(f"| {k} | {v[0] * 100:.1f}% | {v[1]} |" if v[0] is not None else f"| {k} | n/a | 0 |")
        else:
            print(f"| {k} | {v} | |")
    failed = [r["id"] for r in rows if not all(v for k, v in r.items() if k.endswith("_ok"))]
    print("\nfailed cases:", failed or "none")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(run())
