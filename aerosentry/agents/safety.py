"""Safety layer: deterministic policy checks on every proposed action, risk classification,
human-in-the-loop gate and post-approval execution. Autonomy never bypasses this node."""
from ..observability import trace_event
from ..tools import SITES, execute_tool


def policy_check(action: dict, tool_results: list[dict]) -> list[str]:
    """Hard rules that hold regardless of what the LLM proposed."""
    name, args, findings = action["name"], action["arguments"], []
    tele = next((r["result"] for r in tool_results if r["name"] == "get_fleet_telemetry" and r.get("ok")), {})
    wx = next((r["result"] for r in tool_results if r["name"] == "get_weather" and r.get("ok")), {})
    geo = next((r["result"] for r in tool_results if r["name"] == "check_geofence" and r.get("ok")), {})
    if name == "schedule_mission":
        site = SITES.get(args.get("site_id", ""), {})
        if not tele:
            findings.append("no telemetry evidence gathered before scheduling")
        if tele and tele.get("battery_pct", 100) < 40:
            findings.append(f"battery {tele['battery_pct']}% below 40% floor")
        if wx and site and wx.get("wind_ms", 0) > site.get("max_wind_ms", 99):
            findings.append(f"wind {wx['wind_ms']} m/s exceeds site limit {site['max_wind_ms']} m/s")
        if geo and not geo.get("inside", True):
            findings.append("target outside approved geofence")
        if float(args.get("altitude_m", 0)) > site.get("ceiling_m", 120):
            findings.append("altitude above site ceiling")
    if name == "raise_alert" and args.get("severity") in ("high", "critical"):
        img = next((r["result"] for r in tool_results if r["name"] == "analyze_aerial_image" and r.get("ok")), None)
        if img and img.get("severity") in ("low", "medium"):
            findings.append("alert severity not supported by imagery evidence")
    return findings


def safety_gate(state):
    proposed = state.get("proposed_actions", [])
    findings = []
    for a in proposed:
        findings += [f"{a['name']}: {f}" for f in policy_check(a, state.get("tool_results", []))]
    risk = "high" if proposed else "low"
    if state.get("fallback_used") and proposed:
        findings.append("agent operated in degraded (fallback) mode; operator must verify evidence")
    trace_event("safety.verdict", {"risk": risk, "proposed": [a["name"] for a in proposed], "findings": findings})
    return {"risk": risk, "requires_approval": bool(proposed), "policy_findings": findings}


def human_approval(state):
    """Graph interrupts BEFORE this node; the operator sets state['approved'] then resumes."""
    trace_event("hitl.decision", {"approved": state.get("approved")})
    return {}


def execute_actions(state):
    executed = []
    if state.get("approved"):
        for a in state.get("proposed_actions", []):
            if any(f.startswith(a["name"] + ":") for f in state.get("policy_findings", [])):
                executed.append({**a, "ok": False, "error": "blocked by policy despite approval"})
                trace_event("action.blocked", {"tool": a["name"]})
                continue
            res = execute_tool(a["name"], a["arguments"])
            executed.append({**a, **res})
    else:
        executed = [{**a, "ok": False, "error": "rejected by operator"} for a in state.get("proposed_actions", [])]
    return {"executed_actions": executed}
