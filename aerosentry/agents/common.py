"""Shared ReAct tool loop used by every specialist agent, with the fallback protocol baked in:
retry/backoff (llm.resilient_chat) -> rule-based tool plan -> template answer -> escalate flag."""
import json

from ..config import settings
from ..llm import LLMError, get_llm, resilient_chat, rule_tool_calls
from ..observability import trace_event
from ..tools import execute_tool, openai_tool_specs
from .state import AgentState


def run_tool_loop(state: AgentState, system_prompt: str, tool_names: list[str], max_steps: int = 4) -> dict:
    llm, specs = get_llm(), openai_tool_specs(tool_names)
    messages = [{"role": "system", "content": system_prompt}]
    if state.get("memory"):
        messages.append({"role": "system", "content": "Known context:\n- " + "\n- ".join(state["memory"][:6])})
    messages.append({"role": "user", "content": state["user_input"]})
    tool_results, proposed, fallback = [], [], False

    for _ in range(max_steps):
        try:
            out = resilient_chat(llm, messages, tools=specs)
        except LLMError:
            fallback = True
            calls = rule_tool_calls(state["user_input"], tool_names) if not tool_results else []
            out = {"content": "", "tool_calls": calls}
            trace_event("fallback.rule_plan", {"tools": [c["name"] for c in calls]})
        if not out["tool_calls"]:
            answer = out["content"]
            break
        messages.append({"role": "assistant", "content": out["content"] or None,
                         "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in out["tool_calls"]]})
        for c in out["tool_calls"]:
            if c["name"] in settings.high_risk_tools:
                proposed.append({"name": c["name"], "arguments": c["arguments"]})
                payload = {"status": "queued_for_human_approval", "tool": c["name"]}
                trace_event("hitl.queued", {"tool": c["name"]})
            else:
                res = execute_tool(c["name"], c["arguments"])
                tool_results.append({"name": c["name"], "arguments": c["arguments"], **res})
                payload = res
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": json.dumps(payload, default=str)[:4000]})
    else:
        answer, fallback = "", True

    if not answer or answer == "[mock]" or fallback:
        answer = template_answer(state, tool_results, proposed)
    return {"messages": messages, "tool_results": tool_results, "proposed_actions": proposed, "answer": answer, "fallback_used": fallback}


def template_answer(state: AgentState, tool_results: list[dict], proposed: list[dict]) -> str:
    """Deterministic, evidence-grounded summary used in mock mode and when the LLM is unavailable."""
    lines = []
    for r in tool_results:
        res = r.get("result", {})
        if r["name"] == "get_fleet_telemetry" and r.get("ok"):
            lines.append(f"{res['drone_id']}: battery {res['battery_pct']}%, GPS {res['gps_fix']}, wind {res['wind_ms']} m/s, dock {res['dock_state']}, link {res['link_quality']}%.")
        elif r["name"] == "get_weather" and r.get("ok"):
            lines.append(f"Weather: wind {res['wind_ms']} m/s, precip {res['precip_mm']} mm/h, visibility {res['visibility_km']} km.")
        elif r["name"] == "check_geofence" and r.get("ok"):
            lines.append(f"Geofence: {'inside' if res['inside'] else 'OUTSIDE'} ({res['distance_km']} km from centre, ceiling {res['ceiling_m']} m).")
        elif r["name"] == "analyze_aerial_image" and r.get("ok"):
            lines.append(f"Imagery: {res['hotspot_count']} hotspot(s) via {res['detector']}, severity {res['severity']}." + (f" VLM: {res['vlm_summary'][:200]}" if res.get("vlm_summary") else ""))
        elif not r.get("ok"):
            lines.append(f"{r['name']} failed: {res.get('error', r.get('error'))}.")
    for p in proposed:
        lines.append(f"Proposed action pending approval: {p['name']}({', '.join(f'{k}={v}' for k, v in p['arguments'].items())}).")
    return " ".join(lines) or "No tool evidence gathered."
