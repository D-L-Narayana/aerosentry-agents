"""Supervisor: routes each request to a specialist agent and loads persistent memory into state."""
import json

from ..llm import ROUTES, LLMError, get_llm, resilient_chat, rule_route
from ..llm import _site  # noqa: PLC2701 - shared site parser
from ..memory import recall
from ..observability import trace_event

ROUTER_PROMPT = ("ROUTER. You supervise three specialist agents for autonomous drone-dock operations. "
                 "Return JSON {\"route\": one of planner|analyst|knowledge}. "
                 "planner = scheduling/aborting flights and go/no-go checks; "
                 "analyst = analysing aerial/thermal imagery or anomalies; "
                 "knowledge = questions about manuals, SOPs, policies, limits or past incidents.")


def supervisor_node(state):
    text = state["user_input"]
    try:
        out = resilient_chat(get_llm(), [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": text}], json_mode=True)
        route = json.loads(out["content"]).get("route", "")
        fallback = False
        if route not in ROUTES:
            raise ValueError(f"invalid route {route!r}")
    except (LLMError, ValueError, json.JSONDecodeError) as e:
        route, fallback = rule_route(text), True
        trace_event("fallback.rule_route", {"reason": str(e)[:120], "route": route})
    trace_event("router.decision", {"route": route})
    return {"route": route, "memory": recall(_site(text)), "fallback_used": state.get("fallback_used", False) or fallback,
            "tool_results": [], "proposed_actions": [], "policy_findings": [], "executed_actions": []}
