"""The three specialist agents. Each is a ReAct loop over a scoped tool set."""
from ..rag import format_context, retrieve
from .common import run_tool_loop

PLANNER_PROMPT = ("You are the Mission Planner agent for autonomous drone-in-a-box operations. Before proposing a mission "
                  "you MUST gather evidence with get_fleet_telemetry, get_weather and check_geofence, then call schedule_mission "
                  "(or abort_mission when asked to abort). High-risk actions are queued for a human operator; never claim they executed. "
                  "Finish with a concise go/no-go rationale citing the numbers.")

ANALYST_PROMPT = ("You are the Anomaly Analyst agent. Use analyze_aerial_image on the referenced frame and get_fleet_telemetry for the "
                  "drone that captured it. Report hotspot count, severity and a recommended operator action. If severity is high or "
                  "critical, call raise_alert (it will be queued for human approval).")

KNOWLEDGE_PROMPT = ("You are the Knowledge agent. Answer ONLY from the retrieved context below; cite sources as [n]. "
                    "If the context does not contain the answer, say so and suggest which manual to consult.\n\nContext:\n{context}")


def planner_node(state):
    return run_tool_loop(state, PLANNER_PROMPT, ["get_fleet_telemetry", "get_weather", "check_geofence", "schedule_mission", "abort_mission"])


def analyst_node(state):
    return run_tool_loop(state, ANALYST_PROMPT, ["analyze_aerial_image", "get_fleet_telemetry", "raise_alert"])


def knowledge_node(state):
    hits = retrieve(state["user_input"], k=4)
    out = run_tool_loop(state, KNOWLEDGE_PROMPT.format(context=format_context(hits)), [], max_steps=1)
    if out["fallback_used"] or out["answer"] == "No tool evidence gathered.":
        best = hits[0] if hits else None
        out["answer"] = (f"From {best['source']} [1]: {best['text'][:500]}" if best else "No relevant documentation found.")
    out["citations"] = [{"n": i + 1, "id": h["id"], "source": h["source"], "score": h["score"]} for i, h in enumerate(hits)]
    return out
