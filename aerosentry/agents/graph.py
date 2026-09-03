"""LangGraph wiring: supervisor -> specialist -> safety gate -> (interrupt) human approval -> execute -> finalize."""
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..memory import remember_episode
from ..observability import start_run, trace_event
from .safety import execute_actions, human_approval, safety_gate
from .specialists import analyst_node, knowledge_node, planner_node
from .state import AgentState
from .supervisor import supervisor_node


def finalize(state):
    executed = state.get("executed_actions", [])
    tail = ""
    if executed:
        done = [f"{e['name']} -> {'executed' if e.get('ok') else 'NOT executed (' + str(e.get('error') or e.get('result', {}).get('blockers')) + ')'}" for e in executed]
        tail = " Actions: " + "; ".join(done) + "."
    if state.get("policy_findings"):
        tail += " Policy findings: " + "; ".join(state["policy_findings"]) + "."
    answer = (state.get("answer") or "") + tail
    remember_episode(state.get("thread_id", ""), state.get("route", ""), answer)
    trace_event("run.end", {"route": state.get("route"), "risk": state.get("risk"), "fallback": state.get("fallback_used")})
    return {"answer": answer}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_node)
    g.add_node("analyst", analyst_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("safety_gate", safety_gate)
    g.add_node("human_approval", human_approval)
    g.add_node("execute_actions", execute_actions)
    g.add_node("finalize", finalize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"], {"planner": "planner", "analyst": "analyst", "knowledge": "knowledge"})
    for n in ("planner", "analyst", "knowledge"):
        g.add_edge(n, "safety_gate")
    g.add_conditional_edges("safety_gate", lambda s: "human_approval" if s.get("requires_approval") else "finalize",
                            {"human_approval": "human_approval", "finalize": "finalize"})
    g.add_edge("human_approval", "execute_actions")
    g.add_edge("execute_actions", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["human_approval"])


class Orchestrator:
    def __init__(self):
        self.graph = build_graph()

    def _cfg(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def run(self, text: str, thread_id: str | None = None) -> dict:
        thread_id = thread_id or uuid.uuid4().hex[:8]
        run_id = start_run("chat")
        state = self.graph.invoke({"user_input": text, "thread_id": thread_id, "run_id": run_id}, self._cfg(thread_id))
        pending = bool(self.graph.get_state(self._cfg(thread_id)).next)
        return self._view(state, thread_id, pending)

    def approve(self, thread_id: str, approved: bool) -> dict:
        cfg = self._cfg(thread_id)
        if not self.graph.get_state(cfg).next:
            return {"error": "nothing pending for this thread", "thread_id": thread_id}
        self.graph.update_state(cfg, {"approved": approved})
        state = self.graph.invoke(None, cfg)
        return self._view(state, thread_id, False)

    @staticmethod
    def _view(state, thread_id, pending):
        return {"thread_id": thread_id, "run_id": state.get("run_id"), "route": state.get("route"), "risk": state.get("risk"),
                "pending_approval": pending, "answer": state.get("answer"), "proposed_actions": state.get("proposed_actions", []),
                "policy_findings": state.get("policy_findings", []), "executed_actions": state.get("executed_actions", []),
                "tool_calls": [{"name": r["name"], "ok": r.get("ok"), "valid": r.get("valid")} for r in state.get("tool_results", [])],
                "citations": state.get("citations", []), "fallback_used": state.get("fallback_used", False)}
