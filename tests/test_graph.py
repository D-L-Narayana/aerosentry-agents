import aerosentry.llm as llm_mod
from aerosentry.agents import Orchestrator
from aerosentry.llm import rule_route


def test_router_rules():
    assert rule_route("Schedule a thermal inspection for D-01") == "analyst" or rule_route("Schedule a patrol for D-01") == "planner"
    assert rule_route("What is the wind limit in the manual?") == "knowledge"
    assert rule_route("Inspect the frame for hotspots") == "analyst"


def test_planner_requires_human_approval_then_executes():
    o = Orchestrator()
    out = o.run("Schedule a patrol mission for D-01 at solar-farm-01.")
    assert out["route"] == "planner" and out["pending_approval"] and out["risk"] == "high"
    assert {"get_fleet_telemetry", "get_weather", "check_geofence"} <= {t["name"] for t in out["tool_calls"]}
    assert out["proposed_actions"][0]["name"] == "schedule_mission"
    done = o.approve(out["thread_id"], True)
    assert not done["pending_approval"] and done["executed_actions"][0]["ok"]


def test_operator_rejection_blocks_execution():
    o = Orchestrator()
    out = o.run("Schedule a patrol mission for D-01 at solar-farm-01.")
    done = o.approve(out["thread_id"], False)
    assert done["executed_actions"][0]["ok"] is False and "rejected" in done["executed_actions"][0]["error"]


def test_policy_blocks_low_battery_even_when_approved():
    o = Orchestrator()
    out = o.run("Schedule a patrol mission for D-02 at solar-farm-01.")
    assert any("battery" in f for f in out["policy_findings"])
    done = o.approve(out["thread_id"], True)
    assert done["executed_actions"][0]["ok"] is False


def test_knowledge_route_cites_sources_without_approval():
    out = Orchestrator().run("What is the minimum battery percentage before dispatch according to the manual?")
    assert out["route"] == "knowledge" and not out["pending_approval"] and out["citations"]
    assert out["citations"][0]["source"] == "dock_operations_manual.md"


def test_fallback_protocol_when_llm_is_down(monkeypatch):
    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("503 upstream")

    monkeypatch.setattr(llm_mod, "get_llm", lambda: Broken())
    monkeypatch.setattr("aerosentry.agents.common.get_llm", lambda: Broken())
    monkeypatch.setattr("aerosentry.agents.supervisor.get_llm", lambda: Broken())
    monkeypatch.setattr("aerosentry.config.settings.max_retries", 1)
    out = Orchestrator().run("Schedule a patrol mission for D-01 at solar-farm-01.")
    assert out["fallback_used"] and out["route"] == "planner" and out["pending_approval"]
    assert any("degraded" in f for f in out["policy_findings"])
