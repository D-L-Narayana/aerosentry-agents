from fastapi.testclient import TestClient

from aerosentry.api import app

client = TestClient(app)


def test_health_and_console():
    assert client.get("/health").json()["ok"] is True
    assert "AeroSentry" in client.get("/").text


def test_chat_then_approve_flow():
    r = client.post("/chat", json={"message": "Schedule a thermal_survey for D-01 at solar-farm-01"}).json()
    assert r["pending_approval"] is True
    done = client.post("/approve", json={"thread_id": r["thread_id"], "approved": True}).json()
    assert done["executed_actions"][0]["ok"] is True
    traces = client.get("/traces", params={"run_id": r["run_id"]}).json()
    assert any(t["event"] == "safety.verdict" for t in traces)
