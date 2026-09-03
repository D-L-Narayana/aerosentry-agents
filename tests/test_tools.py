from aerosentry.tools import as_langchain_tools, check_geofence, execute_tool, openai_tool_specs, schedule_mission


def test_specs_cover_all_tools():
    names = {s["function"]["name"] for s in openai_tool_specs()}
    assert {"get_fleet_telemetry", "schedule_mission", "analyze_aerial_image", "raise_alert"} <= names


def test_argument_validation_rejects_bad_ids():
    bad = execute_tool("get_fleet_telemetry", {"drone_id": "drone one"})
    assert bad["valid"] is False and bad["ok"] is False
    good = execute_tool("get_fleet_telemetry", {"drone_id": "D-01"})
    assert good["ok"] and good["result"]["battery_pct"] == 92


def test_geofence_math():
    inside = check_geofence(18.5204, 73.8567, "solar-farm-01")
    outside = check_geofence(18.60, 73.95, "solar-farm-01")
    assert inside["inside"] and inside["distance_km"] == 0.0
    assert not outside["inside"] and outside["distance_km"] > 2.5


def test_schedule_mission_preconditions():
    low_batt = schedule_mission("D-02", "solar-farm-01", "patrol", 8, 60)
    windy = schedule_mission("D-05", "refinery-02", "patrol", 8, 60)
    ok = schedule_mission("D-01", "solar-farm-01", "thermal_survey", 8, 60)
    assert not low_batt["scheduled"] and any("battery" in b for b in low_batt["blockers"])
    assert not windy["scheduled"] and any("wind" in b for b in windy["blockers"])
    assert ok["scheduled"] and ok["mission_id"].startswith("M-")


def test_langchain_tool_adapter():
    tools = {t.name: t for t in as_langchain_tools(["check_geofence"])}
    assert tools["check_geofence"].invoke({"lat": 18.5204, "lon": 73.8567, "site_id": "solar-farm-01"})["inside"]
