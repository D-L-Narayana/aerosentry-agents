"""Python functions the agents can execute. Every tool has a Pydantic argument schema; execute_tool()
validates arguments before running anything and records the outcome for tool-call accuracy evals."""
import json
import math
import os
import time
import uuid

from pydantic import BaseModel, Field, ValidationError

from .config import settings
from .observability import trace_event

SITES = {
    "solar-farm-01": {"center": (18.5204, 73.8567), "radius_km": 2.5, "max_wind_ms": 10.0, "ceiling_m": 120},
    "refinery-02": {"center": (19.0760, 72.8777), "radius_km": 1.8, "max_wind_ms": 8.0, "ceiling_m": 90},
    "rail-yard-03": {"center": (28.6139, 77.2090), "radius_km": 3.0, "max_wind_ms": 9.0, "ceiling_m": 100},
}


class TelemetryArgs(BaseModel):
    drone_id: str = Field(..., pattern=r"^D-\d{2}$", description="Fleet drone id, e.g. D-01")


class GeofenceArgs(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    site_id: str = "solar-farm-01"


class WeatherArgs(BaseModel):
    site_id: str = "solar-farm-01"


class MissionArgs(BaseModel):
    drone_id: str = Field(..., pattern=r"^D-\d{2}$")
    site_id: str = "solar-farm-01"
    mission_type: str = Field("inspection", pattern=r"^(inspection|patrol|thermal_survey|incident_response)$")
    waypoints: int = Field(8, ge=1, le=60)
    altitude_m: float = Field(60, ge=20, le=120)


class AlertArgs(BaseModel):
    severity: str = Field(..., pattern=r"^(low|medium|high|critical)$")
    message: str = Field(..., min_length=5, max_length=500)
    site_id: str = "solar-farm-01"


class AbortArgs(BaseModel):
    drone_id: str = Field(..., pattern=r"^D-\d{2}$")
    reason: str = Field(..., min_length=3)


class ImageArgs(BaseModel):
    image_path: str
    question: str = "Identify thermal anomalies or safety hazards."


def _state_path(name: str) -> str:
    p = os.path.join(settings.data_dir, "state")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, name)


def _telemetry_table() -> dict:
    table = {}
    path = os.path.join(settings.data_dir, "logs", "telemetry.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            table[rec["drone_id"]] = rec  # last record wins
    return table


def get_fleet_telemetry(drone_id: str) -> dict:
    rec = _telemetry_table().get(drone_id)
    return rec or {"error": f"unknown drone {drone_id}", "known": sorted(_telemetry_table())}


def get_weather(site_id: str) -> dict:
    with open(os.path.join(settings.data_dir, "logs", "weather.json"), encoding="utf-8") as f:
        wx = json.load(f)
    return wx.get(site_id, {"error": f"no forecast for {site_id}"})


def check_geofence(lat: float, lon: float, site_id: str) -> dict:
    site = SITES.get(site_id)
    if not site:
        return {"error": f"unknown site {site_id}"}
    (lat0, lon0), r = site["center"], site["radius_km"]
    p1, p2 = math.radians(lat0), math.radians(lat)
    dphi, dl = math.radians(lat - lat0), math.radians(lon - lon0)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    d = 2 * 6371.0 * math.asin(math.sqrt(a))
    return {"inside": d <= r, "distance_km": round(d, 3), "radius_km": r, "ceiling_m": site["ceiling_m"]}


def schedule_mission(drone_id: str, site_id: str, mission_type: str, waypoints: int, altitude_m: float) -> dict:
    tele, wx, site = get_fleet_telemetry(drone_id), get_weather(site_id), SITES.get(site_id)
    blockers = []
    if "error" in tele:
        blockers.append(tele["error"])
    else:
        if tele["battery_pct"] < 40:
            blockers.append(f"battery {tele['battery_pct']}% < 40% minimum")
        if tele["dock_state"] != "ready":
            blockers.append(f"dock state is {tele['dock_state']}")
        if tele["gps_fix"] != "3D":
            blockers.append("no 3D GPS fix")
    if site is None:
        blockers.append(f"unknown site {site_id}")
    else:
        if wx.get("wind_ms", 0) > site["max_wind_ms"]:
            blockers.append(f"wind {wx['wind_ms']} m/s > {site['max_wind_ms']} m/s limit")
        if altitude_m > site["ceiling_m"]:
            blockers.append(f"altitude {altitude_m} m > site ceiling {site['ceiling_m']} m")
        if wx.get("precip_mm", 0) > 0.5:
            blockers.append("precipitation above 0.5 mm/h")
    if blockers:
        return {"scheduled": False, "blockers": blockers}
    mission = {"mission_id": "M-" + uuid.uuid4().hex[:8], "drone_id": drone_id, "site_id": site_id,
               "mission_type": mission_type, "waypoints": waypoints, "altitude_m": altitude_m, "ts": time.time()}
    with open(_state_path("missions.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(mission) + "\n")
    return {"scheduled": True, **mission}


def raise_alert(severity: str, message: str, site_id: str) -> dict:
    alert = {"alert_id": "A-" + uuid.uuid4().hex[:8], "severity": severity, "message": message, "site_id": site_id, "ts": time.time()}
    with open(_state_path("alerts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")
    return alert


def abort_mission(drone_id: str, reason: str) -> dict:
    rec = {"drone_id": drone_id, "action": "RTL", "reason": reason, "ts": time.time()}
    with open(_state_path("aborts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {"aborted": True, **rec}


def analyze_aerial_image(image_path: str, question: str) -> dict:
    from .vision.vlm import analyze

    return analyze(image_path, question)


TOOLS = {
    "get_fleet_telemetry": (get_fleet_telemetry, TelemetryArgs, "Latest telemetry for one drone: battery, GPS fix, wind, link quality, dock state."),
    "get_weather": (get_weather, WeatherArgs, "Current site weather: wind (m/s), precipitation (mm/h), visibility (km)."),
    "check_geofence": (check_geofence, GeofenceArgs, "Check whether a coordinate lies inside a site's approved geofence."),
    "schedule_mission": (schedule_mission, MissionArgs, "Schedule an autonomous dock mission. HIGH RISK: requires human approval."),
    "raise_alert": (raise_alert, AlertArgs, "Raise an operator alert. HIGH RISK: requires human approval."),
    "abort_mission": (abort_mission, AbortArgs, "Abort a flight and return the drone to its dock. HIGH RISK: requires human approval."),
    "analyze_aerial_image": (analyze_aerial_image, ImageArgs, "Run the vision-language / thermal analyzer on an aerial image and return findings."),
}


def openai_tool_specs(names=None) -> list[dict]:
    specs = []
    for n, (fn, model, desc) in TOOLS.items():
        if names is None or n in names:
            specs.append({"type": "function", "function": {"name": n, "description": desc, "parameters": model.model_json_schema()}})
    return specs


def as_langchain_tools(names=None):
    from langchain_core.tools import StructuredTool

    return [StructuredTool.from_function(func=fn, name=n, description=desc, args_schema=model)
            for n, (fn, model, desc) in TOOLS.items() if names is None or n in names]


def execute_tool(name: str, arguments: dict) -> dict:
    if name not in TOOLS:
        trace_event("tool.unknown", {"tool": name})
        return {"ok": False, "valid": False, "error": f"unknown tool {name}"}
    fn, model, _ = TOOLS[name]
    t0 = time.time()
    try:
        args = model(**arguments)
    except ValidationError as e:
        trace_event("tool.invalid_args", {"tool": name, "errors": e.errors()[:3]})
        return {"ok": False, "valid": False, "error": "argument validation failed", "details": e.errors()[:3]}
    try:
        result = fn(**args.model_dump())
        ok = "error" not in result
    except Exception as e:  # noqa: BLE001
        result, ok = {"error": str(e)}, False
    trace_event("tool.call", {"tool": name, "ok": ok, "latency_ms": int((time.time() - t0) * 1000)})
    return {"ok": ok, "valid": True, "result": result}
