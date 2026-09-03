"""Provider-agnostic chat client (OpenAI-compatible: OpenAI, Groq, Ollama, Pollinations) plus a
deterministic MockLLM for tests, CI and offline evals. resilient_chat() implements the retry /
backoff layer of the fallback protocol; callers handle LLMError with a rule-based degrade path."""
import json
import random
import re
import time

from .config import settings
from .observability import trace_event


class LLMError(RuntimeError):
    pass


ROUTES = ("planner", "analyst", "knowledge")
_ROUTE_RULES = [
    ("knowledge", r"^(what|why|how|explain|according|which|when|where|does|is|are|can i)\b|\b(manual|procedure|policy|sop|checklist|according to|incident log)\b"),
    ("planner", r"\b(schedule|plan|mission|fly|flight|patrol|launch|abort|dispatch|sortie|takeoff|rtl)\b"),
    ("analyst", r"\b(image|thermal|frame|inspect(ion)?|anomal\w*|hotspot\w*|detect\w*|photo|imagery|camera|analy[sz]e)\b"),
]


def rule_route(text: str) -> str:
    """Question-style / documentation cues win, then action verbs, then imagery cues."""
    t = text.lower().strip()
    for route, pat in _ROUTE_RULES:
        if re.search(pat, t):
            return route
    return "knowledge"


def _drone(text: str) -> str:
    m = re.search(r"\b(D-\d{2})\b", text.upper())
    return m.group(1) if m else "D-01"


def _site(text: str) -> str:
    m = re.search(r"\b(solar-farm-01|refinery-02|rail-yard-03)\b", text.lower())
    return m.group(1) if m else "solar-farm-01"


def rule_tool_calls(text: str, names: list[str]) -> list[dict]:
    """Deterministic tool plan used by MockLLM and by the degrade path when the LLM is down."""
    t, calls = text.lower(), []
    drone, site = _drone(text), _site(text)

    def add(name, **args):
        if name in names:
            calls.append({"id": f"call_{len(calls)}", "name": name, "arguments": args})

    if "analyze_aerial_image" in names:
        add("get_fleet_telemetry", drone_id=drone)
        m = re.search(r"(\S+\.(png|jpg|jpeg))", t)
        add("analyze_aerial_image", image_path=m.group(1) if m else "data/imagery/solar_thermal_anomaly.png",
            question="Identify thermal anomalies or safety hazards.")
    elif "schedule_mission" in names:
        if re.search(r"\babort|rtl|return to (home|dock)\b", t):
            add("abort_mission", drone_id=drone, reason="operator request")
        else:
            add("get_fleet_telemetry", drone_id=drone)
            add("get_weather", site_id=site)
            from .tools import SITES  # local import avoids a cycle at module load

            lat, lon = SITES.get(site, SITES["solar-farm-01"])["center"]
            add("check_geofence", lat=lat, lon=lon, site_id=site)
            add("schedule_mission", drone_id=drone, site_id=site, mission_type="inspection", waypoints=8, altitude_m=60)
    return calls


class MockLLM:
    """Scripted stand-in: routes by rules, plans tools by rules, summarises from tool results."""

    def chat(self, messages, tools=None, json_mode=False, model=None):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "ROUTER" in system:
            return {"content": json.dumps({"route": rule_route(user)}), "tool_calls": []}
        already = any(m["role"] == "tool" for m in messages)
        if tools and not already:
            calls = rule_tool_calls(user, [t["function"]["name"] for t in tools])
            if calls:
                return {"content": "", "tool_calls": calls}
        return {"content": "[mock]", "tool_calls": []}


class OpenAICompatLLM:
    def __init__(self):
        import httpx

        self.client = httpx.Client(timeout=90)
        self.url = settings.resolved_base_url + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {settings.api_key or 'none'}", "Content-Type": "application/json"}

    def chat(self, messages, tools=None, json_mode=False, model=None):
        body = {"model": model or settings.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(self.url, json=body, headers=self.headers)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c["function"].get("arguments")}
            calls.append({"id": c.get("id", f"call_{i}"), "name": c["function"]["name"], "arguments": args})
        return {"content": msg.get("content") or "", "tool_calls": calls}


def get_llm():
    return MockLLM() if not settings.live else OpenAICompatLLM()


def resilient_chat(llm, messages, **kw) -> dict:
    last = None
    for attempt in range(settings.max_retries):
        t0 = time.time()
        try:
            out = llm.chat(messages, **kw)
            trace_event("llm.call", {"attempt": attempt, "latency_ms": int((time.time() - t0) * 1000),
                                     "model": kw.get("model") or settings.model,
                                     "tool_calls": [c["name"] for c in out["tool_calls"]]})
            return out
        except Exception as e:  # noqa: BLE001 - any transport/model error triggers backoff
            last = e
            trace_event("llm.retry", {"attempt": attempt, "error": str(e)[:200]})
            time.sleep(min(0.2 * 2**attempt, 2.0) * (0.5 + random.random()))
    trace_event("llm.failed", {"error": str(last)[:200]})
    raise LLMError(str(last))
