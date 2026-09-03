"""Runtime settings. Everything is env-driven so the same code runs in mock (CI), local (Ollama) or cloud (OpenAI/Groq) mode."""
import os
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PRESETS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "pollinations": "https://text.pollinations.ai/openai",
}


@dataclass
class Settings:
    provider: str = os.getenv("AEROSENTRY_PROVIDER", "mock")  # mock | openai | groq | ollama | pollinations
    model: str = os.getenv("AEROSENTRY_MODEL", "gpt-4o-mini")
    vision_model: str = os.getenv("AEROSENTRY_VISION_MODEL", os.getenv("AEROSENTRY_MODEL", "gpt-4o-mini"))
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "")
    embed_provider: str = os.getenv("AEROSENTRY_EMBED", "hash")  # hash | openai
    data_dir: str = os.getenv("AEROSENTRY_DATA", os.path.join(ROOT, "data"))
    trace_path: str = os.getenv("AEROSENTRY_TRACES", os.path.join(ROOT, "traces", "traces.jsonl"))
    max_retries: int = int(os.getenv("AEROSENTRY_MAX_RETRIES", "3"))
    high_risk_tools: tuple = ("schedule_mission", "abort_mission", "raise_alert")

    @property
    def resolved_base_url(self) -> str:
        return (self.base_url or PRESETS.get(self.provider, "")).rstrip("/")

    @property
    def live(self) -> bool:
        return self.provider != "mock"


settings = Settings()
