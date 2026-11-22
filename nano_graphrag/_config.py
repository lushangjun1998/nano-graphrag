import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str = field(repr=False)
    deepseek_base_url: str
    deepseek_best_model: str
    deepseek_cheap_model: str
    deepseek_thinking: str
    bge_m3_model_path: Path
    embedding_device: str


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to {ENV_FILE}")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    thinking = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
    if thinking not in {"enabled", "disabled"}:
        raise ValueError(
            "DEEPSEEK_THINKING must be either 'enabled' or 'disabled'"
        )

    model_path = Path(_required_env("BGE_M3_MODEL_PATH")).expanduser().resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"BGE-M3 model directory does not exist: {model_path}")

    return Settings(
        deepseek_api_key=_required_env("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).strip().rstrip("/"),
        deepseek_best_model=os.getenv(
            "DEEPSEEK_BEST_MODEL", "deepseek-flash"
        ).strip(),
        deepseek_cheap_model=os.getenv(
            "DEEPSEEK_CHEAP_MODEL", "deepseek-flash"
        ).strip(),
        deepseek_thinking=thinking,
        bge_m3_model_path=model_path,
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
    )
