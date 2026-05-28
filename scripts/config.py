import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

DEEPSEEK_MODEL_FLASH = os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash")
DEEPSEEK_MODEL_PRO = os.getenv("DEEPSEEK_MODEL_PRO", "deepseek-v4-pro")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "flash").strip().lower()

DEEPSEEK_THINKING_ENABLED = os.getenv("DEEPSEEK_THINKING_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DEEPSEEK_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "high").strip().lower()

JINA_API_KEY = os.getenv("JINA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic").strip().lower()


def get_deepseek_model(variant: Optional[str] = None) -> str:
    """Resolve model id from env. variant: 'flash', 'pro', or None for default."""
    if variant is None:
        variant = DEEPSEEK_MODEL

    normalized = variant.strip().lower()
    if normalized in ("flash", "deepseek-v4-flash"):
        return DEEPSEEK_MODEL_FLASH
    if normalized in ("pro", "deepseek-v4-pro"):
        return DEEPSEEK_MODEL_PRO
    if normalized in (DEEPSEEK_MODEL_FLASH.lower(), DEEPSEEK_MODEL_PRO.lower()):
        return variant
    return DEEPSEEK_MODEL_FLASH if DEEPSEEK_MODEL == "flash" else DEEPSEEK_MODEL_PRO


def require_deepseek_api_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return DEEPSEEK_API_KEY


def require_tavily_api_key() -> str:
    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return TAVILY_API_KEY
