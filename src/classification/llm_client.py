"""Provider-independent LLM chat client (OpenAI and Groq)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from config import (
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

def _read_secret(name: str) -> str:
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not value or value.startswith("your_"):
        raise ValueError(
            f"Missing {name} in .env. "
            "Paste the key into .env only — do not paste it into chat."
        )
    return value

def build_chat_client(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[OpenAI, str, str]:
    """
    Return (client, provider_name, model_id).
    Groq uses the OpenAI-compatible endpoint; the OpenAI SDK is reused.
    """
    _load_env()
    name = (provider or LLM_PROVIDER).strip().lower()

    if name == "groq":
        key = _read_secret("GROQ_API_KEY")
        client = OpenAI(
            api_key=key,
            base_url=GROQ_BASE_URL,
            default_headers={"User-Agent": "FlavorScout/0.1"},
            max_retries=0,
        )
        return client, "groq", model or GROQ_MODEL

    if name == "openai":
        key = _read_secret("OPENAI_API_KEY")
        client = OpenAI(api_key=key, max_retries=0)
        return client, "openai", model or OPENAI_MODEL

    raise ValueError(f"Unsupported LLM_PROVIDER={name!r}. Use 'groq' or 'openai'.")

def build_async_chat_client(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[AsyncOpenAI, str, str]:
    """Async OpenAI client. Groq async is not used for the parallel benchmark."""
    _load_env()
    name = (provider or LLM_PROVIDER).strip().lower()
    if name != "openai":
        raise ValueError("Async classification benchmark supports OpenAI only.")
    key = _read_secret("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=key, max_retries=0)
    return client, "openai", model or OPENAI_MODEL
