"""
Universal LLM Client Factory (`get_llm_client`).
Automatically configures an OpenAI client compatible with Google Gemini, xAI Grok, or OpenAI
based on settings.LLM_PROVIDER. Both Gemini and Grok natively support the official OpenAI Python SDK via base_url.
"""

import logging
from typing import Any

from shared.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_llm_client(settings: Settings | None = None, timeout: float = 5.0) -> Any | None:
    """
    Returns an `openai.OpenAI` client configured for the selected provider (`gemini`, `grok`, or `openai`).
    - Gemini: uses https://generativelanguage.googleapis.com/v1beta/openai/ with GEMINI_API_KEY
    - Grok: uses https://api.x.ai/v1 with XAI_API_KEY
    - OpenAI: standard api.openai.com with OPENAI_API_KEY
    """
    settings = settings or get_settings()
    provider = (settings.LLM_PROVIDER or "gemini").lower()

    api_key = None
    base_url = settings.LLM_BASE_URL

    if provider == "gemini" or provider == "google":
        api_key = settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif provider in ("grok", "xai"):
        api_key = settings.XAI_API_KEY or settings.OPENAI_API_KEY
        if not base_url:
            base_url = "https://api.x.ai/v1"
    else:
        # standard openai
        api_key = settings.OPENAI_API_KEY
        # base_url remains None unless explicitly overridden

    if not api_key:
        logger.debug("No API key configured for provider '%s'; running in offline / fallback mode.", provider)
        return None

    try:
        from openai import OpenAI
        logger.info("Initializing universal LLM client for provider '%s'", provider)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    except Exception as exc:
        logger.warning("Failed to initialize client for provider '%s': %s", provider.upper(), exc)
        return None
