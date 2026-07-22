"""
Universal LLM Client Factory (`get_llm_client`).
Automatically configures an OpenAI client compatible with Google Gemini, xAI Grok, or OpenAI
based on settings.LLM_PROVIDER. Both Gemini and Grok natively support the official OpenAI Python SDK via base_url.
"""

import logging
from typing import Any

from shared.config import Settings, get_settings

logger = logging.getLogger(__name__)


_UNAVAILABLE_MODELS: set[str] = set()


def is_model_unavailable(model_name: str) -> bool:
    """Returns True if model was flagged as unavailable (e.g. 404)."""
    return model_name in _UNAVAILABLE_MODELS


def mark_model_unavailable(model_name: str, reason: str = "404 Not Found"):
    """Flags a model as unavailable to fail fast on subsequent calls."""
    if model_name not in _UNAVAILABLE_MODELS:
        _UNAVAILABLE_MODELS.add(model_name)
        logger.error(
            "CONFIGURATION ERROR: LLM model '%s' is unavailable (%s). "
            "Failing fast to local fallback synthesis for all subsequent requests.",
            model_name,
            reason,
        )


def validate_model_config(client: Any, model_name: str) -> bool:
    """
    Validates model availability on startup.
    Returns True if valid, False if model returned 404 or failed.
    """
    if is_model_unavailable(model_name):
        return False
    if not client:
        return False

    try:
        # Dry-run validation call with min tokens
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True
    except Exception as exc:
        err_msg = str(exc)
        if any(token in err_msg.lower() for token in ("404", "429", "not found", "quota", "rate")):
            mark_model_unavailable(model_name, reason=err_msg)
            return False
        logger.debug("Model validation ping failed for '%s': %s", model_name, exc)
        return True


def get_llm_client(settings: Settings | None = None, timeout: float = 5.0) -> Any | None:
    """
    Returns an `openai.OpenAI` client configured for the selected provider (`gemini`, `grok`, or `openai`).
    - Gemini: uses https://generativelanguage.googleapis.com/v1beta/openai/ with GEMINI_API_KEY
    - Grok: uses https://api.x.ai/v1 with XAI_API_KEY
    - OpenAI: standard api.openai.com with OPENAI_API_KEY
    """
    settings = settings or get_settings()
    model_name = settings.LLM_MODEL_NAME

    if is_model_unavailable(model_name):
        logger.debug("Skipping client creation: model '%s' flagged as unavailable.", model_name)
        return None

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
