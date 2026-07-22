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


def get_effective_model_name(client: Any, requested_model: str) -> str:
    """
    Returns effective model name compatible with client base_url.
    - OpenAI API (api.openai.com): 'gpt-4o-mini'
    - Gemini API: 'gemini-1.5-flash' if gemini-3.5-flash requested
    """
    if not client:
        return requested_model
    base_url = str(getattr(client, "base_url", ""))
    if "openai.com" in base_url and "gemini" in requested_model.lower():
        return "gpt-4o-mini"
    if "generativelanguage.googleapis.com" in base_url and "3.5" in requested_model:
        return "gemini-1.5-flash"
    return requested_model
def validate_model_config(client: Any, model_name: str) -> bool:
    """
    Validates model availability on startup.
    Returns True if valid, False if model returned 404 or failed.
    """
    if is_model_unavailable(model_name) or not client:
        return False

    effective_model = get_effective_model_name(client, model_name)
    try:
        # Dry-run validation call with min tokens
        client.chat.completions.create(
            model=effective_model,
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


_VALIDATED_MODELS: set[str] = set()


def get_llm_client(settings: Settings | None = None, timeout: float = 5.0) -> Any | None:
    """
    Returns an `openai.OpenAI` client configured for the selected provider (`gemini`, `grok`, or `openai`).
    If primary provider (e.g. Gemini 429 quota) is unavailable, automatically falls back to OpenAI if OPENAI_API_KEY is configured.
    """
    settings = settings or get_settings()
    model_name = settings.LLM_MODEL_NAME

    provider = (settings.LLM_PROVIDER or "gemini").lower()
    api_key = None
    base_url = settings.LLM_BASE_URL

    if is_model_unavailable(model_name) and is_model_unavailable("gpt-4o-mini"):
        return None

    if provider in ("gemini", "google"):
        api_key = settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif provider in ("grok", "xai"):
        api_key = settings.XAI_API_KEY or settings.OPENAI_API_KEY
        if not base_url:
            base_url = "https://api.x.ai/v1"
    else:
        api_key = settings.OPENAI_API_KEY

    if not api_key:
        logger.debug("No API key configured for provider '%s'; running in offline / fallback mode.", provider)
        return None

    try:
        from openai import OpenAI
        logger.info("Initializing universal LLM client for provider '%s' (model: '%s')", provider, model_name)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

        # Invoke validate_model_config once on startup and cache status
        if model_name not in _VALIDATED_MODELS:
            _VALIDATED_MODELS.add(model_name)
            is_valid = validate_model_config(client, model_name)
            if not is_valid and settings.OPENAI_API_KEY and provider != "openai":
                if is_model_unavailable("gpt-4o-mini"):
                    return None
                logger.info("Primary provider validation failed. Re-trying with OpenAI fallback.")
                fb_client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=timeout, max_retries=0)
                if validate_model_config(fb_client, "gpt-4o-mini"):
                    return fb_client
                return None

        if is_model_unavailable(model_name):
            if settings.OPENAI_API_KEY and not is_model_unavailable("gpt-4o-mini"):
                fb_client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=timeout, max_retries=0)
                if validate_model_config(fb_client, "gpt-4o-mini"):
                    return fb_client
            return None

        return client
    except Exception as exc:
        logger.warning("Failed to initialize client for provider '%s': %s", provider.upper(), exc)
        return None
