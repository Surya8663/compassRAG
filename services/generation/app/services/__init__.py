"""
Generation Service package export index (`services/generation/app/services`).
Exports core generation orchestrator, circuit breaker resilience state machine,
and flagship / fallback synthesis components.
"""

from .circuit_breaker import CircuitBreakerOpenError, CircuitState, GenerationCircuitBreaker
from .fallback import FALLBACK_DISCLAIMER, FallbackSynthesizerService
from .service import GenerationService
from .synthesis import FlagshipSynthesizerService

_generation_service_instance: GenerationService | None = None


def get_generation_service() -> GenerationService:
    """Returns the singleton instance of GenerationService."""
    global _generation_service_instance
    if _generation_service_instance is None:
        _generation_service_instance = GenerationService()
    return _generation_service_instance


__all__ = [
    "FALLBACK_DISCLAIMER",
    "CircuitBreakerOpenError",
    "CircuitState",
    "FallbackSynthesizerService",
    "FlagshipSynthesizerService",
    "GenerationCircuitBreaker",
    "GenerationService",
    "get_generation_service",
]
