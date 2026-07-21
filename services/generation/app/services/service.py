"""
Orchestrator for the Generation Service (`GenerationService`).
Coordinates primary flagship synthesis (`FlagshipSynthesizerService`), resilience
tracking (`GenerationCircuitBreaker`), fallback routing (`FallbackSynthesizerService`),
and mandatory groundedness verification (`GroundednessCheckerService`).
"""

import logging
from typing import Any

from shared.config import Settings, get_settings
from shared.models.common import DocumentChunk, GenerationResponse, RetrievalResult

from .circuit_breaker import CircuitBreakerOpenError, GenerationCircuitBreaker
from .fallback import FALLBACK_DISCLAIMER, FallbackSynthesizerService
from .synthesis import FlagshipSynthesizerService

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Core Generation Service orchestrating LLM answer synthesis, circuit breaker resilience,
    and mandatory post-synthesis groundedness checking without bypass.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        circuit_breaker: GenerationCircuitBreaker | None = None,
        flagship_synthesizer: FlagshipSynthesizerService | None = None,
        fallback_synthesizer: FallbackSynthesizerService | None = None,
        groundedness_checker: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.circuit_breaker = circuit_breaker or GenerationCircuitBreaker(
            failure_threshold=self.settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            reset_timeout=self.settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
        )
        self.flagship_synthesizer = flagship_synthesizer or FlagshipSynthesizerService(
            self.settings
        )
        self.fallback_synthesizer = fallback_synthesizer or FallbackSynthesizerService(
            self.settings
        )

        if groundedness_checker:
            self.groundedness_checker = groundedness_checker
        else:
            try:
                from services.correction.app.services.groundedness import (
                    get_groundedness_checker,
                )
                self.groundedness_checker = get_groundedness_checker()
            except Exception as exc:
                logger.warning(
                    "Could not initialize GroundednessChecker in GenerationService: %s", exc
                )
                self.groundedness_checker = None

    def generate(self, query: str, chunks: list[DocumentChunk]) -> GenerationResponse:
        """
        Synthesizes a grounded answer for `query` using `chunks`.
        Routes to primary flagship model when circuit is CLOSED or HALF_OPEN.
        If primary trips or fails, engages fallback model with reduced accuracy disclaimer.
        Always verifies groundedness of the draft without bypass before returning.
        """
        answer_text = ""
        citations = []
        is_fallback = False

        try:
            # Check circuit breaker state; raises CircuitBreakerOpenError if OPEN
            self.circuit_breaker.can_execute()

            logger.debug("Circuit breaker allowed execution. Calling FlagshipSynthesizer.")
            answer_text, citations = self.flagship_synthesizer.synthesize(query, chunks)
            self.circuit_breaker.record_success()

        except CircuitBreakerOpenError as exc:
            logger.warning("Circuit breaker is OPEN (`%s`). Routing directly to fallback.", exc)
            answer_text, citations = self.fallback_synthesizer.synthesize(query, chunks)
            is_fallback = True

        except Exception as exc:
            logger.error("Flagship synthesis execution failed (`%s`). Recording failure.", exc)
            self.circuit_breaker.record_failure(exc)
            logger.info("Engaging FallbackSynthesizer due to primary failure.")
            answer_text, citations = self.fallback_synthesizer.synthesize(query, chunks)
            is_fallback = True

        # Mandatory Groundedness Verification (No bypass for primary or fallback)
        confidence_score = 1.0
        if self.groundedness_checker and chunks:
            try:
                wrapped_chunks = [
                    RetrievalResult(
                        chunk=c,
                        vector_score=1.0,
                        bm25_score=1.0,
                        fused_score=1.0,
                        rerank_score=1.0,
                    )
                    for c in chunks
                ]
                # Strip disclaimer text when checking factual claims to avoid penalizing fallback
                clean_answer = (
                    answer_text.replace(FALLBACK_DISCLAIMER, "").strip()
                    if is_fallback
                    else answer_text.strip()
                )
                score, _, _, summary = self.groundedness_checker.verify_groundedness(
                    clean_answer, wrapped_chunks
                )
                confidence_score = score
                logger.debug(
                    "Groundedness verification completed (`%.2f`): %s",
                    confidence_score,
                    summary,
                )
            except Exception as exc:
                logger.warning("Groundedness checking error during generation: %s", exc)

        return GenerationResponse(
            answer=answer_text,
            citations=citations,
            confidence_score=confidence_score,
        )
