"""
Comprehensive verification suite for Phase 7: Generation Service (`services/generation`).
Verifies:
1. Exact `GenerationCircuitBreaker` state transitions (`CLOSED` -> `OPEN` -> `HALF_OPEN`).
2. Primary flagship model synthesis (`FlagshipSynthesizerService`) with JSON claim citations.
3. Simulation of 5 consecutive flagship API failures tripping the circuit breaker (`OPEN`),
   engaging `FallbackSynthesizerService` with explicit `reduced accuracy` disclaimer, verifying
   mandatory groundedness checking without bypass, and verifying time-based recovery (`CLOSED`).
4. Integration check for `/generate` API endpoint (`routes.py`).
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from app.api.routes import router as generation_router
from app.services.circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitState,
    GenerationCircuitBreaker,
)
from app.services.fallback import FALLBACK_DISCLAIMER, FallbackSynthesizerService
from app.services.service import GenerationService
from app.services.synthesis import FlagshipSynthesizerService
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.config import get_settings
from shared.models.common import (
    CorrectionVerdict,
    DocumentChunk,
    DocumentMetadata,
    GenerationRequest,
    GenerationResponse,
    SignalType,
)


def _make_test_chunk(chunk_id: str, content: str, source: str = "test_doc.pdf") -> DocumentChunk:
    """Helper creating a valid DocumentChunk fixture."""
    return DocumentChunk(
        id=chunk_id,
        document_id="doc_123",
        content=content,
        metadata=DocumentMetadata(
            source=source,
            page_number=1,
            ingestion_timestamp="2026-07-21T12:00:00Z",
            tenant_id="test_tenant",
            version_id="v1",
        ),
    )


def test_circuit_breaker_state_machine_exact_transitions() -> None:
    """
    Verifies `GenerationCircuitBreaker` transitions cleanly:
    - Starts in CLOSED state.
    - Increments failure counter up to `failure_threshold` (5).
    - Trips to OPEN right when failure_count reaches 5.
    - Raises `CircuitBreakerOpenError` while OPEN inside `reset_timeout`.
    - Transitions to HALF_OPEN after `reset_timeout` elapses.
    - Resets to CLOSED and resets failure_count to 0 upon `record_success()` in HALF_OPEN.
    """
    cb = GenerationCircuitBreaker(failure_threshold=5, reset_timeout=0.3)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # Record 4 consecutive failures -> still CLOSED
    for _ in range(4):
        cb.record_failure(Exception("simulated failure"))
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 4
    assert cb.can_execute() is True

    # Record 5th failure -> trips OPEN
    cb.record_failure(Exception("5th simulated failure"))
    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 5

    # Attempts to execute immediately raise CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        cb.can_execute()
    assert "Circuit breaker is OPEN" in str(exc_info.value)

    # Simulate time passing beyond reset_timeout (0.3s)
    time.sleep(0.35)

    # Next check transitions OPEN -> HALF_OPEN
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Recording success during HALF_OPEN resets state -> CLOSED and failures -> 0
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_flagship_structured_synthesis_with_claim_citations() -> None:
    """
    Verifies `FlagshipSynthesizerService.synthesize()` properly maps structured JSON output
    to exact chunk IDs and supporting snippets.
    """
    settings = get_settings()
    flagship = FlagshipSynthesizerService(settings)

    chunks = [
        _make_test_chunk(
            "c_1", "Compass RAG combines Qdrant vector search and Elasticsearch BM25."
        ),
        _make_test_chunk(
            "c_2", "Reciprocal Rank Fusion (RRF) uses k=60 to merge rankings exactly."
        ),
    ]

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "answer": "Compass RAG combines Qdrant and Elasticsearch using exact RRF.",
                        "claims": [
                            {
                                "claim_text": "Compass RAG combines Qdrant and Elasticsearch.",
                                "chunk_id": "c_1",
                                "quote_snippet": "combines Qdrant vector search and Elasticsearch",
                            },
                            {
                                "claim_text": "Reciprocal Rank Fusion uses k=60 to merge rankings.",
                                "chunk_id": "c_2",
                                "quote_snippet": "Reciprocal Rank Fusion (RRF) uses k=60",
                            },
                        ],
                    }
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    flagship._openai_client = mock_client
    answer, citations = flagship.synthesize("How does retrieval work?", chunks)

    assert "Qdrant and Elasticsearch" in answer
    assert len(citations) == 2
    assert citations[0].chunk_id == "c_1"
    assert "combines Qdrant vector search" in citations[0].quote_snippet
    assert citations[1].chunk_id == "c_2"
    assert "k=60" in citations[1].quote_snippet


def test_generation_circuit_breaker_trip_and_fallback_engagement() -> None:
    """
    Simulates the flagship model failing 5 consecutive times (mocking ONLY the API call),
    asserting:
    - Each failure increments circuit failure_count and engages fallback with disclaimer.
    - After the 5th failure, the circuit breaker trips from CLOSED to OPEN.
    - On the 6th call while OPEN, `CircuitBreakerOpenError` routes immediately to fallback without
      attempting to call the primary flagship model API at all.
    - Both primary and fallback routes pass through GroundednessCheckerService without bypass.
    - After `reset_timeout` elapses, `can_execute()` transitions state to HALF_OPEN.
    - On the 7th call (in HALF_OPEN), primary succeeds, resetting circuit to CLOSED (`failures=0`).
    """
    settings = get_settings()
    cb = GenerationCircuitBreaker(failure_threshold=5, reset_timeout=0.3)
    flagship = FlagshipSynthesizerService(settings)
    fallback = FallbackSynthesizerService(settings)

    # Mock flagship OpenAI client
    mock_flagship_client = MagicMock()
    # First 5 calls raise API error; 6th call (after reset timeout in HALF_OPEN) succeeds
    mock_flagship_client.chat.completions.create.side_effect = [
        Exception("API connection failure 1"),
        Exception("API connection failure 2"),
        Exception("API connection failure 3"),
        Exception("API connection failure 4"),
        Exception("API connection failure 5"),
    ]
    flagship._openai_client = mock_flagship_client

    # Mock fallback OpenAI client to return structured fallback answer
    mock_fallback_client = MagicMock()
    mock_fallback_response = MagicMock()
    mock_fallback_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "answer": "Fallback answer synthesized from c_1.",
                        "claims": [
                            {
                                "claim_text": "Fallback assertion.",
                                "chunk_id": "c_1",
                                "quote_snippet": "Compass RAG combines Qdrant",
                            }
                        ],
                    }
                )
            )
        )
    ]
    mock_fallback_client.chat.completions.create.return_value = mock_fallback_response
    fallback._openai_client = mock_fallback_client

    # Mock Groundedness checker ensuring no bypass
    mock_checker = MagicMock()
    mock_checker.verify_groundedness.return_value = (
        0.88,
        True,
        [
            CorrectionVerdict(
                signal_type=SignalType.GROUNDEDNESS,
                verdict=True,
                confidence=0.88,
                reasoning="Claim verified",
            )
        ],
        "Verified",
    )

    service = GenerationService(
        settings=settings,
        circuit_breaker=cb,
        flagship_synthesizer=flagship,
        fallback_synthesizer=fallback,
        groundedness_checker=mock_checker,
    )

    chunks = [
        _make_test_chunk("c_1", "Compass RAG combines Qdrant vector search and Elasticsearch.")
    ]
    query = "How does retrieval work?"

    # Execute 5 consecutive requests where flagship raises exceptions
    for i in range(1, 6):
        response = service.generate(query, chunks)
        assert FALLBACK_DISCLAIMER in response.answer
        assert "Fallback answer synthesized" in response.answer
        assert response.confidence_score == 0.88
        assert mock_checker.verify_groundedness.call_count == i

    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 5
    assert mock_flagship_client.chat.completions.create.call_count == 5

    # 6th call while OPEN -> immediately catches error without calling flagship client
    resp_open = service.generate(query, chunks)
    assert cb.state == CircuitState.OPEN
    assert FALLBACK_DISCLAIMER in resp_open.answer
    # Flagship call count MUST remain 5 (not incremented because circuit was OPEN)
    assert mock_flagship_client.chat.completions.create.call_count == 5
    # Groundedness verification ran for 6th call (no bypass even on circuit open)
    assert mock_checker.verify_groundedness.call_count == 6

    # Simulate time elapsed > reset_timeout (0.3s)
    time.sleep(0.35)

    # Setup flagship to succeed on next call during HALF_OPEN trial
    mock_flagship_success = MagicMock()
    mock_flagship_success.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "answer": "Primary flagship answer recovered cleanly.",
                        "claims": [
                            {
                                "claim_text": "Primary assertion.",
                                "chunk_id": "c_1",
                                "quote_snippet": "Compass RAG combines Qdrant",
                            }
                        ],
                    }
                )
            )
        )
    ]
    mock_flagship_client.chat.completions.create.side_effect = None
    mock_flagship_client.chat.completions.create.return_value = mock_flagship_success

    # 7th call -> transitions OPEN -> HALF_OPEN, runs flagship, succeeds, transitions -> CLOSED
    resp_recovered = service.generate(query, chunks)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert "Primary flagship answer recovered cleanly." in resp_recovered.answer
    assert FALLBACK_DISCLAIMER not in resp_recovered.answer
    assert mock_flagship_client.chat.completions.create.call_count == 6
    assert mock_checker.verify_groundedness.call_count == 7


def test_api_endpoint_generation() -> None:
    """Verifies POST /generate endpoint executes generation workflow via TestClient."""
    app = FastAPI()
    app.include_router(generation_router)
    client = TestClient(app)

    payload = GenerationRequest(
        query="What databases does Compass RAG use?",
        chunks=[
            _make_test_chunk(
                "c_db", "Compass RAG uses Qdrant for vector embeddings and Elasticsearch."
            )
        ],
    )

    with patch("app.api.routes.get_generation_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.generate.return_value = GenerationResponse(
            answer="Compass RAG uses Qdrant and Elasticsearch.",
            citations=[],
            confidence_score=0.95,
        )
        mock_get_service.return_value = mock_service

        resp = client.post("/generate", json=payload.model_dump(mode="json"))
        assert resp.status_code == 200
        data = resp.json()
        assert "Qdrant and Elasticsearch" in data["answer"]
        assert data["confidence_score"] == 0.95
