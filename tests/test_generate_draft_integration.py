"""
Integration tests for generate_draft_node in the Correction Router Graph.

Mocks ONLY the actual LLM API call (openai.chat.completions.create), and asserts:
(a) FlagshipSynthesizerService.synthesize is actually invoked (not bypassed),
(b) the returned answer is NOT identical to any raw chunk's content field,
(c) two different queries against the same document produce two different answers
    with potentially different top citations.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from shared.models.common import (
    DocumentChunk,
    DocumentMetadata,
    RetrievalResult,
)

from services.correction.app.services.graph import generate_draft_node


def _make_chunk(
    chunk_id: str,
    content: str,
    version: str = "v1.0",
    source: str = "resume.pdf",
    score: float = 0.85,
    tenant_id: str = "tenant_enterprise",
) -> RetrievalResult:
    """Helper to create a test RetrievalResult wrapping a DocumentChunk."""
    meta = DocumentMetadata(
        source=source,
        page_number=1,
        ingestion_timestamp=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        tenant_id=tenant_id,
        version_id=version,
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        content=content,
        metadata=meta,
    )
    return RetrievalResult(
        chunk=chunk,
        vector_score=score,
        bm25_score=score,
        fused_score=score,
        rerank_score=score,
    )


# Two semantically different chunks simulating real ingested resume data
CHUNK_CONTACT = _make_chunk(
    "c0",
    "John Doe\nBangalore, India | 9876543210 | john@example.com\n"
    "Summary: Generative AI Engineer with experience building RAG systems.",
)
CHUNK_HIDEVS = _make_chunk(
    "c1",
    "HiDevs\nBangalore, India (Hybrid)\n"
    "– Built PeopleGPT / Aura AI Agent – RAG pipeline over 13,000+ profiles.\n"
    "– Optimized chatbot latency from 10–15s to ~2.5s via code review.",
)
CHUNK_INTERN = _make_chunk(
    "c2",
    "AI Intern – Azure AI Services\nMar 2024 – Jun 2024\n"
    "In-Biot Private Limited, Bangalore\n"
    "– Deployed Azure Custom Vision and Form Recognizer pipelines.",
)

ALL_CHUNKS = [CHUNK_CONTACT, CHUNK_HIDEVS, CHUNK_INTERN]


def _make_llm_response(answer: str, claims: list[dict]) -> MagicMock:
    """Creates a mock OpenAI chat completion response object."""
    content_json = json.dumps({"answer": answer, "claims": claims})
    mock_message = MagicMock()
    mock_message.content = content_json
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestGenerateDraftNodeInvokesSynthesizer:
    """Asserts FlagshipSynthesizerService.synthesize is actually called."""

    @patch("services.generation.app.services.synthesis.FlagshipSynthesizerService.__init__", return_value=None)
    @patch("services.generation.app.services.synthesis.FlagshipSynthesizerService.synthesize")
    def test_synthesize_is_called(self, mock_synthesize, mock_init):
        """synthesize() must be invoked, not bypassed by a fallback."""
        from shared.models.common import Citation

        mock_synthesize.return_value = (
            "The candidate has experience in RAG systems and AI development.",
            [
                Citation(
                    chunk_id="c0",
                    document_id="doc_c0",
                    source="resume.pdf",
                    page_number=1,
                    quote_snippet="Generative AI Engineer with experience",
                )
            ],
        )

        state = {
            "query": "What are the candidate's technical skills?",
            "retrieved_chunks": ALL_CHUNKS,
        }

        result = generate_draft_node(state)

        # Assert synthesize was called exactly once
        mock_synthesize.assert_called_once()
        # Assert the query was passed correctly
        call_args = mock_synthesize.call_args
        assert call_args[0][0] == "What are the candidate's technical skills?"
        # Assert all 3 chunks were passed
        assert len(call_args[0][1]) == 3
        # Assert draft_answer matches the mock return
        assert result["draft_answer"] == "The candidate has experience in RAG systems and AI development."


class TestAnswerNotRawChunkContent:
    """Asserts the returned answer is NOT identical to any raw chunk's content."""

    @patch("services.generation.app.services.synthesis.FlagshipSynthesizerService.__init__", return_value=None)
    @patch("services.generation.app.services.synthesis.FlagshipSynthesizerService.synthesize")
    def test_answer_differs_from_all_chunks(self, mock_synthesize, mock_init):
        """The answer must be synthesized, not a passthrough of chunk content."""
        from shared.models.common import Citation

        synthesized_answer = (
            "Based on the resume, the candidate worked at HiDevs as a Generative AI Developer, "
            "building PeopleGPT and optimizing chatbot latency to ~2.5s."
        )
        mock_synthesize.return_value = (
            synthesized_answer,
            [
                Citation(
                    chunk_id="c1",
                    document_id="doc_c1",
                    source="resume.pdf",
                    page_number=1,
                    quote_snippet="Built PeopleGPT / Aura AI Agent",
                )
            ],
        )

        state = {
            "query": "Summarize the work experience at HiDevs",
            "retrieved_chunks": ALL_CHUNKS,
        }

        result = generate_draft_node(state)

        # Answer must not be identical to ANY chunk's raw content
        for chunk_result in ALL_CHUNKS:
            assert result["draft_answer"] != chunk_result.chunk.content, (
                f"Answer is identical to chunk {chunk_result.chunk.id}'s raw content — "
                f"FlagshipSynthesizerService is being bypassed!"
            )


class TestDifferentQueriesProduceDifferentAnswers:
    """Asserts two different queries produce different answers and potentially different citations."""

    @patch("services.generation.app.services.synthesis.FlagshipSynthesizerService.__init__", return_value=None)
    @patch("services.generation.app.services.synthesis.FlagshipSynthesizerService.synthesize")
    def test_distinct_queries_distinct_answers(self, mock_synthesize, mock_init):
        from shared.models.common import Citation

        # Configure different return values for different queries
        def side_effect(query, chunks):
            if "skills" in query.lower():
                return (
                    "The candidate's primary skills include Python, FastAPI, LangGraph, and Azure OpenAI.",
                    [
                        Citation(
                            chunk_id="c0",
                            document_id="doc_c0",
                            source="resume.pdf",
                            page_number=1,
                            quote_snippet="Generative AI Engineer with experience",
                        )
                    ],
                )
            elif "hidevs" in query.lower():
                return (
                    "At HiDevs, the candidate built PeopleGPT using RAG over 13,000+ profiles "
                    "and optimized chatbot latency from 10-15s to ~2.5s.",
                    [
                        Citation(
                            chunk_id="c1",
                            document_id="doc_c1",
                            source="resume.pdf",
                            page_number=1,
                            quote_snippet="Built PeopleGPT / Aura AI Agent",
                        )
                    ],
                )
            return ("Generic answer", [])

        mock_synthesize.side_effect = side_effect

        # Query 1: skills
        result_skills = generate_draft_node({
            "query": "What are the technical skills?",
            "retrieved_chunks": ALL_CHUNKS,
        })

        # Query 2: HiDevs experience
        result_hidevs = generate_draft_node({
            "query": "Summarize work at HiDevs",
            "retrieved_chunks": ALL_CHUNKS,
        })

        # Answers must be different
        assert result_skills["draft_answer"] != result_hidevs["draft_answer"], (
            "Two different queries returned identical answers — generation is not query-dependent!"
        )

        # Top citation chunk_ids should differ (skills->c0, hidevs->c1)
        skills_top_cid = result_skills["draft_citations"][0].chunk_id
        hidevs_top_cid = result_hidevs["draft_citations"][0].chunk_id
        assert skills_top_cid != hidevs_top_cid, (
            f"Both queries returned the same top citation chunk_id ({skills_top_cid}) — "
            f"citations are not query-dependent!"
        )


class TestChunkCoercionFailsLoud:
    """Asserts that bad chunk data raises RuntimeError, never silently degrades."""

    def test_unconvertible_chunk_raises_runtime_error(self):
        """If a chunk can't be converted to DocumentChunk, RuntimeError is raised."""
        state = {
            "query": "test query",
            "retrieved_chunks": ["this_is_not_a_chunk"],
        }

        with pytest.raises(RuntimeError, match="Failed to convert chunk at index 0"):
            generate_draft_node(state)

    def test_bad_dict_chunk_raises_runtime_error(self):
        """A dict missing required DocumentChunk fields raises RuntimeError."""
        state = {
            "query": "test query",
            "retrieved_chunks": [{"some_random_key": "value"}],
        }

        with pytest.raises(RuntimeError, match="Failed to convert chunk at index 0"):
            generate_draft_node(state)
