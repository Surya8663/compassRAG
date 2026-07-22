"""
Query Reformulator Service (`QueryReformulatorService`).
Implements a capped 3-attempt query reformulation strategy driven by graph state:
- Attempt 1 (`attempt_count == 0`): HyDE (Hypothetical Document Embeddings)
- Attempt 2 (`attempt_count == 1`): Multi-Query Expansion (3 perspective queries)
- Attempt 3 (`attempt_count == 2`): Keyword Broadening (core high-signal keywords)
"""

import json
import logging
import re
from typing import Any

from shared.config import get_settings

logger = logging.getLogger(__name__)


class QueryReformulatorService:
    """
    Reformulates queries to recover from low retrieval confidence or ungrounded drafts.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER
        self._openai_client: Any = None

        from shared.utils.llm_client import get_llm_client
        self._openai_client = get_llm_client(self.settings, timeout=5.0)

    def _apply_hyde(self, query: str) -> str:
        """
        Attempt 1: Generates a hypothetical answer snippet to match target embedding vectors.
        """
        if self._openai_client:
            try:
                prompt = (
                    "Write a brief, highly factual hypothetical passage answering the "
                    "following query. Do not mention it is hypothetical or unsure; write as an "
                    "authoritative reference document snippet.\n\n"
                    f"Query:\n{query}\n\n"
                    "Respond with only the hypothetical passage text."
                )
                from shared.utils.llm_client import get_effective_model_name
                eff_model = get_effective_model_name(self._openai_client, self.settings.LLM_MODEL_NAME or "gpt-4o-mini")
                response = self._openai_client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                passage = (response.choices[0].message.content or "").strip()
                if passage and len(passage) > 10:
                    return passage
            except Exception as exc:
                logger.warning("OpenAI HyDE generation failed (`%s`). Using local fallback.", exc)

        # Offline / local HyDE passage generation
        clean_q = query.strip(" ?.")
        return clean_q

    def _apply_multi_query_expansion(self, query: str) -> str:
        """
        Attempt 2: Decomposes query into 3 distinct alternative search perspectives.
        """
        if self._openai_client:
            try:
                prompt = (
                    "Generate 3 distinct alternative search queries that rephrase or break down "
                    "the following question from different analytical perspectives.\n\n"
                    f"Query:\n{query}\n\n"
                    "Respond with JSON strictly formatted as:\n"
                    '{"queries": ["query 1", "query 2", "query 3"]}'
                )
                from shared.utils.llm_client import get_effective_model_name
                eff_model2 = get_effective_model_name(self._openai_client, self.settings.LLM_MODEL_NAME or "gpt-4o-mini")
                response = self._openai_client.chat.completions.create(
                    model=eff_model2,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                data = json.loads(response.choices[0].message.content or "{}")
                queries = data.get("queries", [])
                if isinstance(queries, list) and len(queries) >= 2:
                    return " OR ".join([q.strip() for q in queries if q.strip()])
            except Exception as exc:
                logger.warning("OpenAI Multi-Query expansion failed: %s. Using fallback.", exc)

        # Offline / local multi-query expansion
        clean_q = query.strip(" ?.")
        words = [w for w in re.findall(r"\w+", clean_q) if len(w) > 3]
        q1 = f"{clean_q} exact procedure and guidelines"
        q2 = f"summary analysis of {' '.join(words[:4])}"
        q3 = f"technical breakdown and requirements for {clean_q}"
        return f"{q1} OR {q2} OR {q3}"

    def _apply_keyword_broadening(self, query: str) -> str:
        """
        Attempt 3: Strips restrictive qualifiers and extracts core high-signal keywords.
        """
        stop_words = {
            "what",
            "where",
            "when",
            "why",
            "how",
            "does",
            "do",
            "is",
            "are",
            "can",
            "could",
            "would",
            "should",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "about",
            "between",
            "and",
            "or",
            "from",
            "by",
        }
        tokens = re.findall(r"\w+", query.lower())
        keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
        if not keywords:
            return query.strip(" ?.")
        return " ".join(keywords)

    def reformulate(
        self, query: str, attempt_count: int, failure_reason: str
    ) -> tuple[str, str, str]:
        """
        Applies query reformulation strategy based on `attempt_count`.
        Returns: (refined_query: str, strategy_used: str, reasoning: str)
        """
        if attempt_count == 0:
            strategy = "HyDE (Hypothetical Document Embeddings)"
            refined = self._apply_hyde(query)
        elif attempt_count == 1:
            strategy = "Multi-Query Expansion"
            refined = self._apply_multi_query_expansion(query)
        else:
            strategy = "Keyword Broadening"
            refined = self._apply_keyword_broadening(query)

        reasoning = (
            f"Reformulated query via {strategy} (attempt {attempt_count + 1}) "
            f"due to: {failure_reason}. New query: `{refined}`."
        )
        logger.info(reasoning)
        return refined, strategy, reasoning


_query_reformulator: QueryReformulatorService | None = None


def get_query_reformulator() -> QueryReformulatorService:
    """Singleton getter for QueryReformulatorService."""
    global _query_reformulator
    if _query_reformulator is None:
        _query_reformulator = QueryReformulatorService()
    return _query_reformulator
