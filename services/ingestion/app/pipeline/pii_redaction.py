"""
PII Redaction Service using Microsoft Presidio (presidio-analyzer + presidio-anonymizer).
Performs actual NLP entity recognition (names, SSNs, emails, phone numbers, credit cards)
using spaCy (`en_core_web_sm`) and redacts sensitive information before chunking or embedding.
"""

import logging
from typing import ClassVar

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class PIIRedactionService:
    """
    Real PII Redaction Service powered by Microsoft Presidio.
    Enforces mandatory redaction of sensitive entity info before downstream pipeline stages.
    """

    DEFAULT_ENTITIES: ClassVar[list[str]] = [
        "PERSON",
        "US_SSN",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_BANK_NUMBER",
        "IP_ADDRESS",
    ]

    def __init__(self) -> None:
        logger.info("Initializing PIIRedactionService with spaCy en_core_web_sm NLP engine")
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=["en"]
        )
        self.anonymizer = AnonymizerEngine()  # type: ignore[no-untyped-call]

    def redact_text(
        self, text: str, entities: list[str] | None = None
    ) -> str:
        """
        Analyzes and redacts PII entities from input text.
        Returns redacted string where sensitive entities are replaced with <ENTITY_TYPE>.
        If input is empty or non-string, safely returns as string or empty string.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return text if isinstance(text, str) else str(text or "")

        target_entities = entities if entities is not None else self.DEFAULT_ENTITIES
        try:
            results = self.analyzer.analyze(
                text=text, entities=target_entities, language="en"
            )
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,  # type: ignore[arg-type]
            )
            return str(anonymized.text)
        except Exception as exc:
            logger.error("Error during PII redaction: %s", exc, exc_info=True)
            raise RuntimeError(f"PII Redaction failed: {exc}") from exc


# Singleton instance helper for worker usage
_pii_service_instance: PIIRedactionService | None = None


def get_pii_service() -> PIIRedactionService:
    """
    Returns cached singleton instance of PIIRedactionService to avoid reloading models.
    """
    global _pii_service_instance
    if _pii_service_instance is None:
        _pii_service_instance = PIIRedactionService()
    return _pii_service_instance
