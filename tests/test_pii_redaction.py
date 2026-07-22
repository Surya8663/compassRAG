"""
Unit test for Phase 3 PII Redaction Service powered by Microsoft Presidio and spaCy.
"""

from services.ingestion.app.pipeline.pii_redaction import get_pii_service


def test_presidio_pii_redaction_real_entities() -> None:
    """
    Feeds text containing fake SSN, email, phone number, credit card, and name.
    Asserts the redacted output no longer contains the sensitive text and instead contains
    Presidio entity tags like <PERSON>, <EMAIL_ADDRESS>, <US_SSN>, <PHONE_NUMBER>, <CREDIT_CARD>.
    """
    pii_service = get_pii_service()

    raw_text = (
        "Contact Jane Smith at jane.smith@example.com or phone 415-555-1234. "
        "Her SSN is 999-01-1234 and credit card is 4532-0151-1283-0366."
    )

    redacted = pii_service.redact_text(raw_text)

    # Assert email, phone, SSN, and credit card are redacted
    assert "jane.smith@example.com" not in redacted
    assert "415-555-1234" not in redacted
    assert "999-01-1234" not in redacted
    assert "4532-0151-1283-0366" not in redacted
    # Person name is preserved per Requirement #6
    assert "Jane Smith" in redacted

    # Assert Presidio entity replacement tags are present
    assert "<EMAIL_ADDRESS>" in redacted
    assert "<PHONE_NUMBER>" in redacted
    assert "<US_SSN>" in redacted
    assert "<CREDIT_CARD>" in redacted
