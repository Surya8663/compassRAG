"""
Unit tests for ``shared.tenant.resolve_tenant_id``.

These tests guarantee that the canonical tenant resolution function produces
identical results regardless of which service code-path invokes it, and that
the default tenant value is always ``"tenant_enterprise"`` — matching what is
already indexed in Qdrant and Elasticsearch.
"""

import pytest

from shared.tenant import DEFAULT_TENANT_ID, resolve_tenant_id


# ── Basic default behaviour ──────────────────────────────────────────


class TestDefaultTenantId:
    """The canonical default MUST be 'tenant_enterprise'."""

    def test_default_constant_value(self) -> None:
        assert DEFAULT_TENANT_ID == "tenant_enterprise"

    def test_resolve_with_no_inputs(self) -> None:
        assert resolve_tenant_id() == "tenant_enterprise"

    def test_resolve_with_all_none(self) -> None:
        assert resolve_tenant_id(None, None, None) == "tenant_enterprise"

    def test_resolve_with_empty_strings(self) -> None:
        assert resolve_tenant_id("", {}, "") == "tenant_enterprise"

    def test_resolve_with_whitespace_only(self) -> None:
        assert resolve_tenant_id("   ", None, "  ") == "tenant_enterprise"


# ── Priority / resolution order ─────────────────────────────────────


class TestResolutionPriority:
    """explicit_tenant_id > jwt_claims > header_tenant_id > default."""

    def test_explicit_wins_over_everything(self) -> None:
        result = resolve_tenant_id(
            explicit_tenant_id="tenant_alpha",
            jwt_claims={"tenant_id": "tenant_beta"},
            header_tenant_id="tenant_gamma",
        )
        assert result == "tenant_alpha"

    def test_jwt_claim_wins_over_header(self) -> None:
        result = resolve_tenant_id(
            explicit_tenant_id=None,
            jwt_claims={"tenant_id": "tenant_beta"},
            header_tenant_id="tenant_gamma",
        )
        assert result == "tenant_beta"

    def test_jwt_org_field(self) -> None:
        result = resolve_tenant_id(
            jwt_claims={"org": "tenant_org"},
        )
        assert result == "tenant_org"

    def test_jwt_organization_field(self) -> None:
        result = resolve_tenant_id(
            jwt_claims={"organization": "tenant_org2"},
        )
        assert result == "tenant_org2"

    def test_header_used_when_no_explicit_or_jwt(self) -> None:
        result = resolve_tenant_id(
            explicit_tenant_id=None,
            jwt_claims=None,
            header_tenant_id="tenant_header",
        )
        assert result == "tenant_header"

    def test_header_used_when_jwt_has_no_tenant_keys(self) -> None:
        result = resolve_tenant_id(
            explicit_tenant_id=None,
            jwt_claims={"sub": "user123"},
            header_tenant_id="tenant_header",
        )
        assert result == "tenant_header"


# ── Cross-path consistency (ingestion vs. query) ─────────────────────


class TestCrossPathConsistency:
    """
    The critical invariant: given equivalent "no explicit tenant, no JWT"
    inputs, the ingestion code-path and the query code-path MUST produce the
    exact same tenant_id. If someone reintroduces a second hardcoded default
    anywhere, the function itself will still be correct — but any *caller*
    that bypasses it will break these assertions when tested in integration.
    """

    def _simulate_ingestion_path(self, form_tenant_id: str | None) -> str:
        """Mimics what ``services/ingestion/app/api/routes.py`` does."""
        return resolve_tenant_id(explicit_tenant_id=form_tenant_id)

    def _simulate_query_path(
        self,
        payload_tenant_id: str | None,
        jwt_claims: dict | None,
        header_tenant_id: str | None,
    ) -> str:
        """Mimics the auth → orchestrator → graph flow in api-gateway."""
        return resolve_tenant_id(
            explicit_tenant_id=payload_tenant_id,
            jwt_claims=jwt_claims,
            header_tenant_id=header_tenant_id,
        )

    def test_both_paths_match_when_no_inputs(self) -> None:
        ingestion_result = self._simulate_ingestion_path(None)
        query_result = self._simulate_query_path(None, None, None)
        assert ingestion_result == query_result == "tenant_enterprise"

    def test_both_paths_match_with_explicit_tenant(self) -> None:
        ingestion_result = self._simulate_ingestion_path("tenant_acme")
        query_result = self._simulate_query_path("tenant_acme", None, None)
        assert ingestion_result == query_result == "tenant_acme"

    def test_both_paths_match_with_empty_string(self) -> None:
        ingestion_result = self._simulate_ingestion_path("")
        query_result = self._simulate_query_path("", None, None)
        assert ingestion_result == query_result == "tenant_enterprise"


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_whitespace_is_stripped(self) -> None:
        assert resolve_tenant_id(explicit_tenant_id="  tenant_x  ") == "tenant_x"

    def test_jwt_empty_tenant_id_falls_through(self) -> None:
        result = resolve_tenant_id(jwt_claims={"tenant_id": ""})
        assert result == "tenant_enterprise"

    def test_jwt_none_tenant_id_falls_through(self) -> None:
        result = resolve_tenant_id(jwt_claims={"tenant_id": None})
        assert result == "tenant_enterprise"
