"""
Canonical tenant_id resolution logic for the entire CompassRAG platform.

Every service (ingestion, retrieval, correction, generation, api-gateway)
MUST use ``resolve_tenant_id`` instead of hardcoding a fallback string.
This guarantees that the default tenant value is identical across all
code paths, preventing silent mismatches between indexed data (Qdrant /
Elasticsearch) and query-time resolution.
"""

# ── Canonical default ────────────────────────────────────────────────
DEFAULT_TENANT_ID: str = "tenant_enterprise"
"""
The single source-of-truth default tenant identifier.

This value matches the ``tenant_id`` already written into Qdrant points
and Elasticsearch documents by the ingestion pipeline, so changing it
would require a full re-index.
"""


def resolve_tenant_id(
    explicit_tenant_id: str | None = None,
    jwt_claims: dict | None = None,
    header_tenant_id: str | None = None,
) -> str:
    """Resolve the effective tenant_id from all available sources.

    Resolution order (first non-empty wins):
        1. ``explicit_tenant_id`` — caller-provided override (e.g. from a
           request payload or function parameter).
        2. ``jwt_claims`` — extracted from JWT token fields ``tenant_id``,
           ``org``, or ``organization``.
        3. ``header_tenant_id`` — value of the ``X-Tenant-ID`` HTTP header.
        4. :data:`DEFAULT_TENANT_ID` — the canonical fallback.

    All blank / whitespace-only strings are treated as *absent*.
    """
    # 1. Explicit override
    if explicit_tenant_id and explicit_tenant_id.strip():
        return explicit_tenant_id.strip()

    # 2. JWT claim extraction
    if jwt_claims:
        for key in ("tenant_id", "org", "organization"):
            value = jwt_claims.get(key)
            if value and str(value).strip():
                return str(value).strip()

    # 3. HTTP header
    if header_tenant_id and header_tenant_id.strip():
        return header_tenant_id.strip()

    # 4. Canonical default
    return DEFAULT_TENANT_ID
