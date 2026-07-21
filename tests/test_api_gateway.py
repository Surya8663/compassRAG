"""
Verification suite for Phase 8: API Gateway & Auth Layer (`services/api-gateway`).
Verifies:
1. Rejection (`401 Unauthorized`) when JWT is missing/malformed before reaching retrieval.
2. Rejection (`403 Forbidden`) when valid JWT attempts cross-tenant access.
3. Document-level RBAC restrictions against database records (`document_permissions`).
4. End-to-end integration for `POST /v1/ingest`, `POST /v1/query`, `GET /v1/status/{job_id}`.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from app.api.routes import router as gateway_router
from app.db.models import Base, DocumentAccessPermission
from app.db.session import get_sync_session
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from shared.config import get_settings
from shared.models.common import (
    ConfidenceStatus,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(autouse=True)
def setup_gateway_test_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sets up an in-memory SQLite database for gateway RBAC and batch tracking tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_engine() -> Any:
        return engine

    def override_get_sync_session() -> Session:
        return TestingSessionLocal()

    monkeypatch.setattr("app.db.session.get_engine", override_get_engine)
    monkeypatch.setattr("app.db.session.get_sync_session", override_get_sync_session)
    monkeypatch.setattr("app.services.orchestrator.get_sync_session", override_get_sync_session)
    monkeypatch.setattr("app.services.rbac.get_sync_session", override_get_sync_session)


@pytest.fixture
def client() -> TestClient:
    """Provides a FastAPI TestClient configured with gateway routes."""
    app = FastAPI()
    app.include_router(gateway_router)
    return TestClient(app)


def _make_jwt(
    tenant_id: str = "Tenant_A", user_id: str = "user_1", roles: list[str] | None = None
) -> str:
    """Helper generating valid HS256 tokens for gateway auth testing."""
    settings = get_settings()
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles or ["viewer"],
        "permissions": ["read"],
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def test_gateway_rejects_missing_or_invalid_jwt_before_retrieval(client: TestClient) -> None:
    """
    Asserts requests without a valid JWT are rejected (`401 Unauthorized`) before reaching
    any retrieval or orchestrator logic.
    """
    payload = {
        "query": "How does retrieval work?",
        "tenant_context": {
            "tenant_id": "Tenant_A",
            "user_id": "user_1",
            "roles": ["viewer"],
            "permissions": [],
        },
        "top_k": 5,
    }

    with patch("app.api.routes.gateway_orchestrator.process_query") as mock_orchestrator:
        # 1. Missing Authorization header -> 401
        resp_missing = client.post("/v1/query", json=payload)
        assert resp_missing.status_code == 401
        assert "Missing or invalid Authorization header" in resp_missing.json()["detail"]
        assert mock_orchestrator.call_count == 0

        # 2. Malformed token -> 401
        resp_malformed = client.post(
            "/v1/query", json=payload, headers={"Authorization": "Bearer invalid.token.string"}
        )
        assert resp_malformed.status_code == 401
        assert "Invalid or expired JWT token" in resp_malformed.json()["detail"]
        assert mock_orchestrator.call_count == 0


def test_gateway_rejects_cross_tenant_scope_access(client: TestClient) -> None:
    """
    Asserts requests with valid JWT for `Tenant_A` attempting to query `Tenant_B`
    are rejected (`403 Forbidden`).
    """
    token = _make_jwt(tenant_id="Tenant_A", user_id="user_1", roles=["viewer"])
    payload = {
        "query": "What is Tenant B balance?",
        "tenant_context": {
            "tenant_id": "Tenant_B",
            "user_id": "user_1",
            "roles": ["viewer"],
            "permissions": [],
        },
        "top_k": 5,
    }

    resp = client.post("/v1/query", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "Cross-tenant access forbidden" in resp.json()["detail"]


def test_gateway_enforces_document_rbac_permissions(client: TestClient) -> None:
    """
    Verifies document-level RBAC restrictions backed by Postgres permissions records.
    Users without `required_role` are rejected (`403 Forbidden`); authorized users proceed cleanly.
    """
    # Create permission record requiring 'admin' role for 'doc_restricted'
    session = get_sync_session()
    perm = DocumentAccessPermission(
        tenant_id="Tenant_A",
        document_id="doc_restricted",
        required_role="admin",
        allowed_user_id=None,
    )
    session.add(perm)
    session.commit()
    session.close()

    payload = {
        "query": "Summarize confidential report",
        "tenant_context": {
            "tenant_id": "Tenant_A",
            "user_id": "user_viewer",
            "roles": ["viewer"],
            "permissions": [],
        },
        "metadata_filter": {"document_id": "doc_restricted"},
    }

    # 1. Viewer role -> rejected with 403
    token_viewer = _make_jwt(tenant_id="Tenant_A", user_id="user_viewer", roles=["viewer"])
    resp_viewer = client.post(
        "/v1/query", json=payload, headers={"Authorization": f"Bearer {token_viewer}"}
    )
    assert resp_viewer.status_code == 403
    assert "RBAC permission denied" in resp_viewer.json()["detail"]

    # 2. Admin role -> allowed and dispatches to graph
    token_admin = _make_jwt(tenant_id="Tenant_A", user_id="user_admin", roles=["admin"])
    payload_admin = dict(payload)
    payload_admin["tenant_context"]["roles"] = ["admin"]

    with patch(
        "services.correction.app.services.graph.get_correction_graph"
    ) as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_answer": "Confidential report summary.",
            "final_status": ConfidenceStatus.VERIFIED,
            "groundedness_score": 0.95,
            "draft_citations": [],
            "verdicts": [],
        }
        mock_get_graph.return_value = mock_graph

        resp_admin = client.post(
            "/v1/query", json=payload_admin, headers={"Authorization": f"Bearer {token_admin}"}
        )
        assert resp_admin.status_code == 200
        data = resp_admin.json()
        assert data["answer"] == "Confidential report summary."
        assert data["confidence_score"] == 0.95


def test_gateway_end_to_end_ingest_query_status(client: TestClient) -> None:
    """
    Verifies end-to-end integration for `POST /v1/ingest`, `POST /v1/query`,
    and `GET /v1/status/{job_id}` across orchestrator and database.
    """
    token = _make_jwt(tenant_id="Tenant_A", user_id="user_1", roles=["admin"])
    headers = {"Authorization": f"Bearer {token}"}

    # 1. POST /v1/ingest
    ingest_payload = {
        "document_id": "doc_101",
        "filename": "manual.pdf",
        "expected_pages": 12,
        "tenant_id": "Tenant_A",
    }
    with patch(
        "app.services.orchestrator.gateway_orchestrator.celery_app.send_task"
    ) as mock_send_task:
        resp_ingest = client.post("/v1/ingest", json=ingest_payload, headers=headers)
        assert resp_ingest.status_code == 200
        data_ingest = resp_ingest.json()
        job_id = data_ingest["job_id"]
        assert data_ingest["status"] == "PROCESSING"
        mock_send_task.assert_called_once()

    # 2. GET /v1/status/{job_id}
    resp_status = client.get(f"/v1/status/{job_id}", headers=headers)
    assert resp_status.status_code == 200
    data_status = resp_status.json()
    assert data_status["id"] == job_id
    assert data_status["document_id"] == "doc_101"
    assert data_status["expected_pages"] == 12

    # 3. POST /v1/query
    query_payload = {
        "query": "What is in manual?",
        "tenant_context": {
            "tenant_id": "Tenant_A",
            "user_id": "user_1",
            "roles": ["admin"],
            "permissions": [],
        },
    }
    with patch(
        "services.correction.app.services.graph.get_correction_graph"
    ) as mock_get_graph:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_answer": "The manual details pipeline operations.",
            "final_status": ConfidenceStatus.VERIFIED,
            "groundedness_score": 0.99,
            "draft_citations": [],
            "verdicts": [],
        }
        mock_get_graph.return_value = mock_graph

        resp_query = client.post("/v1/query", json=query_payload, headers=headers)
        assert resp_query.status_code == 200
        data_query = resp_query.json()
        assert "manual details pipeline operations" in data_query["answer"]
        assert data_query["confidence_score"] == 0.99
