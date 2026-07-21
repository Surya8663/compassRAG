"""
Role-Based Access Control (RBAC) and multi-tenant scope isolation service for API Gateway.
Enforces strict tenant boundaries and queries Postgres `document_permissions` before retrieval.
"""

from fastapi import HTTPException
from shared.models.common import TenantContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentAccessPermission
from app.db.session import get_sync_session


class RBACService:
    """
    Enforces tenant scope and document-level RBAC checks against real Postgres permissions records.
    """

    @staticmethod
    def verify_tenant_scope(
        tenant_context: TenantContext, requested_tenant_id: str | None
    ) -> None:
        """
        Verifies that user is accessing resources within their own tenant namespace.
        Raises 403 Forbidden immediately if a cross-tenant attempt is detected.
        """
        if requested_tenant_id and requested_tenant_id != tenant_context.tenant_id:
            raise HTTPException(
                status_code=403,
                detail="Cross-tenant access forbidden: wrong tenant scope.",
            )

    @staticmethod
    def verify_document_access(
        tenant_context: TenantContext,
        document_id: str | None,
        session: Session | None = None,
    ) -> None:
        """
        Queries Postgres `DocumentAccessPermission` (`document_permissions`) for required role
        or specific user restrictions before allowing retrieval or ingestion operations.
        """
        if not document_id:
            return

        own_session = False
        if session is None:
            session = get_sync_session()
            own_session = True

        try:
            stmt = select(DocumentAccessPermission).where(
                DocumentAccessPermission.document_id == document_id,
                DocumentAccessPermission.tenant_id == tenant_context.tenant_id,
            )
            records = session.scalars(stmt).all()
            if not records:
                # No specific RBAC restrictions -> access allowed within tenant
                return

            # Check if any permission record grants access to this user/role
            for rec in records:
                if rec.allowed_user_id and rec.allowed_user_id == tenant_context.user_id:
                    return
                if rec.required_role and rec.required_role in tenant_context.roles:
                    return
                if rec.required_role is None and rec.allowed_user_id is None:
                    return

            raise HTTPException(
                status_code=403,
                detail=f"RBAC permission denied for requested document: '{document_id}'",
            )
        finally:
            if own_session:
                session.close()


rbac_service = RBACService()
