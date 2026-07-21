"""
JWT authentication and multi-tenant context extraction service for API Gateway.
Supports both HS256 secret verification and Keycloak RS256 token decoding.
"""

import json
from urllib.request import urlopen

from fastapi import HTTPException, Request
from jose import jwt
from jose.exceptions import JWTError
from shared.config import get_settings
from shared.models.common import TenantContext


class JWTValidator:
    """
    Validates OAuth2/JWT tokens and extracts `TenantContext` containing `tenant_id`,
    `user_id`, `roles`, and `permissions`.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._jwks_keys: dict | None = None

    def _get_jwks(self) -> dict:
        """Fetches public RSA keys from Keycloak JWKS endpoint (`KEYCLOAK_JWKS_URL`)."""
        if self._jwks_keys is None:
            try:
                with urlopen(self.settings.KEYCLOAK_JWKS_URL, timeout=5) as resp:
                    self._jwks_keys = json.loads(resp.read().decode())
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to fetch Keycloak JWKS: {str(e)}"
                ) from e
        return self._jwks_keys

    def validate_and_extract(self, auth_header: str | None, target_tenant: str = "tenant_enterprise") -> TenantContext:
        """
        Decodes token, verifies signature and expiration, and extracts claims.
        Rejects invalid requests with 401 Unauthorized before downstream calls.
        """
        if not self.settings.AUTH_ENABLED:
            return TenantContext(
                tenant_id=target_tenant,
                user_id="dev_admin",
                roles=["admin"],
                permissions=["*"],
            )

        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )

        token = auth_header.split(" ", 1)[1].strip()
        try:
            if self.settings.JWT_ALGORITHM == "RS256":
                jwks = self._get_jwks()
                payload = jwt.decode(
                    token,
                    jwks,
                    algorithms=[self.settings.JWT_ALGORITHM],
                    audience=self.settings.KEYCLOAK_CLIENT_ID,
                )
            else:
                payload = jwt.decode(
                    token,
                    self.settings.JWT_SECRET_KEY,
                    algorithms=[self.settings.JWT_ALGORITHM],
                )
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}") from e

        # Extract tenant_id from claims
        tenant_id = (
            payload.get("tenant_id")
            or payload.get("org")
            or payload.get("organization")
            or "default_tenant"
        )
        user_id = str(payload.get("sub") or payload.get("user_id") or "anonymous")

        # Extract roles from top-level roles list or Keycloak realm_access structure
        roles = payload.get("roles", [])
        if not roles and isinstance(payload.get("realm_access"), dict):
            roles = payload["realm_access"].get("roles", [])
        if not isinstance(roles, list):
            roles = [str(roles)]

        permissions = payload.get("permissions", [])
        if not isinstance(permissions, list):
            permissions = [str(permissions)]

        return TenantContext(
            tenant_id=str(tenant_id),
            user_id=user_id,
            roles=[str(r) for r in roles],
            permissions=[str(p) for p in permissions],
        )


_jwt_validator = JWTValidator()


def get_current_tenant_context(request: Request) -> TenantContext:
    """
    FastAPI dependency that extracts and validates the JWT from incoming HTTP requests.
    """
    auth_header = request.headers.get("Authorization")
    x_tenant_id = request.headers.get("X-Tenant-ID", "tenant_enterprise")
    return _jwt_validator.validate_and_extract(auth_header, target_tenant=x_tenant_id)
