import os
import time
from typing import Dict
from uuid import UUID, uuid4

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)
_INSECURE_DEFAULT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"
DEFAULT_DEMO_TOKEN_TTL_SECONDS = 6 * 60 * 60


def _jwt_secret() -> str:
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret or jwt_secret == _INSECURE_DEFAULT_SECRET:
        raise HTTPException(status_code=503, detail="Authentication service misconfigured")
    return jwt_secret


def create_demo_jwt(session_id: str | None = None) -> Dict:
    """Create a short-lived JWT for the public portfolio demo experience."""
    try:
        demo_session = UUID(session_id) if session_id else uuid4()
    except ValueError:
        demo_session = uuid4()

    now = int(time.time())
    ttl_seconds = int(os.getenv("DEMO_TOKEN_TTL_SECONDS", str(DEFAULT_DEMO_TOKEN_TTL_SECONDS)))
    expires_at = now + ttl_seconds
    payload = {
        "sub": f"portfolio-demo-{demo_session.hex}",
        "role": "demo",
        "tenant_id": "portfolio-demo",
        "session_id": str(demo_session),
        "iat": now,
        "exp": expires_at,
    }
    return {
        "access_token": jwt.encode(payload, _jwt_secret(), algorithm="HS256"),
        "token_type": "bearer",
        "expires_at": expires_at,
        "session_id": str(demo_session),
    }


def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Validate Supabase JWT from Authorization: Bearer <token>."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authentication Token")

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            options={"verify_aud": False, "require": ["sub", "exp"]},
        )
        if not payload.get("sub"):
            raise jwt.InvalidTokenError("Missing subject")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token signature")
