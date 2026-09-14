import os
from typing import Dict

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)
_INSECURE_DEFAULT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"


def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Validate Supabase JWT from Authorization: Bearer <token>."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authentication Token")

    token = credentials.credentials
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret or jwt_secret == _INSECURE_DEFAULT_SECRET:
        raise HTTPException(status_code=503, detail="Authentication service misconfigured")

    try:
        payload = jwt.decode(
            token,
            jwt_secret,
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
