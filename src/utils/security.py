import os
from typing import Dict

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Validate Supabase JWT from Authorization: Bearer <token>."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authentication Token")

    token = credentials.credentials
    # In production, SUPABASE_JWT_SECRET should be set in the environment
    # By default, we use a test secret for development
    jwt_secret = os.getenv(
        "SUPABASE_JWT_SECRET", "super-secret-jwt-token-with-at-least-32-characters-long"
    )

    try:
        # Decode and verify the JWT signature
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"], options={"verify_aud": False})
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token signature")
