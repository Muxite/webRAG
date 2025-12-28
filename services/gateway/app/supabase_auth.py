import os
from dataclasses import dataclass
from typing import Optional

import aiohttp
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt


@dataclass
class SupabaseUser:
    id: str
    email: str
    email_confirmed: bool


class SupabaseJWTVerifier:
    def __init__(self, secret_env: str="SUPABASE_JWT_SECRET") -> None:
        self.secret_env = secret_env

    async def verify(self, token: str) -> SupabaseUser:
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT secret is not configured",
            )

        try:
            claims = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed")

        user_id = claims.get("sub")
        email = claims.get("email") or ""

        email_confirmed = False
        if "email_confirmed_at" in claims and claims["email_confirmed_at"]:
            email_confirmed = True
        if claims.get("email_confirmed") is True:
            email_confirmed = True

        if not user_id or not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing user information")

        return SupabaseUser(id=user_id, email=email, email_confirmed=email_confirmed)


bearer_scheme = HTTPBearer(auto_error=False)
_jwt_verifier = SupabaseJWTVerifier()


async def get_current_supabase_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> SupabaseUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")

    user = await _jwt_verifier.verify(credentials.credentials)

    allow_unconfirmed = os.environ.get("SUPABASE_ALLOW_UNCONFIRMED", "").lower() == "true"

    if not user.email_confirmed and not allow_unconfirmed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email address is not confirmed",
        )

    return user


