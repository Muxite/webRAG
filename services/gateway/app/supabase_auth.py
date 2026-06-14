import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from jose.exceptions import JOSEError

_logger = logging.getLogger(__name__)

# In-process cache of fetched JWKS, keyed by URL: {url: {kid: jwk}}
_jwks_cache: dict[str, dict] = {}


@dataclass
class SupabaseUser:
    id: str
    email: str
    email_confirmed: bool


def _load_static_jwks() -> dict:
    """Parse the SUPABASE_JWT_PUBLIC_JWK env var (a JWKS document) into {kid: jwk}.

    Used as an offline fallback when the JWKS endpoint can't be reached.
    """
    raw = os.environ.get("SUPABASE_JWT_PUBLIC_JWK")
    if not raw:
        return {}
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        _logger.warning("SUPABASE_JWT_PUBLIC_JWK is not valid JSON; ignoring")
        return {}
    keys = doc.get("keys", []) if isinstance(doc, dict) else []
    return {k["kid"]: k for k in keys if k.get("kid")}


async def _fetch_jwks(url: str) -> dict:
    """Fetch a JWKS document and return {kid: jwk}, cached in-process."""
    if url in _jwks_cache:
        return _jwks_cache[url]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                resp.raise_for_status()
                doc = await resp.json()
        mapping = {k["kid"]: k for k in doc.get("keys", []) if k.get("kid")}
        if mapping:
            _jwks_cache[url] = mapping
        return mapping
    except Exception as exc:  # network/parse errors fall back to the static JWK
        _logger.warning("JWKS fetch failed for %s: %s", url, exc)
        return {}


class SupabaseJWTVerifier:
    def __init__(self, secret_env: str="SUPABASE_JWT_SECRET") -> None:
        self.secret_env = secret_env

    async def _resolve_jwk(self, kid: Optional[str]) -> Optional[dict]:
        """Find the asymmetric signing key (by kid) from the JWKS endpoint,
        falling back to the static JWK in SUPABASE_JWT_PUBLIC_JWK."""
        mapping: dict = {}
        url = os.environ.get("SUPABASE_JWKS_URL")
        if url:
            mapping = await _fetch_jwks(url)
        if not mapping or (kid and kid not in mapping):
            static = _load_static_jwks()
            mapping = {**static, **mapping}
        if kid:
            return mapping.get(kid)
        if len(mapping) == 1:
            return next(iter(mapping.values()))
        return None

    async def verify(self, token: str) -> SupabaseUser:
        try:
            header = jwt.get_unverified_header(token)
        except JOSEError as exc:
            _logger.info("JWT header parse failed: %s: %s", type(exc).__name__, exc)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed")

        alg = header.get("alg", "")

        try:
            if alg == "HS256":
                # Legacy shared-secret tokens (and the test suite).
                secret = os.environ.get(self.secret_env)
                if not secret:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="JWT secret is not configured",
                    )
                claims = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
            elif alg in ("ES256", "RS256"):
                # Asymmetric signing keys (muksite). Verify against the project's JWKS.
                signing_key = await self._resolve_jwk(header.get("kid"))
                if not signing_key:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="No matching JWKS signing key is configured",
                    )
                claims = jwt.decode(token, signing_key, algorithms=[alg], options={"verify_aud": False})
            else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unsupported token algorithm")
        except HTTPException:
            raise
        except JOSEError as exc:
            _logger.info("JWT verification failed: %s: %s", type(exc).__name__, exc)
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


