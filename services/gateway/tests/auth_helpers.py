import os
import time
from typing import Dict
from jose import jwt


def create_test_token(user_id: str = "test-user-id", email: str = "test@example.com", email_confirmed: bool = True) -> str:
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET must be set for tests")
    
    now = int(time.time())
    claims = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + 3600,
        "aud": "authenticated",
        "role": "authenticated",
    }
    
    if email_confirmed:
        claims["email_confirmed_at"] = now
    
    return jwt.encode(claims, secret, algorithm="HS256")


def auth_headers(token: str = None) -> Dict[str, str]:
    if token is None:
        token = create_test_token()
    return {"Authorization": f"Bearer {token}"}

