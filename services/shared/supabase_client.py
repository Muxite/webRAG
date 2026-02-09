import os
import base64
import json
from supabase import Client, create_client


def create_user_client(access_token: str) -> Client:
    """
    Create a Supabase client with user's JWT token for RLS-enabled operations.
    
    Uses SUPABASE_ANON_KEY (publishable key) + user's access token.
    This respects Row-Level Security policies - users can only access their own data.
    
    :param access_token: User's JWT access token from Authorization header
    :return: Supabase client configured with user's token
    """
    url = os.environ.get("SUPABASE_URL")
    anon_key = (
        os.environ.get("SUPABASE_ANON_PUBLIC_KEY")
        or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    )
    
    if not url:
        raise RuntimeError("Missing SUPABASE_URL environment variable")
    if not anon_key:
        raise RuntimeError("Missing SUPABASE_ANON_PUBLIC_KEY environment variable (use publishable/anon key, not service role key)")
    
    # Create client with anon key (publishable key) - this respects RLS
    client = create_client(url, anon_key)
    
    # Set user's token so RLS policies can identify the user via auth.uid()
    if hasattr(client, "auth") and hasattr(client.auth, "set_session"):
        try:
            client.auth.set_session(access_token=access_token, refresh_token="")
        except Exception:
            pass
    if hasattr(client, "postgrest") and hasattr(client.postgrest, "auth"):
        try:
            client.postgrest.auth(access_token)
        except Exception:
            pass
    
    return client


def create_service_client() -> Client:
    """
    Create a Supabase client with service role key for server-side operations.
    :return: Supabase client configured with service role key
    """
    url = os.environ.get("SUPABASE_URL")
    service_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SECRET_KEY")
    )

    if not url:
        raise RuntimeError("Missing SUPABASE_URL environment variable")
    if not service_key:
        raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY environment variable")

    try:
        parts = service_key.split(".")
        if len(parts) >= 2:
            payload = parts[1] + "=" * (-len(parts[1]) % 4)
            decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
            claims = json.loads(decoded.decode("utf-8"))
            if claims.get("role") == "anon":
                raise RuntimeError("SUPABASE_API_KEY must be the service role key, not anon")
    except RuntimeError:
        raise
    except Exception:
        pass

    return create_client(url, service_key)

