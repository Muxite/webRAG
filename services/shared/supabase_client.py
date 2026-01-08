import os
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
    anon_key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not url:
        raise RuntimeError("Missing SUPABASE_URL environment variable")
    if not anon_key:
        raise RuntimeError("Missing SUPABASE_ANON_KEY environment variable (use publishable/anon key, not service role key)")

    client = create_client(url, anon_key)

    if hasattr(client, 'auth') and hasattr(client.auth, 'set_session'):
        try:
            client.auth.set_session(access_token=access_token, refresh_token="")
        except Exception:
            pass
    
    return client


