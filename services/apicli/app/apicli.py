import os
from typing import Optional

import requests
from supabase import create_client, Client
from shared.connector_config import ConnectorConfig
from shared.pretty_log import pretty_log_print


class ApiCli:
    def __init__(
        self,
        base_url: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        cfg = ConnectorConfig()
        self.base_url = (base_url or os.environ.get("GATEWAY_URL") or "http://localhost:8080").rstrip("/")
        self.timeout = float(os.environ.get("API_CLI_TIMEOUT", str(cfg.default_timeout if timeout is None else timeout)))
        
        supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        supabase_anon_key = supabase_anon_key or os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_API_KEY")
        
        if not supabase_url or not supabase_anon_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set for API CLI authentication")
        
        self.supabase: Client = create_client(supabase_url, supabase_anon_key)
        self.access_token: Optional[str] = None

    def _ensure_authenticated(self) -> bool:
        if self.access_token:
            try:
                session_response = self.supabase.auth.get_session()
                session = getattr(session_response, "session", None) or (getattr(session_response, "data", {}).get("session") if hasattr(session_response, "data") else None)
                if session and hasattr(session, "access_token"):
                    self.access_token = session.access_token
                    return True
            except Exception as e:
                print(f"Session check failed: {e}")
                self.access_token = None
        
        print("Please sign in to continue:")
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        
        try:
            response = self.supabase.auth.sign_in_with_password({"email": email, "password": password})
            
            if hasattr(response, "session") and response.session:
                session = response.session
                if hasattr(session, "access_token"):
                    self.access_token = session.access_token
                    user_email = email
                    if hasattr(response, "user") and response.user:
                        user = response.user
                        if hasattr(user, "email"):
                            user_email = user.email
                    print(f"Signed in as {user_email}")
                    return True
            
            error_msg = "Authentication failed"
            if hasattr(response, "error") and response.error:
                error = response.error
                if hasattr(error, "message"):
                    error_msg = error.message
                elif isinstance(error, str):
                    error_msg = error
            print(f"Failed to sign in: {error_msg}")
            return False
        except Exception as e:
            print(f"Authentication error: {e}")
            return False

    def _headers(self) -> Optional[dict]:
        if not self.access_token:
            if not self._ensure_authenticated():
                return None
        
        return {"Authorization": f"Bearer {self.access_token}"}

    def _pretty(self, obj: object) -> None:
        print(pretty_log_print(obj))

    def submit_task(self) -> None:
        headers = self._headers()
        if not headers:
            print("Cannot submit task: authentication required")
            return
        
        mandate = input("Mandate: ").strip()
        max_ticks_text = input("Max ticks [50]: ").strip() or "50"
        try:
            max_ticks = int(max_ticks_text)
        except ValueError:
            max_ticks = 50
        payload = {"mandate": mandate, "max_ticks": max_ticks}
        url = f"{self.base_url}/tasks"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            self._pretty({"status_code": resp.status_code, "body": self._safe_json(resp)})
            if resp.status_code == 401:
                self.access_token = None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    def poll_task(self) -> None:
        headers = self._headers()
        if not headers:
            print("Cannot get task: authentication required")
            return
        
        cid = input("Correlation ID: ").strip()
        if not cid:
            print("No correlation id provided")
            return
        url = f"{self.base_url}/tasks/{cid}"
        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            self._pretty({"status_code": resp.status_code, "body": self._safe_json(resp)})
            if resp.status_code == 401:
                self.access_token = None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    def _safe_json(self, resp: requests.Response) -> object:
        try:
            return resp.json()
        except Exception:
            return resp.text

    def run(self) -> None:
        print(f"Gateway: {self.base_url}")
        
        while True:
            print("\nChoose: [1] Submit task  [2] Get task  [s] Sign in  [q] Quit")
            choice = input("> ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                return
            elif choice == "s":
                self._ensure_authenticated()
            elif choice == "1":
                self.submit_task()
            elif choice == "2":
                self.poll_task()
            else:
                print("Unknown choice")


def main() -> None:
    cli = ApiCli()
    cli.run()


if __name__ == "__main__":
    main()
