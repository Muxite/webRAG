import os
from typing import Optional

import requests
from shared.connector_config import ConnectorConfig
from shared.pretty_log import pretty_log_print


class ApiCli:
    """
    Simple interactive CLI for calling the Gateway API.
    :param base_url: Base URL of the Gateway service
    :param api_key: API key for Authorization header (X-API-Key)
    :param timeout: Request timeout in seconds
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        cfg = ConnectorConfig()
        self.base_url = (base_url or os.environ.get("GATEWAY_URL") or "http://localhost:8080").rstrip("/")
        self.api_key = api_key or os.environ.get("GATEWAY_API_KEY")
        self.timeout = float(os.environ.get("API_CLI_TIMEOUT", str(cfg.default_timeout if timeout is None else timeout)))

    def _input_api_key(self) -> None:
        """
        Ensure an API key is present, optionally prompting the user.
        """
        if not self.api_key:
            self.api_key = input("Enter API key: ").strip()

    def _headers(self) -> dict:
        """
        Build request headers.
        :return headers: Dictionary of HTTP headers
        """
        return {"X-API-Key": self.api_key or ""}

    def _pretty(self, obj: object) -> None:
        """
        Print an object using the shared pretty_log formatter.
        :param obj: Any object (dict, list, or primitive)
        """
        print(pretty_log_print(obj))

    def submit_task(self) -> None:
        """
        Submit a task to the Gateway using POST /tasks.
        Prompts for mandate and max_ticks.
        """
        self._input_api_key()
        mandate = input("Mandate: ").strip()
        max_ticks_text = input("Max ticks [50]: ").strip() or "50"
        try:
            max_ticks = int(max_ticks_text)
        except ValueError:
            max_ticks = 50
        payload = {"mandate": mandate, "max_ticks": max_ticks}
        url = f"{self.base_url}/tasks"
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        self._pretty({"status_code": resp.status_code, "body": self._safe_json(resp)})

    def poll_task(self) -> None:
        """
        Retrieve a task by correlation id using GET /tasks/{id}.
        :return None: Nothing is returned
        """
        self._input_api_key()
        cid = input("Correlation ID: ").strip()
        if not cid:
            print("No correlation id provided")
            return
        url = f"{self.base_url}/tasks/{cid}"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        self._pretty({"status_code": resp.status_code, "body": self._safe_json(resp)})

    def _safe_json(self, resp: requests.Response) -> object:
        """
        Extract JSON body from a response safely.
        :param resp: requests.Response object
        :return body: Parsed JSON or plain text fallback
        """
        try:
            return resp.json()
        except Exception:
            return resp.text

    def run(self) -> None:
        """
        Run an interactive loop allowing the user to choose actions.
        """
        print(f"Gateway: {self.base_url}")
        while True:
            print("\nChoose: [1] Submit task  [2] Get task  [q] Quit")
            choice = input("> ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                return
            if choice == "1":
                self.submit_task()
            elif choice == "2":
                self.poll_task()
            else:
                print("Unknown choice")


def main() -> None:
    """
    Entry point for the API CLI program.
    Reads environment variables and starts the interactive loop.
    """
    cli = ApiCli()
    cli.run()


if __name__ == "__main__":
    main()
