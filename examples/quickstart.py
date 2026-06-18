#!/usr/bin/env python3
"""
Euglena / webRAG — minimal end-to-end API client.

Submits a research mandate to the gateway, polls until the task completes, then prints the
answer plus the `evidence` block (pages visited, grounding verdict, token/cost usage).

Stdlib only — no dependencies. Usage:

    export GATEWAY_URL=http://localhost:8080
    export TOKEN=<your Supabase JWT>          # sign in via the frontend to obtain one
    python examples/quickstart.py "Who wrote 'Beloved' and where did she earn her master's?"

See ../docs/CONFIGURATION.md for configuration and ../services/keys.env.example to run the stack.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080").rstrip("/")
TOKEN = os.environ.get("TOKEN", "")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "3"))
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "600"))


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{GATEWAY_URL}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code} on {method} {path}: {exc.read().decode()[:300]}")


def main() -> None:
    mandate = " ".join(sys.argv[1:]).strip() or "Who wrote the novel 'Beloved', and where did she earn her master's degree?"
    if not TOKEN:
        sys.exit("Set TOKEN to a Supabase JWT (see the README 'Submit Your First Query' section).")

    submitted = _request("POST", "/tasks", {"mandate": mandate})
    correlation_id = submitted["correlation_id"]
    print(f"submitted: {correlation_id} (status={submitted['status']})")

    deadline = time.time() + TIMEOUT_SECONDS
    while time.time() < deadline:
        task = _request("GET", f"/tasks/{correlation_id}")
        status = task.get("status")
        print(f"  status={status} tick={task.get('tick')}/{task.get('max_ticks')}")
        if status in ("completed", "failed", "unknown"):
            break
        time.sleep(POLL_SECONDS)
    else:
        sys.exit("timed out waiting for completion")

    result = task.get("result") or {}
    print("\n=== ANSWER ===")
    for d in (result.get("deliverables") or ["(none)"]):
        print(d)

    evidence = result.get("evidence") or {}
    if evidence:
        print("\n=== EVIDENCE ===")
        for src in (evidence.get("sources") or []):
            print(f"  source: {src.get('title') or '(untitled)'} — {src.get('url')}")
        if "grounded" in evidence:
            print(f"  grounded: {evidence['grounded']}  missing={evidence.get('missing_requirements')}")
        if evidence.get("unverified_citations"):
            print(f"  UNVERIFIED citations (not opened): {evidence['unverified_citations']}")
        if evidence.get("truncated"):
            print("  NOTE: answer was truncated at the length cap")
        usage = evidence.get("usage") or {}
        if usage:
            print(f"  usage: {usage.get('llm_calls')} LLM calls, {usage.get('total_tokens')} tokens, "
                  f"{usage.get('cost_str', usage.get('cost_usd', 'n/a'))}, {usage.get('duration_s')}s")
        if evidence.get("failure"):
            print(f"  failure: {evidence['failure']}")


if __name__ == "__main__":
    main()
