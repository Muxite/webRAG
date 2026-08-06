"""Idempotent, tier-scoped Ollama model puller. Stdlib-only (urllib + json) — this only
ever needs two HTTP calls against the ollama container's native API (not the OpenAI-
compatible /v1 surface the agent itself talks to), so no extra dependency is worth adding.

Presence is checked against the ollama container's own /api/tags before pulling, so a
repeat `docker compose up` after the first successful pull re-downloads nothing.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import List, Set

from playground.tier_config import TIER_ROSTER_PATH, _load_yaml

# See the matching note in boot.py: basicConfig only takes effect on its first
# process-wide call, so this format string must stay generic (%(name)s), not hardcode
# "pull_models" — whichever module imports first determines which basicConfig call wins.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("pull_models")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")


def _normalize_tag(tag: str) -> str:
    """Ollama always reports a fully-qualified tag (e.g. `tinyllama` pulled without an
    explicit tag is reported back as `tinyllama:latest`) — normalize before comparing so
    an untagged roster entry doesn't look "missing" forever and get re-pulled every boot.
    """
    return tag if ":" in tag else f"{tag}:latest"


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_present_tags(ollama_url: str = OLLAMA_BASE_URL) -> Set[str]:
    try:
        data = _get_json(f"{ollama_url}/api/tags")
    except Exception as exc:  # ollama not reachable yet, or empty tag store
        logger.warning("Could not list existing ollama tags (%s); assuming none present", exc)
        return set()
    return {m.get("name", "") for m in data.get("models", []) if m.get("name")}


def pull_tag(tag: str, ollama_url: str = OLLAMA_BASE_URL) -> None:
    """Stream one `ollama pull`, logging each distinct NDJSON status line to stdout so
    `docker logs`/`logs.sh` shows first-boot download progress."""
    logger.info("Pulling %s ...", tag)
    payload = json.dumps({"name": tag, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url}/api/pull",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    last_status = None
    last_progress_log = 0.0
    with urllib.request.urlopen(request, timeout=None) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("error"):
                raise RuntimeError(f"ollama pull failed for {tag}: {event['error']}")
            status = event.get("status")
            if status and status != last_status:
                logger.info("  %s: %s", tag, status)
                last_status = status
                last_progress_log = time.monotonic()
                continue
            # Same status (e.g. a multi-GB "pulling <digest>" layer) can run for minutes
            # with no status-string change — log byte progress every ~10s so a slow
            # download on a large model doesn't look stalled in `docker logs`/`logs.sh`.
            total = event.get("total")
            completed = event.get("completed")
            if total and completed and time.monotonic() - last_progress_log >= 10.0:
                pct = 100.0 * completed / total
                logger.info(
                    "  %s: %s (%.0f%%, %.0f/%.0f MB)",
                    tag, status, pct, completed / 1e6, total / 1e6,
                )
                last_progress_log = time.monotonic()
    logger.info("Done: %s", tag)


def tier_pull(tier: str) -> List[str]:
    """Pull every model listed for `tier` in tier_roster.yaml that isn't already
    present. Idempotent — safe to call on every container start. Returns the tags that
    were actually pulled this call (empty list if everything was already present)."""
    roster = _load_yaml(TIER_ROSTER_PATH)
    tier_entry = (roster.get("tiers") or {}).get(tier)
    if not tier_entry:
        raise RuntimeError(f"Unknown tier {tier!r} in tier_roster.yaml")
    tags = list(tier_entry.get("models") or [tier_entry["default"]])

    present = list_present_tags()
    pulled: List[str] = []
    for tag in tags:
        if _normalize_tag(tag) in present:
            logger.info("Already present, skipping: %s", tag)
            continue
        pull_tag(tag)
        pulled.append(tag)
    return pulled
