"""Container CMD for every playground agent service (`command: ["python", "-m",
"playground.boot"]` in docker-compose.yml). Waits for ollama/searxng to answer, pulls
this tier's model roster (idempotent), drops a readiness marker `chat_entrypoint.py`
polls for, then idles with a periodic heartbeat so `docker logs` isn't silent and
`docker compose ps` has something real to report via the healthcheck.

This is deliberately NOT the interactive REPL — that's `chat_entrypoint.py`, reached via
`docker exec` (see chat.sh). Keeping the two separate is what makes `docker compose up`
bring the stack up in the background rather than attaching your terminal to a chat
session.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from playground import pull_models
from playground.tier_config import resolve_tier_config

# NOTE: `logging.basicConfig()` only configures the root handler on its FIRST call
# process-wide — since `playground.pull_models` is imported below and also calls
# basicConfig at module load, whichever import happens first "wins" for every logger in
# the process. Use `%(name)s` (dynamic) rather than a hardcoded module name in the format
# string so the output is correct regardless of import order (found via the live smoke
# test: boot.py's own lines were showing under the literal string "pull_models").
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("boot")

READY_MARKER = Path("/tmp/badmodel-ready")
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")
HEARTBEAT_SECONDS = 300


def _wait_for(url: str, label: str, timeout_s: float = 300.0, interval_s: float = 2.0) -> None:
    """Poll `url` until it answers (any status < 500 counts as "the service is up" —
    e.g. SearXNG's bot-detection limiter may 403 the bare root, that's still "reachable",
    not "still starting"). Connection errors/timeouts/5xx keep polling."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                logger.info("%s reachable (%s, status=%s)", label, url, resp.status)
                return
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                logger.info("%s reachable (%s, status=%s)", label, url, exc.code)
                return
        except Exception:
            pass
        time.sleep(interval_s)
    raise RuntimeError(f"Timed out waiting for {label} at {url}")


def main() -> None:
    tier_cfg = resolve_tier_config()
    logger.info("Booting playground container: tier=%s profile=%s", tier_cfg.tier, tier_cfg.profile)

    _wait_for(f"{pull_models.OLLAMA_BASE_URL}/api/tags", "ollama")
    _wait_for(f"{SEARXNG_URL}/", "searxng")

    pulled = pull_models.tier_pull(tier_cfg.tier)
    if pulled:
        logger.info("Pulled %d model(s): %s", len(pulled), ", ".join(pulled))
    else:
        logger.info("All tier models already present — nothing to pull.")

    READY_MARKER.touch()
    logger.info("Ready. Run ./chat.sh %s from the host to start chatting.", tier_cfg.tier)

    while True:
        time.sleep(HEARTBEAT_SECONDS)
        logger.info(
            "Idle heartbeat — tier=%s default_model=%s profile=%s",
            tier_cfg.tier, tier_cfg.default_model, tier_cfg.profile,
        )


if __name__ == "__main__":
    main()
