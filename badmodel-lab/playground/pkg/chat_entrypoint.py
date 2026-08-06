"""docker exec target: `docker exec -it <container> python -m playground.chat_entrypoint`
(wrapped by chat.sh). Resolves this tier's config, wires transcript/summary logging and
the keyless SearXNG search backend into basic_cli.py via three module-global
monkeypatches (MODEL_CANDIDATES, Agent, ConnectorSearch — all confirmed plain,
reassignable top-of-file imports basic_cli.main() only ever uses as bare names), then
runs basic_cli.main() completely unmodified. Re-running this (e.g. by re-running
chat.sh) naturally opens a fresh session while long-term memory (embedded Chroma) and
all prior logs persist via the container's volumes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import agent.app.basic_cli as basic_cli
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_test_runner import _apply_got_experiment_overrides

from playground.connector_search_searxng import ConnectorSearchXNG
from playground.tier_config import merged_environ, resolve_tier_config
from playground.traced_agent import JsonlLogHandler, SessionTraceRecorder, TracedAgent

READY_MARKER = Path("/tmp/badmodel-ready")
TRANSCRIPTS_DIR = Path("/app/playground_logs/transcripts")


def _wait_until_ready(poll_interval_s: float = 3.0) -> None:
    if READY_MARKER.exists():
        return
    print("Still downloading this tier's models — waiting for first boot to finish...")
    while not READY_MARKER.exists():
        time.sleep(poll_interval_s)
    print("Models ready.\n")


def _new_session_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


async def _run() -> None:
    _wait_until_ready()

    tier_cfg = resolve_tier_config()
    merged_env = merged_environ(tier_cfg)

    # Build the fully-resolved idea_dag_settings once per session: JSON defaults, then
    # this tier's mitigation profile layered on top via the same, unmodified translation
    # function the benchmark harness uses (agent.app.idea_test_runner —
    # _apply_got_experiment_overrides is a leading-underscore/conventionally-private
    # function in an actively-changing file; if a future change there renames or removes
    # it, this import fails loudly at process start, not silently).
    settings = load_idea_dag_settings()
    _apply_got_experiment_overrides(settings, merged_env)

    # basic_cli.main() reads the starting model from ConnectorConfig() (MODEL_NAME env)
    # — default to this tier's model without overriding an explicit operator choice.
    os.environ.setdefault("MODEL_NAME", tier_cfg.default_model)

    session_id = _new_session_id()
    transcript_path = TRANSCRIPTS_DIR / f"{session_id}.jsonl"
    tracer = SessionTraceRecorder(transcript_path, session_id)

    # basic_cli.py's `logging.basicConfig(level=logging.INFO)` (module-level) already
    # sends IdeaDagEngine/LlmExpansionPolicy/SearchLeafAction/etc.'s per-step logging to
    # stdout — attaching to the root logger persists that exact same detail into the
    # transcript file too, since Agent.tracer alone only sees one "init" event (see
    # JsonlLogHandler's docstring for why).
    logging.getLogger().addHandler(JsonlLogHandler(tracer))

    TracedAgent.shared_tracer = tracer
    TracedAgent.shared_settings = settings
    TracedAgent.shared_tier = tier_cfg.tier
    TracedAgent.shared_profile = tier_cfg.profile

    basic_cli.MODEL_CANDIDATES = list(tier_cfg.models)
    basic_cli.Agent = TracedAgent
    basic_cli.ConnectorSearch = ConnectorSearchXNG

    print("=" * 64)
    print("badmodel playground")
    print(f"  tier:          {tier_cfg.tier}")
    print(f"  default model: {tier_cfg.default_model}")
    print(f"  roster:        {', '.join(tier_cfg.models)}")
    print(f"  profile:       {tier_cfg.profile}")
    print(f"  transcript:    {transcript_path}")
    print(f"  summary log:   /app/playground_logs/session_summary.jsonl")
    print("=" * 64)

    try:
        await basic_cli.main()
    finally:
        tracer.close()


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
