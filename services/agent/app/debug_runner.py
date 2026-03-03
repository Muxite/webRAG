"""
agent-debug entry point.

Launch a GDB-style stepping session over the IdeaDAG agent.
Step through expansions, watch leaf actions execute, inspect any node,
set breakpoints, and view the live graph at every pause.

Environment
-----------
INTERACTIVE_MANDATE     Task for the agent (prompted if absent).
INTERACTIVE_TEST_ID     Run against an existing idea-test module.
INTERACTIVE_MAX_STEPS   Engine step cap (default 100).
MODEL_NAME              LLM model (default gpt-5-mini).
INTERACTIVE_LOG_LEVEL   Python log level for agent internals (default WARNING).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from typing import Tuple

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.agent_io import AgentIO
from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_memory import MemoryManager
from agent.app.telemetry import TelemetrySession
from shared.connector_config import ConnectorConfig

from agent.app.interactive import DebugSession, Controller, Renderer


def _setup_logging() -> None:
    level = os.environ.get("INTERACTIVE_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.WARNING),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("IdeaDagEngine").setLevel(logging.WARNING)
    logging.getLogger("agent.app.idea_policies").setLevel(logging.WARNING)


def _resolve_mandate() -> str:
    mandate = os.environ.get("INTERACTIVE_MANDATE", "").strip()
    if mandate:
        return mandate

    test_id = os.environ.get("INTERACTIVE_TEST_ID", "").strip()
    if test_id:
        return _load_test_mandate(test_id)

    print(Renderer.banner("agent-debug"))
    print("  Enter a task for the agent.  End with blank line.\n")
    lines = []
    while True:
        try:
            line = input("  > " if not lines else "  … ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip() and lines:
            break
        lines.append(line)
    mandate = "\n".join(lines).strip()
    if not mandate:
        print("  no mandate — exiting.")
        sys.exit(0)
    return mandate


def _load_test_mandate(test_id: str) -> str:
    from agent.app.testing.runner import discover_test_modules
    from agent.app.testing.test_module import IdeaTestModule

    for path in discover_test_modules():
        mod = IdeaTestModule(path)
        if mod.metadata.get("test_id") == test_id:
            return mod.get_task_statement()
    print(f"  [WARN] test '{test_id}' not found")
    sys.exit(1)


async def _boot_connectors(config: ConnectorConfig) -> Tuple:
    llm = ConnectorLLM(config)
    search = ConnectorSearch(config)
    http = ConnectorHttp(config)
    chroma = ConnectorChroma(config)
    await search.__aenter__()
    await http.__aenter__()
    await llm.__aenter__()
    await search.init_search_api()
    await chroma.init_chroma()
    return llm, search, http, chroma


async def _shutdown_connectors(*connectors) -> None:
    for c in connectors:
        try:
            await c.__aexit__(None, None, None)
        except Exception:
            pass


def _build(llm, search, http, chroma, model, settings, mandate):
    ns = f"idea_dag:{hashlib.sha256(mandate.encode()).hexdigest()[:10]}"
    settings["memo_namespace"] = ns
    telem = TelemetrySession(enabled=True, mandate=mandate, correlation_id=f"debug_{ns}")
    io = AgentIO(
        connector_llm=llm,
        connector_search=search,
        connector_http=http,
        connector_chroma=chroma,
        telemetry=telem,
        collection_name=f"debug_{ns}",
    )
    engine = IdeaDagEngine(io=io, settings=settings, model_name=model)
    engine._current_mandate = mandate
    engine._memory_manager = MemoryManager(connector_chroma=chroma, namespace=ns)
    graph = IdeaDag(root_title=mandate[:200], root_details={"mandate": mandate, "memo_namespace": ns})
    return engine, graph


async def _main() -> None:
    _setup_logging()
    mandate = _resolve_mandate()
    model = os.environ.get("MODEL_NAME", "gpt-5-mini")
    max_steps = int(os.environ.get("INTERACTIVE_MAX_STEPS", "100"))

    settings = load_idea_dag_settings()
    settings["allowed_actions"] = ["search", "visit", "save", "think", "merge"]
    settings["log_dag_ascii"] = False
    settings["log_dag_step_interval"] = 0

    config = ConnectorConfig()
    llm, search, http, chroma = await _boot_connectors(config)

    try:
        engine, graph = _build(llm, search, http, chroma, model, settings, mandate)
        session = DebugSession(engine=engine, graph=graph, ctrl=Controller(), max_steps=max_steps)
        result = await session.run()

        if not result.get("quit_early"):
            print(Renderer.banner("Final Answer"))
            try:
                final = await build_final_payload(
                    io=engine.io, settings=settings,
                    graph=graph, mandate=mandate, model_name=model,
                    memory_manager=engine._memory_manager,
                )
                print(f"\n{final.get('final_deliverable', '(none)')}\n")
            except Exception as exc:
                print(f"\n  [WARN] could not generate answer: {exc}\n")
        else:
            print("\n  session ended early.\n")
    finally:
        await _shutdown_connectors(llm, search, http, chroma)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
