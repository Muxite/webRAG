"""Container ENTRYPOINT for the codebench *badmodel* agent (the compiled-scaffold coding arm).

Run by ``codebench/agents/badmodel/Dockerfile``'s
``ENTRYPOINT ["python", "-m", "app.codebench_run_task"]``, dispatched by
``run_agent_sandbox.sh`` with the same CLI contract the Aider baseline gets::

    python -m app.codebench_run_task --model MODEL --task-dir /task --workdir /work

What it does, in order:
  1. copies ``{task-dir}/repo/`` into ``{workdir}`` — the starter fixture, exactly like the Aider
     baseline's ``run_task.sh`` does, so neither agent kind burns turns discovering its own files;
  2. loads ``{task-dir}/plan.json`` — the ONLY source of the task's compiled plan. It never imports
     ``app.idea_code_tests``: that package holds canonical (sometimes hidden) test content and is
     deliberately deleted from this image at build time, because this container has real code
     execution and nothing hidden may exist anywhere in its filesystem;
  3. builds the sandbox connector rooted at ``{workdir}``, a SearXNG-backed search connector
     (``$SEARXNG_URL`` — the only web surface on the ``--internal`` codebench network) and the
     usual ``ConnectorLLM``/``AgentIO`` pair pointed at ``$MODEL_API_URL``;
  4. drives ``testing/execution_compiled_code.run_compiled_code_plan`` to completion;
  5. prints a short summary and exits 0. Nothing needs to be written anywhere special — the
     harness extracts ``/work`` wholesale after the container exits.

No top-level timeout on purpose: ``run_agent_sandbox.sh`` already enforces a hard 900s wall-clock
kill from outside, and the per-action bounds come from ``SandboxActionConfig``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from shared.connector_config import ConnectorConfig

from agent.app.agent_io import AgentIO
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_sandbox import SandboxConnector
from agent.app.connector_search_searxng import ConnectorSearchXNG
from agent.app.telemetry import TelemetrySession
from agent.app.testing.execution_compiled_code import (
    run_compiled_code_plan,
    sandbox_config,
    summarize_run,
)

_logger = logging.getLogger("codebench_run_task")


def copy_starter_fixture(task_dir: Path, workdir: Path) -> int:
    """Copy ``{task-dir}/repo/`` into ``workdir`` (created if absent). Returns the file count.

    Mirrors ``codebench/agents/aider/run_task.sh``'s ``cp -r "$TASK_DIR/repo/." ...``
    so both agent kinds start from a byte-identical tree.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    repo = task_dir / "repo"
    if not repo.is_dir():
        _logger.warning(f"no starter fixture at {repo}; starting from an empty workdir")
        return 0
    # NOT shutil.copytree: even with copy_function=shutil.copy (content-only, no per-file
    # copystat), copytree's own _copytree() unconditionally calls copystat() on the TOP-LEVEL
    # directory at the end, independent of copy_function — verified by reading shutil's
    # source after this raised "Operation not permitted" from BOTH copy2 and copy. That
    # top-level copystat crosses the bind-mount boundary (host-owned /task -> agent-owned
    # /work under --cap-drop=ALL) and fails. A starter fixture only needs file CONTENT, so
    # walk and copy bytes directly — no shutil call that touches metadata at any level.
    count = 0
    for src in sorted(repo.rglob("*")):
        if src.is_dir():
            continue
        dst = workdir / src.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        count += 1
    return count


def load_plan(task_dir: Path) -> Dict[str, Any]:
    """Read ``{task-dir}/plan.json`` (written host-side by ``codebench/materialize_task.py``).

    Plain JSON on purpose — see this module's docstring for why the task module itself is not
    importable here.
    """
    path = task_dir / "plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"{path} must contain a compiled-plan object, got {type(plan).__name__}")
    return plan


def read_prompt(task_dir: Path) -> str:
    """The natural-language task statement (``prompt.md``), or '' when absent."""
    path = task_dir / "prompt.md"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


def build_context(model: str, workdir: Path, mandate: str):
    """Build the connector set this run needs: sandbox + search + LLM/AgentIO.

    Same construction the ``graph_compiled`` path uses (``ConnectorConfig`` from env, one
    ``ConnectorLLM`` pinned to the runtime model, one ``AgentIO`` carrying a telemetry session) —
    minus Chroma and the browser, which the coding arm has no action for.
    """
    config = ConnectorConfig()
    connector_llm = ConnectorLLM(config)
    connector_llm.set_model(model)
    connector_search = ConnectorSearchXNG(config)
    connector_http = ConnectorHttp(config)

    telemetry = TelemetrySession(enabled=True, mandate=mandate,
                                 correlation_id=f"codebench_{model}_{workdir.name}", trace_path=None)
    # Shipped limits (idea_dag_settings.json's sandbox_* keys), re-rooted at the workdir this
    # container was actually handed — the JSON default is /work, but --workdir is authoritative.
    limits = replace(sandbox_config(), workdir_root=str(workdir))
    sandbox = SandboxConnector(workdir, connector_config=config, limits=limits,
                               connector_search=connector_search)
    # ONE sandbox for the whole run, carried on the AgentIO (which also attaches telemetry to it):
    # every leaf — the compiled-code loop's own dispatch and any sandbox leaf action — then works
    # in the same workdir, so a file one leaf writes is there for the next.
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=None,
        telemetry=telemetry, collection_name="codebench", connector_sandbox=sandbox,
    )
    return sandbox, agent_io, limits, telemetry




#: Arms that can be driven inside the codebench container. ``compiled`` is the historical one;
#: the rest exist so a closed-environment task can be compared ACROSS arms. Before this the
#: container ran only the compiled scaffold, so a sandbox task was measurable on one arm and
#: could not support any claim about the DAG versus a linear agent.
CONTAINER_ARMS = ("compiled", "graph", "sequential", "sequential_react", "langgraph_react")


def _engine_settings(arm: str) -> Dict[str, Any]:
    """Settings for the native-DAG arms, with the sandbox pack armed.

    The pack is opt-in by design (``proximity is not authorization``); a container whose entire
    purpose is manipulating its own workdir is the case it was built for. ``sequential`` is the
    chain-forced control: same engine, siblings never run concurrently.
    """
    from agent.app.idea_dag_settings import load_idea_dag_settings

    settings = dict(load_idea_dag_settings())
    settings["tools_sandbox_pack_enabled"] = True
    if arm == "sequential":
        settings["allow_execute_all_children"] = False
    return settings


async def _run_native(arm: str, mandate: str, model: str, agent_io: Any) -> Dict[str, Any]:
    """Drive the native DAG engine against the workdir."""
    from agent.app.idea_engine import IdeaDagEngine

    settings = _engine_settings(arm)
    engine = IdeaDagEngine(io=agent_io, settings=settings, model_name=model)
    output = await engine.run(mandate, max_steps=int(settings.get("max_steps") or 25),
                              fail_soft=True)
    return {"arm": arm, "answer": str(output.get("final_deliverable") or ""),
            "success": bool(output.get("success")), "graph": output.get("graph")}


async def _run_flat(arm: str, mandate: str, model: str, agent_io: Any) -> Dict[str, Any]:
    """Drive one of the linear arms against the workdir."""
    if arm == "sequential_react":
        from agent.app.testing.execution_sequential import _run_react

        answer = await _run_react(agent_io, mandate, model, max_steps=25, max_tokens=4096)
    else:
        from agent.app.langgraph_solver import LangGraphSolver

        solver = LangGraphSolver(
            connector_llm=agent_io.connector_llm, connector_search=agent_io.connector_search,
            connector_http=agent_io.connector_http, connector_chroma=None,
            connector_browser=None, model_name=model, collection_name="codebench",
        )
        # The solver builds its own AgentIO; hand it the workdir this container was given, or the
        # arm advertises no file actions and the comparison is against a crippled opponent.
        solved = await solver.solve(mandate, connector_sandbox=agent_io.connector_sandbox)
        answer = str(solved.get("final_deliverable") or "") if isinstance(solved, dict) else str(solved)
    return {"arm": arm, "answer": answer, "success": bool(answer)}


async def run_task(model: str, task_dir: Path, workdir: Path, arm: str = "compiled") -> Dict[str, Any]:
    """Copy the fixture, load the plan, and execute it against ``workdir``.

    :param arm: One of :data:`CONTAINER_ARMS`. Only ``compiled`` consumes ``plan.json``; the
        other arms are handed the same prompt and the same workdir and must plan for themselves,
        which is the whole point of comparing them.
    """
    copied = copy_starter_fixture(task_dir, workdir)
    plan = load_plan(task_dir)
    mandate = read_prompt(task_dir) or plan.get("aggregation", "")
    sandbox, agent_io, limits, telemetry = build_context(model, workdir, mandate)
    _logger.info(f"codebench: arm={arm} model={model} task_dir={task_dir} workdir={workdir} "
                 f"starter_files={copied} leaves={len(plan.get('leaves') or [])}")
    try:
        if arm in ("graph", "sequential"):
            return await _run_native(arm, mandate, model, agent_io)
        if arm in ("sequential_react", "langgraph_react"):
            return await _run_flat(arm, mandate, model, agent_io)
        return await run_compiled_code_plan(plan, model, sandbox, agent_io, config=limits)
    finally:
        telemetry.finish(success=True)
        try:
            await agent_io.connector_http.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 — closing a never-opened session is not an error
            pass


#: Optional instrumentation drop, read back host-side by ``codebench/score_and_record.py`` (from
#: the UNFILTERED ``raw/`` dir) to populate the results schema's ``sandbox_actions_count`` column.
#: Nothing depends on it — that reader degrades to ``null`` — so writing it is strictly best effort.
RUN_SUMMARY_FILENAME = ".codebench_run_summary.json"


def write_run_summary(workdir: Path, result: Dict[str, Any]) -> None:
    """Drop the optional run-summary JSON into ``workdir``. Never raises."""
    payload = {
        "sandbox_actions_count": result.get("actions_count", 0),
        "submit_check": result.get("submit_check"),
        "leaves": {
            leaf_id: {"finished": record.get("finished"), "actions": record.get("actions"),
                      "outcome": str(record.get("outcome") or "")[:500]}
            for leaf_id, record in (result.get("leaves") or {}).items()
        },
    }
    try:
        (workdir / RUN_SUMMARY_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        _logger.warning(f"could not write {RUN_SUMMARY_FILENAME}: {exc}")


def main(argv=None) -> int:
    logging.basicConfig(level=os.environ.get("CODEBENCH_LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run one codebench coding task with the compiled scaffold.")
    ap.add_argument("--model", required=True, help="runtime (cheap) model name, e.g. gemma2:2b")
    ap.add_argument("--task-dir", required=True, type=Path, help="read-only task fixture (/task)")
    ap.add_argument("--workdir", required=True, type=Path, help="writable sandbox workdir (/work)")
    ap.add_argument("--arm", default="compiled", choices=CONTAINER_ARMS,
                    help="which agent architecture to run against the workdir")
    args = ap.parse_args(argv)

    try:
        result = asyncio.run(run_task(args.model, args.task_dir, args.workdir, arm=args.arm))
    except Exception as exc:  # noqa: BLE001
        # Exit 0 regardless: the harness grades the extracted /work, and a non-zero exit here would
        # be indistinguishable from the outer 900s timeout's own failure code.
        _logger.error(f"codebench: run failed: {exc}", exc_info=True)
        print(f"codebench: FAILED ({exc})")
        return 0

    write_run_summary(args.workdir, result)
    print("codebench: run complete")
    # ``summarize_run`` reads the compiled loop's own result shape; the other arms return an
    # answer/success pair, so only the compiled arm gets its detailed summary.
    if args.arm == "compiled":
        print(summarize_run(result))
    else:
        print(f"arm={result.get('arm')} success={result.get('success')} "
              f"answer_chars={len(str(result.get('answer') or ''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
