"""Compiled-graph agent for CODE tasks — sibling of ``execution_compiled.py``, wired as the
``graph_compiled_code`` variant.

Same thesis, different domain: an expensive model authors a static DAG plan offline, and the cheap
runtime model only EXECUTES it. The wave/topology machinery is shared verbatim (``compiled_plan``
is domain-agnostic); only two things change:

  * the leaf's action vocabulary — every action in ``sandbox_dispatch.ACTION_NAMES`` (plus
    ``finish``) against a ``connector_sandbox.SandboxConnector`` workdir, instead of
    ``search|visit|finish`` against the web. The dispatch itself is shared with the native
    engine's ``LeafAction`` pack (see :mod:`agent.app.sandbox_dispatch`) so the two cannot drift
    on which argument aliases they accept;
  * the composition step — ``agg_mode: "sandbox_submit"`` runs :func:`_compose_submit`, which only
    CHECKS that the plan's declared deliverable files exist. It deliberately does not score
    correctness: grading happens later, in a separate fresh container that re-runs the canonical
    tests against the extracted submission (``codebench/run_grade.sh``), because an
    agent with real code execution can trivially fake a self-reported pass.

``execution_compiled.py`` is untouched — the QA benchmark path must stay byte-for-byte identical.

Two deliberate divergences from the web sibling, both because leaves here share ONE mutable
workdir rather than each resolving an independent fact:
  * leaves in a wave run SEQUENTIALLY by default (``IDEA_TEST_COMPILED_CODE_CONCURRENCY=1``);
    parallel leaves writing the same tree clobber each other. Wave ORDER is still what the plan's
    ``depends_on`` edges say.
  * there is no free-text aggregation LLM call. The deliverable is the workdir; the "aggregation"
    is a deterministic summary, so a weak model can never talk the run into looking finished.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.agent_io import AgentIO
from agent.app.connector_sandbox import SandboxConnector
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.config import SandboxActionConfig
from agent.app.sandbox_dispatch import ACTION_NAMES, dispatch_sandbox_action
from agent.app.telemetry import TelemetrySession
from agent.app.testing import json_telemetry as _json_telemetry
from agent.app.testing.compiled_plan import (
    plan_structure,
    substitute_deps,
    topological_waves,
    validate_plan,
)
from agent.app.testing.execution import _empty_graph, _parse_decision
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability

_logger = logging.getLogger(__name__)

_CODE_LEAF_SYSTEM = (
    "You complete ONE coding subtask inside a sandbox working directory. Work one step at a "
    "time: think, then call exactly one tool. Tools:\n"
    "- write_file(path, content): create or overwrite a file. Path is relative to the workdir.\n"
    "- read_file(path): read a file back.\n"
    "- list_dir(path): list a directory ('' means the workdir root).\n"
    "- patch_file(path, old, new): replace the first exact occurrence of 'old' with 'new'.\n"
    "- run_python(code): run Python in the workdir (inline code, or the path of a .py file).\n"
    "- run_pytest(path): run pytest on a test file or directory; returns the pass/fail output.\n"
    "- search_web(query): web search; returns titles+URLs+snippets.\n"
    "- find_files(name): list files matching a filename glob, e.g. '*.py'.\n"
    "- head_file(path, lines): show the first N lines of a file (default 10).\n"
    "- count_lines(path, pattern): count a file's lines, or only the lines matching 'pattern'.\n"
    "- word_count(path): count the words in a file.\n"
    "- disk_usage(path): report the size of a file or directory.\n"
    "- finish(summary): stop, and state what you produced.\n"
    "Rules: write COMPLETE, runnable code — never placeholders, '...' or 'TODO'. After writing "
    "code, RUN it (run_pytest when a test file exists) and fix what fails before finishing; do "
    "not claim success you have not observed. Never edit or delete a test file to make it pass. "
    "Each step return ONLY JSON: {\"thought\": \"...\", \"action\": \"write_file|read_file|"
    "list_dir|run_python|run_pytest|patch_file|search_web|find_files|head_file|count_lines|"
    "word_count|disk_usage|finish\", \"args\": {...}}."
)

# Arg aliases a weak model reaches for. The plan authors the instruction, but the model authors
# the JSON, so accept the near-misses instead of burning a step on "INVALID ARGS". The sandbox
# actions' own alias tables live in ``sandbox_dispatch`` (shared with the native engine's pack);
# only ``finish`` is this loop's own verb, so only its aliases are listed here.
_SUMMARY_KEYS = ("summary", "answer", "result", "message", "text")

#: The sandbox vocabulary — what actually counts as a sandbox action (``finish``, an unknown verb
#: and a failed LLM call all touch nothing, so none of them inflate ``sandbox_actions_count``).
#: The full shared set: inside the codebench container, running the agent's own code IS the point,
#: so unlike the native engine's opt-in pack this arm advertises everything the dispatcher offers.
_SANDBOX_ACTIONS = ACTION_NAMES


def _arg(args: Dict[str, Any], *keys: str) -> str:
    """First non-empty value among ``keys`` in ``args``, as a string ('' when none match)."""
    if not isinstance(args, dict):
        return ""
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value)
        if text.strip():
            return text
    return ""


def _leaf_steps() -> int:
    """Per-leaf action budget (``IDEA_TEST_COMPILED_CODE_LEAF_STEPS``, default 10).

    Higher than the web sibling's 4: a coding leaf's proven loop is write -> run -> read the
    failure -> patch -> re-run, which needs several steps before it can honestly finish.
    """
    try:
        return max(1, int(os.environ.get("IDEA_TEST_COMPILED_CODE_LEAF_STEPS", "10")))
    except ValueError:
        return 10


def _max_tokens() -> int:
    """Completion budget for one leaf decision (``IDEA_TEST_COMPILED_CODE_MAX_TOKENS``, default
    2048) — a ``write_file`` action carries a whole source file inside its JSON args, so the web
    sibling's 700 would truncate the file rather than the reasoning."""
    try:
        return max(256, int(os.environ.get("IDEA_TEST_COMPILED_CODE_MAX_TOKENS", "2048")))
    except ValueError:
        return 2048


def _obs_chars() -> int:
    """Chars of one action's output fed back as an observation
    (``IDEA_TEST_COMPILED_CODE_OBS_CHARS``, default 2000)."""
    try:
        return max(200, int(os.environ.get("IDEA_TEST_COMPILED_CODE_OBS_CHARS", "2000")))
    except ValueError:
        return 2000


def _concurrency() -> int:
    """Leaves run in parallel WITHIN a wave (``IDEA_TEST_COMPILED_CODE_CONCURRENCY``, default 1).

    Default 1 on purpose: unlike the web arm's independent facts, code leaves share one mutable
    workdir, so two of them writing concurrently can clobber each other's files. Raise it only for
    a plan whose same-wave leaves provably touch disjoint paths.
    """
    try:
        return max(1, int(os.environ.get("IDEA_TEST_COMPILED_CODE_CONCURRENCY", "1")))
    except ValueError:
        return 1


def _observation(result: Dict[str, Any], obs_chars: int) -> str:
    """Render one action result as the observation text the next step sees."""
    if not isinstance(result, dict):
        return "ERROR: action returned nothing"
    body = str(result.get("output") or "")
    if result.get("ok"):
        return body[:obs_chars] or "OK"
    error = str(result.get("error") or "action failed")
    return (f"ERROR: {error}\n{body}" if body else f"ERROR: {error}")[:obs_chars]


def _syntax_check_written_python(sandbox: SandboxConnector, result: Dict[str, Any]) -> Dict[str, Any]:
    """Downgrade a successful ``write_file`` to a failure when the ``.py`` it just wrote does not
    compile.

    Root cause this guards against (see ``docs/handoffs/AGENT_FAILURE_MODES_2026-08-10.md``): a
    weak local model's ``content`` string can be syntactically VALID JSON — so ``write_file``'s own
    UTF-8/byte checks pass — while being semantically broken Python (a collapsed triple-quote, or
    the whole file landing as one line of literal ``\\n`` text instead of real newlines, both
    observed verbatim in real qwen2.5:14b codebench submissions). Nothing upstream of this ever
    reads the file back, so the leaf loop silently reports success and frequently exhausts its
    entire step budget on a submission that was dead on arrival. Reading the file straight off disk
    (rather than re-deriving ``content`` from ``args``) uses the exact bytes that were actually
    persisted, so this can't drift from whatever alias the dispatcher resolved.
    """
    if result.get("action") != "write_file" or not result.get("ok"):
        return result
    path = result.get("path") or ""
    if not str(path).endswith(".py"):
        return result
    try:
        text = (sandbox.workdir / path).read_text(encoding="utf-8")
    except OSError:
        return result
    try:
        compile(text, path, "exec")
    except SyntaxError as exc:
        return {
            **result,
            "ok": False,
            "error": f"{path} does not compile: SyntaxError: {exc.msg} (line {exc.lineno})",
        }
    return result


async def _dispatch_action(sandbox: SandboxConnector, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run ONE sandbox action for the leaf loop. Never raises: an unknown action or a bad
    argument comes back as a failed result the model can read and correct.

    The translation itself lives in :func:`agent.app.sandbox_dispatch.dispatch_sandbox_action`,
    shared with the native engine's ``LeafAction`` pack; this wrapper only adds ``finish`` — a
    loop verb, not a sandbox action — to the vocabulary an unknown verb is measured against. It
    also runs the codebench-specific ``.py`` syntax check (see
    :func:`_syntax_check_written_python`) — deliberately NOT in the shared dispatcher, which is
    also used by the native engine's general-purpose (language-agnostic) sandbox pack.
    """
    result = await dispatch_sandbox_action(sandbox, action, args,
                                           vocabulary=tuple(_SANDBOX_ACTIONS) + ("finish",))
    return _syntax_check_written_python(sandbox, result)


async def _run_code_leaf(sandbox: SandboxConnector, agent_io: AgentIO, instruction: str, expect: str,
                         model_name: str, leaf_steps: int, obs_chars: int, max_tokens: int,
                         llm_timeout: Optional[float] = None) -> Dict[str, Any]:
    """Drive ONE plan leaf with a small bounded ReAct loop over the sandbox actions.

    Returns a compact record: the model's own closing summary (or a harness-written one when the
    budget ran out), the per-step transcript, how many sandbox actions were actually executed, and
    the files the sandbox OBSERVED this leaf write (``files_written`` — harness-verified, unlike
    the summary, which is whatever the model chose to say). The workdir — not this text — is the
    deliverable; the record exists so a run can be read back and so a dependent leaf can be handed
    something real (see :func:`_dep_text`).

    :param llm_timeout: Wall-clock bound on ONE decision call (``None`` = unbounded). Without it a
        hung backend stalls a step forever inside the outer container budget, making no progress
        against the step budget at all.
    """
    sandbox.reset_leaf_budget()
    scratchpad: List[str] = []
    steps: List[Dict[str, Any]] = []
    task = f"{instruction}\n\nDone when: {expect}" if expect else instruction
    outcome = ""
    finished = False

    for step in range(leaf_steps):
        history = "\n\n".join(scratchpad[-5:]) if scratchpad else "(no actions yet)"
        messages = [
            {"role": "system", "content": _CODE_LEAF_SYSTEM},
            {"role": "user", "content": f"SUBTASK:\n{task}\n\nYOUR STEPS SO FAR:\n{history}\n\n"
                                        f"Return the next step as JSON."},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name,
                                             temperature=0.1, max_tokens=max_tokens)
        try:
            raw = await agent_io.query_llm(payload, model_name=model_name,
                                           timeout_seconds=llm_timeout)
        except Exception as exc:  # noqa: BLE001 — a starved/failed/timed-out completion ends the leaf
            _logger.warning(f"code leaf LLM call failed: {exc}")
            steps.append({"step": step + 1, "action": "(llm_error)", "ok": False,
                          "observation": str(exc) or type(exc).__name__})
            break
        # Fence-tolerant on purpose (shared with the naive arm's ``_parse_decision``): plenty of
        # instruction-tuned models wrap the object in ```json ... ``` even when told to return only
        # JSON, and treating that as a failed step burns the whole budget on zero real actions.
        decision = _parse_decision(raw or "")
        parsed_ok = isinstance(decision, dict) and bool(decision)
        if not isinstance(decision, dict):
            decision = {}
        _json_telemetry.record(model_name, raw, True, parsed_ok, phase="compiled_code_leaf")
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        if action == "finish":
            outcome = _arg(args, *_SUMMARY_KEYS) or "finished"
            finished = True
            steps.append({"step": step + 1, "action": "finish", "ok": True, "observation": outcome[:obs_chars]})
            break

        result = await _dispatch_action(sandbox, action, args)
        obs = _observation(result, obs_chars)
        steps.append({
            "step": step + 1, "action": action or "(none)", "ok": bool(result.get("ok")),
            "path": result.get("path"), "observation": obs,
        })
        scratchpad.append(f"STEP {step + 1}: action={action} args={json.dumps(args)[:200]}\n"
                          f"observation={obs}")

    if not finished:
        last_ok = [s for s in steps if s.get("ok")]
        outcome = (f"step budget exhausted after {len(steps)} steps; "
                   f"last successful action: {last_ok[-1]['action'] if last_ok else 'none'}")
    return {
        "outcome": outcome,
        "finished": finished,
        "steps": steps,
        "actions": len([s for s in steps if s.get("action") in _SANDBOX_ACTIONS]),
        # Read BEFORE the next leaf's reset_leaf_budget() clears it — see _dep_text.
        "files_written": sandbox.files_written_this_leaf,
    }


#: Prefix stamped on a dependent leaf's substituted text when the upstream leaf never called
#: ``finish`` (budget exhausted, or the leaf raised). Without it the harness's own bookkeeping
#: ("step budget exhausted after 10 steps; last successful action: write_file") gets spliced into
#: the downstream instruction as if it described what was built.
UPSTREAM_INCOMPLETE_MARKER = "[UPSTREAM STEP DID NOT COMPLETE]"


def _dep_text(record: Dict[str, Any]) -> str:
    """What a dependent leaf actually sees in place of its ``{dep_id}`` placeholder.

    ``substitute_deps`` is a literal string replace, so whatever this returns lands verbatim inside
    the downstream instruction. The model's own ``finish(summary)`` is therefore never enough on
    its own: nothing forces it to name the files or symbols it produced, and when the leaf did not
    finish there IS no summary — only harness prose. So the text is composed of two parts:

      * the summary (marked with :data:`UPSTREAM_INCOMPLETE_MARKER` when the leaf did not finish,
        so the downstream prompt reads it as a failure notice rather than as content), and
      * the sandbox's OWN record of the files that leaf wrote, which is true regardless of what
        the model said — so a downstream leaf can still import from the real module.
    """
    summary = str(record.get("outcome") or "").strip() or "(no summary)"
    text = summary if record.get("finished") else f"{UPSTREAM_INCOMPLETE_MARKER} {summary}"
    files = [str(f) for f in (record.get("files_written") or [])]
    if files:
        text += f"\n\n[Files written by this step: {', '.join(files)}]"
    return text


# --- Composition: confirm the deliverable exists; do NOT grade it -------------------------------
def _compose_submit(sandbox: SandboxConnector, composition: Dict[str, Any]) -> Dict[str, Any]:
    """``op: "submit_files"`` — report which of the plan's declared deliverable files exist.

    This is a SUBMISSION CHECK, not a score. It never runs the agent's code, never trusts a
    self-reported pytest result, and never emits a pass/fail verdict on the task: correctness is
    decided later by ``codebench/run_grade.sh``, in a fresh container that re-runs the
    canonical tests against the extracted workdir. Its only job is a clean end-of-run summary the
    entrypoint can log (and a harness-side signal that the agent produced nothing at all).
    """
    files = [str(f).strip() for f in (composition.get("files") or []) if str(f).strip()]
    present: List[Dict[str, Any]] = []
    missing: List[str] = []
    for rel in files:
        if sandbox.exists(rel):
            present.append({"path": rel, "bytes": sandbox.file_size(rel)})
        else:
            missing.append(rel)
    return {
        "op": "submit_files",
        "files": files,
        "present": present,
        "missing": missing,
        "all_present": bool(files) and not missing,
        "summary": (f"{len(present)}/{len(files)} declared files present"
                    + (f"; missing: {', '.join(missing)}" if missing else "")),
    }


_CODE_COMPOSERS = {"submit_files": _compose_submit}


def _compose(sandbox: SandboxConnector, composition: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Dispatch a plan's ``composition`` to its composer, or ``None`` when there is nothing to run
    (absent/malformed composition, unrecognized op) or the composer itself failed. Mirrors
    ``execution_compiled._compose``: a composition problem is a logged degradation, never a crash."""
    if not isinstance(composition, dict):
        return None
    op = str(composition.get("op") or "").strip().lower()
    composer = _CODE_COMPOSERS.get(op)
    if composer is None:
        _logger.info(f"code composition op {op!r} is not implemented; skipping submit check")
        return None
    try:
        return composer(sandbox, composition)
    except Exception as exc:  # noqa: BLE001 — a composer must never sink the run
        _logger.warning(f"code composition op '{op}' failed: {exc}")
        return None


def summarize_run(result: Dict[str, Any]) -> str:
    """One compact, human-readable block for the container log (``docker logs``) / deliverable."""
    lines = [f"leaves: {len(result.get('leaves') or {})}, "
             f"sandbox actions: {result.get('actions_count', 0)}"]
    for leaf_id, record in (result.get("leaves") or {}).items():
        status = "finished" if record.get("finished") else "budget-exhausted"
        lines.append(f"- {leaf_id} [{status}, {record.get('actions', 0)} actions]: "
                     f"{str(record.get('outcome') or '')[:200]}")
    check = result.get("submit_check")
    if isinstance(check, dict):
        lines.append(f"submit check: {check.get('summary')}")
    return "\n".join(lines)


async def run_compiled_code_plan(
    plan: Dict[str, Any],
    model_name: str,
    sandbox: SandboxConnector,
    agent_io: AgentIO,
    config: Optional[SandboxActionConfig] = None,
) -> Dict[str, Any]:
    """Execute a compiled DAG plan against ``sandbox`` and return a small result record.

    Leaves are grouped into dependency waves by ``compiled_plan.topological_waves`` (shared with
    the web arm, unchanged); each dependent leaf's ``{dep_id}`` placeholders are filled — via
    :func:`_dep_text`, so what lands there is the upstream summary PLUS the files the sandbox saw
    it write, and is explicitly marked when that leaf never finished. When the plan declares
    ``agg_mode: "sandbox_submit"`` (or any ``composition``), the declared files are checked for
    existence afterwards — see :func:`_compose_submit` for why that is not a grade.

    :param plan: The compiled plan dict (``leaves``/``aggregation``/``agg_mode``/``composition``).
    :param model_name: Runtime (cheap) model that executes the leaves.
    :param sandbox: Workdir-confined sandbox connector.
    :param agent_io: The LLM surface (same ``AgentIO`` the other execution variants use).
    :param config: Optional typed sandbox limits; when given they replace the connector's own, so
        the caller's config is the single source of truth for both.
    :returns: ``{"leaves": {...}, "submit_check": {...}|None, "plan_structure": {...},
        "waves": [[id, ...]], "aggregation": str, "actions_count": int}``.
    """
    if config is not None:
        sandbox.limits = config
    norm = validate_plan(plan)  # normalizes ids/deps and rejects cycles/missing deps
    leaves: List[Dict[str, Any]] = norm["leaves"]
    by_id = {leaf["id"]: leaf for leaf in leaves}
    waves = topological_waves(leaves)

    leaf_steps, obs_chars, max_tokens = _leaf_steps(), _obs_chars(), _max_tokens()
    llm_timeout = float(sandbox.limits.llm_call_timeout_seconds)
    sem = asyncio.Semaphore(_concurrency())
    records: Dict[str, Dict[str, Any]] = {}
    outcomes: Dict[str, str] = {}

    async def _guarded(leaf: Dict[str, Any]) -> None:
        async with sem:
            dep_results = {dep: outcomes.get(dep, "(upstream leaf produced nothing)")
                           for dep in leaf["depends_on"]}
            instruction = substitute_deps(leaf["instruction"], dep_results)
            try:
                record = await _run_code_leaf(sandbox, agent_io, instruction, leaf["expect"],
                                              model_name, leaf_steps, obs_chars, max_tokens,
                                              llm_timeout=llm_timeout)
            except Exception as exc:  # noqa: BLE001 — one bad leaf must not sink the run
                _logger.warning(f"compiled code leaf '{leaf['id']}' failed: {exc}")
                record = {"outcome": f"leaf failed: {exc}", "finished": False, "steps": [],
                          "actions": 0, "files_written": sandbox.files_written_this_leaf}
            records[leaf["id"]] = record
            outcomes[leaf["id"]] = _dep_text(record)

    for wave in waves:
        await asyncio.gather(*[_guarded(by_id[lid]) for lid in wave])

    composition = plan.get("composition") if isinstance(plan, dict) else None
    agg_mode = str((plan.get("agg_mode") if isinstance(plan, dict) else "") or "").strip().lower()
    submit_check = _compose(sandbox, composition) if (composition or agg_mode == "sandbox_submit") else None

    return {
        "leaves": records,
        "submit_check": submit_check,
        "plan_structure": plan_structure(plan),
        "waves": waves,
        "aggregation": norm["aggregation"],
        "actions_count": sum(int(r.get("actions") or 0) for r in records.values()),
    }


def sandbox_config() -> SandboxActionConfig:
    """Typed sandbox limits from the shipped settings (falls back to the dataclass defaults if the
    settings file cannot be read — the limits must never be the reason a run cannot start)."""
    try:
        return SandboxActionConfig.from_settings(load_idea_dag_settings())
    except Exception as exc:  # noqa: BLE001
        _logger.warning(f"could not load idea-DAG settings for sandbox limits ({exc}); using defaults")
        return SandboxActionConfig()


def _resolve_workdir(root: str, name: str) -> Path:
    """A writable per-run workdir under ``root``, falling back to a temp dir when ``root`` is not
    creatable (the configured default ``/work`` only exists inside the codebench container)."""
    try:
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        import tempfile

        return Path(tempfile.mkdtemp(prefix=f"{name}_"))


async def run_compiled_code_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm,
    connector_search,
    connector_http,
    connector_chroma,
    run_stamp: str,
    # Accepted for dispatch parity with every other variant (``testing/runner.py`` passes it
    # uniformly). Unused here: this arm records no JSONL trace of its own -- the sandboxed
    # run's per-leaf record lands in ``.codebench_run_summary.json`` instead.
    cell_tag: str = "",
    summarize_observability_func=summarize_observability,
    connector_browser=None,
) -> Dict[str, Any]:
    """Harness-shaped entrypoint for the ``graph_compiled_code`` variant — same signature and
    return shape as ``execution_compiled.run_compiled_execution``, so ``testing/runner.py``
    dispatches it exactly like the web arm.

    The container entrypoint (``agent.app.codebench_run_task``) does NOT go through here: it
    builds its own sandbox/connectors from ``/task`` and calls :func:`run_compiled_code_plan`
    directly. This adapter exists so the variant is runnable (and testable) from the normal test
    harness against a task module that ships ``get_compiled_plan()`` and, optionally,
    ``get_sandbox_fixture()`` starter files.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_graph_compiled_code_{run_stamp}"

    limits = sandbox_config()
    workdir = _resolve_workdir(limits.workdir_root, f"{test_id}_{run_stamp}")
    sandbox = SandboxConnector(workdir, connector_search=connector_search, limits=limits)

    fixture_fn = getattr(test_module.module, "get_sandbox_fixture", None)
    if callable(fixture_fn):
        try:
            for rel, content in (fixture_fn() or {}).items():
                await sandbox.write_file(str(rel), str(content))
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"could not materialize sandbox fixture for {test_id}: {exc}")
        sandbox.reset_leaf_budget()  # fixture files are the harness's writes, not the leaf's

    plan = None
    plan_fn = getattr(test_module.module, "get_compiled_plan", None)
    if callable(plan_fn):
        try:
            plan = plan_fn()
        except Exception as exc:  # noqa: BLE001
            _logger.error(f"get_compiled_plan() failed for {test_id}: {exc}")

    mandate = test_module.get_task_statement()
    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id,
                                 trace_path=None)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )
    sandbox.set_telemetry(telemetry)

    started = time.perf_counter()
    result: Dict[str, Any] = {}
    if plan is None:
        _logger.error(f"Test {test_id} has no compiled plan; graph_compiled_code cannot run.")
    else:
        try:
            result = await run_compiled_code_plan(plan, model_name, sandbox, agent_io, config=limits)
        except Exception as exc:  # noqa: BLE001
            _logger.error(f"Compiled code execution failed: {exc}", exc_info=True)

    deliverable = summarize_run(result) if result else ""
    output = {
        "final_deliverable": deliverable,
        "success": bool(result),
        "goal_achieved": None,
        "action_summary": "graph_compiled_code",
        "workdir": str(workdir),
        "submit_check": result.get("submit_check"),
        "plan_structure": result.get("plan_structure"),
        "sandbox_actions_count": result.get("actions_count", 0),
    }
    telemetry.finish(success=output["success"])
    observability = summarize_observability_func({"output": output}, telemetry, model_name)
    ended = time.perf_counter()

    return {
        "output": output,
        "graph": _empty_graph(),
        "observability": observability,
        "duration_seconds": round(max(0.0, ended - started), 2),
        "plan_source": "hand" if plan is not None else None,
        "leaves": result.get("leaves", {}),
        "telemetry": {
            "correlation_id": correlation_id,
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
        "telemetry_raw": telemetry.summary(),
    }
