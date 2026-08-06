"""The one-off Python calculator leaf action (``run_python``).

Two things are pinned here, and the second matters more than the first:

1. A model that has the facts can COMPUTE with them: a snippet dispatches, its printed output comes
   back, and a snippet that crashes comes back as a readable failed observation instead of sinking
   the leaf.
2. The capability stays CLOSED unless three independent things are opted into (pack installed,
   name allowed, sandbox on the run's ``AgentIO``) — the same gate ``sandbox_tools`` uses, shared
   rather than re-implemented. Installing the calculator must NOT hand out file/shell access, and
   no existing run changes at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.app.connector_sandbox import SandboxConnector
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.config import SandboxActionConfig
from agent.app.idea_policies.extra_actions import CalculatorToolPack, SandboxToolPack
from agent.app.idea_policies.extra_actions.pack import ExtraActionPack


class _StubIO:
    telemetry = None
    connector_chroma = None

    def __init__(self, sandbox=None):
        self.connector_sandbox = sandbox

    def set_telemetry(self, telemetry):
        return None


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    return root


@pytest.fixture()
def sandbox(workdir: Path) -> SandboxConnector:
    return SandboxConnector(workdir, limits=SandboxActionConfig(workdir_root=str(workdir)))


def _engine(sandbox=None, *, allowed=None, install=True, allow=True, settings=None) -> IdeaDagEngine:
    base = {"allow_unscored_selection": True, "min_score_threshold": 0.0}
    base.update(settings or {})
    if allowed is not None:
        base["allowed_actions"] = allowed
    engine = IdeaDagEngine(io=_StubIO(sandbox), settings=base)
    if install:
        engine.install_action_pack(CalculatorToolPack(settings=engine.settings), allow=allow)
    return engine


async def _run(engine: IdeaDagEngine, graph: IdeaDag, details: dict, action: str = "run_python") -> dict:
    node = graph.add_child(
        graph.root_id(), f"leaf {action}",
        details={DetailKey.ACTION.value: action, **details},
    )
    return await engine._execute_action(graph, graph.root_id(), node.node_id)


# --- it computes --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_working_snippet_dispatches_and_returns_its_printed_output(sandbox):
    """The motivating case: the facts are already gathered, the model just needs the argmax."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")
    code = (
        "peaks = {'Ben Nevis': 1345, 'Snowdon': 1085, 'Scafell Pike': 978}\n"
        "print(max(peaks, key=peaks.get))"
    )

    result = await _run(engine, graph, {"code": code})

    assert result["action"] == "run_python", "the registry must dispatch by name, not fall back to THINK"
    assert result["success"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "Ben Nevis"
    assert "Ben Nevis" in result["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["code", "script", "source", "snippet", "python", "expression"])
async def test_the_code_slot_accepts_the_synonyms_a_weak_model_reaches_for(sandbox, key):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {key: "print(2 + 40)"})

    assert result["success"] is True and result["stdout"].strip() == "42"


@pytest.mark.asyncio
async def test_a_bare_expression_is_printed_for_the_model(sandbox):
    """``max(a, b)`` is unambiguously "tell me this value"; running it unwrapped prints nothing."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "max(8848, 8611, 8586)"})

    assert result["success"] is True
    assert result["stdout"].strip() == "8848"


@pytest.mark.asyncio
async def test_a_snippet_that_already_prints_is_not_double_printed(sandbox):
    """``print(...)`` is itself a valid expression, so a naive auto-print returns "42\\nNone"."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "print(2 + 40)"})

    assert result["stdout"].strip() == "42"
    assert "None" not in result["stdout"]


@pytest.mark.asyncio
async def test_a_statement_is_never_rewritten_into_a_syntax_error(sandbox):
    """The auto-print is gated on ``compile(..., 'eval')``, so real code runs verbatim."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "total = 3 + 4\nprint(total)"})

    assert result["success"] is True and result["stdout"].strip() == "7"


@pytest.mark.asyncio
async def test_fenced_and_indented_code_still_runs(sandbox):
    """Both shapes a model emits inside a JSON slot: a markdown fence, and a uniformly indented
    block (whose SECOND line would otherwise raise ``IndentationError``)."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    fenced = await _run(engine, graph, {"code": "```python\nprint(6 * 7)\n```"})
    indented = await _run(engine, graph, {"code": "    values = [1, 2, 3]\n    print(sum(values))"})
    as_lines = await _run(engine, graph, {"code": ["a = 10", "print(a // 3)"]})

    assert fenced["success"] is True and fenced["stdout"].strip() == "42"
    assert indented["success"] is True and indented["stdout"].strip() == "6"
    assert as_lines["success"] is True and as_lines["stdout"].strip() == "3"


@pytest.mark.asyncio
async def test_a_snippet_that_prints_nothing_says_so(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "x = 1 + 1"})

    assert result["success"] is True
    assert "print(" in result["output"], "an empty result must tell the model what it forgot"


# --- failures are observations, not exceptions ------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, needle",
    [
        ("print(1/0)", "ZeroDivisionError"),
        ("print(undefined_name)", "NameError"),
        ("def broken(:\n    pass", "SyntaxError"),
    ],
)
async def test_a_crashing_snippet_returns_a_clean_failure(sandbox, code, needle):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": code})

    assert result["action"] == "run_python"
    assert result["success"] is False
    assert result["exit_code"] not in (0, None)
    assert needle in result["stderr"], "the traceback is what the model needs to fix its snippet"


@pytest.mark.asyncio
async def test_missing_code_detail_is_an_observation_not_an_exception(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    empty = await _run(engine, graph, {})
    blank = await _run(engine, graph, {"code": "   "})

    assert empty["success"] is False and "code" in empty["error"]
    assert blank["success"] is False


@pytest.mark.asyncio
async def test_a_runaway_loop_is_killed_by_the_calculator_timeout(sandbox):
    """The bound is the calculator's own (tighter) knob, not the coding arm's 15s ``run_python``."""
    engine = _engine(sandbox, settings={"sandbox_calculator_timeout_seconds": 1})
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "while True:\n    pass"})

    assert result["success"] is False
    assert result["timed_out"] is True
    assert "timed out" in result["error"]


def test_the_calculator_timeout_is_tighter_than_the_coding_arms_run_python():
    limits = SandboxActionConfig()

    assert limits.calculator_timeout_seconds < limits.run_python_timeout_seconds


# --- confinement is the connector's, unchanged -------------------------------------------------


@pytest.mark.asyncio
async def test_a_py_path_outside_the_workdir_is_refused(sandbox, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("print('SECRET')\n", encoding="utf-8")
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": str(outside)})

    assert result["success"] is False
    assert "SECRET" not in str(result)


@pytest.mark.asyncio
async def test_a_workdir_py_file_runs_as_a_file_not_as_an_expression(sandbox, workdir):
    """``calc.py`` also PARSES as an expression; auto-printing it would break the connector's own
    file mode (which a run with both packs installed relies on)."""
    (workdir / "calc.py").write_text("print(sum(range(5)))\n", encoding="utf-8")
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "calc.py"})

    assert result["success"] is True
    assert result["stdout"].strip() == "10"


# --- the capability stays closed unless opted into ---------------------------------------------


def test_no_default_registry_or_pack_exposure():
    """Nothing installs the calculator by default, and it is deliberately not part of the curated
    web-research ``ExtraActionPack`` — those are read-only public APIs, this runs a subprocess."""
    engine = IdeaDagEngine(io=_StubIO())

    for name in CalculatorToolPack().names():
        assert not engine.actions.has(name), f"{name} must not be registered by default"
        assert name not in engine.settings.get("allowed_actions", [])
    assert not set(CalculatorToolPack().names()) & set(ExtraActionPack().names())


def test_installing_the_calculator_does_not_hand_out_file_or_shell_access():
    """The whole point of the second pack: "let it compute" must not imply "let it write files"."""
    engine = _engine(None)

    assert engine.actions.has("run_python")
    for name in SandboxToolPack().names():
        assert not engine.actions.has(name), f"{name} leaked in with the calculator"
        assert name not in engine.settings.get("allowed_actions", [])
    assert not set(CalculatorToolPack().names()) & set(SandboxToolPack().names())


@pytest.mark.asyncio
async def test_registered_but_not_allowed_falls_back_to_think(sandbox, workdir):
    engine = _engine(sandbox, allowed=["think", "search"], allow=False)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "open('pwned.txt', 'w').write('x')"})

    assert result["action"] == IdeaActionType.THINK.value
    assert not (workdir / "pwned.txt").exists()


@pytest.mark.asyncio
async def test_never_registered_is_unreachable(sandbox, workdir):
    """Even if the name is somehow allowed, an uninstalled action cannot run."""
    engine = _engine(sandbox, allowed=["think", "run_python"], install=False)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, {"code": "open('pwned.txt', 'w').write('x')"})

    assert result["action"] == IdeaActionType.THINK.value
    assert not (workdir / "pwned.txt").exists()


@pytest.mark.asyncio
async def test_no_sandbox_on_the_run_means_nothing_executes(tmp_path):
    """The gate that survives every mistake above it: a run whose AgentIO carries no sandbox runs
    no code, whatever the plan says."""
    engine = _engine(None)
    graph = IdeaDag(root_title="root")
    target = tmp_path / "pwned.txt"

    result = await _run(engine, graph, {"code": f"open({str(target)!r}, 'w').write('x')"})

    assert result["action"] == "run_python"
    assert result["success"] is False
    assert "no sandbox" in result["error"]
    assert not target.exists()


# --- it is describable to the model ------------------------------------------------------------


def test_the_calculator_describes_itself_for_the_prompt():
    for cls in CalculatorToolPack.ACTION_CLASSES:
        assert cls.menu_description(), f"{cls.__name__} has no description"
        assert cls.args_hint, f"{cls.__name__} has no args hint"
        assert cls.menu_line().startswith(f"- {cls.name}: details=")


@pytest.mark.asyncio
async def test_the_allowed_calculator_is_described_in_the_expansion_prompt(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    prompt = engine.expansion._build_messages(graph, graph.get_node(graph.root_id()))[0]["content"]

    assert "- run_python: details={code}. Run a short Python snippet" in prompt


@pytest.mark.asyncio
async def test_the_computed_number_reaches_the_finalize_stage(sandbox):
    """A calculator the final answer never sees would be theatre: the leaf-result collector
    finalize falls back on is action-agnostic, so the snippet's stdout rides into the synthesis."""
    from agent.app.idea_finalize import _collect_leaf_results_fallback

    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    await _run(engine, graph, {"code": "print(max(4148, 3642, 3585))"})
    collected = _collect_leaf_results_fallback(graph)

    assert any(entry["action"] == "run_python" and "4148" in str(entry["result"])
               for entry in collected)


def test_an_untouched_engine_is_byte_identical_without_the_pack():
    """Nothing about an existing run changes: same registry, same allowed actions, same menu."""
    plain = IdeaDagEngine(io=_StubIO())
    also_plain = IdeaDagEngine(io=_StubIO())
    also_plain.install_action_pack(CalculatorToolPack(settings=also_plain.settings), allow=False)

    assert sorted(plain.actions.names() + ["run_python"]) == sorted(also_plain.actions.names())
    assert plain.settings.get("allowed_actions") == also_plain.settings.get("allowed_actions")
    assert plain.actions.menu_lines(plain.settings.get("allowed_actions")) == \
        also_plain.actions.menu_lines(also_plain.settings.get("allowed_actions"))
