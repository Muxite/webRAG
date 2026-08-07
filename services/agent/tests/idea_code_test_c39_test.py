"""
Adversarial offline checks for codebench task c39 (bugfix-template-renderer) -- no Docker, no
LLM.

Debug-task variant of idea_code_test_c01_test.py's pattern (independently re-derive ground
truth, check keystone/grading/plan shape, exercise materialize_task.py end-to-end) PLUS the
directions this task shape (two independent seeded bugs, one visible-obvious, one hidden-only-
latent; a hidden canonical superset) needs proven, exactly as its predecessor's own validator
did:
  (1) the shipped BUGGY starter genuinely fails Group A (the "obvious" cases, bug (1)) when
      actually run;
  (2) a targeted patch of ONLY bug (1) (the "obvious" fix) makes every VISIBLE case agree with
      ground truth -- i.e. it looks like a complete fix from the visible suite's point of view
      -- while every HIDDEN case (which exercises bug (2), a structurally independent defect
      the visible suite never touches at all) still disagrees with ground truth;
  (3) the fully correct joint patch (both bugs fixed) agrees with ground truth on EVERY case in
      the canonical (visible + hidden) superset;
  (4) the hidden cases are genuinely absent from the sandbox fixture (so the agent cannot just
      read them), while still being a strict superset built ON TOP of the visible content at
      the identical relpath (so the manifest-drop mechanism still works);
  (5) unlike its predecessor, this task has NO hang risk at all (every branch of the scanning
      loop advances its index by at least one character every iteration, for any input, under
      any combination of the two seeded bugs) -- checked directly here via a wall-clock bound
      on every single case, rather than assumed.

Ground-truth case data (template, values, expected outcome) is hand-derived independently here
(never copied from the task module's own reasoning) and then cross-checked against the actual
embedded test file content via plain substring presence, so there is no way for this file's
notion of "what the canonical suite checks" to silently drift from what it actually contains.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c39_bugfix_template_renderer as c39


# --------------------------------------------------------------------------------------------
# Ground truth: a hand-written reference implementation, independently coded here (never
# importing the task module's own buggy starter source as anything but a thing to EXECUTE and
# check against this reference).
# --------------------------------------------------------------------------------------------
def _independent_render(template: str, values: dict) -> str:
    out = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            if template[i:i + 2] == "{{":
                out.append("{")
                i += 2
                continue
            j = template.find("}", i)
            if j == -1:
                raise ValueError("unmatched open brace")
            inner = template[i + 1:j]
            if ":" in inner:
                name, default = inner.split(":", 1)
            else:
                name, default = inner, None
            if name in values:
                out.append(str(values[name]))
            elif default is not None:
                out.append(default)
            else:
                raise KeyError(name)
            i = j + 1
        elif ch == "}":
            if template[i:i + 2] == "}}":
                out.append("}")
                i += 2
            else:
                raise ValueError("unmatched close brace")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


class _Hang(Exception):
    """Raised when a wrapped call does not return within the wall-clock bound. This task is
    engineered to have NO input that can ever hang (every loop branch advances the scan index
    by >= 1 every iteration, for any input, under any combination of the two seeded bugs) --
    this bound exists purely to PROVE that claim rather than assume it, never because any
    variant tested below is expected to actually trip it."""


@contextmanager
def _wall_clock_limit(seconds: float = 2.0):
    def _handler(signum, frame):
        raise _Hang(f"did not return within {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _call_bounded(fn, *args, seconds: float = 2.0):
    with _wall_clock_limit(seconds):
        return fn(*args)


def _exec_render(source: str):
    import importlib.util

    spec = importlib.util.spec_from_loader("c39_template_render_under_test", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)  # noqa: S102 - trusted in-repo fixture
    return module.render


# Ground-truth case table. Each case is (template, values, expected) where expected is either
# a plain string (the render() must equal it) or one of the sentinel tuples below.
_KEY_ERROR = "KEY_ERROR"
_VALUE_ERROR = "VALUE_ERROR"

# Group A: visible, currently BROKEN on the starter by bug (1) alone.
_GROUP_A_CASES = [
    ("hi {name}!", {"name": "Ann"}, "hi Ann!"),
    ("{a}-{b}-{c}", {"a": "1", "b": "2", "c": "3"}, "1-2-3"),
    ("count={n}", {"n": 5}, "count=5"),
    ("hi {name:stranger}!", {"name": "Bo"}, "hi Bo!"),
    ("{missing}", {}, (_KEY_ERROR, "missing")),
    ("{a}middle{b}", {"a": "X", "b": "Y"}, "XmiddleY"),
]
# Bonus: visible, already correct on the starter, stay correct under any fix.
_BONUS_CASES = [
    ("hello world", {}, "hello world"),
    ("hi {name:stranger}!", {}, "hi stranger!"),
    ("bad {open", {}, (_VALUE_ERROR, None)),
]
# Group C: hidden-only, exercise bug (2) (the {{ / }} escaping path).
_HIDDEN_CASES = [
    ("{{not a placeholder}}", {}, "{not a placeholder}"),
    ("{{literal}} then {name}", {"name": "Q"}, "{literal} then Q"),
    ("{{{{}}}}", {}, "{{}}"),
    ("{{ok}}extra}", {}, (_VALUE_ERROR, None)),
]
_ALL_CASES = _GROUP_A_CASES + _BONUS_CASES + _HIDDEN_CASES


def _expected_outcome(template, values):
    """Compute ground truth via the independent reference, normalized to the same sentinel
    shape used in the case tables above."""
    try:
        return _independent_render(template, values)
    except KeyError as e:
        return (_KEY_ERROR, e.args[0])
    except ValueError:
        return (_VALUE_ERROR, None)


def _actual_outcome(fn, template, values, seconds=2.0):
    try:
        result = _call_bounded(fn, template, values, seconds=seconds)
        return result
    except _Hang:
        raise
    except KeyError as e:
        return (_KEY_ERROR, e.args[0] if e.args else None)
    except ValueError:
        return (_VALUE_ERROR, None)


def test_ground_truth_case_tables_are_internally_consistent():
    """Every hand-derived expected outcome above must actually match what the independent
    reference implementation computes -- i.e. the case tables are not just asserted by
    authorial confidence."""
    for template, values, expected in _ALL_CASES:
        assert _expected_outcome(template, values) == expected, template


def test_case_tables_correspond_to_what_the_embedded_test_file_actually_contains():
    """Cross-check: every case in the hand-derived tables above must be traceable to an actual
    literal in the embedded canonical test content, so this validator cannot silently drift
    from what the shipped fixture really checks."""
    visible_content = c39._TEST_FILE_CONTENT
    hidden_content = c39._HIDDEN_TEST_ADDENDUM
    canonical = c39.get_grading_payload()["tests"][c39.VISIBLE_TEST_PATH]
    assert canonical == c39._CANONICAL_TEST_FILE_CONTENT

    for template, _values, _expected in _GROUP_A_CASES + _BONUS_CASES:
        assert repr(template) in visible_content or template in visible_content, template
    for template, _values, _expected in _HIDDEN_CASES:
        assert repr(template) in hidden_content or template in hidden_content, template
        assert template not in visible_content, (
            f"hidden-only template {template!r} must not appear in the visible test content"
        )


def test_starter_module_genuinely_fails_every_group_a_case_when_run():
    buggy_render = _exec_render(c39._TEMPLATE_RENDER_PY_CONTENT)
    for template, values, expected in _GROUP_A_CASES:
        actual = _actual_outcome(buggy_render, template, values)
        assert actual != expected, f"{template!r} was expected to be BROKEN by bug (1)"


def test_starter_module_passes_bonus_cases_when_run():
    buggy_render = _exec_render(c39._TEMPLATE_RENDER_PY_CONTENT)
    for template, values, expected in _BONUS_CASES:
        actual = _actual_outcome(buggy_render, template, values)
        assert actual == expected, template


def test_starter_module_never_hangs_on_any_canonical_case():
    buggy_render = _exec_render(c39._TEMPLATE_RENDER_PY_CONTENT)
    for template, values, _expected in _ALL_CASES:
        try:
            _actual_outcome(buggy_render, template, values)
        except _Hang:
            pytest.fail(f"starter hung on {template!r} -- this task must never be able to hang")


def _obvious_fix_source() -> str:
    """Bug (1) fixed in isolation, bug (2) left in place -- the single most plausible
    near-miss: a model that spots the visibly-failing Group A cases and fixes exactly what
    those point at."""
    assert c39._TEMPLATE_RENDER_PY_CONTENT.count("inner = template[i:j]") == 1
    fixed = c39._TEMPLATE_RENDER_PY_CONTENT.replace(
        "inner = template[i:j]", "inner = template[i + 1:j]"
    )
    assert fixed != c39._TEMPLATE_RENDER_PY_CONTENT
    assert 'i += 1\n                continue' in fixed  # bug (2) must still be present
    return fixed


def _full_fix_source() -> str:
    fixed = c39._TEMPLATE_RENDER_PY_CONTENT.replace(
        "inner = template[i:j]", "inner = template[i + 1:j]"
    )
    assert fixed.count('i += 1\n                continue') == 1
    fixed = fixed.replace(
        'i += 1\n                continue', 'i += 2\n                continue'
    )
    # the plain-character branch (`else: out.append(ch); i += 1`) legitimately keeps its own,
    # unrelated "i += 1" -- only the escape-handling one (immediately followed by `continue`)
    # is bug (2) and must be gone after this fix
    assert 'i += 1\n                continue' not in fixed
    assert 'template[i + 1:j]' in fixed
    return fixed


def test_obvious_fix_looks_complete_from_the_visible_suite_alone():
    """Direction (2a): bug-(1)-only fix agrees with ground truth on EVERY case a visible-only
    test run would show -- Group A (now fixed) AND every bonus case."""
    fn = _exec_render(_obvious_fix_source())
    for template, values, expected in _GROUP_A_CASES + _BONUS_CASES:
        actual = _actual_outcome(fn, template, values)
        assert actual == expected, template


def test_obvious_fix_fails_every_hidden_escaping_case():
    """Direction (2b), the actual trap: the SAME fix that looks complete above still
    disagrees with ground truth on every case that exercises bug (2), which the visible suite
    never touches at all."""
    fn = _exec_render(_obvious_fix_source())
    for template, values, expected in _HIDDEN_CASES:
        actual = _actual_outcome(fn, template, values)
        assert actual != expected, (
            f"calibration invariant violated: the obvious (bug-1-only) fix should disagree "
            f"with ground truth on hidden case {template!r}"
        )


def test_full_fix_matches_ground_truth_on_every_canonical_case_no_hangs():
    """Direction (3): the fully correct joint patch agrees with ground truth everywhere,
    with no hang anywhere."""
    fn = _exec_render(_full_fix_source())
    for template, values, expected in _ALL_CASES:
        actual = _actual_outcome(fn, template, values)
        assert actual == expected, template


def test_keystone_ids_reference_real_test_functions_in_the_canonical_grading_file():
    canonical_content = c39.get_grading_payload()["tests"][c39.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", canonical_content, re.MULTILINE))
    assert len(defined) == 13
    for node_id in c39.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c39.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the canonical file"
    assert set(c39.KEYSTONE_TEST_IDS) == {
        f"{c39.VISIBLE_TEST_PATH}::test_single_placeholder_is_substituted",
        f"{c39.VISIBLE_TEST_PATH}::test_multiple_placeholders_are_all_substituted",
        f"{c39.VISIBLE_TEST_PATH}::test_non_string_value_is_stringified",
        f"{c39.VISIBLE_TEST_PATH}::test_default_is_ignored_when_name_is_present",
        f"{c39.VISIBLE_TEST_PATH}::test_missing_name_without_default_raises_key_error",
        f"{c39.VISIBLE_TEST_PATH}::test_placeholder_at_start_and_end_of_template",
        f"{c39.VISIBLE_TEST_PATH}::test_escaped_braces_alone_render_as_literal_braces",
        f"{c39.VISIBLE_TEST_PATH}::test_escaped_braces_combined_with_a_real_placeholder",
        f"{c39.VISIBLE_TEST_PATH}::test_consecutive_escaped_pairs_render_correctly",
        f"{c39.VISIBLE_TEST_PATH}::test_unmatched_close_brace_after_valid_escapes_raises_value_error",
    }


def test_category_is_hard():
    assert c39.get_test_metadata()["category"] == "hard"


def test_visibility_is_visible():
    assert c39.get_visibility() == "visible"


def test_sandbox_fixture_excludes_hidden_tests():
    fixture = c39.get_sandbox_fixture()
    assert fixture[c39.STARTER_MODULE_PATH] == c39._TEMPLATE_RENDER_PY_CONTENT
    assert fixture[c39.VISIBLE_TEST_PATH] == c39._TEST_FILE_CONTENT
    assert "inner = template[i:j]" in fixture[c39.STARTER_MODULE_PATH]
    hidden_names = set(re.findall(r"^def (test_\w+)\(", c39._HIDDEN_TEST_ADDENDUM, re.MULTILINE))
    assert len(hidden_names) == 4
    for name in hidden_names:
        assert name not in fixture[c39.VISIBLE_TEST_PATH]


def test_grading_payload_is_a_genuine_superset_of_the_visible_fixture():
    payload = c39.get_grading_payload()
    canonical = payload["tests"][c39.VISIBLE_TEST_PATH]
    visible = c39.get_sandbox_fixture()[c39.VISIBLE_TEST_PATH]
    assert canonical != visible
    assert canonical.startswith(visible)
    assert canonical == c39._CANONICAL_TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "template_render", "functions": ["render"]}
    assert payload["keystone_test_ids"] == c39.KEYSTONE_TEST_IDS


def test_compiled_plan_structure_and_no_leaked_values():
    plan = c39.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["fix_bug"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "template_render.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["template_render.py"]}

    instruction = leaf["instruction"]
    for leaked in (
        "template[i:j]", "template[i + 1:j]", "template[i+1:j]", "i += 1", "i += 2",
        '"{{"', "'{{' ", "Ann", "stranger", "middle",
    ):
        assert leaked not in instruction, f"plan leaks the mechanism via {leaked!r}"
    json.dumps(plan)


def test_task_statement_does_not_leak_the_seeded_bug_mechanism():
    statement = c39.get_task_statement()
    for leaked in ("template[i:j]", "template[i + 1:j]", "i += 1", "i += 2", "Ann", "stranger"):
        assert leaked not in statement, f"statement leaks {leaked!r}"


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c39", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c39"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c39.get_task_statement()
    assert (public / "repo" / c39.STARTER_MODULE_PATH).read_text() == c39._TEMPLATE_RENDER_PY_CONTENT
    assert (public / "repo" / c39.VISIBLE_TEST_PATH).read_text() == c39._TEST_FILE_CONTENT
    assert json.loads((public / "plan.json").read_text()) == c39.get_compiled_plan()

    assert (private / c39.VISIBLE_TEST_PATH).read_text() == c39.get_grading_payload()["tests"][c39.VISIBLE_TEST_PATH]
    assert (private / c39.VISIBLE_TEST_PATH).read_text() == c39._CANONICAL_TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c39.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c39.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
