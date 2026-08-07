"""
Adversarial offline checks for codebench task c29 (mailmerge-template-substitution) — no
Docker, no LLM. Mirrors idea_code_test_c04_test.py's exec-based structure: every test_*
function in the embedded canonical test file is run for real against an
INDEPENDENTLY-WRITTEN reference implementation (a single combined tokenizing regex
consumed via `finditer`, rather than the task module's own manual index-walking
while-loop) to catch a mistake in the embedded literal expectations, particularly around
the escaping/backslash handling, which is easy to get subtly wrong by hand.

Also mutation-tests the two new hardening cases against two plausible near-miss "naive"
implementations to prove each one has real discriminating power, not just cosmetic
novelty:
  - the escape-adjacent-to-real-placeholder case, against an implementation that
    unescapes braces in one pass and THEN regex-substitutes placeholders in a second
    pass (a common, idiomatic way to write this kind of function);
  - the non-strict-fallback-not-at-template-start case, against a minimal near-miss
    that is IDENTICAL to a correct manual scan except for a single off-by-index slice
    in the non-strict fallback branch (`template[i:i+end+2]` instead of
    `template[i:end+2]`) -- this specific bug is not hypothetical: it is exactly what
    an actual submitted solution for this task shipped during live calibration, and it
    passed every one of the original 9 canonical tests because their one non-strict
    fallback case happened to have the placeholder flush at index 0, where the bug is
    invisible.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

from agent.app.idea_code_tests import test_c29_mailmerge_template as c29

# Matches, in order of preference: an escaped brace (\{ or \}), or a {{...}} placeholder
# (non-greedy), or any single other character. re.DOTALL isn't needed since '.' already
# matches everything except newline and templates in this task are single-line/short.
# Python's alternation tries branches in the order they're written at each match
# position, which is exactly what makes this regex-driven tokenizer structurally
# different from (yet spec-equivalent to) the task module's manual char-by-char scan.
_TOKEN_RE = re.compile(r"\\[{}]|\{\{(.*?)\}\}|.", re.DOTALL)


def _independent_render_template(template: str, context: dict, strict: bool = True) -> str:
    """Independently-written via a single combined tokenizing regex consumed with
    `finditer` (escaped-brace / placeholder / any-char alternatives, tried in that
    priority order by regex alternation) rather than the task module's manual
    index-walking while-loop — a structurally different way of arriving at the same
    spec. Critically, because the regex alternation is evaluated in ONE combined pass
    over the string (not "unescape everything, then separately find placeholders"),
    it naturally respects that a backslash-escape only ever consumes a single following
    brace character, never accidentally "completing" a {{ pair that a two-pass
    unescape-then-substitute strategy would."""
    # The token grammar alone can't express "raise ValueError for a non-match", so
    # detect an unterminated "{{" (one with no LATER "}}") via a direct pre-scan using
    # the same escape rule, independently of the tokenizing pass below.
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "\\" and i + 1 < n and template[i + 1] in "{}":
            i += 2
            continue
        if template[i] == "{" and i + 1 < n and template[i + 1] == "{" and "}}" not in template[i + 2:]:
            raise ValueError(f"unterminated placeholder at {i}")
        i += 1

    out_parts = []
    for m in _TOKEN_RE.finditer(template):
        text = m.group(0)
        if text.startswith("\\") and len(text) == 2:
            out_parts.append(text[1])
        elif m.group(1) is not None:
            key = m.group(1).strip()
            if key in context:
                out_parts.append(str(context[key]))
            elif strict:
                raise KeyError(key)
            else:
                out_parts.append(text)
        else:
            out_parts.append(text)
    return "".join(out_parts)


def _buggy_two_pass_render_template(template: str, context: dict, strict: bool = True) -> str:
    """A plausible near-miss implementation: unescape braces in one pass, THEN
    regex-substitute "{{...}}" placeholders in a second pass. This is a common,
    idiomatic-looking way to write a mini template engine, and it passes every
    pre-existing c29 canonical test — but it mishandles a backslash sitting
    immediately next to a real placeholder, because unescaping first can accidentally
    "complete" a {{ pair the true single left-to-right priority-ordered scan would
    never have formed. Used only to prove the new canonical test actually discriminates
    (mutation testing), never treated as ground truth."""
    unescaped = template.replace("\\{", "{").replace("\\}", "}")

    def repl(m: re.Match) -> str:
        key = m.group(1).strip()
        if key in context:
            return str(context[key])
        if strict:
            raise KeyError(key)
        return m.group(0)

    return re.sub(r"\{\{(.*?)\}\}", repl, unescaped)


def _buggy_offset_slice_render_template(template: str, context: dict, strict: bool = True) -> str:
    """A minimal near-miss: otherwise a byte-for-byte correct manual left-to-right scan
    (same structure as the task's own reference algorithm), except the non-strict
    verbatim-fallback line slices `template[i:i+end+2]` instead of the correct
    `template[i:end+2]` -- an easy transcription slip (re-using the loop's "current
    position" variable `i` a second time where the slice's own start, which already IS
    `i`, was needed instead). This exact bug was found in an actual submitted solution
    for this task during live calibration. Used only to prove the new positional
    canonical test discriminates (mutation testing), never treated as ground truth."""
    out = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "\\" and i + 1 < n and template[i + 1] in "{}":
            out.append(template[i + 1])
            i += 2
            continue
        if template[i] == "{" and i + 1 < n and template[i + 1] == "{":
            end = template.find("}}", i + 2)
            if end == -1:
                raise ValueError(f"unterminated placeholder at {i}")
            key = template[i + 2:end].strip()
            if key in context:
                out.append(str(context[key]))
            elif strict:
                raise KeyError(key)
            else:
                out.append(template[i:i + end + 2])  # BUG: should be template[i:end + 2]
            i = end + 2
            continue
        out.append(template[i])
        i += 1
    return "".join(out)


def _run_embedded_tests_against(render_fn) -> set[str]:
    """Exec c29's own embedded canonical test file with `mailmerge.render_template`
    monkeypatched to `render_fn`, returning the set of test_* function names that
    failed (via AssertionError, any other exception including pytest.raises' internal
    BaseException-derived Failed signal, or a straightforward propagated exception)."""
    content = c29.get_sandbox_fixture()[c29.VISIBLE_TEST_PATH]
    fake_module = types.ModuleType("mailmerge")
    fake_module.render_template = render_fn
    original = sys.modules.get("mailmerge")
    sys.modules["mailmerge"] = fake_module
    try:
        namespace: dict = {}
        exec(compile(content, "<embedded c29 test file, mutation subject>", "exec"), namespace)
        test_functions = {
            k: v for k, v in namespace.items() if k.startswith("test_") and callable(v)
        }
        assert test_functions, "expected at least one test_ function in the embedded file"

        failures: set[str] = set()
        for name, fn in test_functions.items():
            try:
                fn()
            except BaseException:
                # BaseException (not just Exception) on purpose: pytest.raises' own
                # failure signal (raised when the expected exception never occurs
                # inside its `with` block) is a BaseException subclass, not an
                # Exception subclass.
                failures.add(name)
        return failures
    finally:
        if original is not None:
            sys.modules["mailmerge"] = original
        else:
            sys.modules.pop("mailmerge", None)


def test_ground_truth_basic_cases():
    assert _independent_render_template("Hello, {{name}}!", {"name": "Alice"}) == "Hello, Alice!"
    assert _independent_render_template("Hi {{ name }}", {"name": "Bob"}) == "Hi Bob"
    assert _independent_render_template("{{a}}-{{b}}-{{a}}", {"a": "x", "b": "y"}) == "x-y-x"
    assert _independent_render_template("Count: {{n}}", {"n": 42}) == "Count: 42"
    assert _independent_render_template("plain text, no braces here", {}) == "plain text, no braces here"


def test_ground_truth_escaping():
    template = "Use \\{\\{ and \\}\\} for literals"
    assert _independent_render_template(template, {}) == "Use {{ and }} for literals"


def test_ground_truth_missing_key_policies():
    try:
        _independent_render_template("{{missing}}", {})
        raise AssertionError("strict missing key should have raised KeyError")
    except KeyError:
        pass
    assert _independent_render_template("{{missing}}", {}, strict=False) == "{{missing}}"


def test_ground_truth_unterminated_placeholder():
    try:
        _independent_render_template("{{oops", {"oops": "x"})
        raise AssertionError("unterminated placeholder should have raised ValueError")
    except ValueError:
        pass


def test_ground_truth_escaped_brace_immediately_before_real_placeholder():
    # Independently re-derive the tricky literal embedded in c29's canonical test file,
    # via the regex-tokenizer implementation (a structurally different code path from
    # the task module's manual scan).
    template = r"\{{not_a_var}}{{real}}"
    assert _independent_render_template(template, {"real": "R"}) == "{{not_a_var}}R"


def test_ground_truth_non_strict_missing_placeholder_not_at_start():
    template = "Dear {{missing}}, thanks"
    assert _independent_render_template(template, {}, strict=False) == template


def test_naive_two_pass_implementation_is_actually_wrong_on_the_new_case():
    # Sanity-check the escape-adjacency mutation itself: confirm the "unescape then
    # regex-substitute" near-miss really does disagree with the spec here (otherwise
    # the new canonical test would have zero discriminating power against it).
    template = r"\{{not_a_var}}{{real}}"
    try:
        _buggy_two_pass_render_template(template, {"real": "R"})
        raise AssertionError(
            "expected the naive two-pass implementation to incorrectly raise KeyError "
            "here, but it didn't -- the mutation no longer demonstrates anything"
        )
    except KeyError as exc:
        assert exc.args == ("not_a_var",)


def test_offset_slice_bug_implementation_is_actually_wrong_on_the_new_case():
    # Sanity-check the offset-slice mutation itself: confirm it really does disagree
    # with the spec on the new positional canonical test (otherwise that test would
    # have zero discriminating power against it).
    template = "Dear {{missing}}, thanks"
    result = _buggy_offset_slice_render_template(template, {}, strict=False)
    assert result != template, (
        "expected the offset-slice near-miss to corrupt this output, but it matched "
        "the correct answer -- the mutation no longer demonstrates anything"
    )


def test_escape_adjacent_case_discriminates_the_buggy_two_pass_implementation():
    # Run c29's own embedded canonical test file against the buggy two-pass
    # implementation and confirm the failures are EXACTLY the tests a two-pass
    # unescape-then-substitute strategy is known to mishandle (proving the new
    # escape-adjacent case adds real, non-redundant discriminating power alongside
    # the pre-existing escaping/unterminated tests, not that it alone catches this
    # particular near-miss).
    failures = _run_embedded_tests_against(_buggy_two_pass_render_template)
    assert failures == {
        "test_escaped_braces_literal",
        "test_unterminated_placeholder_raises_value_error",
        "test_escaped_brace_immediately_before_real_placeholder",
    }, f"unexpected failure set against the buggy two-pass implementation: {failures!r}"


def test_new_positional_case_discriminates_the_offset_slice_bug():
    # Run c29's own embedded canonical test file against the offset-slice near-miss
    # (structurally identical to a correct scan except for one line) and confirm
    # EXACTLY the new positional test fails against it -- proving that case adds real,
    # newly-needed discriminating power the original 9-test suite did not have (its
    # only non-strict-fallback case had the placeholder flush at index 0, which this
    # exact bug happens to get right by coincidence).
    failures = _run_embedded_tests_against(_buggy_offset_slice_render_template)
    assert failures == {"test_non_strict_missing_placeholder_not_at_template_start"}, (
        f"expected exactly the new positional test to fail against the offset-slice "
        f"near-miss, got failures={failures!r}"
    )


def test_embedded_test_file_asserts_match_ground_truth():
    content = c29.get_sandbox_fixture()[c29.VISIBLE_TEST_PATH]
    fake_module = types.ModuleType("mailmerge")
    fake_module.render_template = _independent_render_template
    original = sys.modules.get("mailmerge")
    sys.modules["mailmerge"] = fake_module
    try:
        namespace: dict = {}
        exec(compile(content, "<embedded c29 test file>", "exec"), namespace)
        test_functions = [
            v for k, v in namespace.items() if k.startswith("test_") and callable(v)
        ]
        assert test_functions, "expected at least one test_ function in the embedded file"
        for fn in test_functions:
            fn()  # raises if an embedded expected value / pytest.raises() is wrong
    finally:
        if original is not None:
            sys.modules["mailmerge"] = original
        else:
            sys.modules.pop("mailmerge", None)


def test_keystone_ids_reference_real_test_functions():
    content = c29.get_sandbox_fixture()[c29.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c29.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c29.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_shallow_checks():
    # Whitespace-trimming, repeated-key reuse, non-string stringification, and plain
    # passthrough are real but shallow checks that a naive str.replace()-based
    # implementation would likely also pass — bonus credit only, not gating.
    for shallow in (
        "test_whitespace_inside_braces_stripped",
        "test_multiple_placeholders_repeated_key",
        "test_non_string_value_stringified",
        "test_no_placeholders_passthrough",
    ):
        assert f"{c29.VISIBLE_TEST_PATH}::{shallow}" not in c29.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c29.get_visibility() == "visible"


def test_sandbox_fixture_and_grading_tests_share_relpath():
    fixture = c29.get_sandbox_fixture()
    tests = c29.get_grading_payload()["tests"]
    assert set(fixture.keys()) == set(tests.keys())
    for path in fixture:
        assert fixture[path] == tests[path]


def test_grading_payload_shape():
    payload = c29.get_grading_payload()
    assert payload["tests"][c29.VISIBLE_TEST_PATH] == c29.get_sandbox_fixture()[c29.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "mailmerge", "functions": ["render_template"]}
    assert payload["keystone_test_ids"] == c29.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c29.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "mailmerge.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["mailmerge.py"]}
    json.dumps(plan)  # must be JSON-serializable as-is


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c29", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c29"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c29.get_task_statement()
    assert (public / "repo" / c29.VISIBLE_TEST_PATH).read_text() == c29.get_sandbox_fixture()[c29.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c29.get_compiled_plan()

    assert (private / c29.VISIBLE_TEST_PATH).read_text() == c29.get_grading_payload()["tests"][c29.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c29.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c29.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
