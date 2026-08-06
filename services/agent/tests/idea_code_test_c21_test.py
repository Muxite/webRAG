"""
Adversarial offline checks for codebench task c21 (http-path-router) — no Docker, no LLM.

Mirrors idea_code_test_c06_test.py's pattern for a non-numeric-literal task: re-derive ground
truth with a SECOND, differently-coded implementation (here: regex-based route matching instead
of c21's manual segment-split-and-zip approach) run against the same input sequences the
canonical test file uses, then separately confirm via string search that those exact scenarios
are actually what's embedded in the canonical test file (binding the independent computation to
what's really shipped, not to authorial memory of intent).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c21_path_router as c21


def _independent_pattern_to_regex(pattern: str):
    """Regex-based reimplementation of pattern matching, structurally unrelated to c21's own
    manual split()/zip() approach: converts a path_pattern into a compiled regex where {name}
    becomes a named capture group over one-or-more non-slash characters (which, being a '+'
    quantifier, naturally REJECTS an empty segment with no special-casing needed)."""
    segments = pattern.split("/")[1:]
    parts = []
    specificity = 0
    for seg in segments:
        if seg.startswith("{") and seg.endswith("}") and len(seg) > 2:
            parts.append(f"(?P<{seg[1:-1]}>[^/]+)")
        else:
            parts.append(re.escape(seg))
            specificity += 1
    return re.compile("^" + "/".join(parts) + "$"), specificity


def _independent_dispatch(routes, method, path):
    """routes: list of (method, pattern, handler) tuples in registration order. Independently
    re-implements Router.dispatch's contract via regex matching + a separate specificity/tie-
    break pass, as a cross-check against c21's spec (manual segment comparison)."""
    path_body = path[1:]
    candidates = []
    for idx, (m, pattern, handler) in enumerate(routes):
        regex, specificity = _independent_pattern_to_regex(pattern)
        match = regex.match(path_body)
        if match:
            candidates.append((specificity, idx, m, handler, match.groupdict()))
    if not candidates:
        return (404, None)
    best_specificity = max(c[0] for c in candidates)
    best = sorted((c for c in candidates if c[0] == best_specificity), key=lambda c: c[1])[0]
    _, _, matched_method, handler, params = best
    if matched_method.upper() != method.upper():
        return (405, None)
    return (200, handler(**params))


def test_ground_truth_static_beats_dynamic_precedence():
    calls = []
    routes = [
        ("GET", "/users/{id}", lambda **kw: calls.append(("dynamic", kw)) or "dynamic"),
        ("GET", "/users/me", lambda **kw: calls.append(("static", kw)) or "static"),
    ]
    assert _independent_dispatch(routes, "GET", "/users/me") == (200, "static")
    assert calls[-1] == ("static", {})
    assert _independent_dispatch(routes, "GET", "/users/42") == (200, "dynamic")
    assert calls[-1] == ("dynamic", {"id": "42"})


def test_ground_truth_multi_param_capture():
    routes = [("GET", "/orgs/{org}/repos/{repo}", lambda **kw: kw)]
    status, body = _independent_dispatch(routes, "GET", "/orgs/acme/repos/widgets")
    assert status == 200
    assert body == {"org": "acme", "repo": "widgets"}


def test_ground_truth_method_mismatch_is_405_not_404():
    routes = [("GET", "/widgets/{id}", lambda **kw: kw)]
    assert _independent_dispatch(routes, "DELETE", "/widgets/5") == (405, None)


def test_ground_truth_unknown_path_is_404():
    routes = [("GET", "/widgets/{id}", lambda **kw: kw)]
    assert _independent_dispatch(routes, "GET", "/gadgets/5") == (404, None)


def test_ground_truth_segment_count_mismatch_is_404():
    # The classic zip()-truncation trap: a naive matcher using zip(pattern_segs, path_segs)
    # without a length check would wrongly match here (capturing id="1", ignoring "/2").
    routes = [("GET", "/users/{id}", lambda **kw: kw)]
    assert _independent_dispatch(routes, "GET", "/users/1/2") == (404, None)


def test_ground_truth_empty_segment_does_not_satisfy_a_param():
    routes = [("GET", "/users/{id}/profile", lambda **kw: kw)]
    assert _independent_dispatch(routes, "GET", "/users//profile") == (404, None)


def test_independent_regex_matcher_itself_rejects_a_bad_match():
    # Sanity-check the independent matcher: a pattern with more segments than the path must
    # never accidentally match.
    routes = [("GET", "/a/{x}/{y}", lambda **kw: kw)]
    assert _independent_dispatch(routes, "GET", "/a/1") == (404, None)


def test_embedded_test_file_scenarios_match_the_independent_reimplementation():
    """Binds the independent computation above to what's actually shipped: the canonical test
    file's literal scenario strings and expected outcomes must appear verbatim."""
    content = c21.get_grading_payload()["tests"][c21._TEST_FILE_PATH]
    assert 'r.add_route("GET", "/users/{id}"' in content
    assert 'r.add_route("GET", "/users/me"' in content
    assert 'status, body = r.dispatch("GET", "/users/me")' in content
    assert '(status, body) == (200, "static")' in content
    assert 'calls[-1] == ("dynamic", {"id": "42"})' in content
    assert 'r.add_route("GET", "/orgs/{org}/repos/{repo}"' in content
    assert '("repo", {"org": "acme", "repo": "widgets"})' in content
    assert 'r.dispatch("DELETE", "/widgets/5") == (405, None)' in content
    assert 'r.dispatch("GET", "/gadgets/5") == (404, None)' in content
    assert 'r.dispatch("GET", "/users/1/2") == (404, None)' in content
    assert 'r.dispatch("GET", "/users//profile") == (404, None)' in content


def test_keystone_ids_reference_real_test_functions():
    content = c21.get_grading_payload()["tests"][c21._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 10
    for node_id in c21.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c21._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_non_discriminating_cases():
    non_keystone = [
        "test_static_route_exact_match",
        "test_unknown_path_returns_404",
        "test_empty_segment_does_not_satisfy_param",
        "test_method_case_insensitive",
        "test_route_overwrite_on_duplicate_registration",
    ]
    for name in non_keystone:
        assert f"{c21._TEST_FILE_PATH}::{name}" not in c21.KEYSTONE_TEST_IDS
    assert len(c21.KEYSTONE_TEST_IDS) == 5


def test_visibility_is_hidden():
    assert c21.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c21.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c21.get_grading_payload()
    assert payload["tests"] == {c21._TEST_FILE_PATH: c21._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "mini_router",
        "class": "Router",
        "methods": ["add_route", "dispatch"],
    }
    assert payload["keystone_test_ids"] == c21.KEYSTONE_TEST_IDS


def test_compiled_plan_structure_and_no_leaked_private_literals():
    plan = c21.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "mini_router.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["mini_router.py"]}
    json.dumps(plan)  # must be plain JSON-safe data

    full_text = json.dumps(plan)
    # Private test literals that must never leak into the (agent-visible) plan: the specific
    # multi-param values, and the specific 404/405 scenario paths, from the hidden canonical
    # test. (The static-vs-dynamic "/users/me" vs "/users/{id}" example is fine to restate — it
    # already appears verbatim in get_task_statement() itself, which is always agent-visible.)
    for leaked in ("acme", "widgets/{id}", "gadgets", "users/1/2", "users//profile"):
        assert leaked not in full_text, f"plan leaks private test literal {leaked!r}"


def test_task_statement_does_not_leak_segment_mismatch_or_multiparam_cases():
    statement = c21.get_task_statement()
    for leaked in ("acme", "widgets", "gadgets", "1/2", "//profile"):
        assert leaked not in statement, f"task statement leaks private test literal {leaked!r}"


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c21", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c21"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c21.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c21.get_compiled_plan()

    assert (private / c21._TEST_FILE_PATH).read_text() == c21._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c21._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c21.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
