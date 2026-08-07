"""
Adversarial offline checks for codebench task c22 (conditional-etag-resource-store) — no Docker,
no LLM.

Independently re-derives ground truth with a SECOND, differently-coded conditional-request
simulator (a plain dict-of-(etag, body) plus free functions operating on it, rather than c22's
spec of a stateful ResourceStore class with internal comparison helpers) run through the same
scenarios the canonical test file uses, then separately confirms via string search that those
exact scenarios are what's actually embedded in the canonical test file. Also proves — by
constructing and running two independent plausible-near-miss mutants — that this task's keystone
set discriminates a "structurally looks complete" implementation from a genuinely correct one,
which is the entire reason this task exists in its current (replaced) form; see the module
docstring for the three prior live-calibration failures of the token-bucket version this task
used to be.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c22_conditional_etag_resource_store as c22


# ---------------------------------------------------------------------------
# Independent re-derivation #1: free functions over a plain dict, deliberately NOT structured
# as a class with named _strong_match/_weak_match helpers, to cross-check c22's own spec.
# ---------------------------------------------------------------------------

def _etag_of(body: str) -> str:
    return '"' + hashlib.sha256(body.encode("utf-8")).hexdigest() + '"'


def _tokens(header):
    header = header.strip()
    if header == "*":
        return ["*"]
    return [t.strip() for t in header.split(",")]


def _opaque_of(token):
    if token.startswith("W/"):
        token = token[2:]
    if len(token) >= 2 and token[0] == token[-1] == '"':
        token = token[1:-1]
    return token


def _independent_put(data, path, body, if_match=None):
    """data: dict[path] -> (etag, body), mutated in place. Returns (status, etag)."""
    existed = path in data
    if if_match is None:
        etag = _etag_of(body)
        data[path] = (etag, body)
        return (201, etag) if not existed else (200, etag)

    if not existed:
        return (412, None)

    current_etag, _ = data[path]
    ok = False
    for tok in _tokens(if_match):
        if tok == "*":
            ok = True
            break
        # strong comparison: reject if EITHER side is weak-marked
        if tok.startswith("W/") or current_etag.startswith("W/"):
            continue
        if _opaque_of(tok) == _opaque_of(current_etag):
            ok = True
            break
    if not ok:
        return (412, None)
    etag = _etag_of(body)
    data[path] = (etag, body)
    return (200, etag)


def _independent_get(data, path, if_none_match=None):
    if path not in data:
        return (404, None, None)
    current_etag, body = data[path]
    if if_none_match is None:
        return (200, current_etag, body)
    ok = False
    for tok in _tokens(if_none_match):
        if tok == "*" or _opaque_of(tok) == _opaque_of(current_etag):
            ok = True
            break
    if ok:
        return (304, current_etag, None)
    return (200, current_etag, body)


def test_ground_truth_put_new_and_overwrite_status_codes():
    data = {}
    assert _independent_put(data, "/widgets/1", "hello") == (201, _etag_of("hello"))
    assert _independent_put(data, "/widgets/1", "world") == (200, _etag_of("world"))


def test_ground_truth_get_nonexistent_and_existing():
    data = {}
    assert _independent_get(data, "/nope") == (404, None, None)
    _independent_put(data, "/widgets/1", "hello")
    assert _independent_get(data, "/widgets/1") == (200, _etag_of("hello"), "hello")


def test_ground_truth_same_body_same_etag():
    data = {}
    _, e1 = _independent_put(data, "/x", "same")
    _independent_put(data, "/x", "changed")
    _, e2 = _independent_put(data, "/x", "same")
    assert e1 == e2 == _etag_of("same")


def test_ground_truth_if_match_exact_and_stale():
    data = {}
    _, etag1 = _independent_put(data, "/x", "v1")
    assert _independent_put(data, "/x", "v2", if_match=etag1) == (200, _etag_of("v2"))

    data2 = {}
    _independent_put(data2, "/x", "v1")
    assert _independent_put(data2, "/x", "v2", if_match='"stale"') == (412, None)
    assert data2["/x"] == (_etag_of("v1"), "v1")  # unchanged


def test_ground_truth_wildcard_requires_existing_resource():
    data = {}
    assert _independent_put(data, "/brand-new", "v1", if_match="*") == (412, None)
    assert "/brand-new" not in data
    _independent_put(data, "/x", "v1")
    assert _independent_put(data, "/x", "v2", if_match="*") == (200, _etag_of("v2"))


def test_ground_truth_weak_validator_strong_vs_weak_split():
    data = {}
    _, etag1 = _independent_put(data, "/x", "v1")
    weak_token = "W/" + etag1
    # If-Match (strong): weak-marked token must NEVER match, even with identical opaque content
    assert _independent_put(data, "/x", "v2", if_match=weak_token) == (412, None)
    assert data["/x"] == (etag1, "v1")
    # If-None-Match (weak): the SAME weak-marked token must match
    assert _independent_get(data, "/x", if_none_match=weak_token) == (304, etag1, None)


def test_ground_truth_strong_etag_also_satisfies_weak_comparison():
    data = {}
    _, etag1 = _independent_put(data, "/x", "v1")
    assert _independent_get(data, "/x", if_none_match=etag1) == (304, etag1, None)


def test_ground_truth_multi_value_lists_match_any_token():
    data = {}
    _, etag1 = _independent_put(data, "/x", "v1")
    put_header = '"decoy-one", ' + etag1 + ', "decoy-two"'
    assert _independent_put(data, "/x", "v2", if_match=put_header) == (200, _etag_of("v2"))

    data2 = {}
    _, etag1b = _independent_put(data2, "/x", "v1")
    get_header = '"decoy-one", "decoy-two", ' + etag1b
    assert _independent_get(data2, "/x", if_none_match=get_header) == (304, etag1b, None)


# ---------------------------------------------------------------------------
# Independent re-derivation #2: a pure closed-form comparison-semantics check, isolated from any
# store/dict state at all, to cross-check the strong/weak split itself isn't an artifact of
# _independent_put/_independent_get sharing a bug.
# ---------------------------------------------------------------------------

def _pure_strong_matches(client_token: str, store_etag: str) -> bool:
    if client_token.startswith("W/") or store_etag.startswith("W/"):
        return False
    return _opaque_of(client_token) == _opaque_of(store_etag)


def _pure_weak_matches(client_token: str, store_etag: str) -> bool:
    return _opaque_of(client_token) == _opaque_of(store_etag)


def test_pure_comparison_semantics_diverge_only_on_weak_tokens():
    strong_etag = _etag_of("v1")
    weak_of_same = "W/" + strong_etag
    # identical opaque content on both sides
    assert _pure_weak_matches(weak_of_same, strong_etag) is True
    assert _pure_strong_matches(weak_of_same, strong_etag) is False
    # two strong tokens with identical opaque content: both comparisons agree
    assert _pure_weak_matches(strong_etag, strong_etag) is True
    assert _pure_strong_matches(strong_etag, strong_etag) is True
    # different opaque content: both comparisons agree (no match either way)
    other = _etag_of("v2")
    assert _pure_weak_matches(other, strong_etag) is False
    assert _pure_strong_matches(other, strong_etag) is False


def test_embedded_test_file_scenarios_match_the_independent_reimplementation():
    content = c22.get_grading_payload()["tests"][c22._TEST_FILE_PATH]
    assert "import hashlib" in content
    assert 'sha256(body.encode("utf-8")).hexdigest()' in content
    assert 'store.put("/widgets/1", "hello")' in content
    assert 'store.put("/x", "v1")' in content
    assert 'if_match="*"' in content
    assert 'if_none_match="*"' in content
    assert 'weak_token = "W/" + etag1' in content
    assert content.count('weak_token') == 4  # used once in each of the two weak-validator tests
    assert '"decoy-one", ' in content and '"decoy-two"' in content
    assert content.count('"decoy-one"') == 2  # one in the If-Match test, one in If-None-Match


def test_keystone_ids_reference_real_test_functions():
    content = c22.get_grading_payload()["tests"][c22._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 16
    for node_id in c22.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c22._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_non_discriminating_cases():
    non_keystone = [
        "test_put_new_resource_returns_201_with_content_hash_etag",
        "test_put_overwrite_unconditional_returns_200_with_new_etag",
        "test_get_nonexistent_returns_404",
        "test_get_existing_returns_current_body_and_etag",
        "test_same_body_produces_same_etag_different_body_produces_different_etag",
        "test_put_if_match_exact_strong_etag_succeeds",
        "test_put_if_match_wildcard_succeeds_on_existing_resource",
        "test_get_if_none_match_exact_strong_etag_also_satisfies_weak_comparison",
        "test_get_if_none_match_stale_etag_returns_200_with_body",
        "test_get_if_none_match_wildcard_matches_existing_resource",
    ]
    for name in non_keystone:
        assert f"{c22._TEST_FILE_PATH}::{name}" not in c22.KEYSTONE_TEST_IDS, name
    keystone_only = [
        "test_put_if_match_stale_etag_returns_412_and_leaves_resource_unchanged",
        "test_put_if_match_wildcard_fails_on_nonexistent_resource",
        "test_put_if_match_weak_validator_never_satisfies_strong_comparison",
        "test_get_if_none_match_weak_validator_satisfies_weak_comparison",
        "test_put_if_match_multi_value_list_matches_any_listed_etag",
        "test_get_if_none_match_multi_value_list_matches_any_listed_etag",
    ]
    for name in keystone_only:
        assert f"{c22._TEST_FILE_PATH}::{name}" in c22.KEYSTONE_TEST_IDS, name
    assert len(c22.KEYSTONE_TEST_IDS) == 6


def test_visibility_is_hidden():
    assert c22.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c22.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c22.get_grading_payload()
    assert payload["tests"] == {c22._TEST_FILE_PATH: c22._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "resource_store",
        "class": "ResourceStore",
        "methods": ["put", "get"],
    }
    assert payload["keystone_test_ids"] == c22.KEYSTONE_TEST_IDS


def test_compiled_plan_structure_and_no_leaked_private_literals():
    plan = c22.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "resource_store.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["resource_store.py"]}
    json.dumps(plan)  # must be plain JSON-safe data

    full_text = json.dumps(plan)
    # The exact literal test-scenario strings (paths, bodies, decoy tokens) that gate the score
    # must never leak into the agent-visible plan.
    for leaked in (
        "/widgets/1", "/brand-new", "/nope", "decoy-one", "decoy-two",
        '"v1"', '"v2"', '"hello"', '"world"', "not-the-real-etag",
    ):
        assert leaked not in full_text, f"plan leaks private test literal {leaked!r}"


def test_task_statement_does_not_leak_specific_numeric_scenarios():
    statement = c22.get_task_statement()
    for leaked in (
        "/widgets/1", "/brand-new", "/nope", "decoy-one", "decoy-two",
        '"v1"', '"v2"', '"hello"', '"world"', "not-the-real-etag",
    ):
        assert leaked not in statement, f"task statement leaks private test literal {leaked!r}"


# ---------------------------------------------------------------------------
# Mutation testing: construct the strongest plausible near-miss implementations and confirm the
# canonical suite actually catches them (this is what proves the keystone set has real
# discriminating power, not just plausible-sounding rationale in a docstring).
# ---------------------------------------------------------------------------

_REFERENCE_IMPL = '''\
import hashlib


def _compute_etag(body):
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f'"{digest}"'


def _is_weak(token):
    return token.startswith("W/")


def _opaque(token):
    if _is_weak(token):
        token = token[2:]
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        token = token[1:-1]
    return token


def _parse_etag_list(value):
    value = value.strip()
    if value == "*":
        return ["*"]
    return [tok.strip() for tok in value.split(",")]


def _strong_match(client_token, store_etag):
    if _is_weak(client_token) or _is_weak(store_etag):
        return False
    return _opaque(client_token) == _opaque(store_etag)


def _weak_match(client_token, store_etag):
    return _opaque(client_token) == _opaque(store_etag)


class ResourceStore:
    def __init__(self):
        self._data = {}

    def put(self, path, body, if_match=None):
        exists = path in self._data
        if if_match is None:
            etag = _compute_etag(body)
            self._data[path] = (etag, body)
            return (201 if not exists else 200, etag)

        if not exists:
            return (412, None)

        current_etag, _ = self._data[path]
        tokens = _parse_etag_list(if_match)
        matched = "*" in tokens or any(_strong_match(tok, current_etag) for tok in tokens)
        if not matched:
            return (412, None)

        etag = _compute_etag(body)
        self._data[path] = (etag, body)
        return (200, etag)

    def get(self, path, if_none_match=None):
        if path not in self._data:
            return (404, None, None)

        current_etag, body = self._data[path]
        if if_none_match is None:
            return (200, current_etag, body)

        tokens = _parse_etag_list(if_none_match)
        matched = "*" in tokens or any(_weak_match(tok, current_etag) for tok in tokens)
        if matched:
            return (304, current_etag, None)
        return (200, current_etag, body)
'''

# Mutant A: "half-fix" — routes If-Match to a distinctly-named _strong_match function and
# If-None-Match to _weak_match (the routing is entirely correct), but _strong_match's body was
# written identically to _weak_match's (forgot to reject weak-marked tokens). This is the
# plausible near-miss the module docstring's mutation analysis is built around.
_MUTANT_HALF_FIX = _REFERENCE_IMPL.replace(
    "def _strong_match(client_token, store_etag):\n"
    "    if _is_weak(client_token) or _is_weak(store_etag):\n"
    "        return False\n"
    "    return _opaque(client_token) == _opaque(store_etag)",
    "def _strong_match(client_token, store_etag):\n"
    "    return _opaque(client_token) == _opaque(store_etag)",
)
assert _MUTANT_HALF_FIX != _REFERENCE_IMPL, "mutant A patch did not apply"

# Mutant B: treats '*' as "always matches" instead of "matches only if a current representation
# already exists" — an independent, differently-shaped near-miss on the wildcard rule.
_MUTANT_WILDCARD = _REFERENCE_IMPL.replace(
    "        if not exists:\n"
    "            return (412, None)\n"
    "\n"
    "        current_etag, _ = self._data[path]\n"
    "        tokens = _parse_etag_list(if_match)\n"
    "        matched = \"*\" in tokens or any(_strong_match(tok, current_etag) for tok in tokens)\n"
    "        if not matched:\n"
    "            return (412, None)\n"
    "\n"
    "        etag = _compute_etag(body)\n"
    "        self._data[path] = (etag, body)\n"
    "        return (200, etag)",
    "        tokens = _parse_etag_list(if_match)\n"
    "        if \"*\" not in tokens:\n"
    "            if not exists:\n"
    "                return (412, None)\n"
    "            current_etag, _ = self._data[path]\n"
    "            if not any(_strong_match(tok, current_etag) for tok in tokens):\n"
    "                return (412, None)\n"
    "\n"
    "        etag = _compute_etag(body)\n"
    "        self._data[path] = (etag, body)\n"
    "        return (201 if not exists else 200, etag)",
)
assert _MUTANT_WILDCARD != _REFERENCE_IMPL, "mutant B patch did not apply"

# Fully naive baseline: ignores if_match/if_none_match entirely.
_NAIVE_IGNORES_CONDITIONS = '''\
import hashlib


def _compute_etag(body):
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f'"{digest}"'


class ResourceStore:
    def __init__(self):
        self._data = {}

    def put(self, path, body, if_match=None):
        exists = path in self._data
        etag = _compute_etag(body)
        self._data[path] = (etag, body)
        return (201 if not exists else 200, etag)

    def get(self, path, if_none_match=None):
        if path not in self._data:
            return (404, None, None)
        current_etag, body = self._data[path]
        return (200, current_etag, body)
'''


def _run_pytest_against_impl(tmp_path, impl_source: str):
    (tmp_path / "resource_store.py").write_text(impl_source)
    test_content = c22.get_grading_payload()["tests"][c22._TEST_FILE_PATH]
    (tmp_path / "test_resource_store.py").write_text(test_content)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", str(tmp_path / "test_resource_store.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    return result.stdout + result.stderr


def _failed_test_names(pytest_output: str):
    return set(re.findall(r"FAILED test_resource_store\.py::(test_\w+)", pytest_output))


def test_reference_implementation_passes_every_canonical_test(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _REFERENCE_IMPL)
    assert "16 passed" in output, output


def test_half_fix_mutant_fails_exactly_the_targeted_keystone(tmp_path):
    """The plausible near-miss (correct routing, wrong strong-comparison mechanics) must fail
    EXACTLY the one keystone case built to catch it, and nothing else — proving this is a
    narrow, structurally-interacting trap rather than a broad correctness gap."""
    output = _run_pytest_against_impl(tmp_path, _MUTANT_HALF_FIX)
    failed = _failed_test_names(output)
    assert failed == {"test_put_if_match_weak_validator_never_satisfies_strong_comparison"}, output
    assert "15 passed" in output, output


def test_wildcard_mutant_fails_a_different_single_keystone(tmp_path):
    """An independent near-miss (wildcard treated as unconditional) must fail a DIFFERENT
    single keystone case, confirming that edge is independently load-bearing."""
    output = _run_pytest_against_impl(tmp_path, _MUTANT_WILDCARD)
    failed = _failed_test_names(output)
    assert failed == {"test_put_if_match_wildcard_fails_on_nonexistent_resource"}, output
    assert "15 passed" in output, output


def test_naive_baseline_fails_most_keystones(tmp_path):
    """A submission that ignores conditional headers entirely must fail the large majority of
    keystone cases, confirming the conditional machinery is not redundant with basic put/get."""
    output = _run_pytest_against_impl(tmp_path, _NAIVE_IGNORES_CONDITIONS)
    failed = _failed_test_names(output)
    keystone_names = {node_id.split("::")[1] for node_id in c22.KEYSTONE_TEST_IDS}
    failed_keystones = failed & keystone_names
    assert len(failed_keystones) >= 5, (failed, keystone_names)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c22", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c22"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c22.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c22.get_compiled_plan()

    assert (private / c22._TEST_FILE_PATH).read_text() == c22._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c22._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c22.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
