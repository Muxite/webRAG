"""Offline tests for the mechanical task-shape registry (`scripts/task_shapes.py`).

Deliberately absent: any assertion on how MANY tasks land in a shape. The adversarial review
that motivated this registry flagged exactly that -- a count assertion turns the target count
into an input and re-contaminates the classification. These tests check that the registry is
*well-formed, complete and deterministic*, never that it agrees with a prior number.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import task_shapes as ts  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return ts.build_registry()


@pytest.fixture(scope="module")
def shipped():
    return ts.load_registry()


def _suite_ids():
    return set(ts.suite59_ids())


def test_registry_covers_exactly_suite59(registry):
    ids = {k for k in registry if not k.startswith("_")}
    suite = _suite_ids()
    assert ids - suite == set(), f"registry has ids outside suite59: {sorted(ids - suite)}"
    assert suite - ids == set(), f"suite59 ids missing from registry: {sorted(suite - ids)}"


def test_every_shape_is_a_valid_enum(registry):
    for test_id, entry in registry.items():
        if test_id.startswith("_"):
            continue
        assert entry["shape"] in ts.SHAPES, f"{test_id} has invalid shape {entry['shape']!r}"
        assert isinstance(entry["breadth"], bool)
        assert isinstance(entry["evidence"], list)


def test_classified_tasks_carry_evidence(registry):
    """A non-'other' classification must name the predicate(s) that produced it."""
    for test_id, entry in registry.items():
        if test_id.startswith("_"):
            continue
        if entry["shape"] != "other":
            assert entry["evidence"], f"{test_id} classified {entry['shape']} with no evidence"
        else:
            assert entry["evidence"] == [], f"{test_id} is 'other' but fired {entry['evidence']}"


def test_breadth_implies_aggregation(registry):
    for test_id, entry in registry.items():
        if test_id.startswith("_"):
            continue
        if entry["breadth"]:
            assert entry["shape"] == "aggregation", (
                f"{test_id} is flagged breadth but shape is {entry['shape']}"
            )


def test_rule_string_is_non_empty(registry):
    assert isinstance(registry["_rule"], str)
    assert len(registry["_rule"].strip()) > 200
    assert registry["_written"]


def test_aggregation_ids_are_stable_across_two_runs():
    """Determinism: source in, same answer out. No dict-ordering or import-order drift."""
    first = ts.build_registry()
    second = ts.build_registry()
    assert ts.aggregation_ids(first) == ts.aggregation_ids(second)
    assert ts.shape_map(first) == ts.shape_map(second)
    assert ts.breadth_ids(first) == ts.breadth_ids(second)


def test_shape_map_drops_underscore_keys_and_is_flat(registry):
    flat = ts.shape_map(registry)
    assert all(not k.startswith("_") for k in flat)
    assert all(isinstance(v, str) and v in ts.SHAPES for v in flat.values())
    assert set(flat) == {k for k in registry if not k.startswith("_")}


def test_aggregation_ids_subset_of_shape_map(registry):
    agg = ts.aggregation_ids(registry)
    assert agg == sorted(agg)
    flat = ts.shape_map(registry)
    assert all(flat[i] == "aggregation" for i in agg)


def test_shipped_json_matches_a_fresh_derivation(shipped, registry):
    """The committed artifact is reproducible from source alone."""
    assert ts.shape_map(shipped) == ts.shape_map(registry)
    assert ts.breadth_ids(shipped) == ts.breadth_ids(registry)
    assert shipped["_rule"] == ts.RULE


def test_cli_writes_registry_and_flat_map(tmp_path):
    out = tmp_path / "reg.json"
    flat = tmp_path / "flat.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "services"), str(REPO_ROOT / "agent")]
    )
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "task_shapes.py"),
         "--out", str(out), "--flat", str(flat)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    written = json.loads(out.read_text())
    flat_map = json.loads(flat.read_text())
    assert "_rule" in written
    # compare_arms --shapes consumes exactly this form: {id: "<shape>"}.
    assert flat_map == ts.shape_map(written)
    assert all(isinstance(v, str) for v in flat_map.values())


def test_ladder_comment_labels_parse_but_are_not_an_input():
    """The cross-check source parses; it is used only by the diagnostic table."""
    labels = ts.ladder_comment_labels()
    assert labels, "failed to parse any in-comment label from core_long24"
    assert set(labels.values()) <= {"aggregation", "chain", "mixed-comment", "unlabelled"}
    rows = ts.agreement_table(ts.build_registry())
    assert rows
    assert all(r[3] in ("agree", "DISAGREE", "n/a") for r in rows)
