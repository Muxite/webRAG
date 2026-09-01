"""The KPI specification is frozen; changing it must be impossible to do quietly.

``docs/LEDGER_KPI_SPEC.md`` and its machine form ``scripts/ledger_kpi_spec.json`` were written
BEFORE the build work of the phase they measure, so that a later result cannot be manufactured by
moving a definition. The hash assertion below is the enforcement: any edit to a threshold, an
operation set, a tolerance or the holdout split fails this test until the constant here is
updated in the SAME commit, which puts the change in the diff and in ``git log`` where a reviewer
and a future session will both see it.

This is the anti-gaming guard the phase was scoped around. It is not a formality: the phase's
stated goal is to build a system that scores well on metrics this repo authors itself, and the
only thing separating that from marking your own homework is that the homework was written down
first and cannot be edited invisibly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_PATH = _ROOT / "scripts" / "ledger_kpi_spec.json"
_DOC_PATH = _ROOT / "docs" / "LEDGER_KPI_SPEC.md"

#: sha256 of ``scripts/ledger_kpi_spec.json`` as frozen on 2026-09-01. Updating this constant is
#: the deliberate, reviewable act of amending a frozen specification -- see the amendment section
#: at the bottom of the doc, which must be updated in the same commit.
FROZEN_SHA256 = "5be51a6da4365c31b2b817b065d8cb94e26bb35052a796f37749c478381fdc9a"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(_SPEC_PATH.read_bytes())


def test_spec_file_is_unchanged_since_it_was_frozen():
    digest = hashlib.sha256(_SPEC_PATH.read_bytes()).hexdigest()
    assert digest == FROZEN_SHA256, (
        "scripts/ledger_kpi_spec.json changed. If that was deliberate, update FROZEN_SHA256 in "
        "this test AND add a dated entry to the Amendments section of docs/LEDGER_KPI_SPEC.md, "
        "in the same commit. If it was not deliberate, revert it -- every number reported "
        "against the old spec is invalidated by a silent edit."
    )


def test_the_prose_spec_exists_and_declares_itself_frozen():
    text = _DOC_PATH.read_text()
    assert "frozen" in text.lower()
    assert "## Amendments" in text


def test_clusters_partition_the_numeric_suite(spec):
    tasks = [t for group in spec["task_clusters"].values() for t in group]
    assert len(tasks) == len(set(tasks)), "a task appears in two clusters"
    assert sorted(tasks) == list(range(210, 232))


def test_holdout_is_exactly_what_the_stated_rule_produces(spec):
    """The split must be recomputable from the rule, never a hand-picked list.

    The rule reads cluster membership only -- never a score -- so the split cannot be steered by
    looking at results. Recomputing it here means a hand-edit to the ``holdout`` array fails.
    """
    assert spec["split_rule"] == "max_task_id_per_cluster_is_holdout"
    recomputed = sorted(max(group) for group in spec["task_clusters"].values())
    assert recomputed == sorted(spec["holdout"])


def test_holdout_and_tuning_are_a_disjoint_cover(spec):
    holdout, tuning = set(spec["holdout"]), set(spec["tuning"])
    assert not holdout & tuning
    assert holdout | tuning == set(range(210, 232))


def test_cluster_membership_matches_the_live_task_modules(spec):
    """A task whose category moves must break the split, not silently re-stratify it.

    Categories are read from the task modules themselves, so adding a task to the suite or
    editing one's ``category`` fails here rather than quietly changing what the holdout covers.
    """
    metadata = pytest.importorskip("agent.app.idea_tests", reason="task modules unavailable")
    del metadata
    import importlib
    import pkgutil

    import agent.app.idea_tests as suite

    by_id: dict[int, str] = {}
    for info in pkgutil.iter_modules(suite.__path__):
        if not info.name.startswith("test_2"):
            continue
        module = importlib.import_module(f"agent.app.idea_tests.{info.name}")
        getter = getattr(module, "get_test_metadata", None)
        if getter is None:
            continue
        meta = getter()
        try:
            task_id = int(meta.get("test_id"))
        except (TypeError, ValueError):
            continue
        if 210 <= task_id <= 231:
            by_id[task_id] = str(meta.get("category") or "")

    if not by_id:
        pytest.skip("no numeric-suite modules discovered")

    assert set(by_id) == set(range(210, 232)), "the numeric suite gained or lost a task"

    # Every task in a cluster must share a category prefix with the others in that cluster: the
    # clusters encode the MECHANISM under test, which is what the holdout stratifies over.
    for name, group in spec["task_clusters"].items():
        categories = {by_id[t] for t in group if t in by_id}
        assert categories, f"cluster {name} matched no live task"
        prefixes = {c.split("(")[0].strip() for c in categories}
        assert len(prefixes) == 1, (
            f"cluster {name} spans more than one category ({sorted(prefixes)}); the frozen split "
            "no longer stratifies by mechanism and must be re-derived and re-frozen"
        )


def test_power_declaration_forbids_an_arm_ranking_claim(spec):
    """The spec must keep saying out loud what this n cannot support."""
    power = spec["power"]
    assert power["paired_unit"] == "task"
    assert power["arm_ranking_claim_permitted"] is False
    assert power["n_paired_tasks"] < min(power["required_for_0_10_effect"])


def test_unit_conversion_stays_forbidden(spec):
    """LEDGER_PLAN section 7 non-goal, encoded so a support rule cannot quietly acquire it."""
    assert spec["support"]["unit_conversion_allowed"] is False


def test_false_positive_ceiling_is_declared(spec):
    """A recomputability search can match by coincidence; the floor must be reported, not assumed."""
    assert 0 < spec["support"]["cross_cell_false_positive_ceiling"] <= 0.05
