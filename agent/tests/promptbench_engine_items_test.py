"""The followup and goal_achieved families.

Both reach 28 clusters because they need only candidate NAMES plus the statement.
The risk they carry is different from keystone_claim's: not a wrong label, but a
label a model can guess from surface form without reading anything. These tests
close the shortcuts one at a time.
"""

from __future__ import annotations

import re

import pytest

from agent.app.promptbench.availability import Availability, OracleLeak, PromptContext
from agent.app.promptbench.factors import build_prompt, is_applicable
from agent.app.promptbench.items import load_all_specs
from agent.app.promptbench.items_engine import (
    build_followup_items,
    build_goal_achieved_items,
    census,
)

SPECS = load_all_specs()
FOLLOWUP = build_followup_items(SPECS)
GOAL = build_goal_achieved_items(SPECS)


@pytest.mark.parametrize("items", [FOLLOWUP, GOAL], ids=["followup", "goal_achieved"])
def test_families_are_non_empty_and_balanced(items):
    assert items
    polarities = [i.posthoc["polarity"] for i in items]
    assert len(set(polarities)) == 2
    counts = {p: polarities.count(p) for p in set(polarities)}
    assert len(set(counts.values())) == 1, f"unbalanced: {counts}"


@pytest.mark.parametrize("items", [FOLLOWUP, GOAL], ids=["followup", "goal_achieved"])
def test_families_reach_more_clusters_than_the_description_bearing_ones(items):
    """The point of these two families: they do not need per-candidate `desc`, so
    they cover the nine sets `select` must drop."""
    assert len({i.cluster for i in items}) >= 25


@pytest.mark.parametrize("items", [FOLLOWUP, GOAL], ids=["followup", "goal_achieved"])
def test_item_ids_are_unique(items):
    ids = [i.item_id for i in items]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Shortcut closure
# ---------------------------------------------------------------------------

def test_goal_achieved_length_does_not_predict_the_label():
    """An earlier draft made the NOT_ACHIEVED synthesis 2.4x longer than ACHIEVED,
    so a model could score well above chance on character count alone, without
    reading a word."""
    by = {"ACHIEVED": [], "NOT_ACHIEVED": []}
    for i in GOAL:
        by[i.posthoc["polarity"]].append(i.posthoc["synthesis_chars"])
    a = sum(by["ACHIEVED"]) / len(by["ACHIEVED"])
    n = sum(by["NOT_ACHIEVED"]) / len(by["NOT_ACHIEVED"])
    assert abs(a - n) / max(a, n) < 0.10, f"length cue: ACHIEVED {a:.0f} vs NOT_ACHIEVED {n:.0f}"


def test_goal_achieved_polarities_share_their_preamble():
    """They must differ in whether an answer is named, and in nothing else."""
    pairs = {}
    for i in GOAL:
        pairs.setdefault(i.cluster, {})[i.posthoc["polarity"]] = i
    for cluster, both in pairs.items():
        a, n = both["ACHIEVED"].runtime["synthesis"], both["NOT_ACHIEVED"].runtime["synthesis"]
        shared = "The approach follows the task's structure closely."
        assert a.startswith(shared) and n.startswith(shared), cluster
        assert a != n


def test_goal_achieved_negative_names_no_answer():
    for i in GOAL:
        if i.posthoc["polarity"] == "NOT_ACHIEVED":
            assert "ANSWER:" not in i.runtime["synthesis"]
        else:
            assert "ANSWER:" in i.runtime["synthesis"]


def test_followup_coverage_matches_the_label():
    """YES means candidates named in the statement are not covered by the listed
    sub-tasks; NO means they all are. The label is that set difference, so a
    mismatch here is a wrong label, not a hard item."""
    for i in FOLLOWUP:
        covered = i.posthoc["n_covered"]
        total = i.posthoc["n_candidates"]
        expected = "NO" if covered >= total else "YES"
        assert i.posthoc["polarity"] == expected, i.item_id


def test_followup_split_point_is_not_always_the_same_index():
    """A fixed split point would make position, rather than coverage, the cue."""
    cuts = {i.posthoc["n_covered"] for i in FOLLOWUP if i.posthoc["polarity"] == "YES"}
    assert len(cuts) > 1, f"every positive item stops at the same index: {cuts}"


def test_no_authored_field_contains_its_own_label_token():
    """The fields this repo writes must not state the answer.

    Scoped to the authored fields on purpose. The task statement is hand-written
    prose from the source module, and ``YES``/``NO`` occur in it as ordinary
    English ("You are given NO URLs..."). Banning the substring there would fail on
    language rather than on leakage -- and the statement cannot leak the label
    anyway, which the next test establishes structurally rather than lexically.
    """
    authored = {"completed_task", "completed_result", "siblings", "synthesis"}
    for items in (FOLLOWUP, GOAL):
        for item in items:
            token = item.posthoc["polarity"]
            for key in authored & set(item.runtime):
                assert not re.search(rf"\b{re.escape(token)}\b", str(item.runtime[key])), (
                    f"{item.item_id}: {token} appears in authored field {key}")


def test_the_statement_is_identical_across_both_polarities():
    """A field that is byte-identical for TRUE and FALSE carries zero bits about
    which one this is. That is a stronger guarantee than any keyword scan."""
    for items in (FOLLOWUP, GOAL):
        by_cluster = {}
        for item in items:
            by_cluster.setdefault(item.cluster, {})[item.posthoc["polarity"]] = item
        for cluster, both in by_cluster.items():
            statements = {i.runtime["statement"] for i in both.values()}
            assert len(statements) == 1, cluster


# ---------------------------------------------------------------------------
# Anti-oracle, and rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("items", [FOLLOWUP, GOAL], ids=["followup", "goal_achieved"])
def test_labels_remain_unreadable(items):
    with pytest.raises(OracleLeak):
        str(items[0].label)
    assert items[0].label.expose("test").availability is Availability.ORACLE


@pytest.mark.parametrize("family,items", [("followup", FOLLOWUP), ("goal_achieved", GOAL)])
@pytest.mark.parametrize("variant", ["A0", "A1", "A2", "A3", "A4", "SHIPPED",
                                     "F_json", "G_nostatement"])
def test_every_applicable_variant_renders(family, items, variant):
    assert is_applicable(family, variant)
    prompt = build_prompt(items[0].runtime,
                          PromptContext(family=family, variant=variant, model="t"))
    assert prompt.strip() and "OPTIONS:" in prompt


@pytest.mark.parametrize("family,items", [("followup", FOLLOWUP), ("goal_achieved", GOAL)])
def test_g_nostatement_withholds_the_statement(family, items):
    item = items[0]
    with_stmt = build_prompt(item.runtime, PromptContext(family=family, variant="A1", model="t"))
    without = build_prompt(item.runtime,
                           PromptContext(family=family, variant="G_nostatement", model="t"))
    assert item.runtime["statement"][:60] in with_stmt
    assert item.runtime["statement"][:60] not in without


def test_census_reports_balance_and_drops():
    c = census(SPECS)
    assert c["followup_positive"] == c["followup_negative"]
    assert c["goal_achieved_positive"] == c["goal_achieved_negative"]
    assert "dropped" in c
