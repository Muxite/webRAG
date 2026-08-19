"""The keystone_claim family's guarantees, especially the one that is easy to get wrong.

The family's whole validity rests on the false twin being provably false. The
tempting construction -- read a two-alternative KEYSTONE_RX as a right/wrong pair --
would produce a family whose entire negative population is TRUE, and every accuracy
number on it would be meaningless while looking completely ordinary.
"""

from __future__ import annotations

import re

import pytest

from agent.app.promptbench.availability import Availability, OracleLeak, PromptContext
from agent.app.promptbench.factors import build_prompt, is_applicable
from agent.app.promptbench.items import load_all_specs
from agent.app.promptbench.items_keystone import (
    _clean_evidence,
    _false_twin,
    _perturbations,
    build_keystone_claim_items,
    census,
)

SPECS = load_all_specs()
ITEMS = build_keystone_claim_items(SPECS)
BY_MODULE = {s["module"]: s for s in SPECS}


def test_the_family_is_not_empty():
    assert ITEMS, "no keystone items built -- every downstream assertion would vacuously pass"


def test_family_is_balanced_by_construction():
    pos = sum(1 for i in ITEMS if i.posthoc["polarity"] == "TRUE")
    assert pos * 2 == len(ITEMS)


def test_enough_clusters_to_survive_the_loco_rule():
    assert len({i.cluster for i in ITEMS}) >= 5


def test_item_ids_are_unique():
    ids = [i.item_id for i in ITEMS]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# The falsity oracle -- the assertion this family exists or dies on
# ---------------------------------------------------------------------------

def test_every_false_claim_fails_its_own_modules_keystone_regex():
    """The 565/165 trap: those are one value in two units, not a true/false pair."""
    for item in ITEMS:
        if item.posthoc["polarity"] != "FALSE":
            continue
        rx = re.compile(BY_MODULE[item.cluster]["keystone_pattern"], re.IGNORECASE)
        assert not rx.search(item.runtime["claim"]), (
            f"{item.item_id}: the 'false' claim still matches KEYSTONE_RX, so it is true")


def test_every_true_claim_matches_its_own_modules_keystone_regex():
    for item in ITEMS:
        if item.posthoc["polarity"] != "TRUE":
            continue
        rx = re.compile(BY_MODULE[item.cluster]["keystone_pattern"], re.IGNORECASE)
        assert rx.search(item.runtime["claim"]), f"{item.item_id}: the 'true' claim is not true"


def test_a_false_value_never_appears_in_its_own_evidence():
    """A number printed in the evidence for another reason is confirmable, which
    would make the item's own label wrong rather than merely hard."""
    for item in ITEMS:
        if item.posthoc["polarity"] != "FALSE":
            continue
        value = item.posthoc["value"]
        assert not re.search(rf"(?<!\d){re.escape(value)}(?!\d)", item.runtime["evidence"]), (
            f"{item.item_id}: the false value {value!r} is stated in the evidence")


def test_twin_generation_refuses_rather_than_guesses_when_nothing_is_provable():
    """A pattern matching every perturbation must yield no twin, not a wrong one."""
    permissive = re.compile(r"\d+")
    assert _false_twin("565", permissive, "") is None


def test_perturbations_preserve_shape_so_the_twin_is_not_identifiable_by_form():
    for candidate in list(_perturbations("763 mph"))[:8]:
        assert candidate.endswith(" mph")
        assert len(candidate) == len("763 mph")
        assert candidate != "763 mph"


# ---------------------------------------------------------------------------
# Leak guards
# ---------------------------------------------------------------------------

def test_no_statement_states_its_own_keystone():
    for item in ITEMS:
        rx = re.compile(BY_MODULE[item.cluster]["keystone_pattern"], re.IGNORECASE)
        assert not rx.search(item.runtime["statement"]), (
            f"{item.item_id}: the statement gives the answer away")


def test_every_true_item_is_answerable_from_its_evidence():
    """The complement of the leak guard. If the TRUE item's evidence lacked the
    datum, TRUE would be unanswerable while FALSE stayed answerable -- a
    polarity-dependent difficulty gap that manufactures an effect."""
    for item in ITEMS:
        if item.posthoc["polarity"] != "TRUE":
            continue
        rx = re.compile(BY_MODULE[item.cluster]["keystone_pattern"], re.IGNORECASE)
        assert rx.search(item.runtime["evidence"]), (
            f"{item.item_id}: evidence does not contain the keystone")


def test_evidence_cleaning_strips_the_authoring_banner_but_keeps_the_prose():
    raw = ("Test 143: Tier 5 (graph) - a thing.\n"
           "Level: graph   Weight: long   Difficulty: 9/10\n\n"
           "The crater diameter is 630 km.  [KEYSTONE = 630 km]\n")
    cleaned = _clean_evidence(raw)
    assert "Test 143" not in cleaned
    assert "Difficulty" not in cleaned
    assert "630" in cleaned
    assert "[KEYSTONE" not in cleaned


def test_the_two_polarities_differ_only_in_the_claim():
    """Everything else held fixed, so a difference is attributable to the claim."""
    pairs = {}
    for item in ITEMS:
        pairs.setdefault(item.cluster, {})[item.posthoc["polarity"]] = item
    for cluster, both in pairs.items():
        if len(both) != 2:
            continue
        assert both["TRUE"].runtime["evidence"] == both["FALSE"].runtime["evidence"], cluster
        assert both["TRUE"].runtime["statement"] == both["FALSE"].runtime["statement"], cluster
        assert both["TRUE"].runtime["claim"] != both["FALSE"].runtime["claim"], cluster


# ---------------------------------------------------------------------------
# Anti-oracle, and rendering
# ---------------------------------------------------------------------------

def test_the_label_still_cannot_be_read_as_text():
    with pytest.raises(OracleLeak):
        str(ITEMS[0].label)


def test_the_label_still_cannot_be_compared():
    with pytest.raises(OracleLeak):
        _ = ITEMS[0].label == "TRUE"


def test_exposing_a_label_taints_the_signal_oracle():
    assert ITEMS[0].label.expose("test").availability is Availability.ORACLE


def test_no_rendered_prompt_contains_its_own_label_token():
    """TRUE and FALSE are the option tokens, so both appear in OPTIONS. What must
    not happen is the label leaking through some other field."""
    for item in ITEMS:
        for field in ("statement", "evidence", "claim"):
            assert "SATISFIES" not in item.runtime[field]


@pytest.mark.parametrize("variant", ["A0", "A1", "A2", "A3", "A4", "SHIPPED",
                                     "F_json", "G_nostatement", "G_noevidence"])
def test_every_applicable_variant_renders(variant):
    assert is_applicable("keystone_claim", variant)
    prompt = build_prompt(ITEMS[0].runtime,
                          PromptContext(family="keystone_claim", variant=variant, model="t"))
    assert prompt.strip() and "OPTIONS: TRUE | FALSE" in prompt


def test_g_noevidence_actually_withholds_the_evidence():
    item = ITEMS[0]
    with_ev = build_prompt(item.runtime,
                           PromptContext(family="keystone_claim", variant="A1", model="t"))
    without = build_prompt(item.runtime,
                           PromptContext(family="keystone_claim", variant="G_noevidence", model="t"))
    assert item.runtime["evidence"][:60] in with_ev
    assert item.runtime["evidence"][:60] not in without


def test_census_names_what_it_dropped():
    c = census(SPECS)
    assert c["keystone_claim_positive"] == c["keystone_claim_negative"]
    assert isinstance(c["dropped"], dict)
