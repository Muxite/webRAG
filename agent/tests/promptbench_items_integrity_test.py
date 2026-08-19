"""Integrity of the constructed benchmark items, and the SHIPPED-arm parity guard.

These are the checks that make the promptbench numbers mean anything:
the negative population is real, no prompt contains its own answer, and the
SHIPPED arm is still the text the engine actually sends.
"""

from __future__ import annotations

import re

import pytest

from agent.app.promptbench.availability import Availability, OracleLeak, PromptContext
from agent.app.promptbench.factors import (
    ALL_VARIANTS,
    PRIMARY_VARIANTS,
    build_prompt,
    shipped_instruction,
)
from agent.app.promptbench.items import (
    build_select_items,
    build_verify_items,
    census,
    load_specs,
)

SPECS = load_specs()
VERIFY = build_verify_items(SPECS)
SELECT = build_select_items(SPECS)


# ---------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------

def test_verify_family_is_balanced_so_a_precision_number_is_defined():
    pos = sum(1 for i in VERIFY if i.posthoc["polarity"] == "SATISFIES")
    neg = len(VERIFY) - pos
    assert pos > 0 and neg > 0
    assert neg >= 0.5 * pos, "negative control too small for a defined false-positive rate"


def test_both_families_rest_on_enough_clusters_to_survive_the_loco_rule():
    assert len({i.cluster for i in VERIFY}) >= 5
    assert len({i.cluster for i in SELECT}) >= 5


def test_item_ids_are_unique_within_each_family():
    for items in (VERIFY, SELECT):
        ids = [i.item_id for i in items]
        assert len(ids) == len(set(ids))


def test_census_reports_what_was_dropped_rather_than_hiding_it():
    c = census(SPECS)
    assert "dropped_undescribed" in c and "dropped_statement_leak" in c
    assert c["verify_positive"] == c["verify_negative"]


# ---------------------------------------------------------------------------
# No prompt may contain its own answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", PRIMARY_VARIANTS)
def test_select_prompts_do_not_single_out_the_answer(variant):
    """Every candidate is named in a select prompt -- that is the options list.
    The failure this guards against is the answer being singled out, e.g. the
    survivor named where the distractors are not."""
    for item in SELECT:
        ctx = PromptContext(family="select", variant=variant, model="test")
        prompt = build_prompt(item.runtime, ctx)
        named = [c for c in item.runtime["choices"]
                 if re.search(re.escape(c), prompt, re.IGNORECASE)]
        assert len(named) == len(item.runtime["choices"]), (
            f"{item.item_id}: only {len(named)} of {len(item.runtime['choices'])} "
            "candidates appear -- the option set is asymmetric")


def test_prompt_builder_cannot_reach_the_label():
    """The structural guarantee: build_prompt takes (runtime, ctx), and ctx
    carries no label. Reaching for one raises rather than leaking."""
    item = VERIFY[0]
    with pytest.raises(OracleLeak):
        str(item.label)


def test_exposing_a_label_taints_the_signal_oracle():
    signal = VERIFY[0].label.expose("integrity test")
    assert signal.availability is Availability.ORACLE


# ---------------------------------------------------------------------------
# Every variant renders, and the shapes actually differ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_every_variant_renders_for_both_families(variant):
    for family, items in (("verify", VERIFY), ("select", SELECT)):
        ctx = PromptContext(family=family, variant=variant, model="test")
        prompt = build_prompt(items[0].runtime, ctx)
        assert prompt.strip()
        assert "OPTIONS:" in prompt


def test_the_primary_ladder_produces_distinct_prompts():
    """If two arms rendered identically the comparison would be vacuous."""
    item = VERIFY[0]
    rendered = {v: build_prompt(item.runtime, PromptContext(family="verify", variant=v, model="t"))
                for v in PRIMARY_VARIANTS}
    assert len(set(rendered.values())) == len(PRIMARY_VARIANTS)


def test_g_nostatement_actually_withholds_the_statement():
    item = VERIFY[0]
    with_stmt = build_prompt(item.runtime, PromptContext(family="verify", variant="A1", model="t"))
    without = build_prompt(item.runtime, PromptContext(family="verify", variant="G_nostatement", model="t"))
    assert item.runtime["statement"][:80] in with_stmt
    assert item.runtime["statement"][:80] not in without


# ---------------------------------------------------------------------------
# SHIPPED parity -- the arm must be the engine's real text
# ---------------------------------------------------------------------------

def test_shipped_arm_is_imported_from_the_engine_not_retyped():
    from agent.app.idea_policies.actions import VerifyLeafAction

    assert shipped_instruction() == VerifyLeafAction._DEFAULT_SYSTEM_PROMPT


def test_shipped_prompt_still_asks_for_the_answer_before_the_justification():
    """The premise of the whole cycle. If the engine is changed to put the
    reasoning first, this test fails and the pre-registered comparison has to
    be re-stated rather than silently re-interpreted."""
    text = shipped_instruction()
    assert text.index('"verdict"') < text.index('"reasoning"')


def test_shipped_arm_appears_in_the_rendered_prompt():
    item = VERIFY[0]
    prompt = build_prompt(item.runtime, PromptContext(family="verify", variant="SHIPPED", model="t"))
    assert "Graph-of-Thought" in prompt
