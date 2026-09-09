"""Offline adversarial tests for task 232 (undersea tunnel sections under on-page decoys).

232 exists to measure ATTRIBUTION, not arithmetic: every wrong answer it is designed to catch is
a number English Wikipedia really prints, on one of the two pages the task names. These tests pin
that each such wrong read scores 0 on the keystone, and that the honest read scores 1.

No network, no model: the validators are pure functions over a result dict.
"""
from __future__ import annotations

import importlib

import pytest

M = importlib.import_module(
    "agent.app.idea_tests.test_232_tier5_undersea_section_decoy_difference")

SEEN = {"visit": {"count": 2}}
UNSEEN = {"visit": {"count": 0}}


def _result(text: str) -> dict:
    return {"deliverables": [text], "final_deliverable": text}


def test_the_honest_read_passes_the_keystone():
    out = M.validate_keystone(_result("The difference is 14.6 km."), SEEN)
    assert out["passed"] and out["score"] == 1.0


def test_the_keystone_requires_an_actual_page_visit():
    """A parametric guess that happens to be right earns nothing."""
    assert not M.validate_keystone(_result("14.6 km"), UNSEEN)["passed"]


@pytest.mark.parametrize("label,answer", [
    ("both TOTAL lengths (the summary-shaped read)", "3.39 km"),
    ("Seikan total minus the Channel figure quoted on Seikan's page", "15.95 km"),
    ("Channel total minus Seikan undersea", "27.2 km"),
    ("sum instead of difference", "61.2 km"),
    ("read Seikan's page for both operands", "0 km"),
])
def test_every_on_page_decoy_fails_the_keystone(label: str, answer: str):
    assert not M.validate_keystone(_result(f"The difference is {answer}."), SEEN)["passed"], label


def test_reporting_the_rival_figure_for_both_tunnels_is_caught_by_coverage():
    """Seikan's body prints the Channel Tunnel's 37.9 km beside the words "undersea section".
    An agent that copies it for BOTH tunnels must not be credited with two operands."""
    text = "Channel Tunnel: 37.9 km undersea. Seikan Tunnel: 37.9 km undersea."
    out = M.validate_coverage(_result(text), SEEN)
    assert out["score"] == 0.5 and not out["passed"], out


def test_coverage_credits_both_only_for_the_two_distinct_correct_operands():
    text = ("Channel Tunnel: 37.9 km undersea (wiki/Channel_Tunnel). "
            "Seikan Tunnel: 23.3 km undersea (wiki/Seikan_Tunnel).")
    assert M.validate_coverage(_result(text), SEEN)["score"] == 1.0


def test_citations_are_not_credited_without_the_keystone():
    text = "See https://en.wikipedia.org/wiki/Seikan_Tunnel — the difference is 3.39 km."
    assert M.validate_citations(_result(text), SEEN)["score"] == 0.0


def test_the_derived_value_is_printed_on_neither_source_page():
    """Anti-leak: the task statement and the compiled plan must never contain the answer."""
    assert str(M.DERIVED) not in M.get_task_statement()
    plan = M.get_compiled_plan()
    blob = " ".join([leaf["instruction"] for leaf in plan["leaves"]] + [plan["aggregation"]])
    assert str(M.DERIVED) not in blob
    for operand in (M.OP_A["value"], M.OP_B["value"]):
        assert str(operand) not in blob, "the plan must not leak a raw operand either"
