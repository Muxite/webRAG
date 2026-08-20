"""Tests for the deterministic task-shape classifier and its expansion.py wiring."""

from __future__ import annotations

import importlib
import os

import pytest

from agent.app.idea_policies.shape_classifier import classify_answer_shape, classify_shape
from agent.app.idea_policies import expansion


def _stmt(module_name: str) -> str:
    mod = importlib.import_module(f"agent.app.idea_tests.{module_name}")
    return mod.get_task_statement()


# ---------------------------------------------------------------------------
# Classification against REAL task statements in the repo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name, expected",
    [
        ("test_095_tier5_branch_eliminate_chain", "branch_eliminate"),
        ("test_065_tier5_leak_resistant_chain", "chain"),
        ("test_051_tier4_dependent_chain", "chain"),
        ("test_055_tier5_multichain_arithmetic", "parallel_merge"),
        ("test_061_tier5_director_birthyear_arithmetic", "parallel_merge"),
        # Fan-out / breadth shapes that are none of the three -> fail open to None.
        ("test_052_tier5_breadth_aggregation", None),
        ("test_059_tier5_computed_ratio_argmax", None),
    ],
)
def test_classify_shape_real_tasks(module_name, expected):
    assert classify_shape(_stmt(module_name)) == expected


def test_classify_shape_empty_fails_open():
    assert classify_shape("") is None
    assert classify_shape(None) is None  # type: ignore[arg-type]


def test_classify_shape_plain_prose_none():
    assert classify_shape("Find the capital of France and report it.") is None


# ---------------------------------------------------------------------------
# Answer-shape classifier (finalize reconcile-chain gate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mandate, expected",
    [
        # single_value: report/give-a-specific-scalar phrasings (the adaptive target tasks)
        ("Report the maximum depth of Quesnel Lake in metres. Give the number.", "single_value"),
        ("Report the land area of Amsterdam Island in square kilometres. Give the number.", "single_value"),
        ("What is the capital of France?", "single_value"),
        ("In what year was the bridge completed?", "single_value"),
        # count
        ("How many novels did the author publish?", "count"),
        # computation
        ("Compute the absolute difference between the two founding years.", "computation"),
        ("What is the ratio of the two populations?", "computation"),
        # argmax: superlative + a selection word
        ("Which of these lakes is the deepest?", "argmax"),
        # disambiguation
        ("Which of the four rivers empties into the English Channel?", "disambiguation"),
        # narrative / open-ended -> None (skip the passes)
        ("Summarize the history and cultural significance of Quesnel Lake.", None),
        ("Describe the ecosystem of the lake in detail.", None),
        ("", None),
    ],
)
def test_classify_answer_shape(mandate, expected):
    assert classify_answer_shape(mandate) == expected


def test_answer_shape_attribute_superlative_is_single_value_not_argmax():
    # "maximum depth" is an attribute NAME, not an argmax over candidates -> single_value.
    assert classify_answer_shape("Report the maximum depth of the lake.") == "single_value"


def test_answer_shape_reuses_classify_shape_labels():
    # A branch-eliminate task is an answer-shaped disambiguation; a chain resolves to single_value.
    assert classify_answer_shape(_stmt("test_095_tier5_branch_eliminate_chain")) == "disambiguation"
    assert classify_answer_shape(_stmt("test_065_tier5_leak_resistant_chain")) == "single_value"
    assert classify_answer_shape(_stmt("test_055_tier5_multichain_arithmetic")) == "computation"


# ---------------------------------------------------------------------------
# Auto-selection wiring in expansion.py
# ---------------------------------------------------------------------------

def _clear_env(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_REASONING_RULES", raising=False)


def test_auto_injects_branch_eliminate_when_env_unset(monkeypatch):
    _clear_env(monkeypatch)
    block = expansion._auto_reasoning_rules(_stmt("test_095_tier5_branch_eliminate_chain"))
    assert "Mandatory reasoning rules" in block
    # A distinctive line from reasoning_rules/branch_eliminate.md.
    assert "before you may name a survivor" in block


def test_auto_injects_chain_when_env_unset(monkeypatch):
    _clear_env(monkeypatch)
    assert classify_shape(_stmt("test_065_tier5_leak_resistant_chain")) == "chain"
    block = expansion._auto_reasoning_rules(_stmt("test_065_tier5_leak_resistant_chain"))
    assert "Mandatory reasoning rules" in block
    # A distinctive line from reasoning_rules/chain.md.
    assert "Propose ONE next hop at a time" in block


def test_auto_injects_parallel_merge_when_env_unset(monkeypatch):
    _clear_env(monkeypatch)
    assert classify_shape(_stmt("test_055_tier5_multichain_arithmetic")) == "parallel_merge"
    block = expansion._auto_reasoning_rules(_stmt("test_055_tier5_multichain_arithmetic"))
    assert "Mandatory reasoning rules" in block
    # A distinctive line from reasoning_rules/parallel_merge.md.
    assert "BOTH chains have actually produced their final value" in block


def test_auto_falls_back_to_local_text_when_root_unclassified(monkeypatch):
    # Root mandate doesn't classify; the node-local goal does -> additive fallback
    # fires. This must never happen the other way around (see the next test).
    _clear_env(monkeypatch)
    root = "Find the capital of France and report it."
    local = _stmt("test_065_tier5_leak_resistant_chain")
    assert classify_shape(root) is None
    block = expansion._auto_reasoning_rules(root, local_text=local)
    assert "Mandatory reasoning rules" in block
    assert "Propose ONE next hop at a time" in block


def test_auto_local_text_never_overrides_a_classified_root(monkeypatch):
    # Root classifies as branch_eliminate; local text alone would classify as chain.
    # The root verdict must win -- local text is a fallback for an UNCLASSIFIED root
    # only, never a competing signal against a root that already resolved.
    _clear_env(monkeypatch)
    root = _stmt("test_095_tier5_branch_eliminate_chain")
    local = _stmt("test_065_tier5_leak_resistant_chain")
    block = expansion._auto_reasoning_rules(root, local_text=local)
    assert "before you may name a survivor" in block
    assert "Propose ONE next hop at a time" not in block


def test_auto_no_injection_for_unclassified(monkeypatch):
    _clear_env(monkeypatch)
    assert expansion._auto_reasoning_rules("Just report a plain fact.") == ""


def test_manual_env_overrides_auto_classification(monkeypatch):
    # Mandate classifies as branch_eliminate, but the operator explicitly set the env
    # var; the manual selection must win. Here the manual value is a valid name that
    # differs in mechanism (explicit) from auto — confirm _load_reasoning_rules is what
    # drives the block when the env var is set. All three shapes are now valid manual
    # names (all three have rule files).
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "branch_eliminate")
    manual = expansion._load_reasoning_rules()
    assert "Mandatory reasoning rules" in manual
    assert "before you may name a survivor" in manual
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "chain")
    manual = expansion._load_reasoning_rules()
    assert "Mandatory reasoning rules" in manual
    assert "Propose ONE next hop at a time" in manual
    # When the env var is set to a name that isn't a recognised shape at all, the
    # manual path returns "" and auto is NOT consulted (the call site only
    # auto-selects when unset).
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "not_a_real_shape")
    assert expansion._load_reasoning_rules() == ""


def test_manual_none_disables_and_does_not_auto(monkeypatch):
    # Explicit "none" disables injection; the call-site guard also blocks auto because
    # the env var is set (non-empty).
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "none")
    assert expansion._load_reasoning_rules() == ""
    assert os.environ.get("IDEA_TEST_REASONING_RULES", "").strip() != ""
