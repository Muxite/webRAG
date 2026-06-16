"""
Unit tests for the lean settings overlay (A2) — free, no LLM/network.

The graph over-spends on easy tasks; the lean overlay caps token budgets +
branching for short-`weight` tasks while leaving hard (`long`) tasks at full budget.
"""
import importlib

runner = importlib.import_module("agent.app.idea_test_runner")


def test_short_weight_caps_budgets(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_LEAN", raising=False)
    base = {"final_max_tokens": 120000, "max_branching": 5, "max_total_nodes": 500}
    out = runner._apply_lean_overlay(base, {"weight": "short"})
    assert out["final_max_tokens"] == 4096
    assert out["max_branching"] == 3
    assert out["max_total_nodes"] == 40
    assert out["max_links_per_visit"] == 5  # caps heavy per-visit link/Chroma work
    # original dict untouched (copy semantics)
    assert base["final_max_tokens"] == 120000


def test_long_weight_unchanged(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_LEAN", raising=False)
    base = {"final_max_tokens": 120000, "max_branching": 5}
    out = runner._apply_lean_overlay(base, {"weight": "long"})
    assert out["final_max_tokens"] == 120000
    assert out["max_branching"] == 5


def test_env_force_off_overrides_short(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LEAN", "0")
    base = {"final_max_tokens": 120000}
    out = runner._apply_lean_overlay(base, {"weight": "short"})
    assert out["final_max_tokens"] == 120000


def test_env_force_on_applies_to_long(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LEAN", "1")
    base = {"final_max_tokens": 120000}
    out = runner._apply_lean_overlay(base, {"weight": "long"})
    assert out["final_max_tokens"] == 4096


def test_existing_smaller_budget_preserved(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_LEAN", raising=False)
    base = {"final_max_tokens": 1024, "max_steps": 8}  # already tighter than caps
    out = runner._apply_lean_overlay(base, {"weight": "short"})
    assert out["final_max_tokens"] == 1024  # min(1024, 4096)
    assert out["max_steps"] == 8           # min(8, 12)
