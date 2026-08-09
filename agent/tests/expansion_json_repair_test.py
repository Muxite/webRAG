"""Malformed-expansion-JSON repair — offline, no LLM.

A cheap executor model regularly returns a plan wrapped in a markdown fence or prose, or a plan
TRUNCATED at the stage's token ceiling. The old salvage was a single non-nesting regex
(``\\{[^{}]*"candidates"[^{}]*\\}``) that cannot match once ``candidates`` holds objects — i.e.
always — so the run silently continued with an EMPTY plan (no children, no tool use). These tests
pin the brace-balanced / truncation repair, including nested JSON inside a candidate, and the
fail-safe (empty plan, never an exception) when the text is genuinely unrepairable.
"""
from __future__ import annotations

import json
import logging

from agent.app.idea_policies.expansion import LlmExpansionPolicy, _repair_json_object


def _policy() -> LlmExpansionPolicy:
    policy = LlmExpansionPolicy.__new__(LlmExpansionPolicy)  # no LLM/connector needed for parsing
    policy._logger = logging.getLogger("expansion-repair-test")
    policy._cfg = None
    return policy


def _parse(content):
    """``_parse_candidates`` needs only the logger and (when the flag is read) the config."""
    from agent.app.idea_policies.config import IdeaConfig

    policy = _policy()
    policy._cfg = IdeaConfig.from_settings({})
    return policy._parse_candidates(content)


_NESTED = (
    '{"candidates": ['
    '{"title": "Find the depth", "action": "search", "details": {"action": "search", '
    '"query": "Quesnel Lake depth", "filters": {"lang": "en", "sites": ["wikipedia.org"]}}},'
    '{"title": "Read it", "action": "visit", "details": {"action": "visit", '
    '"urls": ["https://en.wikipedia.org/wiki/Quesnel_Lake"]}}'
    '], "meta": {"strategy": "chain"}}'
)


# ---------------------------------------------------------------------------
# _repair_json_object
# ---------------------------------------------------------------------------
def test_repair_recovers_fenced_plan_with_nested_json_in_candidates():
    # The exact shape the old regex could never match: nested objects AND arrays inside a candidate.
    fenced = "Here is the plan:\n```json\n" + _NESTED + "\n```\nLet me know."
    data = _repair_json_object(fenced, required_key="candidates")
    assert data is not None
    assert [c["title"] for c in data["candidates"]] == ["Find the depth", "Read it"]
    assert data["candidates"][0]["details"]["filters"]["sites"] == ["wikipedia.org"]
    assert data["meta"] == {"strategy": "chain"}


def test_repair_recovers_truncated_plan_at_the_last_complete_candidate():
    truncated = _NESTED[: _NESTED.index('{"title": "Read it"') + 40]
    data = _repair_json_object(truncated, required_key="candidates")
    assert data is not None
    # The first (complete) candidate survives with its nested details; the cut one is dropped or
    # kept only up to its last complete value — either way the plan is non-empty and well-formed.
    assert data["candidates"][0]["details"]["query"] == "Quesnel Lake depth"
    assert all(isinstance(c, dict) for c in data["candidates"])


def test_repair_handles_bare_top_level_array_and_trailing_commas():
    bare = '[{"title": "a", "details": {"action": "search", "query": "x"}},]'
    data = _repair_json_object(bare, required_key="candidates")
    assert data == {"candidates": [{"title": "a", "details": {"action": "search", "query": "x"}}]}


def test_repair_ignores_inner_braces_inside_strings():
    noisy = 'text {"candidates": [{"title": "use {curly} braces", "details": {"q": "a}b"}}]} tail'
    data = _repair_json_object(noisy, required_key="candidates")
    assert data["candidates"][0]["title"] == "use {curly} braces"
    assert data["candidates"][0]["details"]["q"] == "a}b"


def test_repair_fails_safe_on_unrepairable_text():
    for junk in ("", "I cannot help with that.", "{{{", '{"candidates": '):
        assert _repair_json_object(junk, required_key="candidates") is None


def test_repair_does_not_relabel_an_unrelated_inner_array():
    # An object whose list lives under a different key is returned AS IS (the caller then logs
    # "no candidates") — inventing candidates out of an unrelated array would be worse.
    data = _repair_json_object('{"notes": ["a", "b"]} trailing', required_key="candidates")
    assert data == {"notes": ["a", "b"]}


# ---------------------------------------------------------------------------
# _parse_candidates integration
# ---------------------------------------------------------------------------
def test_parse_candidates_recovers_a_plan_the_old_regex_dropped():
    cleaned, meta = _parse("```json\n" + _NESTED + "\n```")
    assert [c["title"] for c in cleaned] == ["Find the depth", "Read it"]
    assert cleaned[0]["details"]["query"] == "Quesnel Lake depth"
    assert meta == {"strategy": "chain"}


def test_parse_candidates_recovers_a_truncated_plan():
    cleaned, _meta = _parse(_NESTED[: _NESTED.index('{"title": "Read it"')])
    assert len(cleaned) == 1
    assert cleaned[0]["title"] == "Find the depth"


def test_parse_candidates_still_degrades_to_an_empty_plan_when_unrepairable():
    assert _parse("Sorry, I can't produce a plan.") == ([], {})
    assert _parse(None) == ([], {})


def test_parse_candidates_accepts_a_bare_valid_top_level_array():
    # Valid JSON, wrong shape: a bare array parses fine, so it never reaches the repair path —
    # it must not blow up on ``data.get`` (it used to raise AttributeError).
    cleaned, meta = _parse('[{"title": "a", "action": "search", "details": {"query": "x"}}]')
    assert [c["title"] for c in cleaned] == ["a"]
    assert meta == {}


def test_parse_candidates_ignores_a_truthy_non_dict_meta():
    # Live-observed (2026-08-08, qwen2.5:1.5b): a malformed but TRUTHY "meta" field survives
    # `meta or {}` (only falsy values get replaced) and then `dict(meta)` throws TypeError,
    # killing the whole expansion step instead of just ignoring the bad field.
    for bad_meta in (True, ["a"], "oops"):
        cleaned, meta = _parse(
            '{"candidates": [{"title": "a", "details": {"action": "search", "query": "x"}}], '
            f'"meta": {json.dumps(bad_meta)}}}'
        )
        assert [c["title"] for c in cleaned] == ["a"]
        assert meta == {}
