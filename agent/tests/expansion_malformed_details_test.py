"""Malformed candidate ``details`` must not kill the whole expansion step — offline, no LLM.

LIVE FAILURE (2026-08-15 capability-spectrum sweep, `llama3.2:3b` via ollama, task 134):

    [EXPANSION] Exception during expansion:
        dictionary update sequence element #0 has length 1; 2 is required
      File "agent/app/idea_policies/expansion.py", line 1335, in _parse_candidates
    [EXPANSION] Policy returned 0 candidates
    [EXPANSION] EXPANSION FAILED: Expansion policy returned no candidates!
    [EXPANSION] Created fallback candidate: Analyze and plan next steps...   <- action=None
    [GROUNDING-GATE] zero opened pages on a grounded-research mandate (stripped 3 citations)
    [134] llama3.2:3b [graph]: FAILED (score: 0.00)

Root cause: ``details = candidate.get("details") or {}`` substitutes only on FALSY values, so a
truthy-but-wrong-shaped ``details`` (a weak model emitting a LIST or a STRING where an object
belongs) survives to ``dict(details)``, which raises ``ValueError``. The exception escapes
``_parse_candidates`` and takes the entire expansion step with it — every candidate is lost, not
just the malformed one. The engine then falls back to an action-less node, never calls a tool, and
the grounding gate scores the run 0.

This is the SAME bug class already guarded 8 lines earlier for ``meta`` (whose comment documents
the identical live failure). These tests pin the fix for ``details`` and, critically, that ONE bad
candidate does not destroy its well-formed siblings.
"""
from __future__ import annotations

import logging

import pytest

from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.expansion import LlmExpansionPolicy


def _parse(content):
    policy = LlmExpansionPolicy.__new__(LlmExpansionPolicy)  # no LLM/connector needed for parsing
    policy._logger = logging.getLogger("expansion-malformed-details-test")
    policy._cfg = IdeaConfig.from_settings({})
    return policy._parse_candidates(content)


# The exact shape that raises: `dict(["a"])` -> "sequence element #0 has length 1; 2 is required".
# A truthy non-mapping only reaches `dict()` when `action` is set, so every case here sets one.
@pytest.mark.parametrize("bad_details", [
    '["a"]',              # list of short strings — the live shape
    '[["q"]]',            # list of 1-element lists
    '"search the web"',   # a bare string
    'true',               # a bare bool (mirrors the `meta` guard's recorded live case)
    '42',                 # a number
])
def test_malformed_details_does_not_raise(bad_details):
    content = (
        '{"candidates": [{"title": "Find the engineer", "action": "search", '
        f'"details": {bad_details}}}]}}'
    )
    candidates, _meta = _parse(content)  # must not raise
    assert len(candidates) == 1, "the candidate itself must survive a malformed details field"
    details = candidates[0]["details"]
    assert isinstance(details, dict)
    # The action is the load-bearing field: without it the node gets action=None, never calls a
    # tool, and the grounding gate zeroes the run. Recovering it is the whole point of the fix.
    assert details.get("action") == "search"


def test_one_malformed_candidate_does_not_destroy_its_siblings():
    """The live blast radius: the exception killed the ENTIRE plan, not just the bad candidate."""
    content = (
        '{"candidates": ['
        '{"title": "good one", "action": "search", "details": {"action": "search", '
        '"query": "Statue of Liberty armature engineer"}},'
        '{"title": "malformed", "action": "visit", "details": ["oops"]},'
        '{"title": "good two", "action": "visit", "details": {"action": "visit", '
        '"urls": ["https://en.wikipedia.org/wiki/Gustave_Eiffel"]}}'
        ']}'
    )
    candidates, _meta = _parse(content)
    titles = [c["title"] for c in candidates]
    assert titles == ["good one", "malformed", "good two"]
    assert candidates[0]["details"]["query"] == "Statue of Liberty armature engineer"
    assert candidates[2]["details"]["urls"] == ["https://en.wikipedia.org/wiki/Gustave_Eiffel"]


def test_wellformed_details_is_untouched():
    """Guard against the fix quietly discarding legitimate nested detail payloads."""
    content = (
        '{"candidates": [{"title": "Read it", "action": "visit", "details": '
        '{"action": "visit", "urls": ["https://en.wikipedia.org/wiki/Garabit_viaduct"], '
        '"filters": {"lang": "en"}}}]}'
    )
    candidates, _meta = _parse(content)
    d = candidates[0]["details"]
    assert d["urls"] == ["https://en.wikipedia.org/wiki/Garabit_viaduct"]
    assert d["filters"] == {"lang": "en"}
    assert d["action"] == "visit"
