"""
Unit tests for the graph mandate-URL fix (A1) — free, no LLM/network.

The pilot showed the GoT planner emitting `visit` candidates with no URL on
explicit-URL mandates, so the graph visited 0 pages and failed where naive_rag
passed. These tests lock in the three-part fix:
  1. expansion recovers a visit URL from the mandate (or drops the zombie node),
  2. the MandateUrlInjectionHook self-heals a URL-less visit child in place and
     does NOT inject a duplicate for an already-covered URL,
  3. title<->URL slug matching.
"""
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.expansion import LlmExpansionPolicy
from agent.app.idea_policies.post_expansion_hooks import (
    MandateUrlInjectionHook,
    _best_url_for_title,
)

EIFFEL = "https://en.wikipedia.org/wiki/Eiffel_Tower"
LIBERTY = "https://en.wikipedia.org/wiki/Statue_of_Liberty"
MANDATE = (
    f"Compare the completion dates using these pages: {EIFFEL} (Eiffel Tower) "
    f"and {LIBERTY} (Statue of Liberty). Which was first and by how many years?"
)


def _visit_urls(graph, parent_id):
    out = []
    for cid in graph.get_node(parent_id).children:
        c = graph.get_node(cid)
        if NodeDetailsExtractor.get_action(c.details) == IdeaActionType.VISIT.value:
            out.append(c.details.get(DetailKey.URL.value) or c.details.get("optional_url"))
    return out


# ---- slug matching ------------------------------------------------------------

def test_best_url_for_title_matches_by_slug():
    urls = [EIFFEL, LIBERTY]
    assert _best_url_for_title("Visit Eiffel Tower Wikipedia page", urls) == EIFFEL
    assert _best_url_for_title("Read the Statue of Liberty article", urls) == LIBERTY


def test_best_url_for_title_single_url_always_returns_it():
    assert _best_url_for_title("anything at all", [EIFFEL]) == EIFFEL


def test_best_url_for_title_no_match_returns_none():
    assert _best_url_for_title("totally unrelated topic", [EIFFEL, LIBERTY]) is None
    assert _best_url_for_title("x", []) is None


# ---- expansion helpers (no constructor needed) --------------------------------

def _bare_policy():
    pol = object.__new__(LlmExpansionPolicy)  # skip heavy __init__
    pol._logger = logging.getLogger("test")
    return pol


def test_mandate_urls_extracted_from_root():
    graph = IdeaDag(root_title=MANDATE, root_details={"mandate": MANDATE})
    pol = _bare_policy()
    urls = pol._mandate_urls(graph)
    assert EIFFEL in urls and LIBERTY in urls


def test_match_mandate_url_picks_right_page():
    pol = _bare_policy()
    assert pol._match_mandate_url("Visit Eiffel Tower page", [EIFFEL, LIBERTY]) == EIFFEL
    assert pol._match_mandate_url("Visit Statue of Liberty page", [EIFFEL, LIBERTY]) == LIBERTY


# ---- hook self-heal + no-duplicate -------------------------------------------

def test_hook_repairs_urlless_visit_child_and_injects_missing_only():
    graph = IdeaDag(root_title=MANDATE, root_details={"mandate": MANDATE})
    root = graph.root_id()
    # Planner emitted a URL-less visit node that merely names Eiffel in its title.
    child = graph.add_child(
        parent_id=root,
        title="Visit Eiffel Tower Wikipedia page to extract completion date",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                 DetailKey.GOAL.value: "extract Eiffel completion date"},
    )

    MandateUrlInjectionHook().apply(graph, root, 0, MANDATE, logging.getLogger("test"))

    # The malformed child was repaired in place (not left empty, not duplicated).
    assert child.details.get(DetailKey.URL.value) == EIFFEL
    urls = _visit_urls(graph, root)
    assert urls.count(EIFFEL) == 1, f"Eiffel should not be duplicated: {urls}"
    # The other mandate URL (never covered) was injected.
    assert LIBERTY in urls
