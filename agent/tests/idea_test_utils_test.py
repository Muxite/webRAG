"""Unit tests for the shared chain-coverage grounding helpers in idea_test_utils.py.

``waypoint_evidence_ok`` / ``waypoint_chain_coverage`` are the fix for the ``validate_chain_coverage``
over-crediting defect (2026-08-16): the original per-task implementations matched each waypoint's
``name_rx`` only against the model's own final-answer text, with credit capped by the AGGREGATE
``observability.visit.count`` -- any N successful visits, regardless of WHICH pages, could bank
credit for up to N named waypoints. Independently re-derived: 20 of 60 stored cells across tasks
135-139 credited strictly more distinct waypoints than were actually visited.

These tests exercise the shared helper directly (offline, no LLM, no live corpus dependency) plus
one regression built from a REAL over-crediting cell in agent/idea_test_results (task 139,
cschain_g/l1b/good_adaptive): a completely off-topic single visit (a health-manual page, nothing
to do with Gaudi or the Sagrada Familia) that the OLD validator credited 1/3 purely because
n_visits >= 1 and "Sagrada Familia" happened to appear somewhere in the echoed mandate text.
"""
from agent.app.idea_test_utils import (
    visited_evidence,
    waypoint_evidence_ok,
    waypoint_chain_coverage,
)


# ---------------------------------------------------------------------------------------------
# waypoint_evidence_ok
# ---------------------------------------------------------------------------------------------

_BRIDGE_WAYPOINT = {
    "name": "John A. Roebling Suspension Bridge (Cincinnati)",
    "name_rx": r"cincinnati|covington|roebling_suspension|roebling\s+suspension",
    "slug_rx": r"roebling_suspension_bridge|wiki/[^ ]*cincinnati",
}


def test_waypoint_evidence_ok_matches_on_slug():
    visited = [{"url": "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge",
                "content": "some page text with no name-level mention"}]
    assert waypoint_evidence_ok(_BRIDGE_WAYPOINT, visited) is True


def test_waypoint_evidence_ok_matches_on_content_from_non_wikipedia_source():
    # A legitimate, non-Wikipedia source (e.g. structuremag.org) whose URL cannot match a
    # `wiki/...` slug pattern, but whose fetched CONTENT genuinely covers the terminal bridge.
    visited = [{"url": "https://www.structuremag.org/?p=12345",
                "content": "The Roebling Suspension Bridge spans the Ohio River between "
                           "Cincinnati and Covington."}]
    assert waypoint_evidence_ok(_BRIDGE_WAYPOINT, visited) is True


def test_waypoint_evidence_ok_rejects_unrelated_page():
    visited = [{"url": "https://www.reddit.com/r/todayilearned/comments/abc/",
                "content": "TIL something about frogs."}]
    assert waypoint_evidence_ok(_BRIDGE_WAYPOINT, visited) is False


def test_waypoint_evidence_ok_false_on_empty_evidence():
    assert waypoint_evidence_ok(_BRIDGE_WAYPOINT, []) is False


# ---------------------------------------------------------------------------------------------
# waypoint_chain_coverage -- direct regression for the donation-page / Reddit / TikTok pattern
# ---------------------------------------------------------------------------------------------

_CHAIN = [
    {"name": "Brooklyn Bridge", "name_rx": r"brooklyn\s+bridge", "slug_rx": r"wiki/brooklyn_bridge"},
    {"name": "John A. Roebling", "name_rx": r"john\s+a\.?\s+roebling|\broebling\b",
     "slug_rx": r"wiki/john_a\.?_roebling(?!_suspension)"},
    _BRIDGE_WAYPOINT,
]

_ANSWER_NAMES_ALL_THREE = {
    "output": {"final_deliverable": (
        "The Brooklyn Bridge was designed by John A. Roebling, whose Cincinnati-Covington "
        "suspension bridge over the Ohio River has a main span of 1,057 ft."
    )}
}


def test_donation_reddit_tiktok_visits_do_not_credit_a_bridge_engineer_waypoint():
    """The literal case this repair exists for: several real, successful visits to pages that have
    nothing to do with any chain waypoint must not bank credit for that waypoint, no matter how
    many of them there are or how high the aggregate visit count runs."""
    junk_visits = [
        {"url": "https://wikimediafoundation.org/give", "content": "Donate to the Wikimedia Foundation..."},
        {"url": "https://www.reddit.com/r/todayilearned/comments/xyz/", "content": "TIL an unrelated fact."},
        {"url": "https://www.tiktok.com/@someone/video/12345", "content": "unrelated video caption"},
    ]
    observability = {"visit": {"count": len(junk_visits)}, "evidence": {"visited": junk_visits}}
    result = waypoint_chain_coverage(_CHAIN, {}, observability, _ANSWER_NAMES_ALL_THREE["output"]["final_deliverable"])
    assert result["check"] == "chain_coverage"
    assert result["score"] == 0.0
    assert result["passed"] is False


def test_one_genuine_visit_among_junk_credits_only_that_waypoint():
    visits = [
        {"url": "https://wikimediafoundation.org/give", "content": "Donate to the Wikimedia Foundation..."},
        {"url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
         "content": "The Brooklyn Bridge was designed by John A. Roebling."},
        {"url": "https://www.reddit.com/r/todayilearned/comments/xyz/", "content": "TIL an unrelated fact."},
    ]
    observability = {"visit": {"count": len(visits)}, "evidence": {"visited": visits}}
    result = waypoint_chain_coverage(_CHAIN, {}, observability, _ANSWER_NAMES_ALL_THREE["output"]["final_deliverable"])
    # Brooklyn Bridge page names both "start" and "creator" (Roebling appears in its own text);
    # "terminal" (the Cincinnati bridge) was never visited by anything.
    assert abs(result["score"] - 2 / 3) < 1e-9


def test_full_evidence_credits_all_waypoints():
    visits = [
        {"url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge", "content": "designed by John A. Roebling"},
        {"url": "https://en.wikipedia.org/wiki/John_A._Roebling", "content": "engineer, bridge designer"},
        {"url": "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge",
         "content": "Cincinnati Covington Ohio River main span 1,057 ft"},
    ]
    observability = {"visit": {"count": 3}, "evidence": {"visited": visits}}
    result = waypoint_chain_coverage(_CHAIN, {}, observability, _ANSWER_NAMES_ALL_THREE["output"]["final_deliverable"])
    assert result["score"] == 1.0
    assert result["passed"] is True


def test_not_named_in_answer_is_never_credited_even_with_evidence():
    """A page can be genuinely visited and support a waypoint, but if the model's own answer never
    names that waypoint at all, it still should not bank "traversal" credit for it -- chain_coverage
    measures what was both walked AND reported, not just walked."""
    visits = [
        {"url": "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge",
         "content": "Cincinnati Covington Ohio River main span 1,057 ft"},
    ]
    observability = {"visit": {"count": 1}, "evidence": {"visited": visits}}
    answer_names_only_start = {"output": {"final_deliverable": "I read about the Brooklyn Bridge."}}
    result = waypoint_chain_coverage(_CHAIN, {}, observability, answer_names_only_start["output"]["final_deliverable"])
    assert result["score"] == 0.0  # "Brooklyn Bridge" is named but has no visited-page evidence here


# ---------------------------------------------------------------------------------------------
# Fail-open for graph-less arms: visited_evidence must not silently zero an arm (e.g.
# langgraph_react) that never populates result["graph"] but DOES populate
# observability["evidence"]["visited"] via telemetry.documents_seen at run time.
# ---------------------------------------------------------------------------------------------

def test_visited_evidence_sourced_from_observability_not_graph():
    empty_graph_result = {"output": {"final_deliverable": "..."}, "graph": {}}
    observability = {"evidence": {"visited": [{"url": "https://example.com/x", "content": "hello"}]}}
    ev = visited_evidence(empty_graph_result, observability)
    assert ev == [{"url": "https://example.com/x", "content": "hello"}]


def test_chain_coverage_zero_when_no_evidence_channel_at_all():
    """When NEITHER observability.evidence NOR result.graph carries anything (e.g. a rescore of a
    verbosity-stripped, non-graph-variant result), the helper degrades to 0 credit -- fail-closed
    per-waypoint, not a crash and not a silent full-credit fallback to the old aggregate-count
    behavior."""
    empty_result = {"output": {"final_deliverable": _ANSWER_NAMES_ALL_THREE["output"]["final_deliverable"]}}
    observability = {"visit": {"count": 5}}  # high aggregate count, zero page-level evidence
    result = waypoint_chain_coverage(_CHAIN, empty_result, observability,
                                      _ANSWER_NAMES_ALL_THREE["output"]["final_deliverable"])
    assert result["score"] == 0.0


# ---------------------------------------------------------------------------------------------
# Real-corpus regression: task 139 (Gaudi -> Casa Mila), cschain_g_l1b_good_adaptive_rep1.
# A weak model (llama-3.2-1b) visited exactly one page -- a Missouri health department manual
# about nontuberculous mycobacterium, entirely unrelated to the task -- yet the OLD validator
# still credited "1/3 chain waypoints traversed" because n_visits (1) >= 1 named waypoint and
# "Sagrada Família" happened to appear somewhere in the model's confused output.
# ---------------------------------------------------------------------------------------------

_GAUDI_CHAIN = [
    {"name": "Sagrada Família", "name_rx": r"sagrada\s+fam[ií]lia", "slug_rx": r"wiki/sagrada_fam"},
    {"name": "Antoni Gaudí", "name_rx": r"gaud[ií]", "slug_rx": r"wiki/antoni_gaud"},
    {"name": "Casa Milà (La Pedrera)", "name_rx": r"casa\s+mil[àa]|pedrera", "slug_rx": r"wiki/casa_mil"},
]


def test_real_corpus_off_topic_single_visit_does_not_credit_the_named_waypoint():
    off_topic_visit = [{
        "url": "https://health.mo.gov/providers/manuals/communicable-disease-investigation-reference-manual"
               "/nontuberculous-mycobacterium",
        "content": "Nontuberculous Mycobacterium (NTM) / MOTT: treatment guidance for clinicians...",
    }]
    observability = {"visit": {"count": 1}, "evidence": {"visited": off_topic_visit}}
    confused_answer = "Nontuberculous Mycobacterium (NTM) / MOTT ... Sagrada Família ..."
    result = waypoint_chain_coverage(_GAUDI_CHAIN, {}, observability, confused_answer)
    assert result["score"] == 0.0
    assert result["passed"] is False
