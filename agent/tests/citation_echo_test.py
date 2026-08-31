"""Citation echo: N per-entity claims, one pasted URL, fewer pages opened than facts asserted.

Measured live on 2026-08-28, task 162 (a 7-branch fan-out: which STS-61 crew member was oldest
at launch). The graph engine made 5 real visits, ran out of visit budget partway through the
fan-out, and finished the remaining branches from parametric memory -- then dressed every one of
the seven astronauts with the SAME generic URL as its source. One date of birth was fabricated
outright (F. Story Musgrave: 1938-10-29 / age 55, against the real 1935-08-19 / 58), and that
fabricated value is the one the answer elected as the winner.

Every existing check passes on that run: five pages really were opened, ``grounded`` is true,
``goal_achieved`` is true, and the single cited URL is reported as one ``unverified_citations``
entry -- a count of one, indistinguishable from a run that cited one page it happened not to
open. Nothing anywhere counts how many DISTINCT per-entity claims lean on that one URL, and
nothing compares the number of asserted per-entity facts against the number of pages the run
actually opened.

The deliverable fixtures below are the run's real text (``_EVIDENCE_DELIVERABLE`` is a verbatim
excerpt of the capture named in ``_EVIDENCE_PATH``), so the numbers under test are the ones the
engine really produced: 7 claims, 1 URL, 5 visits -> max_reuse 7, claims_per_visit 1.4.

Detection is unconditional; enforcement is gated behind ``final_citation_echo_enforcement_enabled``,
default OFF. No network, no LLM.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.candidate_coverage import enumerated_items
from agent.app.idea_policies.citation_echo import (
    CITATION_ECHO,
    CitationEchoResult,
    attach_citation_echo,
    audit_citation_echo,
    entity_claims,
)
from agent.app.idea_policies.config import IdeaConfig

_NASA = "https://science.nasa.gov/mission/hubble/team/astronauts/meet-the-astronauts"

#: Verbatim excerpt of the live capture: the enumerated per-astronaut block, in which the same
#: URL is pasted as the source of all seven dates of birth.
_EVIDENCE_DELIVERABLE = (
    "The full crew of this mission included Richard Covey (Mission Commander), Kenneth "
    "Bowersox (Pilot), Kathryn Thornton (Mission Specialist), F. Story Musgrave (Mission "
    "Specialist), Claude Nicollier (Mission Specialist), Jeffrey Hoffman (Mission "
    "Specialist), and Thomas Akers (Mission Specialist). Here are their dates of birth, ages "
    "at the time of mission launch, and the citation from each astronaut's page:\n\n"
    f"1. Richard Covey: Born on March 9, 1945, he was 48 years old during STS-61. Quoted "
    f"source URL for his date of birth: {_NASA}.\n\n"
    f"2. Kenneth Bowersox: Born on September 17, 1953, he was 40 years old during STS-61. "
    f"Quoted source URL for his date of birth: {_NASA}.\n\n"
    f"3. Kathryn Thornton: Born on June 28, 1957, she was 36 years old during STS-61. Quoted "
    f"source URL for her date of birth: {_NASA}.\n\n"
    f"4. F. Story Musgrave: Born on October 29, 1938, he was 55 years old during STS-61. "
    f"Quoted source URL for his date of birth: {_NASA}.\n\n"
    f"5. Claude Nicollier: Born on October 20, 1944, he was 49 years old during STS-61. "
    f"Quoted source URL for his date of birth: {_NASA}.\n\n"
    f"6. Jeffrey Hoffman: Born on January 23, 1950, he was 43 years old during STS-61. Quoted "
    f"source URL for his date of birth: {_NASA}.\n\n"
    f"7. Thomas Akers: Born on November 23, 1953, he was 40 years old during STS-61. Quoted "
    f"source URL for his date of birth: {_NASA}.\n\n"
    "The oldest astronaut at the time of mission launch was F. Story Musgrave who was 55 "
    "years old."
)

#: The five pages that run really opened -- none of them an astronaut page.
_EVIDENCE_SOURCES = [
    {"url": "https://www.quora.com/Why-did-the-Hubble-Space-Telescope-have-a-slot", "title": ""},
    {"url": "https://www.quora.com/unanswered/What-is-the-reason-why-the-mirror", "title": ""},
    {"url": "https://science.nasa.gov/mission/hubble/observatory/missions-to-hubble/"
            "servicing-mission-1/", "title": ""},
    {"url": "https://science.nasa.gov/mission/hubble/observatory/design/optics/"
            "hubbles-mirror-flaw/", "title": ""},
    {"url": "https://airandspace.si.edu/collection-objects/costar-hubble-flown", "title": ""},
]

_EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "idea_test_results"
    / "seqfan14b_g_rest_162_qwen2.5:14b_graph_cfg77435a2a_r1.json"
)


def _payload(deliverable: str, sources=None) -> dict:
    return {
        "final_deliverable": deliverable,
        "sources": list(sources or []),
        "goal_achieved": True,
        "grounded": True,
        "coverage_ratio": 1.0,
        "claim_verification_ratio": 0.0,
        "success": True,
    }


# --- the measured failure -----------------------------------------------------------------


def test_one_url_pasted_across_seven_astronauts_is_counted_as_seven_way_reuse():
    result = audit_citation_echo(_payload(_EVIDENCE_DELIVERABLE, _EVIDENCE_SOURCES))
    assert result.active
    assert result.distinct_claims == 7
    assert result.distinct_urls == 1
    assert result.max_reuse == 7
    assert result.echoed


def test_the_echoed_url_names_the_entities_that_lean_on_it():
    result = audit_citation_echo(_payload(_EVIDENCE_DELIVERABLE, _EVIDENCE_SOURCES))
    (echoed,) = result.echoed_urls
    assert echoed.claim_count == 7
    assert echoed.entities[0] == "Richard Covey"
    assert "F. Story Musgrave" in echoed.entities
    assert _NASA.split("//", 1)[1] in echoed.url


def test_more_asserted_facts_than_pages_opened_gives_a_ratio_above_one():
    result = audit_citation_echo(_payload(_EVIDENCE_DELIVERABLE, _EVIDENCE_SOURCES))
    assert result.visits == 5
    assert result.claims_per_visit == pytest.approx(1.4)
    assert result.over_asserted


def test_the_explicit_visit_count_overrides_the_opened_page_list():
    result = audit_citation_echo(_payload(_EVIDENCE_DELIVERABLE, _EVIDENCE_SOURCES), visits=5)
    assert result.visits == 5
    assert result.claims_per_visit == pytest.approx(1.4)


def test_the_real_report_capture_reproduces_the_evidence_numbers():
    if not _EVIDENCE_PATH.exists():
        pytest.skip(f"report capture not present: {_EVIDENCE_PATH}")
    capture = json.loads(_EVIDENCE_PATH.read_text())
    execution = capture["execution"]
    result = audit_citation_echo(
        execution["output"], visits=execution["observability"]["visit"]["count"]
    )
    assert (result.distinct_claims, result.distinct_urls, result.visits) == (7, 1, 5)
    assert result.max_reuse == 7
    assert result.claims_per_visit == pytest.approx(1.4)
    assert execution["output"]["goal_achieved"] is True
    assert len(execution["output"]["unverified_citations"]) == 1


# --- the healthy shape --------------------------------------------------------------------


_HEALTHY = (
    "Ages at launch:\n"
    "1. Richard Covey: born March 9, 1945. Source: https://en.wikipedia.org/wiki/Dick_Covey\n"
    "2. Kathryn Thornton: born August 17, 1952. Source: "
    "https://en.wikipedia.org/wiki/Kathryn_C._Thornton\n"
    "3. Jeffrey Hoffman: born November 2, 1944. Source: "
    "https://en.wikipedia.org/wiki/Jeffrey_A._Hoffman\n"
)
_HEALTHY_SOURCES = [
    {"url": "https://en.wikipedia.org/wiki/Dick_Covey"},
    {"url": "https://en.wikipedia.org/wiki/Kathryn_C._Thornton"},
    {"url": "https://en.wikipedia.org/wiki/Jeffrey_A._Hoffman"},
]


def test_one_page_per_entity_flags_no_echo():
    result = audit_citation_echo(_payload(_HEALTHY, _HEALTHY_SOURCES))
    assert result.active
    assert (result.distinct_claims, result.distinct_urls, result.max_reuse) == (3, 3, 1)
    assert result.echoed_urls == []
    assert not result.echoed


def test_one_page_per_entity_asserts_no_more_than_it_opened():
    result = audit_citation_echo(_payload(_HEALTHY, _HEALTHY_SOURCES))
    assert result.claims_per_visit == pytest.approx(1.0)
    assert not result.over_asserted


# --- a legitimately shared source ---------------------------------------------------------


_SHARED = (
    "Both twins are documented on the same NASA biography page:\n"
    "1. Mark Kelly: born February 21, 1964. Source: https://www.nasa.gov/twin-astronauts\n"
    "2. Scott Kelly: born February 21, 1964. Source: https://www.nasa.gov/twin-astronauts\n"
)
_SHARED_SOURCES = [
    {"url": "https://www.nasa.gov/twin-astronauts"},
    {"url": "https://en.wikipedia.org/wiki/STS-61"},
]


def test_a_genuinely_shared_page_is_reported_rather_than_ignored():
    result = audit_citation_echo(_payload(_SHARED, _SHARED_SOURCES))
    assert result.max_reuse == 2
    assert result.echoed
    assert result.echoed_urls[0].entities == ("Mark Kelly", "Scott Kelly")


def test_a_genuinely_shared_page_is_not_called_over_asserted():
    result = audit_citation_echo(_payload(_SHARED, _SHARED_SOURCES))
    assert result.claims_per_visit == pytest.approx(1.0)
    assert not result.over_asserted


def test_one_page_legitimately_listing_many_items_still_reads_as_over_asserted():
    """The documented limitation of the ratio: a roster page is one visit and many items.

    Live shape (task 012, gpt-4.1-nano): ten search results enumerated off one opened page.
    """
    roster = "".join(
        f"{i}. Result {i}: https://example.org/r{i}\n" for i in range(1, 11)
    )
    result = audit_citation_echo(_payload(roster, [{"url": "https://example.org/search"}]))
    assert result.claims_per_visit == pytest.approx(10.0)
    assert result.over_asserted
    assert not result.echoed


def test_a_distinct_fake_url_per_entity_escapes_echo_but_not_the_ratio():
    """The sibling failure the ratio catches and URL matching cannot (live task 152 shape)."""
    fanned = (
        "1. Mont Blanc - 1786 (Source: https://en.wikipedia.org/wiki/Mont_Blanc)\n"
        "2. Matterhorn - 1865 (Source: https://en.wikipedia.org/wiki/Matterhorn)\n"
        "3. Aconcagua - 1897 (Source: https://en.wikipedia.org/wiki/Aconcagua)\n"
    )
    result = audit_citation_echo(
        _payload(fanned, [{"url": "https://en.wikipedia.org/wiki/Mont_Blanc"}])
    )
    assert result.max_reuse == 1
    assert not result.echoed
    assert result.claims_per_visit == pytest.approx(3.0)
    assert result.over_asserted


def test_reuse_alone_never_becomes_a_verdict():
    shared = audit_citation_echo(_payload(_SHARED, _SHARED_SOURCES)).as_dict()
    assert "verdict" not in shared and "goal_achieved" not in shared
    assert set(shared) == {
        "active", "max_reuse", "echoed_urls", "distinct_urls", "distinct_claims",
        "visits", "claims_per_visit", "reason",
    }


# --- degenerate and malformed input -------------------------------------------------------


def test_zero_visits_reports_an_unknown_ratio_rather_than_a_fabricated_zero():
    result = audit_citation_echo(_payload(_HEALTHY, sources=[]))
    assert result.visits == 0
    assert result.claims_per_visit is None
    assert not result.over_asserted


def test_an_uncited_enumeration_reports_no_urls_and_no_echo():
    uncited = "1. Richard Covey: born 1945.\n2. Kathryn Thornton: born 1952.\n"
    result = audit_citation_echo(_payload(uncited, _HEALTHY_SOURCES))
    assert result.active
    assert (result.distinct_urls, result.max_reuse, result.echoed_urls) == (0, 0, [])
    assert result.distinct_claims == 2


def test_prose_with_no_per_entity_enumeration_is_inert():
    result = audit_citation_echo(_payload("The oldest crew member was Story Musgrave."))
    assert not result.active
    assert result.reason
    assert result.as_dict()["claims_per_visit"] is None


def test_an_instruction_list_is_not_read_as_per_entity_claims():
    instructions = (
        "1. Identify the corrective-optics package.\n"
        "2. Open the mission page and read the launch date.\n"
    )
    assert not audit_citation_echo(_payload(instructions)).active


def test_a_missing_payload_is_inert_rather_than_a_crash():
    for payload in (None, {}, [], "deliverable", 7):
        result = audit_citation_echo(payload)
        assert not result.active
        assert result.visits is None
        assert result.claims_per_visit is None


def test_malformed_payload_fields_are_inert_rather_than_a_crash():
    assert not audit_citation_echo({"final_deliverable": 7, "sources": 3}).active
    assert not audit_citation_echo({"final_deliverable": None, "sources": None}).active
    weird = audit_citation_echo({"final_deliverable": _HEALTHY, "sources": [None, 5, {}]})
    assert weird.active
    assert weird.visits == 3


def test_a_negative_visit_count_is_treated_as_unknown():
    result = audit_citation_echo(_payload(_HEALTHY, _HEALTHY_SOURCES), visits=-1)
    assert result.visits is None
    assert result.claims_per_visit is None


def test_a_non_numeric_visit_count_is_treated_as_unknown():
    result = audit_citation_echo(_payload(_HEALTHY, _HEALTHY_SOURCES), visits="five")
    assert result.visits is None
    assert result.claims_per_visit is None


def test_the_audit_never_raises_on_hostile_input():
    for deliverable in ("", "\n\n", "1. \n2. \n", "1) : \n2) : \n", "https://x/", None):
        audit_citation_echo({"final_deliverable": deliverable})


# --- claim parsing ------------------------------------------------------------------------


def test_a_claim_keeps_the_entity_name_without_its_description():
    claims = entity_claims(_EVIDENCE_DELIVERABLE)
    assert [c.entity for c in claims][:2] == ["Richard Covey", "Kenneth Bowersox"]
    assert all(len(c.urls) == 1 for c in claims)


def test_the_same_entity_listed_twice_is_one_claim():
    repeated = (
        "1. Richard Covey: born 1945. Source: https://a.example/covey\n"
        "2. Richard Covey: commander. Source: https://a.example/covey\n"
    )
    result = audit_citation_echo(_payload(repeated, _HEALTHY_SOURCES))
    assert result.distinct_claims == 1
    assert result.max_reuse == 1


def test_the_shared_enumeration_helper_is_the_one_candidate_coverage_uses():
    assert enumerated_items(_HEALTHY)[0].startswith("Richard Covey")
    assert enumerated_items("1. Identify the package.\n2. Open the page.\n") == []


# --- payload wiring -----------------------------------------------------------------------


def test_the_marker_is_attached_to_an_active_payload():
    payload = _payload(_EVIDENCE_DELIVERABLE, _EVIDENCE_SOURCES)
    attach_citation_echo(payload)
    assert payload[CITATION_ECHO]["max_reuse"] == 7
    assert payload[CITATION_ECHO]["claims_per_visit"] == pytest.approx(1.4)


def test_the_marker_is_absent_when_the_audit_is_inert():
    payload = _payload("The oldest crew member was Story Musgrave.")
    attach_citation_echo(payload)
    assert CITATION_ECHO not in payload


def test_attaching_the_marker_changes_no_score_gate_or_verdict():
    payload = _payload(_EVIDENCE_DELIVERABLE, _EVIDENCE_SOURCES)
    before = copy.deepcopy(payload)
    attach_citation_echo(payload)
    after = {k: v for k, v in payload.items() if k != CITATION_ECHO}
    assert after == before


def test_attaching_the_marker_never_raises_on_a_hostile_payload():
    for payload in ({}, {"final_deliverable": 7}, {"sources": "nope"}):
        attach_citation_echo(payload)
        assert CITATION_ECHO not in payload


# --- the enforcement flag -----------------------------------------------------------------


def test_enforcement_is_off_by_default():
    cfg = IdeaConfig.from_settings(load_idea_dag_settings())
    assert cfg.final.citation_echo_enforcement_enabled is False


def test_enforcement_can_be_turned_on_from_settings():
    settings = dict(load_idea_dag_settings())
    settings["final_citation_echo_enforcement_enabled"] = True
    cfg = IdeaConfig.from_settings(settings)
    assert cfg.final.citation_echo_enforcement_enabled is True


def test_the_flag_is_a_shipped_json_default():
    assert load_idea_dag_settings()["final_citation_echo_enforcement_enabled"] is False


def test_the_result_is_pure_telemetry_whatever_the_flag_says():
    result = CitationEchoResult()
    assert not hasattr(result, "missing_requirements")
