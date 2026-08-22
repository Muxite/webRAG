"""One page-attributable verdict per enumerated candidate (strong-agent trace B6/B7/B8).

The fixture is test 122's real mandate, imported verbatim from the task module rather than
retyped: four large radio telescopes, exactly one is the largest filled-aperture single dish
CURRENTLY in operation. The 2026-08-21 diagnostic established that this task's real 0.50 -> 0.80
improvement came from a config cap FORCING all-four coverage, never from the engine deciding to
check every candidate -- and that a "checked" candidate is recorded as a bare visit with no
field saying why it lost. The page texts below are short paraphrases of the four Wikipedia
pages' relevant sentences (Arecibo's December 2020 collapse, RATAN-600's ring reflector, Green
Bank's steerable superlative, FAST's 300 m illuminated circle).

Detection is unconditional; the downgrade is gated behind ``merge_candidate_roster_enabled``,
default OFF. No network: every LLM response is scripted.
"""
from __future__ import annotations

import asyncio
import json
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.candidate_roster import (
    CANDIDATE_ROSTER,
    DISPOSED,
    ROSTER_MAGNITUDE_MARKER,
    ROSTER_UNDISPOSED_MARKER,
    UNDATED,
    UNRESOLVED,
    UNSOURCED,
    audit_candidate_roster,
)
from agent.app.idea_policies.mandate_requirements import parse_mandate_requirements
from agent.app.idea_tests.test_122_tier5_filled_aperture_telescope_survivor import (
    get_task_statement,
)

MANDATE = get_task_statement()

_PAGES = {
    "Arecibo Telescope": (
        "https://en.wikipedia.org/wiki/Arecibo_Telescope",
        "Arecibo Telescope. The 305 m spherical reflector dish collapsed on 1 December 2020 "
        "after cable failures and is no longer operational; the observatory was decommissioned.",
    ),
    "RATAN-600": (
        "https://en.wikipedia.org/wiki/RATAN-600",
        "RATAN-600 is a radio telescope in Russia with a 576 m ring reflector, in operation "
        "since 1974. It is not a filled aperture: the reflecting elements form a circle rather "
        "than a continuous dish.",
    ),
    "Green Bank Telescope": (
        "https://en.wikipedia.org/wiki/Green_Bank_Telescope",
        "The Green Bank Telescope is a 100 m dish in West Virginia and has been the world's "
        "largest fully steerable radio telescope since 2001.",
    ),
    "FAST": (
        "https://en.wikipedia.org/wiki/Five-hundred-meter_Aperture_Spherical_Telescope",
        "The Five-hundred-meter Aperture Spherical Telescope (FAST) has been fully operational "
        "since January 2020. The reflector diameter is 500 m, but only a circle of 300 m "
        "diameter is illuminated and useful at any one time.",
    ),
}

# What a run that did the work says: a dated, page-attributable statement per candidate.
_COMPLETE_SUMMARY = (
    "Arecibo Telescope collapsed on 1 December 2020 and is no longer operational. "
    "RATAN-600 is a 576 m ring reflector in operation since 1974, not a filled aperture. "
    "The Green Bank Telescope is a 100 m dish, the largest fully steerable radio telescope "
    "since 2001. "
    "FAST is the survivor: a 500 m filled-aperture dish fully operational since January 2020, "
    "of which only a 300 m circle is illuminated."
)


def _visit(node_id: str, candidate: str) -> dict:
    url, content = _PAGES[candidate]
    return {
        "node_id": node_id,
        "title": f"visit {candidate}",
        "status": "done",
        "is_merge": False,
        "result": {
            "success": True,
            "action": "visit",
            "url": url,
            "page_title": candidate,
            "content_full": content,
        },
    }


def _graph(candidates=tuple(_PAGES)) -> IdeaDag:
    graph = IdeaDag(root_title="root")
    root = graph.get_node(graph.root_id())
    root.details["mandate"] = MANDATE
    for i, candidate in enumerate(candidates):
        child = graph.add_child(graph.root_id(), f"visit {candidate}",
                                status=IdeaNodeStatus.DONE)
        child.details[DetailKey.ACTION_RESULT.value] = _visit(f"c{i}", candidate)["result"]
    return graph


def _children(candidates=tuple(_PAGES)) -> list:
    return [_visit(f"c{i}", c) for i, c in enumerate(candidates)]


def _audit(summary=_COMPLETE_SUMMARY, candidates=tuple(_PAGES), mandate=MANDATE):
    return audit_candidate_roster(_graph(candidates), mandate, summary, _children(candidates))


def _entry(audit, candidate):
    return next(e for e in audit.entries if e.candidate == candidate)


# --- mandate detection ------------------------------------------------------------------

def test_the_real_task_122_mandate_yields_a_roster():
    req = parse_mandate_requirements(MANDATE)
    assert req.roster_candidates == [
        "Arecibo Telescope", "RATAN-600", "Green Bank Telescope", "FAST",
    ]
    assert req.individual_disposition is True
    assert req.time_indexed is True          # "CURRENTLY IN OPERATION"
    assert req.requires_candidate_roster is True


def test_an_enumerated_list_without_a_disposition_cue_is_not_a_roster():
    """The breadth-argmax shape: pick the deepest of six lakes. Nothing to disqualify."""
    req = parse_mandate_requirements(
        "Compare these lakes and report the deepest.\n"
        "1. Lake Baikal\n2. Lake Tanganyika\n3. Lake Superior\n"
    )
    assert req.roster_candidates
    assert req.individual_disposition is False
    assert req.requires_candidate_roster is False


def test_a_numbered_question_list_is_not_a_roster():
    """``extract_named_candidates`` returns whole sentences here; a roster must be NAMES."""
    req = parse_mandate_requirements(
        "Research the Downfall vulnerability. Exactly one CPU feature is abused.\n"
        "1. What class of attack is it, and what CPU feature does it abuse?\n"
        "2. Which instruction set extensions are primarily involved?\n"
    )
    assert req.roster_candidates == []
    assert req.requires_candidate_roster is False


def test_a_time_cue_inside_a_longer_word_does_not_count():
    """``active`` is a substring of ``radioactive``; the cue is matched on word boundaries."""
    assert parse_mandate_requirements("Which sample is radioactive?").time_indexed is False
    assert parse_mandate_requirements("Which telescope is still active?").time_indexed is True


def test_the_audit_is_inert_off_shape():
    audit = audit_candidate_roster(_graph(), "Report the height of the Eiffel Tower", "x", [])
    assert audit.active is False
    assert audit.entries == []
    assert audit.missing_requirements() == []


# --- the audit --------------------------------------------------------------------------

def test_a_complete_roster_passes():
    audit = _audit()
    assert audit.active is True
    assert audit.complete is True
    assert [e.status for e in audit.entries] == [DISPOSED] * 4
    assert audit.missing_requirements() == []
    assert audit.magnitude_tripwire == []


def test_every_entry_carries_a_quote_and_its_source_url():
    audit = _audit()
    arecibo = _entry(audit, "Arecibo Telescope")
    assert "1 December 2020" in arecibo.disqualifier_quote
    assert arecibo.source_url == _PAGES["Arecibo Telescope"][0]
    survivor = _entry(audit, "FAST")
    assert survivor.survivor is True
    assert survivor.source_url == _PAGES["FAST"][0]


def test_a_candidate_nobody_mentioned_is_unresolved():
    summary = _COMPLETE_SUMMARY.replace(
        "RATAN-600 is a 576 m ring reflector in operation since 1974, not a filled aperture. ",
        "",
    )
    audit = _audit(summary)
    assert _entry(audit, "RATAN-600").status == UNRESOLVED
    assert audit.complete is False
    assert any("no disposition was recorded" in line and "RATAN-600" in line
               for line in audit.missing_requirements())


def test_a_verdict_with_no_page_behind_it_is_unsourced():
    """Every candidate is described, but RATAN-600's page was never opened -- the parametric
    memory shape the whole contract exists to separate from a real check."""
    visited = ("Arecibo Telescope", "Green Bank Telescope", "FAST")
    audit = audit_candidate_roster(
        _graph(visited), MANDATE, _COMPLETE_SUMMARY, _children(visited)
    )
    ratan = _entry(audit, "RATAN-600")
    assert ratan.status == UNSOURCED
    assert ratan.source_url == ""
    assert "576 m ring reflector" in ratan.disqualifier_quote
    assert audit.complete is False


def test_a_survivor_page_naming_a_competitor_does_not_dispose_of_it():
    """FAST's page mentions the telescopes it beat; identity matching is title/url only, so the
    survivor's page cannot stand in for the pages never opened."""
    audit = audit_candidate_roster(
        _graph(("FAST",)),
        MANDATE,
        "Arecibo Telescope collapsed in 2020. FAST is the survivor, 500 m, operational 2020.",
        [{
            "node_id": "c0", "title": "visit FAST", "status": "done", "is_merge": False,
            "result": {
                "success": True, "action": "visit", "url": _PAGES["FAST"][0],
                "page_title": "FAST",
                "content_full": _PAGES["FAST"][1] + " It surpassed the Arecibo Telescope, "
                                "which collapsed in 2020.",
            },
        }],
    )
    assert _entry(audit, "Arecibo Telescope").status == UNSOURCED


def test_a_nested_merge_cannot_certify_a_roster_entry():
    """A merge hands up prose, not pages; letting one supply a verdict would self-certify."""
    nested = {"node_id": "m0", "title": "Merge", "status": "done", "is_merge": True,
              "result": {"summary": _COMPLETE_SUMMARY, "goal_achieved": True}}
    audit = audit_candidate_roster(_graph(()), MANDATE, "", [nested])
    assert [e.status for e in audit.entries] == [UNRESOLVED] * 4


def test_a_childs_direct_text_disposes_of_a_candidate():
    """The verdict does not have to come from the synthesis -- a leaf's own extract counts."""
    audit = audit_candidate_roster(
        _graph(), MANDATE, "", _children() + [{
            "node_id": "x", "title": "extract", "status": "done", "is_merge": False,
            "result": {"success": True, "action": "extract",
                       "summary": _COMPLETE_SUMMARY},
        }],
    )
    assert audit.complete is True


def test_a_candidates_page_is_found_under_its_mandate_alias():
    """The roster name is ``FAST``; every page the run opens for it is titled
    *Five-hundred-meter Aperture Spherical Telescope*. Replaying the 15 captured task-122
    reports showed both runs that opened all four candidate pages would otherwise be scored as
    never having opened the survivor's."""
    graph = _graph(())
    node = graph.add_child(graph.root_id(), "visit the survivor", status=IdeaNodeStatus.DONE)
    node.details[DetailKey.ACTION_RESULT.value] = {
        "success": True, "action": "visit",
        "url": "https://en.wikipedia.org/wiki/Five-hundred-meter_Aperture_Spherical_Telescope",
        "page_title": "Five-hundred-meter Aperture Spherical Telescope",
        "content_full": _PAGES["FAST"][1],
    }
    audit = audit_candidate_roster(
        graph,
        MANDATE,
        "FAST is the survivor, a 500 m dish operational since 2020 with a 300 m "
        "illuminated circle.",
        [],
    )
    fast = _entry(audit, "FAST")
    assert fast.status == DISPOSED
    assert fast.source_url.endswith("Five-hundred-meter_Aperture_Spherical_Telescope")


def test_a_descriptive_parenthetical_is_not_treated_as_an_alias():
    """Arecibo's mandate line reads "(a 305 m spherical dish)" -- a description, not a name."""
    from agent.app.idea_policies.candidate_roster import _aliases

    assert _aliases(MANDATE, "Arecibo Telescope") == ["Arecibo Telescope"]
    assert "Five-hundred-meter Aperture Spherical Telescope" in _aliases(MANDATE, "FAST")


# --- B8: time-indexed superlatives ------------------------------------------------------

def test_an_undated_status_fails_a_time_indexed_mandate():
    summary = _COMPLETE_SUMMARY.replace("collapsed on 1 December 2020", "has collapsed")
    audit = _audit(summary)
    arecibo = _entry(audit, "Arecibo Telescope")
    assert arecibo.status == UNDATED
    assert arecibo.source_url == _PAGES["Arecibo Telescope"][0]   # the page WAS opened
    assert any("carries no date" in line for line in audit.missing_requirements())


def test_the_same_undated_status_passes_when_the_mandate_is_not_time_indexed():
    """Only the mandate's own cue words impose the year token."""
    mandate = MANDATE.replace("CURRENTLY IN OPERATION", "the widest").replace(
        "currently", "").replace("still operational, has it collapsed",
                                 "collapsed, has it collapsed")
    assert parse_mandate_requirements(mandate).time_indexed is False
    summary = _COMPLETE_SUMMARY.replace("collapsed on 1 December 2020", "has collapsed")
    audit = _audit(summary, mandate=mandate)
    assert _entry(audit, "Arecibo Telescope").status == DISPOSED


#: The live shape (2026-08-21 ollama probe, 2 of 3 correct answers downgraded): all four pages
#: genuinely opened, the CORRECT answer given, and the two candidates that are simply still
#: running described by their status with no year to cite -- because there is no dated event to
#: cite for a telescope nothing happened to.
_ONGOING_STATE_SUMMARY = (
    "Arecibo Telescope collapsed on 1 December 2020. "
    "RATAN-600: Operational, a 576 m ring reflector, not a filled aperture. "
    "The Green Bank Telescope is a 100 m dish, fully steerable and operational. "
    "FAST is the survivor: a 500 m filled-aperture dish, operational, of which only a 300 m "
    "circle is illuminated."
)


def test_an_ongoing_state_disposition_needs_no_year():
    """The live false positive. "Operational" is page-corroborated and dateless by nature."""
    from agent.app.idea_policies.candidate_roster import _YEAR_RE

    audit = _audit(_ONGOING_STATE_SUMMARY)
    for name in ("RATAN-600", "Green Bank Telescope", "FAST"):
        entry = _entry(audit, name)
        assert entry.status == DISPOSED, name
        assert not _YEAR_RE.search(entry.disqualifier_quote), name
    assert _entry(audit, "Arecibo Telescope").status == DISPOSED   # dated event, dated quote
    assert audit.complete is True
    assert audit.missing_requirements() == []


def test_the_magnitude_tripwire_accepts_an_ongoing_state_disposition():
    """B7 inherits roster status, so the false positive propagated: RATAN-600 (576 m) beats the
    survivor (500 m) on size and is eliminated by geometry, not by a date."""
    audit = _audit(_ONGOING_STATE_SUMMARY)
    assert _entry(audit, "RATAN-600").magnitude_m == 576.0
    assert _entry(audit, "FAST").magnitude_m == 500.0
    assert audit.magnitude_tripwire == []


def test_an_event_shaped_disposition_with_no_year_still_fails():
    """The ORIGINAL B8 target, unchanged: a collapse recalled without its date is exactly what
    stale parametric memory produces, so the year stays load-bearing for event-shaped claims."""
    summary = _ONGOING_STATE_SUMMARY.replace(
        "collapsed on 1 December 2020", "collapsed after cable failures"
    )
    audit = _audit(summary)
    arecibo = _entry(audit, "Arecibo Telescope")
    assert arecibo.status == UNDATED
    assert audit.complete is False
    assert any("carries no date" in line and "Arecibo" in line
               for line in audit.missing_requirements())


def test_a_past_or_negated_state_word_is_an_event_not_a_state():
    """"No longer operational" / "was in operation until" are collapses wearing a state word."""
    from agent.app.idea_policies.candidate_roster import _year_required

    assert _year_required("Arecibo Telescope is no longer operational") is True
    assert _year_required("Arecibo was operational until the cable failures") is True
    assert _year_required("The dish is not in service") is True
    assert _year_required("RATAN-600: Operational") is False
    assert _year_required("Green Bank Telescope is fully steerable and operational") is False
    assert _year_required("FAST remains in operation") is False


def test_a_dateless_descriptive_claim_still_needs_a_year():
    """Neither cue fires: the requirement stays, so the fix narrows B8 without inverting it."""
    from agent.app.idea_policies.candidate_roster import _year_required

    assert _year_required("Arecibo Telescope is the largest single-dish radio telescope") is True
    summary = _ONGOING_STATE_SUMMARY.replace(
        "Arecibo Telescope collapsed on 1 December 2020.",
        "The Arecibo Telescope is a 305 m spherical reflector dish in Puerto Rico.",
    )
    assert _entry(_audit(summary), "Arecibo Telescope").status == UNDATED


def test_a_dated_statement_elsewhere_in_the_run_satisfies_the_year_requirement():
    """The first mention need not be the dated one; the audit takes the best available."""
    audit = _audit("Arecibo Telescope is no longer operational. " + _COMPLETE_SUMMARY)
    assert _entry(audit, "Arecibo Telescope").status == DISPOSED
    assert "2020" in _entry(audit, "Arecibo Telescope").disqualifier_quote


#: The second live shape (2026-08-22 fresh-completion re-validation): the WHOLE roster in one
#: sentence, which is how models actually report it. Arecibo's "collapsed" is the only
#: event-shaped clause in it.
_ONE_SENTENCE_SUMMARY = (
    "Arecibo Telescope has collapsed, RATAN-600 is operational, the Green Bank Telescope is "
    "operational, and FAST is operational with a 300 m illuminated aperture, so FAST is the "
    "survivor"
)


def _wide_audit(summary):
    """The real pages cross-reference each other (FAST's page names the telescopes it beat), so
    a whole-roster sentence corroborates against any one of them -- which is why the live shape
    reached the year gate at all instead of stopping at ``unsourced``."""
    graph = _graph()
    for node in graph.iter_depth_first():
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if isinstance(result, dict) and result.get("content_full"):
            result["content_full"] += (
                " Comparison: the Arecibo Telescope collapsed, RATAN-600 is a ring, the Green "
                "Bank Telescope is steerable and FAST is the largest survivor, operational, "
                "with a 300 m illuminated aperture circle."
            )
    return audit_candidate_roster(graph, MANDATE, summary, _children())


def test_one_candidates_terminal_event_does_not_bleed_onto_its_siblings():
    """Bug A: ``_year_required`` used to read the whole multi-candidate sentence, so one
    "collapsed" forced a year onto the three ongoing-state clauses that cannot have one."""
    audit = _wide_audit(_ONE_SENTENCE_SUMMARY)
    for name in ("RATAN-600", "Green Bank Telescope", "FAST"):
        assert _entry(audit, name).status == DISPOSED, name
    # Arecibo's own clause is still event-shaped and still dateless.
    assert _entry(audit, "Arecibo Telescope").status == UNDATED
    assert [line for line in audit.missing_requirements() if "Arecibo" in line]
    assert not any("RATAN-600" in line or "Green Bank" in line
                   for line in audit.missing_requirements())


def test_a_clause_is_scoped_to_its_own_candidate():
    from agent.app.idea_policies.candidate_roster import _candidate_clause

    others = ["Arecibo Telescope", "Green Bank Telescope", "FAST"]
    assert _candidate_clause(_ONE_SENTENCE_SUMMARY, ["RATAN-600"], others) == (
        "RATAN-600 is operational"
    )
    assert _candidate_clause(
        _ONE_SENTENCE_SUMMARY, ["Arecibo Telescope"], ["RATAN-600", "FAST"],
    ) == "Arecibo Telescope has collapsed"
    # A continuation clause naming nobody belongs to the candidate before it.
    assert _candidate_clause(
        "Arecibo Telescope collapsed in 2020, RATAN-600 uses a ring reflector, not a filled "
        "aperture",
        ["RATAN-600"], ["Arecibo Telescope"],
    ) == "RATAN-600 uses a ring reflector, not a filled aperture"
    # Nothing to separate: a single-candidate statement is returned untouched.
    single = "RATAN-600 is a 576 m ring reflector, in operation since 1974"
    assert _candidate_clause(single, ["RATAN-600"], others) == single


def test_a_coordinated_subject_falls_open_instead_of_scoping_to_a_bare_name():
    """A real live wording: the two candidates SHARE one predicate, so RATAN-600's own clause is
    just its name. Scoping to a name asserts nothing, so the whole statement is kept."""
    from agent.app.idea_policies.candidate_roster import _candidate_clause

    shared = ("The RATAN-600 and FAST telescopes are operational but do not meet the criteria "
              "of being the largest filled-aperture single-dish radio telescope")
    assert _candidate_clause(shared, ["RATAN-600"], ["FAST"]) == shared


def test_a_dated_sibling_clause_does_not_satisfy_this_candidates_year():
    """The scoping cuts both ways: Arecibo's date is not RATAN-600's evidence."""
    audit = _wide_audit(
        "Arecibo Telescope collapsed on 1 December 2020, RATAN-600 is the largest ring "
        "telescope, the Green Bank Telescope is operational, and FAST is the survivor at 500 m"
    )
    assert _entry(audit, "Arecibo Telescope").status == DISPOSED
    assert _entry(audit, "RATAN-600").status == UNDATED


#: The third live shape: the disqualifier the MANDATE ITSELF supplies for RATAN-600 is a
#: geometry fact. It has no date and never will.
_STRUCTURAL_SUMMARY = (
    "Arecibo Telescope collapsed on 1 December 2020. "
    "RATAN-600 uses a ring reflector, not a filled-aperture dish. "
    "The Green Bank Telescope is a 100 m fully steerable dish, smaller than the others. "
    "FAST is the survivor, a 500 m filled-aperture dish with a 300 m illuminated circle."
)


def test_a_structural_disqualifier_needs_no_year():
    """Bug B: a categorical fact about what KIND of thing a candidate is cannot be dated, and
    on task 122 it is the elimination the mandate hands the run."""
    from agent.app.idea_policies.candidate_roster import _YEAR_RE, _year_required

    assert _year_required("RATAN-600 uses a ring reflector, not a filled-aperture dish") is False
    assert _year_required("Green Bank is a smaller fully steerable dish") is False
    assert _year_required("It is a ring antenna rather than a continuous dish") is False

    audit = _audit(_STRUCTURAL_SUMMARY)
    for name in ("RATAN-600", "Green Bank Telescope", "FAST"):
        entry = _entry(audit, name)
        assert entry.status == DISPOSED, name
        assert not _YEAR_RE.search(entry.disqualifier_quote), name
    assert audit.complete is True
    assert audit.missing_requirements() == []


def test_a_structural_cue_does_not_rescue_a_terminal_event():
    """The exemption is a third category, not a loosening: a dateable event still owes its
    date however much geometry the same clause carries."""
    from agent.app.idea_policies.candidate_roster import _year_required

    assert _year_required(
        "Arecibo's filled-aperture dish collapsed after cable failures"
    ) is True
    audit = _audit(_STRUCTURAL_SUMMARY.replace(
        "collapsed on 1 December 2020",
        "collapsed after cable failures and its filled-aperture dish is gone",
    ))
    assert _entry(audit, "Arecibo Telescope").status == UNDATED


# --- B7: the magnitude tripwire ---------------------------------------------------------

def test_a_larger_undisqualified_candidate_trips_the_wire():
    """RATAN-600 (576 m) is PHYSICALLY LARGER than the elected survivor (500 m) and is only
    correctly eliminated by a geometry predicate read off its own page. Asserting its size
    without opening it leaves the survivor election resting on nothing."""
    visited = ("Arecibo Telescope", "Green Bank Telescope", "FAST")
    audit = audit_candidate_roster(
        _graph(visited), MANDATE, _COMPLETE_SUMMARY, _children(visited)
    )
    assert audit.magnitude_tripwire == ["RATAN-600"]
    lines = audit.missing_requirements()
    assert any("LARGER than the elected survivor" in line for line in lines)
    # The sharper line replaces the generic gap rather than doubling it.
    assert len([line for line in lines if "RATAN-600" in line]) == 1


def test_a_smaller_undisqualified_candidate_does_not_trip_the_wire():
    visited = ("Arecibo Telescope", "RATAN-600", "FAST")
    audit = audit_candidate_roster(
        _graph(visited), MANDATE, _COMPLETE_SUMMARY, _children(visited)
    )
    assert _entry(audit, "Green Bank Telescope").status == UNSOURCED   # 100 m, still a gap
    assert audit.magnitude_tripwire == []
    assert any("not supported by any page opened for it" in line
               and "Green Bank Telescope" in line
               for line in audit.missing_requirements())


def test_a_disqualified_larger_candidate_never_trips_the_wire():
    """The correct solve: RATAN-600 IS bigger, and its own page says why it does not count."""
    audit = _audit()
    assert _entry(audit, "RATAN-600").magnitude_m == 576.0
    assert _entry(audit, "FAST").magnitude_m == 500.0
    assert audit.magnitude_tripwire == []


def test_no_identifiable_survivor_means_no_tripwire():
    """Fail open: with nobody elected there is nothing to outrank."""
    summary = _COMPLETE_SUMMARY.replace("FAST is the survivor:", "FAST is described as")
    visited = ("Arecibo Telescope", "Green Bank Telescope", "FAST")
    audit = audit_candidate_roster(
        _graph(visited), MANDATE, summary, _children(visited)
    )
    assert all(not e.survivor for e in audit.entries)
    assert audit.magnitude_tripwire == []


# --- wiring into the merge action -------------------------------------------------------

class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _merge(summary=_COMPLETE_SUMMARY, candidates=tuple(_PAGES), achieved=True, **overrides):
    graph = _graph(candidates)
    parent = graph.add_child(graph.root_id(), "eliminate to one survivor",
                             status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = MANDATE
    merge = graph.add_child(parent.node_id, "Merge: eliminate to one survivor",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = _children(candidates)
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({
        "summary": summary, "goal_achieved": achieved,
        "goal_evaluation": "the survivor is FAST", "missing_requirements": [],
    })
    result = asyncio.run(
        MergeLeafAction(settings=settings).execute(graph, merge.node_id, _ScriptedIO(response))
    )
    return graph, parent.node_id, merge.node_id, result


def test_detection_is_unconditional_and_the_default_verdict_is_untouched(caplog):
    visited = ("Arecibo Telescope", "Green Bank Telescope", "FAST")
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _merge(candidates=visited)
    details = graph.get_node(merge_id).details
    assert details[ROSTER_UNDISPOSED_MARKER] == ["RATAN-600"]
    assert details[ROSTER_MAGNITUDE_MARKER] == ["RATAN-600"]
    assert result["goal_achieved"] is True
    assert graph.get_node(parent_id).status == IdeaNodeStatus.DONE
    assert any("candidate roster incomplete" in r.message for r in caplog.records)
    assert not any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_audit_is_stamped_on_the_root_whatever_the_flag_says():
    graph, _, _, _ = _merge()
    roster = graph.get_node(graph.root_id()).details[CANDIDATE_ROSTER]
    assert [e["candidate"] for e in roster] == [
        "Arecibo Telescope", "RATAN-600", "Green Bank Telescope", "FAST",
    ]
    assert all(e["status"] == DISPOSED for e in roster)
    assert all(e["disqualifier_quote"] and e["source_url"] for e in roster)


def test_the_flag_downgrades_an_incomplete_roster(caplog):
    visited = ("Arecibo Telescope", "Green Bank Telescope", "FAST")
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _merge(
            candidates=visited, merge_candidate_roster_enabled=True
        )
    details = graph.get_node(merge_id).details
    assert result["goal_achieved"] is False
    assert details[DetailKey.GOAL_ACHIEVED.value] is False
    assert details["merge_incomplete"] is True
    assert any("LARGER than the elected survivor" in req
               for req in details["missing_requirements"])
    assert graph.get_node(parent_id).status == IdeaNodeStatus.ACTIVE
    assert any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_raw_model_output_survives_the_downgrade():
    visited = ("FAST",)
    _, _, _, result = _merge(candidates=visited, merge_candidate_roster_enabled=True)
    assert result["synthesized"]["goal_achieved"] is True
    assert result["goal_achieved"] is False


def test_a_complete_roster_is_untouched_with_the_flag_on(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _merge(merge_candidate_roster_enabled=True)
    assert result["goal_achieved"] is True
    assert ROSTER_UNDISPOSED_MARKER not in graph.get_node(merge_id).details
    assert graph.get_node(parent_id).status == IdeaNodeStatus.DONE
    assert not any("candidate roster" in r.message for r in caplog.records)


def test_an_off_shape_mandate_stamps_nothing():
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "measure a bridge", status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = "Report the Hardanger Bridge's main span in metres"
    merge = graph.add_child(parent.node_id, "Merge: measure a bridge",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = []
    settings = load_idea_dag_settings()
    settings["merge_candidate_roster_enabled"] = True
    response = json.dumps({"summary": "1,310 m", "goal_achieved": True,
                           "goal_evaluation": "done", "missing_requirements": []})
    asyncio.run(MergeLeafAction(settings=settings).execute(graph, merge.node_id,
                                                          _ScriptedIO(response)))
    assert CANDIDATE_ROSTER not in graph.get_node(graph.root_id()).details
    assert ROSTER_UNDISPOSED_MARKER not in graph.get_node(merge.node_id).details
