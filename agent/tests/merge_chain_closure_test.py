"""A chain hop that resolved to the WRONG entity, caught by the relation the mandate hops on.

The 2026-08-21/22 strong-agent trace (recommendation C1) solved task 065 -- collection -> poet
-> that poet's BIRTHPLACE town -> that town's elevation -- and reported refusing a real
chain-poisoning decoy: an early search snippet called Temuco Neruda's "native town". Trusting
it makes hop 3 execute FLAWLESSLY against the wrong page, because Temuco is a real city whose
real infobox states a real elevation (360 m). Visit-count grounding passes, the page-identity
guard passes, and ``grounding.answer_numeric_provenance`` passes too -- 360 m genuinely appears
in genuinely fetched text. The error is in the RELATION between hop 2 and hop 3.

What the trace agent checked instead: the NEXT-hop page's own back-reference to the
PREVIOUS-hop entity through the SAME relation the mandate named. Live-verified against all
three real candidate pages -- Parral's page reads "is the birthplace of poet Pablo Neruda"
(closes), Temuco names him only as having lived there (does not close), Hidalgo del Parral, the
famous Mexican homonym, never mentions him (does not close).

The ``_PAGE_*`` fixtures below are HAND-BUILT reconstructions of those three pages, written
from the trace's own description of what each one does and does not say, not captured page
text: no report capture in this repo stores the raw article bodies of a run that went to the
decoy. They are deliberately unhelpful to the mechanism where the real pages are -- Temuco's
carries Neruda's name, his profession and his childhood there, and its own elevation figure,
so only the relation wording separates it from Parral's.

Detection is unconditional (the ``chain_closure`` audit + ``chain_closure_open`` marker); the
downgrade is gated behind ``merge_chain_closure_enabled``, default OFF.

No network, no LLM: every merge response is scripted.
"""
from __future__ import annotations

import asyncio
import json
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.chain_closure import (
    CHAIN_CLOSURE,
    CHAIN_CLOSURE_OPEN_MARKER,
    audit_chain_closure,
    mandate_relation,
    visited_pages,
)
from agent.app.idea_policies.shape_classifier import classify_shape
from agent.app.idea_tests.test_065_tier5_leak_resistant_chain import get_task_statement

#: The real task-065 mandate, imported rather than paraphrased so the shape/relation detection
#: under test is exercised against the text the benchmark actually ships.
_MANDATE = get_task_statement()

# --- hand-built reconstructions of the three real candidate pages -------------------------

_URL_NERUDA = "https://en.wikipedia.org/wiki/Pablo_Neruda"
_URL_PARRAL = "https://en.wikipedia.org/wiki/Parral,_Chile"
_URL_TEMUCO = "https://en.wikipedia.org/wiki/Temuco"
_URL_HIDALGO = "https://en.wikipedia.org/wiki/Hidalgo_del_Parral"

_PAGE_NERUDA = (
    "Pablo Neruda\n"
    "Pablo Neruda was the pen name and later legal name of the Chilean poet-diplomat and "
    "politician Ricardo Eliecer Neftali Reyes Basoalto.\n"
    "Born\tRicardo Eliecer Neftali Reyes Basoalto\n12 July 1904\nParral, Chile\n"
    "Died\t23 September 1973 (aged 69)\nSantiago, Chile\n"
    "Awards\tNobel Prize in Literature (1971)\n"
    "Neruda became known as a poet when he was 13 years old. He wrote in a variety of styles, "
    "including the surrealist poems of Twenty Love Poems and a Song of Despair (1924)."
)

#: The CORRECT hop-3 page: its lead sentence carries the back-reference.
_PAGE_PARRAL = (
    "Parral, Chile\n"
    "Parral is a Chilean city and commune in Linares Province, Maule Region. "
    "It is the birthplace of the poet Pablo Neruda, who was born there in 1904 and whose "
    "family left the town shortly afterwards.\n"
    "Elevation\t162 m (531 ft)\nPopulation (2012)\t37,822\n"
)

#: The DECOY: a real page, a real elevation, Neruda named -- and only as a resident.
_PAGE_TEMUCO = (
    "Temuco\n"
    "Temuco is a city and commune, capital of the Cautin Province and of the Araucania "
    "Region of Chile.\n"
    "Elevation\t360 m (1,181 ft)\nPopulation (2017)\t282,415\n"
    "The poet Pablo Neruda lived in Temuco through his childhood, attended the Liceo de "
    "Hombres there and described the region's rain in his memoirs.\n"
    "Notable people: Pablo Neruda, poet; Gabriela Mistral, poet, who taught at the girls' "
    "school in the city."
)

#: The HOMONYM: a real, more famous "Parral" that never mentions the poet at all.
_PAGE_HIDALGO = (
    "Hidalgo del Parral\n"
    "Hidalgo del Parral, commonly known as Parral, is a city and seat of the surrounding "
    "municipality of Hidalgo del Parral in the Mexican state of Chihuahua.\n"
    "Elevation\t1,620 m (5,315 ft)\n"
    "The revolutionary general Pancho Villa was assassinated in the city in 1923."
)


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


_NEUTRAL_CHILD = {
    "node_id": "c9", "title": "reflect", "status": "done", "is_merge": False,
    "result": {"success": True, "action": "think", "thought": "walked the chain"},
}


def _graph(pages, *, mandate: str = _MANDATE) -> IdeaDag:
    """A run whose visits are ``(url, title, text)`` triples, in the order they happened."""
    graph = IdeaDag(root_title=mandate)
    parent = graph.add_child(graph.root_id(), "walk the chain", status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = "read the birthplace town's elevation"
    for i, (url, title, text) in enumerate(pages):
        visit = graph.add_child(parent.node_id, f"visit {i}", status=IdeaNodeStatus.DONE)
        visit.details[DetailKey.ACTION_RESULT.value] = {
            "action": "visit", "success": True, "url": url,
            "page_title": title, "content_full": text,
        }
    return graph


def _run(pages, *, summary, mandate: str = _MANDATE, achieved=True, **overrides):
    """One merge execute over a run that fetched ``pages``."""
    graph = _graph(pages, mandate=mandate)
    parent_id = graph.get_node(graph.root_id()).children[0]
    merge = graph.add_child(parent_id, "Merge: walk the chain", status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = [_NEUTRAL_CHILD]
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({
        "summary": summary, "key_findings": [], "goal_achieved": achieved,
        "goal_evaluation": "answered", "missing_requirements": [],
    })
    result = asyncio.run(
        MergeLeafAction(settings=settings).execute(graph, merge.node_id, _ScriptedIO(response))
    )
    return graph, parent_id, merge.node_id, result


_CORRECT_CHAIN = [
    (_URL_NERUDA, "Pablo Neruda", _PAGE_NERUDA),
    (_URL_PARRAL, "Parral, Chile", _PAGE_PARRAL),
]
_DECOY_CHAIN = [
    (_URL_NERUDA, "Pablo Neruda", _PAGE_NERUDA),
    (_URL_TEMUCO, "Temuco", _PAGE_TEMUCO),
]
_HOMONYM_CHAIN = [
    (_URL_NERUDA, "Pablo Neruda", _PAGE_NERUDA),
    (_URL_HIDALGO, "Hidalgo del Parral", _PAGE_HIDALGO),
]


# --- the mandate side ---------------------------------------------------------------------

def test_the_real_task_065_mandate_is_chain_shaped():
    """The trigger the whole mechanism hangs off, asserted against the shipped task text."""
    assert classify_shape(_MANDATE) == "chain"


def test_the_relation_is_mined_from_the_mandates_own_wording():
    relation, cues = mandate_relation(_MANDATE)
    assert relation == "birthplace"
    assert "birthplace" in cues


def test_a_mandate_naming_no_hop_relation_yields_nothing():
    assert mandate_relation("read the tower's completion year from its infobox") == ("", ())


def test_the_first_named_relation_wins_rather_than_the_union():
    """A mandate mentioning two relations hops on one; widening the cue set only weakens the
    closure test."""
    relation, cues = mandate_relation("report the poet's birthplace and who designed the town hall")
    assert relation == "birthplace"
    assert "designed by" not in cues


# --- the three real candidate pages -------------------------------------------------------

def test_the_correct_chain_closes_on_parrals_back_reference():
    result = audit_chain_closure(_graph(_CORRECT_CHAIN))
    assert result.active and result.closed
    assert not result.drifted
    assert result.missing_requirements() == []
    earlier, later, token, cue = result.closure
    assert (earlier, later) == (_URL_NERUDA, _URL_PARRAL)
    assert (token, cue) == ("pablo", "birthplace")


def test_the_temuco_decoy_is_flagged_as_chain_drift():
    """Neruda is named on the page, with his profession and his childhood there -- but never
    through the relation the mandate asked about."""
    result = audit_chain_closure(_graph(_DECOY_CHAIN))
    assert result.active and not result.closed
    assert result.drifted
    assert "birthplace" in result.missing_requirements()[0]
    assert _URL_TEMUCO in result.missing_requirements()[0]


def test_the_mexican_homonym_is_flagged_as_chain_drift():
    result = audit_chain_closure(_graph(_HOMONYM_CHAIN))
    assert result.drifted


def test_the_decoy_page_would_have_passed_numeric_provenance():
    """Why this check has to exist: the wrong page sources its own figure perfectly well."""
    from agent.app.idea_policies.grounding import answer_numeric_provenance

    numeric = answer_numeric_provenance(_graph(_DECOY_CHAIN), "the elevation is 360 m")
    assert [c.value for c in numeric.verified] == ["360"]
    assert not numeric.unsupported


# --- scope and direction ------------------------------------------------------------------

def test_the_back_reference_must_run_forward_along_the_chain():
    """The poet's OWN page says "Born ... Parral, Chile" whatever town the run then opened, so
    accepting an earlier page's reference to a later one would certify the decoy too. The same
    two real pages fetched in reverse order therefore do NOT close."""
    result = audit_chain_closure(_graph(list(reversed(_CORRECT_CHAIN))))
    assert result.drifted


def test_a_titles_disambiguating_qualifier_is_not_an_entity_token():
    """"Parral, Chile" is an entity named Parral; letting "chile" close a chain would credit
    any Chilean page at all."""
    pages = visited_pages(_graph([(_URL_PARRAL, "Parral, Chile", _PAGE_PARRAL)]))
    assert pages[0].tokens == ("parral",)


def test_a_country_wide_coincidence_does_not_close_a_chain():
    result = audit_chain_closure(_graph([
        (_URL_PARRAL, "Parral, Chile", _PAGE_PARRAL),
        ("https://en.wikipedia.org/wiki/Santa_Cruz", "Santa Cruz",
         "Santa Cruz is the birthplace of several public figures in Chile."),
    ]))
    assert result.drifted


def test_a_revisit_cannot_reorder_the_chain():
    """Deduped by URL keeping the first occurrence, so the poet's page stays the earlier hop."""
    result = audit_chain_closure(_graph(_CORRECT_CHAIN + [
        (_URL_NERUDA + "#Early_life", "Pablo Neruda", _PAGE_NERUDA),
    ]))
    assert result.closed


def test_a_search_only_run_is_inert():
    graph = IdeaDag(root_title=_MANDATE)
    node = graph.add_child(graph.root_id(), "search", status=IdeaNodeStatus.DONE)
    node.details[DetailKey.ACTION_RESULT.value] = {
        "action": "search", "success": True,
        "results": [{"title": "Temuco", "description": "Neruda's native town"}],
    }
    result = audit_chain_closure(graph)
    assert not result.active
    assert "fewer than two distinct pages" in result.reason


def test_a_single_page_run_is_inert():
    result = audit_chain_closure(_graph(_CORRECT_CHAIN[:1]))
    assert not result.active


def test_a_chain_mandate_with_no_relation_cue_is_inert():
    mandate = (
        "You are given NO URLs. Follow a dependency chain in which each step's answer is "
        "required to find the next page: identify the tallest completed building of 1998, then "
        "open its page and read the number of floors from the infobox."
    )
    result = audit_chain_closure(_graph(_DECOY_CHAIN, mandate=mandate))
    assert not result.active
    assert result.reason == "mandate names no hop relation"


def test_a_non_chain_mandate_is_inert():
    mandate = "For each of the six novels, report its first-publication year and pick the earliest."
    result = audit_chain_closure(_graph(_DECOY_CHAIN, mandate=mandate))
    assert not result.active
    assert result.reason == "mandate is not chain-shaped"


def test_a_page_with_no_text_makes_the_pair_unjudgeable_rather_than_drifted():
    result = audit_chain_closure(_graph([
        (_URL_NERUDA, "Pablo Neruda", _PAGE_NERUDA),
        (_URL_PARRAL, "Parral, Chile", ""),
    ]))
    assert not result.active
    assert "page text" in result.reason


# --- the merge action ---------------------------------------------------------------------

def test_detection_is_unconditional_and_the_default_verdict_is_untouched(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _run(
            _DECOY_CHAIN, summary="Neruda's native town Temuco sits at 360 m.",
        )
    details = graph.get_node(merge_id).details
    assert details[CHAIN_CLOSURE_OPEN_MARKER] == [_URL_NERUDA, _URL_TEMUCO]
    assert details[CHAIN_CLOSURE]["relation"] == "birthplace"
    assert result["goal_achieved"] is True
    assert graph.get_node(merge_id).status == IdeaNodeStatus.DONE
    assert graph.get_node(parent_id).status == IdeaNodeStatus.DONE
    assert any("chain closure OPEN" in r.message for r in caplog.records)
    assert not any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_flag_downgrades_a_drifted_chain(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _run(
            _DECOY_CHAIN, summary="Neruda's native town Temuco sits at 360 m.",
            merge_chain_closure_enabled=True,
        )
    details = graph.get_node(merge_id).details
    assert result["goal_achieved"] is False
    assert details[DetailKey.GOAL_ACHIEVED.value] is False
    assert details["merge_incomplete"] is True
    assert details["merge_should_skip"] is True
    assert graph.get_node(parent_id).status == IdeaNodeStatus.ACTIVE
    assert any("birthplace" in m for m in result["missing_requirements"])
    assert any("merge_chain_closure_enabled" in r.message for r in caplog.records)


def test_the_raw_model_output_survives_the_downgrade():
    _, _, _, result = _run(_DECOY_CHAIN, summary="Temuco, 360 m.",
                           merge_chain_closure_enabled=True)
    assert result["synthesized"]["goal_achieved"] is True
    assert result["goal_achieved"] is False


def test_the_correct_chain_is_untouched_with_the_flag_on(caplog):
    with caplog.at_level(logging.WARNING):
        graph, _, merge_id, result = _run(
            _CORRECT_CHAIN, summary="Parral, Chile, the poet's birthplace, sits at 162 m.",
            merge_chain_closure_enabled=True,
        )
    details = graph.get_node(merge_id).details
    assert result["goal_achieved"] is True
    assert details[CHAIN_CLOSURE]["closed"] is True
    assert CHAIN_CLOSURE_OPEN_MARKER not in details
    assert not any("chain closure OPEN" in r.message for r in caplog.records)


def test_a_non_chain_task_is_never_examined_even_with_the_flag_on():
    graph, _, merge_id, result = _run(
        _DECOY_CHAIN, summary="Temuco, 360 m.",
        mandate="Summarize what these pages say about the region.",
        merge_chain_closure_enabled=True,
    )
    details = graph.get_node(merge_id).details
    assert result["goal_achieved"] is True
    assert CHAIN_CLOSURE not in details
    assert CHAIN_CLOSURE_OPEN_MARKER not in details


def test_an_already_not_achieved_verdict_gains_the_gap_but_not_a_second_downgrade():
    _, _, merge_id, result = _run(
        _DECOY_CHAIN, summary="Temuco, 360 m.", achieved=False,
        merge_chain_closure_enabled=True,
    )
    assert result["goal_achieved"] is False
    assert any("birthplace" in m for m in result["missing_requirements"])
