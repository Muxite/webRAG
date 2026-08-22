"""Do a race group's routes agree on the VALUE they raced for?

``race_route_evidence`` (``alternative_branch_structural_test.py``) establishes only that a
race group's members took DIFFERENT ROUTES; it never looks at what those routes returned.
This file pins the missing half, ``alternative_branch.race_value_agreement``, plus its two
merge-time consumers:

  1. the four verdicts (``agree``/``disagree``/``single``/``unknown``) and the SCOPING that
     makes them meaningful -- only the datum the mandate asks for is compared, so a ranked-list
     position drifting 20 -> 21 between two sources is structurally excluded rather than
     separately ignored;
  2. the reconstructed live failure: a 2026-08-21 completion narrating "all three routes
     confirm 575 meters" about a figure in no fetched page. Nothing computed route agreement,
     so nothing could contradict it. This does;
  3. ``SimpleMergePolicy`` (verdict stamping, unconditional; ``disagree`` suppressing winner
     selection; ``agree`` admitting a tier 2 inferred group) and ``MergeLeafAction`` (the
     conflict reaching ``missing_requirements`` and the achieved verdict), both gated behind
     ``merge_race_value_agreement_enabled``, default OFF.

No network and no LLM: every verdict is deterministic string work, and the one merge
completion is scripted.
"""
from __future__ import annotations

import asyncio
import json
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies import SimpleMergePolicy
from agent.app.idea_policies import alternative_branch as alt
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.merge import race_group_value_verdicts, race_value_conflicts

_MANDATE = "Report the main span of the Hardanger Bridge in metres. Do not guess."
_PARENT_TITLE = "Find the main span of the Hardanger Bridge"

# The same figure as three real sources write it, plus an incidental ranked-list position that
# genuinely differs between them (the page was read at different times).
_EN_WIKI = "The Hardanger Bridge has a main span of 1,310 m, ranked 20 among suspension bridges."
_NO_WIKI = "Hardangerbrua: main span 1 310 metres. Ranked 21 by span worldwide."
_STRUCTURAE = "Hardanger Bridge - main span length 1310 m (opened 2013)."

# The live hallucination: a route ASSERTING a figure no page carries.
_HALLUCINATED = "All three routes confirm the main span is 575 meters."


def _visit(url, content, **extra):
    details = {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.IS_LEAF.value: True,
        "optional_url": url,
        DetailKey.ACTION_RESULT.value: {
            "action": IdeaActionType.VISIT.value, "success": True, "url": url,
            "title": "Hardanger Bridge", "content": content,
        },
    }
    details.update(extra)
    return details


def _search(query, *hits, **extra):
    details = {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.IS_LEAF.value: True,
        DetailKey.QUERY.value: query,
        DetailKey.ACTION_RESULT.value: {
            "action": IdeaActionType.SEARCH.value, "success": True, "query": query,
            "results": [{"title": "hit", "url": f"https://e/{i}", "snippet": hit}
                        for i, hit in enumerate(hits)],
        },
    }
    details.update(extra)
    return details


def _graph(*specs, mandate=_MANDATE):
    """(graph, parent, [members]) for ``specs`` of ``(title, details)``, all DONE."""
    graph = IdeaDag(root_title="root", root_details={"mandate": mandate})
    parent = graph.add_child(graph.root_id(), _PARENT_TITLE, details={})
    members = []
    for title, det in specs:
        child = graph.add_child(parent.node_id, title, details=dict(det))
        child.status = IdeaNodeStatus.DONE
        members.append(child)
    return graph, parent, members


# ======================================================================================
# the four verdicts
# ======================================================================================


def test_the_same_value_written_three_ways_agrees():
    """"1,310" / "1 310 metres" / "1310 m" are one fact under normalization."""
    _g, _p, members = _graph(
        ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("Read the Norwegian encyclopedia entry", _visit("https://no.wikipedia.org/x", _NO_WIKI)),
        ("Read the structures database", _visit("https://structurae.net/x", _STRUCTURAE)),
    )
    evidence = alt.race_value_evidence(members, _MANDATE)
    assert evidence.verdict == alt.RACE_VALUE_AGREE
    assert evidence.datum == "span"
    assert set(evidence.values.values()) == {"1310 m"}
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_AGREE


def test_genuinely_different_values_disagree():
    _g, _p, members = _graph(
        ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("Read the bridge register", _visit(
            "https://register.example/x", "Hardanger Bridge: main span 1,380 m.")),
    )
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_DISAGREE


def test_one_member_carrying_a_value_is_single_not_agreement():
    """One route confirming itself is not confirmation -- there is nothing to cross-check."""
    _g, _p, members = _graph(
        ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("Read the tourism page", _visit(
            "https://visit.example/x", "The bridge crosses the Hardangerfjord in Vestland.")),
    )
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_SINGLE


def test_no_member_carrying_a_value_is_unknown_not_agreement():
    """``unknown`` is insufficient data. Reading it as agreement would confirm silence."""
    _g, _p, members = _graph(
        ("Read the tourism page", _visit(
            "https://visit.example/x", "The bridge crosses the Hardangerfjord in Vestland.")),
        ("Read the history page", _visit(
            "https://history.example/x", "Construction began in February and ran for years.")),
    )
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_UNKNOWN


def test_a_mandate_asking_for_no_measurable_datum_is_unknown():
    """No scopeable ask -> no comparison at all, rather than comparing arbitrary numbers."""
    _g, _p, members = _graph(
        ("Read page A", _visit("https://a.example/x", "Route A reports 1,310 and 20.")),
        ("Read page B", _visit("https://b.example/x", "Route B reports 4,298 and 21.")),
        mandate="Name the architect of the Hardanger Bridge.",
    )
    assert alt.race_value_agreement(members, "Name the architect of the Hardanger Bridge.") == (
        alt.RACE_VALUE_UNKNOWN
    )


def test_a_count_style_ask_is_not_compared():
    """"how many" has no stable page wording, so its extractor grabs the FIRST number
    anywhere -- comparing that across two pages would manufacture disagreement."""
    mandate = "How many lanes does the Hardanger Bridge carry?"
    _g, _p, members = _graph(
        ("Read page A", _visit("https://a.example/x", "Opened in 2013. It carries 2 lanes.")),
        ("Read page B", _visit("https://b.example/x", "Length 1380 m. It carries 2 lanes.")),
        mandate=mandate,
    )
    assert alt.race_value_agreement(members, mandate) == alt.RACE_VALUE_UNKNOWN


def test_differing_units_report_unknown_rather_than_a_false_disagreement():
    """1,310 m and 4,298 ft are one agreeing fact written twice, not two answers."""
    _g, _p, members = _graph(
        ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("Read the imperial-units page", _visit(
            "https://us.example/x", "Hardanger Bridge: main span 4,298 ft.")),
    )
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_UNKNOWN


def test_a_search_members_snippets_count_as_what_its_route_returned():
    _g, _p, members = _graph(
        ("Search the web for the span", _search(
            "hardanger bridge main span", "Main span of 1,310 m, completed 2013.")),
        ("Read the structures database", _visit("https://structurae.net/x", _STRUCTURAE)),
    )
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_AGREE


# ======================================================================================
# scoping -- the load-bearing property
# ======================================================================================


def test_an_incidental_number_difference_does_not_trigger_disagreement():
    """The ranked-list position differs (20 vs 21) while the asked-for span matches.

    The strong-agent trace's own example of load-bearing vs incidental disagreement. Scoping
    to the mandate's quantity excludes it BY CONSTRUCTION -- there is no "ignore ranks" rule
    anywhere in the mechanism.
    """
    _g, _p, members = _graph(
        ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("Read the Norwegian encyclopedia entry", _visit("https://no.wikipedia.org/x", _NO_WIKI)),
    )
    assert "20" in _EN_WIKI and "21" in _NO_WIKI, "fixture: the incidental numbers differ"
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_AGREE


def test_the_scoped_datum_wins_over_an_agreeing_incidental_one():
    """Mirror image: the incidental numbers MATCH while the asked-for span does not."""
    _g, _p, members = _graph(
        ("Read the English encyclopedia entry", _visit(
            "https://en.wikipedia.org/x", "Main span 1,310 m; ranked 20 worldwide.")),
        ("Read the register", _visit(
            "https://register.example/x", "Main span 1,380 m; ranked 20 worldwide.")),
    )
    assert alt.race_value_agreement(members, _MANDATE) == alt.RACE_VALUE_DISAGREE


def test_member_contracts_scope_the_comparison_when_the_mandate_cannot():
    """No mandate text -> fall back to the members' own ``expect``/goal contracts."""
    _g, _p, members = _graph(
        ("Route A", _visit("https://a.example/x", _EN_WIKI, **{
            DetailKey.EXPECT.value: "the main span in metres and the source URL"})),
        ("Route B", _visit("https://b.example/x", _NO_WIKI, **{
            DetailKey.EXPECT.value: "the main span in metres and the source URL"})),
        mandate="",
    )
    assert alt.race_value_agreement(members) == alt.RACE_VALUE_AGREE


# ======================================================================================
# the 2026-08-21 live probe's false positives: a decoy datum and a document-wide number
# ======================================================================================

# Task 150's own mandate shape: it asks for the MAIN SPAN and says in the same breath that the
# TOTAL LENGTH one infobox line above is the wrong answer. Both are measurable datums, so
# ``_required_datums`` returns both -- ``length`` first.
_DECOY_MANDATE = (
    "ONE value is wanted: the length of the MAIN SPAN, in metres, of the Hardanger Bridge in "
    "Norway. ROUTE 1 - the bridge's own English Wikipedia article (its infobox also prints the "
    "bridge's TOTAL length). ROUTE 2 - English Wikipedia's ranked list of the longest "
    "suspension bridge spans. Careful: the bridge's total length is NOT its main span."
)

# ROUTE 1 as the real article prints it: the trap and the answer, one line apart.
_INFOBOX = ("Hardanger Bridge. Characteristics: design suspension bridge, total length 1,380 "
            "metres (4,530 ft), width 20 metres (66 ft), longest span 1,310 metres (4,300 ft).")

# ROUTE 2 as the real 60k-character ranked list prints it: a cue-proximate number in an
# unrelated lead sentence, the record-holder's span in another, and the target's own row far
# below with no cue word anywhere near it.
_RANKED_LIST = (
    "List of longest suspension bridge spans. The 1915 Canakkale Bridge in Turkey has the "
    "longest central span (2,023 m) of any suspension bridge. Therefore, as of October 2025, "
    "the 28 longest bridges on this list are the 28 longest spans of all bridge types.\n"
    + "".join(f"{i} Some Other Bridge {1400 + i} m (4,600 ft) 19{i} Somewhere, Elsewhere\n"
             for i in range(60, 80))
    + "Tsing Ma Bridge 1,377 m (4,518 ft) 1997 Tsing Yi, Hong Kong\n"
    "Hardanger Bridge 1,310 m (4,298 ft) 2013 Ulvik - Eidfjord, Vestland\n"
    "Verrazzano-Narrows Bridge 1,298 m (4,260 ft) 1964 New York City, New York\n"
)


def test_the_mandates_decoy_datum_cannot_veto_agreement_on_the_asked_for_one():
    """The live false positive: two routes agreeing on the span, reported as disagreeing.

    ``_required_datums`` yields ``length`` before ``span`` for this mandate, and the two pages
    genuinely differ there -- ROUTE 1 prints the 1,380 m total length the mandate warns about,
    the ranked list prints no total length at all. Comparing the first datum that yields any
    value reported ``disagree`` and killed a correct answer.
    """
    _g, _p, members = _graph(
        ("ROUTE 1 - the bridge's own article", _visit("https://en.wikipedia.org/x", _INFOBOX)),
        ("ROUTE 2 - the ranked list", _visit("https://en.wikipedia.org/y", _RANKED_LIST)),
        mandate=_DECOY_MANDATE,
    )
    evidence = alt.race_value_evidence(members, _DECOY_MANDATE)

    assert [d[0] for d in alt._scoped_datums(members, _DECOY_MANDATE)][0] == "span", (
        "the mandate's most specific cue ('main span') is the ask, 'longest' is the decoy"
    )
    assert evidence.verdict == alt.RACE_VALUE_AGREE
    assert evidence.datum == "span"
    assert set(evidence.values.values()) == {"1310 m"}


def test_a_number_beside_the_cue_but_nowhere_near_the_entity_is_not_this_entitys_value():
    """The second live false positive, isolated: on the ranked list the unanchored cue search
    finds "the 28 longest bridges" and the record holder's 2,023 m span, both thousands of
    characters from the row being raced for."""
    from agent.app.idea_policies.alternative_branch import _RACE_NUMBER_RE, _member_value
    from agent.app.idea_policies.contract_satisfaction import _find_number_near

    page = _RANKED_LIST.lower()
    unanchored = _find_number_near(page, ("span", "main span"), _RACE_NUMBER_RE)
    assert unanchored is not None and unanchored.group() == "2,023", (
        "fixture: without an entity anchor the cue search reads the record holder's span"
    )
    assert _find_number_near(page, ("length", "long"), _RACE_NUMBER_RE).group() == "28"

    anchors = alt._entity_anchors(_DECOY_MANDATE)
    assert "hardanger" in anchors
    assert _member_value(page, ("span", "main span"), anchors,
                         entity_fallback=True).render() == "1310 m"


def test_a_row_naming_no_datum_cue_at_all_still_yields_the_entitys_measurement():
    """A table row prints ``Hardanger Bridge | 1,310 m | 2013`` with the quantity named only in
    a header far above, so cue proximity finds nothing beside the entity. The unit is what
    separates the measurement from the row's year and rank."""
    from agent.app.idea_policies.alternative_branch import _member_value

    row = ("tsing ma bridge 1,377 m (4,518 ft) 1997 hong kong [25]\n"
           "hardanger bridge 1,310 m (4,298 ft) 2013 ulvik - eidfjord [26]\n")
    value = _member_value(row, ("span", "main span"), ["hardanger"], entity_fallback=True)

    assert value is not None and value.render() == "1310 m"


def test_a_genuine_disagreement_survives_when_no_datum_agrees():
    """Precedence prefers agreement, never manufactures it: the register contradicts the
    article on the span and carries no total length to agree on either."""
    _g, _p, members = _graph(
        ("ROUTE 1 - the bridge's own article", _visit("https://en.wikipedia.org/x", _INFOBOX)),
        ("ROUTE 2 - the register", _visit(
            "https://register.example/x", "Hardanger Bridge: main span 1,380 m.")),
        mandate=_DECOY_MANDATE,
    )
    assert alt.race_value_agreement(members, _DECOY_MANDATE) == alt.RACE_VALUE_DISAGREE


def test_the_anchor_argument_is_off_by_default_for_every_other_caller():
    """The shared cue search is unchanged unless a caller asks for entity scoping."""
    from agent.app.idea_policies.contract_satisfaction import _find_number_near

    page = "widget catalogue: length 512 mm."
    assert _find_number_near(page, ("length",)).group() == "512"
    assert _find_number_near(page, ("length",), anchors=("widget",)).group() == "512"
    assert _find_number_near(page, ("length",), anchors=("sprocket",)).group() == "512", (
        "an anchor the page never names cannot scope anything, so the search is left alone"
    )


# ======================================================================================
# the 2026-08-22 fresh-data false positive: a comparative clause
# ======================================================================================

# The real Serper snippet block, from the fresh re-validation capture. It states the raced-for
# figure TWICE and compares it once -- and the comparative clause's own "main span" cue sits
# about one character closer to its "30" than the first sentence's cue sits to its "1310", so
# nearest-cue proximity read the DIFFERENCE as the value.
_COMPARATIVE_SNIPPET = (
    "The bridge is 1380 meters long with a main span of 1310 meters, which is Norway's "
    "longest. The main span is 30 meters longer than the Golden Gate Bridge.\n"
    "The Hardanger Bridge in Norway. With a main span of 1310 metres, it is the 15th longest "
    "suspension bridge in the world."
)


def test_a_comparative_clauses_difference_is_not_read_as_the_value():
    """The live false ``disagree``: a search route quoting "30 meters longer than the Golden
    Gate Bridge" against a route correctly reading 1,310 m. Both state the same span."""
    _g, _p, members = _graph(
        ("ROUTE 1 - the bridge's own article", _visit("https://en.wikipedia.org/x", _INFOBOX)),
        ("ROUTE 2 - a web search", _search("hardanger bridge main span", _COMPARATIVE_SNIPPET)),
        mandate=_DECOY_MANDATE,
    )
    evidence = alt.race_value_evidence(members, _DECOY_MANDATE)

    assert evidence.verdict == alt.RACE_VALUE_AGREE
    assert set(evidence.values.values()) == {"1310 m"}


def test_the_comparative_occurrence_loses_to_the_stated_value_of_the_same_datum():
    """Isolated: the guard REORDERS occurrences rather than dropping the search. The rejected
    "30" is replaced by another occurrence of the figure the same text states directly."""
    from agent.app.idea_policies.alternative_branch import _member_value

    text = _COMPARATIVE_SNIPPET.lower()
    assert _member_value(text, ("span", "main span"), ["hardanger"]).render() == "1310 m"


def test_a_value_that_is_then_compared_is_still_read_as_the_value():
    """The guard's own precision bar: "1,310 m, which is longer than X" STATES the span and then
    compares it, unlike "30 m longer than X", which states only the difference. Only the unit
    word may sit between the number and the comparative, so the two shapes stay separable."""
    from agent.app.idea_policies.alternative_branch import _member_value

    text = ("hardanger bridge has a main span of 1,310 m, which is longer than the golden "
            "gate bridge.")
    assert _member_value(text, ("span", "main span"), ["hardanger"]).render() == "1310 m"


def test_a_bound_stated_before_the_number_is_rejected_too():
    """The other direction of the same construction: "more than"/"compared to" introduce a
    bound or a foreign entity's figure, not this entity's own value."""
    from agent.app.idea_policies.alternative_branch import _comparative_number, _RACE_NUMBER_RE

    for text in ("main span of more than 1,200 m",
                 "main span, compared to 1,280 m for the older crossing"):
        match = _RACE_NUMBER_RE.search(text)
        assert _comparative_number(text, match), text


def test_an_only_comparative_reading_goes_quiet_instead_of_disagreeing():
    """When the ONLY cue-proximate number is a difference, no value is read at all and the
    verdict degrades to ``single``. A missed ``disagree`` is this check's safe failure; using
    the difference figure anyway is how a correct answer gets destroyed."""
    _g, _p, members = _graph(
        ("ROUTE 1 - the bridge's own article", _visit("https://en.wikipedia.org/x", _INFOBOX)),
        ("ROUTE 2 - a web search", _search(
            "hardanger bridge main span",
            "The Hardanger Bridge's main span is 30 metres longer than the Golden Gate Bridge.")),
        mandate=_DECOY_MANDATE,
    )
    evidence = alt.race_value_evidence(members, _DECOY_MANDATE)

    assert evidence.verdict == alt.RACE_VALUE_SINGLE
    assert set(evidence.values.values()) == {"1310 m"}


def test_a_comparative_figure_cannot_answer_a_cue_less_table_row_either():
    """The anchor fallback accepts any unit-carrying number near the entity, and a difference
    carries a unit, so it needs the same guard as the cue search."""
    from agent.app.idea_policies.alternative_branch import _member_value

    row = "hardanger bridge is 30 m longer than the golden gate bridge.\n"
    assert _member_value(row, ("span", "main span"), ["hardanger"],
                         entity_fallback=True) is None


def test_the_reject_argument_is_off_by_default_for_every_other_caller():
    """The shared cue search vetoes nothing unless a caller passes a predicate."""
    from agent.app.idea_policies.contract_satisfaction import _find_number_near

    page = "widget catalogue: length 30 mm longer than the mk i."
    assert _find_number_near(page, ("length",)).group() == "30"
    assert _find_number_near(page, ("length",), reject=lambda m: m.group() == "30") is None


# ======================================================================================
# a cross-language route
# ======================================================================================

# The real third route of task 150: the Norwegian article labels the raced-for quantity
# 'hovedspenn', so an English-only cue vocabulary reads nothing off it at all.
_NO_ROUTE = ("Hardangerbrua er en bro over Eidfjorden mellom Vallavik og Bu. "
             "Broen er 1380 meter lang, med et hovedspenn pa 1310 meter.")


def test_a_norwegian_route_confirms_the_english_one_instead_of_going_quiet():
    _g, _p, members = _graph(
        ("ROUTE 1 - English Wikipedia", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("ROUTE 3 - Norwegian Wikipedia", _visit("https://no.wikipedia.org/x", _NO_ROUTE)),
    )
    assert "span" not in _NO_ROUTE.lower(), (
        "fixture: no English cue word appears, which is why this route used to report 'single'"
    )
    evidence = alt.race_value_evidence(members, _MANDATE)

    assert evidence.verdict == alt.RACE_VALUE_AGREE
    assert set(evidence.values.values()) == {"1310 m"}


# ======================================================================================
# the reconstructed live case: "all three routes confirm 575 meters"
# ======================================================================================


def _hallucination_graph():
    """Two routes that fetched 1,310 m, one that asserts 575 -- the real 2026-08-21 shape."""
    return _graph(
        ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
        ("Read the Norwegian encyclopedia entry", _visit("https://no.wikipedia.org/x", _NO_WIKI)),
        ("Summarize the routes", _visit("https://c.example/x", _HALLUCINATED)),
    )


def test_the_575_hallucination_is_caught_as_disagreement():
    _g, _p, members = _hallucination_graph()
    evidence = alt.race_value_evidence(members, _MANDATE)
    assert evidence.verdict == alt.RACE_VALUE_DISAGREE
    assert sorted(set(evidence.values.values())) == ["1310 m", "575 m"]
    assert not any("575" in page for page in (_EN_WIKI, _NO_WIKI)), (
        "fixture: 575 appears in no fetched page, exactly as it appeared in no fetched page live"
    )


def test_the_575_conflict_reaches_the_merge_as_a_missing_requirement():
    graph, parent, members = _hallucination_graph()
    parent.details[alt.RACE_GROUPS] = {"span": [m.node_id for m in members]}
    merge_node = graph.add_child(parent.node_id, "Merge: span", details={})

    assert race_value_conflicts(graph, merge_node) == ["span"]


# ======================================================================================
# merge-time consumption
# ======================================================================================


def _race_graph(*specs, authored=True, mandate=_MANDATE):
    graph, parent, members = _graph(*specs, mandate=mandate)
    key = alt.RACE_GROUPS if authored else alt.RACE_GROUPS_INFERRED
    parent.details[key] = {"span": [m.node_id for m in members]}
    return graph, parent, members


def _policy(**overrides):
    settings = {
        "merge_race_winner_selection_enabled": True,
        "merge_race_winner_selection_includes_inferred_groups_enabled": False,
        "merge_race_value_agreement_enabled": False,
    }
    settings.update(overrides)
    return SimpleMergePolicy(settings=settings)


_AGREEING = (
    ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
    ("Read the Norwegian encyclopedia entry", _visit("https://no.wikipedia.org/x", _NO_WIKI)),
)
_CONFLICTING = (
    ("Read the English encyclopedia entry", _visit("https://en.wikipedia.org/x", _EN_WIKI)),
    ("Read the register", _visit(
        "https://register.example/x", "Hardanger Bridge: main span 1,380 m.")),
)


def test_the_verdict_is_stamped_even_with_every_merge_flag_off(caplog):
    """Detection is unconditional: the disagree rate is measurable before anything acts."""
    graph, parent, _members = _race_graph(*_CONFLICTING)
    with caplog.at_level(logging.WARNING):
        excluded = SimpleMergePolicy(settings={})._race_excluded_ids(graph, parent.node_id)

    assert excluded == set()
    assert parent.details[alt.RACE_VALUE_AGREEMENT] == {"span": alt.RACE_VALUE_DISAGREE}
    assert any("returned DIFFERENT values" in r.message for r in caplog.records)


def test_a_disagreeing_group_still_elects_a_winner_with_the_flag_off():
    graph, parent, (a, b) = _race_graph(*_CONFLICTING)
    excluded = _policy()._race_excluded_ids(graph, parent.node_id)

    assert excluded == {b.node_id}
    assert a.details.get(alt.RACE_LOSER) is None and b.details[alt.RACE_LOSER] is True


def test_the_flag_keeps_every_route_of_a_disagreeing_group(caplog):
    """Discarding N-1 contradicting routes would hide the disagreement from synthesis."""
    graph, parent, (a, b) = _race_graph(*_CONFLICTING)
    with caplog.at_level(logging.WARNING):
        policy = _policy(merge_race_value_agreement_enabled=True)
        merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]

    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}
    assert alt.RACE_LOSER not in b.details
    assert b.status == IdeaNodeStatus.DONE
    assert any("winner selection skipped" in r.message for r in caplog.records)


def test_an_agreeing_group_is_resolved_exactly_as_before():
    """The flag touches ``disagree`` only -- an agreeing race collapses as it always did."""
    graph, parent, (a, b) = _race_graph(*_AGREEING)
    policy = _policy(merge_race_value_agreement_enabled=True)
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]

    assert [item["node_id"] for item in merged] == [a.node_id]
    assert b.status == IdeaNodeStatus.SKIPPED
    assert parent.details[alt.RACE_VALUE_AGREEMENT] == {"span": alt.RACE_VALUE_AGREE}


# --- the positive half: value agreement rehabilitates a tier 2 inferred group -------------


def _tier2_graph(*specs):
    graph, parent, members = _graph(*specs)
    parent.details[alt.RACE_GROUPS_INFERRED] = {"inferred:span": [m.node_id for m in members]}
    parent.details[alt.RACE_GROUPS_INFERRED_TIERS] = {"inferred:span": 2}
    return graph, parent, members


def test_a_tier2_group_stays_unconsumable_without_value_agreement():
    graph, parent, (a, b) = _tier2_graph(*_CONFLICTING)
    policy = _policy(
        merge_race_winner_selection_includes_inferred_groups_enabled=True,
        merge_race_value_agreement_enabled=True,
    )
    assert policy._race_excluded_ids(graph, parent.node_id) == set()
    assert race_group_value_verdicts(graph, parent) == {"inferred:span": alt.RACE_VALUE_DISAGREE}


def test_value_agreement_admits_a_tier2_group_into_winner_selection():
    """A route-independent signal the title-overlap heuristic cannot produce for itself: if
    the routes returned the same value, collapsing them discards nothing."""
    graph, parent, (a, b) = _tier2_graph(*_AGREEING)
    policy = _policy(
        merge_race_winner_selection_includes_inferred_groups_enabled=True,
        merge_race_value_agreement_enabled=True,
    )
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]

    assert [item["node_id"] for item in merged] == [a.node_id]
    assert b.status == IdeaNodeStatus.SKIPPED


def test_the_tier2_rescue_needs_the_inferred_flag_too():
    graph, parent, (a, b) = _tier2_graph(*_AGREEING)
    policy = _policy(merge_race_value_agreement_enabled=True)
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]

    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}


# --- the merge action's verdict ------------------------------------------------------------


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _run_merge_action(specs, **overrides):
    graph, parent, members = _race_graph(*specs)
    parent.status = IdeaNodeStatus.ACTIVE
    parent.details[DetailKey.GOAL.value] = _PARENT_TITLE
    merge_node = graph.add_child(parent.node_id, "Merge: span", status=IdeaNodeStatus.PENDING)
    merge_node.details[DetailKey.MERGED_RESULTS.value] = [
        {"node_id": m.node_id, "title": m.title, "status": "done", "is_merge": False,
         "result": m.details[DetailKey.ACTION_RESULT.value]}
        for m in members
    ]
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({
        "summary": "All routes confirm the main span.", "goal_achieved": True,
        "goal_evaluation": "answered", "missing_requirements": [],
    })
    result = asyncio.run(
        MergeLeafAction(settings=settings).execute(graph, merge_node.node_id, _ScriptedIO(response))
    )
    return graph, merge_node, result


def test_the_merge_action_detects_the_conflict_without_changing_its_verdict(caplog):
    with caplog.at_level(logging.WARNING):
        _graph_, merge_node, result = _run_merge_action(_CONFLICTING)

    assert merge_node.details["race_value_disagreement"] == ["span"]
    assert result["goal_achieved"] is True
    assert any("do not agree" in r.message for r in caplog.records)
    assert not any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_flag_turns_the_conflict_into_a_missing_requirement(caplog):
    with caplog.at_level(logging.WARNING):
        _graph_, merge_node, result = _run_merge_action(
            _CONFLICTING, merge_race_value_agreement_enabled=True
        )

    assert result["goal_achieved"] is False
    assert merge_node.details[DetailKey.GOAL_ACHIEVED.value] is False
    assert any("conflicting" in req for req in merge_node.details["missing_requirements"])
    assert merge_node.details["merge_incomplete"] is True
    assert any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_an_agreeing_race_leaves_the_merge_verdict_alone():
    _graph_, merge_node, result = _run_merge_action(
        _AGREEING, merge_race_value_agreement_enabled=True
    )

    assert result["goal_achieved"] is True
    assert "race_value_disagreement" not in merge_node.details


def test_a_merge_with_no_race_registry_above_it_is_untouched():
    graph = IdeaDag(root_title="root", root_details={"mandate": _MANDATE})
    parent = graph.add_child(graph.root_id(), _PARENT_TITLE, details={})
    merge_node = graph.add_child(parent.node_id, "Merge: span", details={})

    assert race_value_conflicts(graph, merge_node) == []
