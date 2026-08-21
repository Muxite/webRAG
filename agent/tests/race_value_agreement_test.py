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
