"""Structural race-group inference: the same relationship as ``race_group``, no tag required.

Companion to ``alternative_branch_test.py``, which pins the MODEL-TAGGED path. This file pins
the tag-free one (``expansion_race_group_structural_inference_enabled``), built because the
live emission probe found the authored tag is simply never emitted below the 14b tier, so a
prompt-side fix cannot reach the cheap models this repo exists to boost.

Layers pinned, in the order they run:

  1. ``idea_sequencing.disjoint_approach_reason`` — the "different routes" half, lifted out of
     ``siblings_are_independent`` so the inference can reuse it without a graph — and
     ``alternative_branch.race_route_evidence``, the stricter race-only fork of it;
  2. ``alternative_branch.infer_race_groups`` — tier 1 (``expect``), tier 2 (target/source
     route decomposition), and the same-target check that keeps a breadth fan-out out;
  3. the engine call site, on and off;
  4. ``SimpleMergePolicy``'s second, independent gate: an inferred group is instrumentation
     until ``merge_race_winner_selection_includes_inferred_groups_enabled`` says otherwise,
     and a TIER 2 group stays instrumentation even then.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies import alternative_branch as alt
from agent.app.idea_policies.base import (
    DecompositionPolicy,
    DetailKey,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_sequencing import disjoint_approach_reason

_MANDATE = "Report the main span of the Hardanger Bridge in metres. Do not guess."
_PARENT_TITLE = "Find the main span of the Hardanger Bridge"
_DATUM_PAGE = "The Hardanger Bridge has a main span of 1310 m and opened in 2013."
_SUBJECT_ONLY_PAGE = "The Hardanger Bridge is a suspension bridge in Vestland, Norway."


def _leaf(action=IdeaActionType.SEARCH.value, **extra):
    details = {DetailKey.ACTION.value: action, DetailKey.IS_LEAF.value: True}
    details.update(extra)
    return details


def _graph_with_children(*specs, parent_title=_PARENT_TITLE):
    """(graph, parent, [children]) for ``specs`` of ``(title, details)``."""
    graph = IdeaDag(root_title="root", root_details={"mandate": _MANDATE})
    parent = graph.add_child(graph.root_id(), parent_title, details={})
    children = [graph.add_child(parent.node_id, title, details=dict(det))
                for title, det in specs]
    return graph, parent, children


def _visit(url, **extra):
    return _leaf(IdeaActionType.VISIT.value, **{"optional_url": url, **extra})


def _search(query, **extra):
    return _leaf(IdeaActionType.SEARCH.value, **{DetailKey.QUERY.value: query, **extra})


# ======================================================================================
# layer 1 — the extracted "different routes" predicate
# ======================================================================================


def test_disjoint_approach_reason_reports_each_shape():
    _g, _p, (a, b) = _graph_with_children(
        ("A", _visit("https://en.wikipedia.org/wiki/Hardanger_Bridge")),
        ("B", _visit("https://structurae.net/hardanger")),
    )
    assert disjoint_approach_reason([a, b]) == "concrete_urls"

    _g, _p, (c, d) = _graph_with_children(
        ("C", _search("hardanger bridge main span")),
        ("D", _search("hardanger bru hovedspenn")),
    )
    assert disjoint_approach_reason([c, d]) == "disjoint_searches"

    _g, _p, (e, f) = _graph_with_children(
        ("E", _search("hardanger bridge main span")),
        ("F", _visit("https://structurae.net/hardanger")),
    )
    assert disjoint_approach_reason([e, f]) == "mixed_search_visit"


def test_disjoint_approach_reason_is_none_without_evidence():
    _g, _p, (a, b) = _graph_with_children(
        ("A", _search("same query")),
        ("B", _search("same query")),
    )
    assert disjoint_approach_reason([a, b]) is None
    assert disjoint_approach_reason([]) is None


def test_siblings_are_independent_still_reports_the_same_reasons():
    """Regression on the extraction: the three heuristics moved out of that function verbatim."""
    import logging

    from agent.app.idea_sequencing import siblings_are_independent

    graph, _p, (a, b) = _graph_with_children(
        ("A", _visit("https://a.example/x")),
        ("B", _visit("https://b.example/y")),
    )
    assert siblings_are_independent(
        graph, [a.node_id, b.node_id], graph.get_node(graph.root_id()),
        logging.getLogger("t"),
    ) == (True, "concrete_urls")


# ======================================================================================
# layer 2 — inference itself
# ======================================================================================


_EXPECT_SPAN = "the main span in metres and the source URL it was read from"
_EXPECT_SPAN_SHORT = "the main span in metres and the source URL"


def test_tier1_matching_expect_and_distinct_urls_registers_a_group():
    graph, parent, (a, b) = _graph_with_children(
        ("Read the encyclopedia entry", _visit(
            "https://en.wikipedia.org/wiki/Hardanger_Bridge",
            **{DetailKey.EXPECT.value: _EXPECT_SPAN})),
        ("Read the bridge database entry", _visit(
            "https://structurae.net/hardanger",
            **{DetailKey.EXPECT.value: _EXPECT_SPAN_SHORT})),
    )

    tiers = alt.infer_race_groups(parent, [a, b])

    assert list(tiers.values()) == [1]
    label = next(iter(tiers))
    assert parent.details[alt.RACE_GROUPS_INFERRED] == {label: [a.node_id, b.node_id]}
    assert parent.details[alt.RACE_GROUPS_INFERRED_TIERS] == {label: 1}
    assert a.details[alt.RACE_GROUP_INFERRED] == b.details[alt.RACE_GROUP_INFERRED] == label
    # The authored registry and the authored per-node tag are untouched: provenance is the
    # entire reason this is a second key, and writing DetailKey.RACE_GROUP would additionally
    # hand the group a dispatch-independence pass on heuristic evidence alone.
    assert alt.RACE_GROUPS not in parent.details
    for node in (a, b):
        assert DetailKey.RACE_GROUP.value not in node.details
        assert node.status == IdeaNodeStatus.PENDING


def test_tier1_ignores_a_breadth_fanout_with_unrelated_expect():
    """Task-052-shaped: disjoint searches, entirely different targets. The approach half of
    the race definition passes here — so this is the test that fails if the same-target half
    is missing or broken."""
    graph, parent, (a, b) = _graph_with_children(
        ("Find the Tesla Model 3 range", _search("tesla model 3 epa range", **{
            DetailKey.EXPECT.value: "the Tesla Model 3 EPA range in miles and the source URL"})),
        ("Find the Nissan Leaf range", _search("nissan leaf epa range", **{
            DetailKey.EXPECT.value: "the Nissan Leaf EPA range in miles and the source URL"})),
        parent_title="Compare the range of several electric cars",
    )
    assert disjoint_approach_reason([a, b]) == "disjoint_searches"

    assert alt.infer_race_groups(parent, [a, b]) == {}
    assert alt.RACE_GROUPS_INFERRED not in parent.details
    assert alt.RACE_GROUP_INFERRED not in a.details


def test_same_target_alone_is_not_enough_without_disjoint_approaches():
    """The other half: two leaves promising the same datum via the SAME query are a duplicate,
    not a race."""
    graph, parent, (a, b) = _graph_with_children(
        ("Route A", _search("hardanger bridge main span", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN})),
        ("Route B", _search("hardanger bridge main span", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN})),
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}


def test_a_singleton_group_is_dropped():
    graph, parent, (a, b) = _graph_with_children(
        ("Route A", _visit("https://a.example/x", **{DetailKey.EXPECT.value: _EXPECT_SPAN})),
        ("Route B", _visit("https://b.example/y", **{
            DetailKey.EXPECT.value: "the opening year of the bridge and the source URL"})),
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}
    assert alt.RACE_GROUPS_INFERRED not in parent.details


def test_authored_and_parked_candidates_are_skipped():
    """An authored tag is better evidence than this heuristic, and a parked A->B fallback is
    sequential by construction — neither is re-decided here."""
    graph, parent, (tagged, parked, plain) = _graph_with_children(
        ("Route A", _visit("https://a.example/x", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN,
            DetailKey.RACE_GROUP.value: "span"})),
        ("Route B", _visit("https://b.example/y", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN,
            alt.ALTERNATIVE_PENDING: True})),
        ("Route C", _visit("https://c.example/z", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN})),
    )
    # Only ``plain`` survives the filter, and one candidate is not a race.
    assert alt.infer_race_groups(parent, [tagged, parked, plain]) == {}


def test_a_merge_child_is_not_a_race_member():
    graph, parent, (a, b, merge_child) = _graph_with_children(
        ("Route A", _visit("https://a.example/x", **{DetailKey.EXPECT.value: _EXPECT_SPAN})),
        ("Route B", _visit("https://b.example/y", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN_SHORT})),
        ("Merge: parent goal", {DetailKey.ACTION.value: IdeaActionType.MERGE.value,
                                DetailKey.EXPECT.value: _EXPECT_SPAN}),
    )
    tiers = alt.infer_race_groups(parent, [a, b, merge_child])
    label = next(iter(tiers))
    assert parent.details[alt.RACE_GROUPS_INFERRED][label] == [a.node_id, b.node_id]


def test_three_way_group_registers_every_member():
    graph, parent, nodes = _graph_with_children(
        ("Route A", _visit("https://a.example/x", **{DetailKey.EXPECT.value: _EXPECT_SPAN})),
        ("Route B", _visit("https://b.example/y", **{
            DetailKey.EXPECT.value: _EXPECT_SPAN_SHORT})),
        ("Route C", _visit("https://c.example/z", **{DetailKey.EXPECT.value: _EXPECT_SPAN})),
    )
    tiers = alt.infer_race_groups(parent, nodes)
    label = next(iter(tiers))
    assert parent.details[alt.RACE_GROUPS_INFERRED][label] == [n.node_id for n in nodes]


#: The four leaves qwen2.5:14b emitted for ``race_akashi_span`` (struct+expect, replicate 0) in
#: the 2026-08-21 probe: two searches of DIFFERENT sources plus each one's own follow-up visit,
#: all four carrying the identical ``expect``. Verbatim from that capture, including the
#: unresolved URL placeholder the model wrote in place of a real link.
_AKASHI_EXPECT = (
    "the exact length of the main span in metres from the Wikipedia page AND the source URL"
)
_AKASHI_PLACEHOLDER_URL = "<URL found in search results>"


def _akashi_span_cluster(visit_urls=(_AKASHI_PLACEHOLDER_URL, _AKASHI_PLACEHOLDER_URL)):
    expect = {DetailKey.EXPECT.value: _AKASHI_EXPECT}
    return _graph_with_children(
        ("Find Akashi Kaikyo Bridge Wikipedia article", _leaf(**expect)),
        ("Visit Akashi Kaikyo Bridge Wikipedia article", _visit(visit_urls[0], **expect)),
        ("Find longest suspension bridge spans list on Wikipedia", _leaf(**expect)),
        ("Visit longest suspension bridge spans list on Wikipedia",
         _visit(visit_urls[1], **expect)),
    )


def test_tier1_recovers_the_race_hiding_inside_a_route_rejected_cluster():
    """Clustering groups by TARGET alone, so the two searches of this real capture arrive in
    one cluster with the two visits they feed — and ``race_route_evidence`` rejects that
    foursome, correctly. Before subset recovery the genuine two-search race inside it was
    discarded along with it and the whole cell registered NOTHING; now the largest valid
    strict subset is registered instead."""
    graph, parent, nodes = _akashi_span_cluster()
    search_a, _visit_a, search_b, _visit_b = nodes

    assert alt.race_route_evidence(nodes) is None  # the cluster as a whole is no race
    tiers = alt.infer_race_groups(parent, nodes)

    assert list(tiers.values()) == [1]
    label = next(iter(tiers))
    assert parent.details[alt.RACE_GROUPS_INFERRED] == {
        label: [search_a.node_id, search_b.node_id]}
    # The visits stay out: neither carries a resolved URL, so they are no race of their own.
    assert alt.RACE_GROUP_INFERRED not in _visit_a.details
    assert alt.RACE_GROUP_INFERRED not in _visit_b.details


def test_recovered_subsets_never_share_a_member():
    """Same shape with the visits' URLs resolved, which makes the visit pair its own valid
    race (two sources, one fact). Both disjoint subsets register; no candidate lands in two
    groups, since one node in two readings would be double-counted at merge time."""
    graph, parent, nodes = _akashi_span_cluster((
        "https://en.wikipedia.org/wiki/Akashi_Kaikyo_Bridge",
        "https://en.wikipedia.org/wiki/List_of_longest_suspension_bridge_spans",
    ))
    search_a, visit_a, search_b, visit_b = nodes

    tiers = alt.infer_race_groups(parent, nodes)

    assert sorted(tiers.values()) == [1, 1]
    registry = parent.details[alt.RACE_GROUPS_INFERRED]
    assert sorted(registry.values()) == sorted([
        [search_a.node_id, search_b.node_id], [visit_a.node_id, visit_b.node_id]])
    members = [node_id for ids in registry.values() for node_id in ids]
    assert len(members) == len(set(members))


def test_subset_recovery_finds_nothing_when_no_subset_is_a_race():
    """Recovery re-tests subsets against the SAME gate, so a cluster with no race inside it
    stays rejected: two identical queries are one route, and pairing either with the visit is
    the chain-step shape ``mixed_search_visit`` exists to refuse."""
    expect = {DetailKey.EXPECT.value: _EXPECT_SPAN}
    graph, parent, nodes = _graph_with_children(
        ("Search the encyclopedia", _search("hardanger bridge main span", **expect)),
        ("Search the encyclopedia again", _search("hardanger bridge main span", **expect)),
        ("Read the encyclopedia entry",
         _visit("https://en.wikipedia.org/wiki/Hardanger_Bridge", **expect)),
    )
    assert alt.infer_race_groups(parent, nodes) == {}


# --- tier 2 ---------------------------------------------------------------------------


def test_route_decomposition_splits_a_title_into_target_and_source():
    """The verb names the action and never the fact, so it is dropped; the first
    source-introducing preposition is the seam between what is sought and where."""
    route = alt._route_of("Search for Hardanger Bridge main span length on English Wikipedia")
    assert route.target == {"hardanger", "bridge", "main", "span", "length"}
    assert route.source == {"english", "wikipedia"}

    # Two-word introducer, and a later preposition belongs to the source phrase itself.
    assert alt._route_of("Check the span according to the bridge database").source == {
        "bridge", "database"}
    assert alt._route_of("Read the figure from the infobox on English Wikipedia").source == {
        "infobox", "english", "wikipedia"}

    # No source-introducing preposition at all -> no route evidence.
    assert alt._route_of("Verify Jane Austen's birth year").source == set()


def test_tier2_registers_one_target_reached_through_two_sources():
    """The rep2/150 shape from the live probe: one fact, two named routes to it."""
    graph, parent, (a, b) = _graph_with_children(
        ("Search for Hardanger Bridge main span length on English Wikipedia",
         _search("hardanger bridge main span english wikipedia")),
        ("Search for Hardanger Bridge main span length on the ranked list of longest "
         "suspension bridges", _search("list of longest suspension bridges")),
    )
    tiers = alt.infer_race_groups(parent, [a, b])

    assert list(tiers.values()) == [2]
    label = next(iter(tiers))
    assert parent.details[alt.RACE_GROUPS_INFERRED] == {label: [a.node_id, b.node_id]}
    # The tier is recorded separately from tier 1 precisely so this weaker signal's
    # false-positive rate stays measurable — and so merge consumption can refuse it outright.
    assert parent.details[alt.RACE_GROUPS_INFERRED_TIERS] == {label: 2}


def test_tier2_rejects_a_search_and_its_own_following_visit():
    """The false positive that retired the old symmetric title-Jaccard signal. This pair scored
    ABOVE genuine cross-route pairs on that metric; decomposed, it fails both halves — the two
    share a source (one route) and differ in target (the fact vs "the top search result")."""
    graph, parent, (a, b) = _graph_with_children(
        ("Search for Hardanger Bridge main span length on English Wikipedia",
         _search("hardanger bridge main span english wikipedia")),
        ("Visit the top search result for Hardanger Bridge main span length on English "
         "Wikipedia", _visit("https://en.wikipedia.org/wiki/Hardanger_Bridge")),
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}


def test_tier2_recovers_the_race_hiding_inside_a_route_rejected_cluster():
    """Subset recovery is in the clustering core both tiers share, so the ``race_liskov_turing_year``
    capture (qwen2.5:7b, replicate 1) recovers too: the ACM-side visit joins the two searches'
    cluster on target, its concrete URL makes the batch ``mixed_search_visit``, and the two
    searches alone are the race left when it is dropped."""
    graph, parent, nodes = _graph_with_children(
        ("Search for Barbara Liskov's ACM A.M. Turing Award year on Wikipedia",
         _search("barbara liskov turing award year wikipedia")),
        ("Visit Barbara Liskov's ACM A.M. Turing Award year on Wikipedia",
         _visit("https://en.wikipedia.org/wiki/Barbara_Liskov")),
        ("Search for Barbara Liskov's ACM A.M. Turing Award year on ACM website",
         _search("barbara liskov turing award year acm")),
        ("Visit Barbara Liskov's ACM A.M. Turing Award year on ACM website",
         _visit("https://www.acm.org/awards/turing.html")),
    )
    tiers = alt.infer_race_groups(parent, nodes)

    assert list(tiers.values()) == [2]
    label = next(iter(tiers))
    assert parent.details[alt.RACE_GROUPS_INFERRED] == {
        label: [nodes[0].node_id, nodes[2].node_id]}


def test_tier2_rejects_two_candidates_naming_the_same_source():
    """Same target AND same route is a duplicate step, not a race."""
    graph, parent, (a, b) = _graph_with_children(
        ("Read the recorded main span measurement in metres", _visit("https://a.example/x")),
        ("Read the published main span measurement in metres", _visit("https://b.example/y")),
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}


def test_tier2_ignores_candidates_that_name_no_source():
    """Task-052-shaped breadth: six author pages, none of which says WHERE it looks. A
    candidate that never names a route cannot be shown to take a different one."""
    graph, parent, (a, b) = _graph_with_children(
        ("Visit Jane Austen's author page", _visit("https://a.example/austen")),
        ("Visit Virginia Woolf's author page", _visit("https://b.example/woolf")),
        parent_title="Find the earliest-born of six novelists",
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}


def test_tier2_ignores_different_targets_reached_through_different_sources():
    graph, parent, (a, b) = _graph_with_children(
        ("Read the Tesla Model 3 range from the manufacturer specification",
         _visit("https://a.example/tesla")),
        ("Read the Nissan Leaf range from the EPA database", _visit("https://b.example/leaf")),
        parent_title="Compare the range of several electric cars",
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}


def test_tier2_allows_two_sources_that_share_only_a_generic_token():
    """"English Wikipedia" against "Norwegian Wikipedia" is the canonical race of the whole
    mechanism, and a zero-overlap bar on sources would throw it away over the shared word
    ``wikipedia`` — hence :data:`RACE_ROUTE_SOURCE_OVERLAP` rather than strict disjointness."""
    graph, parent, (a, b) = _graph_with_children(
        ("Read the main span figure on English Wikipedia", _visit("https://en.example/x")),
        ("Read the main span figure on Norwegian Wikipedia", _visit("https://no.example/y")),
    )
    assert list(alt.infer_race_groups(parent, [a, b]).values()) == [2]


# --- the race-specific route gate -------------------------------------------------------


def test_race_route_evidence_rejects_mixed_search_visit():
    """``disjoint_approach_reason`` is right to call this pair dispatch-independent and wrong
    to call it a race: a search and its own resulting visit is the definition of a chain step
    pair. The fork exists so the dispatch verdict stays byte-identical."""
    _g, _p, (a, b) = _graph_with_children(
        ("A", _search("hardanger bridge main span")),
        ("B", _visit("https://structurae.net/hardanger")),
    )
    assert disjoint_approach_reason([a, b]) == "mixed_search_visit"
    assert alt.race_route_evidence([a, b]) is None


def test_race_route_evidence_requires_every_visit_member_to_carry_a_url():
    _g, _p, (a, b) = _graph_with_children(
        ("A", _visit("https://a.example/x")),
        ("B", _visit("https://b.example/y")),
    )
    assert alt.race_route_evidence([a, b]) == "concrete_urls"

    b.details["optional_url"] = ""
    assert alt.race_route_evidence([a, b]) is None


def test_race_route_evidence_passes_the_remaining_reasons_through():
    _g, _p, (a, b) = _graph_with_children(
        ("A", _search("hardanger bridge main span")),
        ("B", _search("hardanger bru hovedspenn")),
    )
    assert alt.race_route_evidence([a, b]) == "disjoint_searches"
    assert alt.race_route_evidence([]) is None


def test_tiers_are_never_mixed_inside_one_group():
    """A leaf WITH a contract and a leaf without are compared on different evidence, so they
    are clustered separately rather than one being scored against the other's proxy."""
    graph, parent, (a, b) = _graph_with_children(
        ("Confirm the published span figure in metres", _visit(
            "https://a.example/x", **{DetailKey.EXPECT.value: _EXPECT_SPAN})),
        ("Confirm the published span figure in metres", _visit("https://b.example/y")),
    )
    assert alt.infer_race_groups(parent, [a, b]) == {}


def test_infer_race_groups_tolerates_detail_less_and_titleless_nodes():
    class Bare:
        node_id = "x"
        title = ""
        details = None

    graph, parent, (a,) = _graph_with_children(("Route A", _visit("https://a.example/x")))
    assert alt.infer_race_groups(parent, [Bare(), a]) == {}


# ======================================================================================
# layer 3 — the engine call site
# ======================================================================================


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    def __init__(self, settings=None, candidates=None):
        super().__init__(settings=settings)
        self._candidates = candidates or []

    async def expand(self, graph, node_id, memories=None):
        return [json.loads(json.dumps(c)) for c in self._candidates]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class FakeActionRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):  # pragma: no cover - no action is executed here
        return None


_RACE_CANDIDATES = [
    {
        "title": "Read the encyclopedia entry",
        "details": {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "https://en.wikipedia.org/wiki/Hardanger_Bridge",
            DetailKey.EXPECT.value: _EXPECT_SPAN,
        },
    },
    {
        "title": "Read the bridge database entry",
        "details": {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "https://structurae.net/hardanger",
            DetailKey.EXPECT.value: _EXPECT_SPAN_SHORT,
        },
    },
]


def _make_engine(**overrides):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "got_reexpand_enabled": False,
        "auto_parallel_siblings": False,
        "semantic_dedup_visits_enabled": False,
        "plan_library_enabled": False,
    }
    settings.update(overrides)
    return IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=FakeExpansion(settings, candidates=_RACE_CANDIDATES),
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeActionRegistry(settings),
        post_expansion_hooks=[],
    )


async def _expand_through_engine(**overrides):
    engine = _make_engine(**overrides)
    graph = IdeaDag(root_title="root", root_details={"mandate": _MANDATE})
    parent = graph.add_child(graph.root_id(), _PARENT_TITLE, details={})
    await engine._handle_expansion_node(graph, parent.node_id, 0, None)
    return graph, parent


@pytest.mark.asyncio
async def test_engine_populates_the_inferred_registry_when_the_flag_is_on():
    _graph, parent = await _expand_through_engine(
        expansion_race_group_structural_inference_enabled=True,
    )
    registry = parent.details[alt.RACE_GROUPS_INFERRED]
    assert len(registry) == 1
    assert len(next(iter(registry.values()))) == 2
    assert set(parent.details[alt.RACE_GROUPS_INFERRED_TIERS].values()) == {1}


@pytest.mark.asyncio
async def test_engine_works_with_the_branching_schema_variant_switched_off():
    """The whole point: no tag is asked for, so ``alternative_branch_enabled`` stays off."""
    _graph, parent = await _expand_through_engine(
        expansion_race_group_structural_inference_enabled=True,
        expansion_alternative_branch_enabled=False,
    )
    assert parent.details[alt.RACE_GROUPS_INFERRED]
    assert alt.RACE_GROUPS not in parent.details


@pytest.mark.asyncio
async def test_engine_flag_off_never_touches_the_registry(monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("infer_race_groups ran with the flag off")

    monkeypatch.setattr(alt, "infer_race_groups", _boom)
    _graph, parent = await _expand_through_engine()

    assert alt.RACE_GROUPS_INFERRED not in parent.details
    assert alt.RACE_GROUPS_INFERRED_TIERS not in parent.details


# ======================================================================================
# layer 4 — the second, independent merge-time gate
# ======================================================================================


def _visit_result(content):
    return {
        "action": IdeaActionType.VISIT.value, "success": True,
        "url": "https://en.wikipedia.org/wiki/Hardanger_Bridge",
        "title": "Hardanger Bridge", "content": content,
    }


def _inferred_race_graph():
    """Two DONE routes to one datum, registered by inference: one verifies it, one does not."""
    graph, parent, (a, b) = _graph_with_children(
        ("Read the encyclopedia entry", _visit(
            "https://en.wikipedia.org/wiki/Hardanger_Bridge", **{
                DetailKey.GOAL.value: _PARENT_TITLE,
                DetailKey.EXPECT.value: _EXPECT_SPAN,
                DetailKey.ACTION_RESULT.value: _visit_result(_SUBJECT_ONLY_PAGE),
                alt.RACE_COMPLETED_STEP: 1})),
        ("Read the bridge database entry", _visit(
            "https://structurae.net/hardanger", **{
                DetailKey.GOAL.value: _PARENT_TITLE,
                DetailKey.EXPECT.value: _EXPECT_SPAN_SHORT,
                DetailKey.ACTION_RESULT.value: _visit_result(_DATUM_PAGE),
                alt.RACE_COMPLETED_STEP: 2})),
    )
    a.status = b.status = IdeaNodeStatus.DONE
    tiers = alt.infer_race_groups(parent, [a, b])
    assert tiers, "fixture precondition: the group is inferred"
    return graph, parent, a, b, next(iter(tiers))


def _policy(*, winner_selection=True, include_inferred=False):
    return SimpleMergePolicy(settings={
        "merge_race_winner_selection_enabled": winner_selection,
        "merge_race_winner_selection_includes_inferred_groups_enabled": include_inferred,
    })


def test_inferred_groups_are_instrumentation_only_by_default():
    graph, parent, a, b, label = _inferred_race_graph()
    policy = _policy()

    assert policy._race_excluded_ids(graph, parent.node_id) == set()
    assert policy.select_winner(graph, parent.node_id, label) is None
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}
    assert a.status == b.status == IdeaNodeStatus.DONE
    assert alt.RACE_LOSER not in a.details


def test_both_flags_on_resolves_an_inferred_group_like_an_authored_one():
    graph, parent, a, b, label = _inferred_race_graph()
    policy = _policy(include_inferred=True)

    # Same mechanical chain as a tagged group: the datum-verified route wins even though it
    # completed later, and the loser is dropped from synthesis and marked SKIPPED.
    assert policy.select_winner(graph, parent.node_id, label) == b.node_id
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]
    assert [item["node_id"] for item in merged] == [b.node_id]
    assert a.status == IdeaNodeStatus.SKIPPED
    assert a.details[alt.RACE_LOSER] is True


def test_the_inferred_flag_alone_still_needs_winner_selection_armed():
    graph, parent, a, b, label = _inferred_race_graph()
    policy = _policy(winner_selection=False, include_inferred=True)

    assert policy._race_excluded_ids(graph, parent.node_id) == set()
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}


def test_an_authored_group_is_unaffected_by_the_inferred_registry():
    graph, parent, a, b, _label = _inferred_race_graph()
    plain = graph.add_child(parent.node_id, "Unrelated fact", details=_visit(
        "https://c.example/z", **{DetailKey.ACTION_RESULT.value: _visit_result("Other page.")}))
    plain.status = IdeaNodeStatus.DONE
    parent.details[alt.RACE_GROUPS] = {"span": [a.node_id, b.node_id]}

    merged = _policy().merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {b.node_id, plain.node_id}


def test_a_tier2_group_is_never_consumable_even_with_both_flags_on():
    """Tier 2 was measured at 50% live precision, and consuming a wrong group DISCARDS correct
    findings — so the flag that opens merge consumption to inferred groups opens it to tier 1
    only. Tier 2 keeps being written for instrumentation and stays unreachable here."""
    graph, parent, (a, b) = _graph_with_children(
        ("Read the main span figure on English Wikipedia", _visit(
            "https://en.example/x", **{
                DetailKey.GOAL.value: _PARENT_TITLE,
                DetailKey.ACTION_RESULT.value: _visit_result(_SUBJECT_ONLY_PAGE)})),
        ("Read the main span figure on Norwegian Wikipedia", _visit(
            "https://no.example/y", **{
                DetailKey.GOAL.value: _PARENT_TITLE,
                DetailKey.ACTION_RESULT.value: _visit_result(_DATUM_PAGE)})),
    )
    a.status = b.status = IdeaNodeStatus.DONE
    tiers = alt.infer_race_groups(parent, [a, b])
    assert list(tiers.values()) == [2], "fixture precondition: a tier 2 group"
    label = next(iter(tiers))

    policy = _policy(include_inferred=True)
    assert policy._race_registry(parent) == {}
    assert policy.select_winner(graph, parent.node_id, label) is None
    assert policy._race_excluded_ids(graph, parent.node_id) == set()
    merged = policy.merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}
    # Still recorded: the demotion is about consumption, not about losing the observation.
    assert parent.details[alt.RACE_GROUPS_INFERRED_TIERS] == {label: 2}


def test_an_authored_label_wins_a_collision_with_an_inferred_one():
    graph, parent, a, b, label = _inferred_race_graph()
    parent.details[alt.RACE_GROUPS] = {label: [b.node_id, a.node_id]}
    policy = _policy(include_inferred=True)
    # Authored membership order decides the tie-break, proving the authored entry was read.
    assert policy._race_registry(parent)[label] == [b.node_id, a.node_id]
