"""Resolve authored branching hints into real sibling relationships.

Two structural shapes are authored as optional per-candidate strings by the expansion
schema variant ``EXPANSION_JSON_SCHEMA_WITH_BRANCHING`` (opt-in,
``expansion_alternative_branch_enabled``):

* ``alternative_of: "<other candidate's title>"`` — a SEQUENTIAL fallback. Try A; only if A
  does not work out, try B.
* ``race_group: "<short label>"`` — a CONCURRENT race. Two or more different routes to the
  SAME fact, worth running at once and resolving to one winner at the merge point.

Both hints name *titles*, which only become node ids once ``graph.expand()`` has minted the
children — hence this module, run as a post-expand pass exactly like
:func:`plan_library.link_dependencies`, whose two-phase shape it copies.

Why a ``details`` marker rather than a new status: the fallback node is parked in
``IdeaNodeStatus.BLOCKED`` because adding a status enum member ripples across every
status-comparison site in the codebase. BLOCKED on its own gives no dispatch protection
whatsoever — the intermediate-node eligibility loop, :func:`idea_node_state.is_action_ready`
and the single-select score loop all wave BLOCKED nodes straight through, and
``_has_required_data`` only gates on ``requires_data``, which this design deliberately does
not use. So the parked node also carries :data:`ALTERNATIVE_PENDING`, and every
ready/eligible/parallel-candidate filter consults :func:`is_alternative_pending`. Without
that the "sequential" fallback would simply race its primary the first time both cleared
``siblings_are_independent``, which is the whole mechanism defeated.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaNode


#: Authored, pre-resolution hint: the TITLE of the sibling this candidate falls back for.
#: Consumed and replaced by :data:`DetailKey.ALTERNATIVE_OF_NODE` (a real node id) here.
ALTERNATIVE_OF_HINT = "alternative_of"

#: Dispatch guard on a parked fallback node. Present => the node is excluded from every
#: ready/eligible/parallel-candidate list until ``_maybe_promote_alternative_branch``
#: clears it. Cleared exactly once, on promotion.
ALTERNATIVE_PENDING = "alternative_pending"

#: Post-hoc attribution for a promoted fallback: which trigger fired (primary FAILED vs
#: primary DONE-but-no-datum-verified). Pure instrumentation.
ALTERNATIVE_TRIGGER_REASON = "alternative_trigger_reason"

#: Written on a fallback that was retired unused, because its primary reached a terminal
#: status without tripping either trigger. Retiring matters: a fallback left BLOCKED forever
#: would stall its parent's ``are_children_ready_to_merge`` check.
ALTERNATIVE_RETIRED = "alternative_retired"

#: Ancestor-side race registry: ``{label: [member_node_id, ...]}`` written onto the shared
#: parent. ``SimpleMergePolicy.select_winner`` reads group membership from HERE rather than
#: from graph topology, which is what lets the merge node keep its ordinary single-parent
#: shape (see that method's docstring for the compatibility bug this avoids).
RACE_GROUPS = "race_groups"

#: Same shape and same ancestor as :data:`RACE_GROUPS` (``{label: [member_node_id, ...]}``),
#: but produced by :func:`infer_race_groups` from PLAN SHAPE rather than from a model-emitted
#: tag. Deliberately a second key rather than an update of the first: provenance survives into
#: every report capture, and merge-time consumption of inferred groups is separately gated
#: (``merge_race_winner_selection_includes_inferred_groups_enabled``).
RACE_GROUPS_INFERRED = "race_groups_inferred"

#: Ancestor-side ``{label: 1|2}`` companion to :data:`RACE_GROUPS_INFERRED` recording WHICH
#: signal registered each group — tier 1 (near-duplicate ``expect`` contracts) or the weaker
#: tier 2 (title overlap). Kept parallel rather than folded into the registry so the registry
#: keeps one shape for both producers, and so tier 2's false-positive rate can be measured on
#: its own before it is ever trusted at merge time.
RACE_GROUPS_INFERRED_TIERS = "race_groups_inferred_tiers"

#: Member-side marker: the inferred label this node was grouped under. Instrumentation only —
#: notably NOT :data:`DetailKey.RACE_GROUP`, which would additionally hand the group a
#: dispatch-independence pass in ``siblings_are_independent`` on heuristic evidence alone.
RACE_GROUP_INFERRED = "race_group_inferred"

#: Step index at which a race member completed. The only ordering signal available at merge
#: time — the graph records no per-node completion timestamp — so the winner chain's
#: "earliest completion" tie-break is (step, declared order).
RACE_COMPLETED_STEP = "race_completed_step"

#: Written on a race member that lost the winner selection. Excluded from the merge's
#: synthesis input; the node itself is left to finish naturally and marked SKIPPED.
RACE_LOSER = "race_loser"


def is_alternative_pending(node: Any) -> bool:
    """Is ``node`` a parked fallback that must not be dispatched yet?

    The single shared predicate behind every eligibility filter that has to know about
    parked fallbacks, so the four call sites cannot drift apart. Tolerates ``None`` and
    detail-less objects so callers need no defensive branch of their own.
    """
    if node is None:
        return False
    details = getattr(node, "details", None)
    if not isinstance(details, dict):
        return False
    return bool(details.get(ALTERNATIVE_PENDING))


def link_alternatives(nodes: Sequence["IdeaNode"]) -> int:
    """Turn ``alternative_of`` title hints into real primary/fallback node references.

    Run on the children ``graph.expand()`` just created. For each tagged node whose named
    sibling exists, writes :data:`DetailKey.ALTERNATIVE_OF_NODE` on the fallback and the
    :data:`DetailKey.HAS_ALTERNATIVE_NODE` back-pointer on the primary, then parks the
    fallback (BLOCKED + :data:`ALTERNATIVE_PENDING`).

    Titles are matched case-insensitively on whitespace-normalized text: the model is
    re-typing its own title into a second field, so exact-byte matching loses real pairs to
    trailing punctuation. A hint naming a missing, ambiguous or self-referential sibling is
    dropped silently, leaving an ordinary independent candidate.

    A primary may own at most ONE fallback (the first declared): a chain of fallbacks would
    need a promotion cursor the trigger does not have, and a primary with two fallbacks has
    no defined promotion order.

    :returns: how many fallbacks were parked.
    """
    by_title: Dict[str, List["IdeaNode"]] = {}
    for node in nodes:
        key = _norm(getattr(node, "title", ""))
        if key:
            by_title.setdefault(key, []).append(node)

    linked = 0
    for node in nodes:
        details = getattr(node, "details", None)
        if not isinstance(details, dict):
            continue
        hint = details.get(ALTERNATIVE_OF_HINT)
        if not isinstance(hint, str) or not hint.strip():
            continue
        details.pop(ALTERNATIVE_OF_HINT, None)
        matches = by_title.get(_norm(hint)) or []
        if len(matches) != 1:
            continue
        primary = matches[0]
        if primary is node:
            continue
        if primary.details.get(DetailKey.HAS_ALTERNATIVE_NODE.value):
            continue
        primary.details[DetailKey.HAS_ALTERNATIVE_NODE.value] = node.node_id
        details[DetailKey.ALTERNATIVE_OF_NODE.value] = primary.node_id
        details[ALTERNATIVE_PENDING] = True
        node.status = IdeaNodeStatus.BLOCKED
        linked += 1
    return linked


def link_race_groups(parent: "IdeaNode", nodes: Sequence["IdeaNode"]) -> int:
    """Register each authored race group on its members AND on their shared parent.

    Members keep their normal PENDING status and dispatch as ordinary independent siblings —
    racing IS concurrent execution, so there is nothing to hold back.

    Takes the parent node explicitly (unlike :func:`link_alternatives`) because an
    ``IdeaNode`` carries only a parent *id*, and the ancestor-side registry
    (:data:`RACE_GROUPS`) is the whole point: ``select_winner`` must be able to recover
    group membership without inferring it from topology.

    A label claimed by only one candidate is not a race and is dropped, so a stray label
    cannot make a lone sibling look like a group of one at merge time.

    :returns: how many member nodes were registered.
    """
    groups: Dict[str, List["IdeaNode"]] = {}
    for node in nodes:
        details = getattr(node, "details", None)
        if not isinstance(details, dict):
            continue
        label = details.get(DetailKey.RACE_GROUP.value)
        if not isinstance(label, str) or not label.strip():
            continue
        groups.setdefault(_norm(label), []).append(node)

    registered = 0
    registry: Dict[str, List[str]] = {}
    for label, members in groups.items():
        if len(members) < 2:
            for lone in members:
                lone.details.pop(DetailKey.RACE_GROUP.value, None)
            continue
        for member in members:
            member.details[DetailKey.RACE_GROUP.value] = label
            registered += 1
        registry[label] = [m.node_id for m in members]

    if registry:
        existing = parent.details.get(RACE_GROUPS)
        if not isinstance(existing, dict):
            existing = {}
            parent.details[RACE_GROUPS] = existing
        existing.update(registry)
    return registered


#: Two token sets count as the SAME target above this Jaccard similarity. 0.7 is a
#: near-duplicate bar, not a topical-relatedness one: a pair of n-token sets differing by one
#: token each way scores ``(n-1)/(n+1)``, so it takes 7+ tokens to clear it by wording alone.
RACE_INFERENCE_SIMILARITY = 0.7

#: Two route SOURCES count as different routes below this Jaccard similarity. Deliberately not
#: "strictly disjoint": the canonical race of the whole mechanism is "English Wikipedia" against
#: "Norwegian Wikipedia", which shares the generic token ``wikipedia`` and would fail a
#: zero-overlap bar. 0.5 admits that pair (0.33) while still rejecting identical sources (1.0)
#: and one-token-different restatements of the same source.
RACE_ROUTE_SOURCE_OVERLAP = 0.5

#: Dropped before every similarity comparison. Function words are boilerplate shared by every
#: sibling of every plan, so leaving them in inflates the score of unrelated candidates.
_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "this", "to", "via", "with",
})

#: Leading imperative verbs stripped off a title before it is decomposed into target/source.
#: Every plan title starts with one and it says which ACTION is taken, never which FACT is
#: sought — leaving it in makes two routes to one fact look different whenever the author
#: reached them with different verbs ("Search for X on A" vs "Verify X against B").
_TITLE_VERBS = frozenset({
    "aggregate", "browse", "check", "collect", "compare", "compute", "confirm", "consult",
    "cross", "determine", "extract", "fetch", "find", "gather", "get", "identify", "inspect",
    "list", "locate", "look", "obtain", "open", "query", "read", "retrieve", "review",
    "scan", "search", "summarise", "summarize", "verify", "visit",
})

#: Words that introduce the SOURCE half of a title. Everything before the first one is the
#: target (the fact being sought), everything after it is the route taken to that fact. Kept
#: to source-introducing prepositions only: ``for``/``of``/``about`` introduce more target,
#: not a route, so splitting on them would shred the target instead.
_SOURCE_PREPOSITIONS = frozenset({"on", "from", "against", "in", "via"})

#: Two-word source introducer, matched before the single-word set.
_SOURCE_PREPOSITION_PAIRS = (("according", "to"),)


def infer_race_groups(parent: "IdeaNode", nodes: Sequence["IdeaNode"]) -> Dict[str, int]:
    """Register race groups read off PLAN SHAPE, with no authored tag of any kind.

    :func:`link_race_groups` can only see a race the model explicitly labelled, and the live
    emission probe (``scripts/alt_branch_emission_probe.py``, 2026-08-21) found that tag never
    emitted at the 0.5b/7b tiers this repo targets — prompt text alone had hit its ceiling. So
    the same relationship is recovered structurally instead, from the module docstring's own
    definition of a race: **the same target reached by different routes**. Both halves are
    required, because either half alone is a well-known false positive:

    * different routes alone — :func:`race_route_evidence` — is exactly what a six-way breadth
      fan-out over six unrelated entities also looks like;
    * the same target alone is what any two steps of one sequential chain look like.

    Same-target evidence comes in two tiers, never mixed inside one group:

    * **tier 1**: near-duplicate ``expect`` contracts (the opt-in
      ``expansion_expect_contract_enabled`` field). Two leaves promising to return the same
      measurable datum ARE two routes to one fact, stated by the author for an unrelated
      reason, which is the strongest signal available here for free. Known limitation: a
      breadth fan-out whose leaves all promise the same FIELD without naming their differing
      entity ("the prominence in metres and the source URL", six times) is indistinguishable
      from a race by this signal alone — which is exactly why merge-time consumption is gated
      separately and a breadth negative control belongs in the live probe.
    * **tier 2**: asymmetric target/route decomposition of the title
      (:func:`_route_of`), for plans with no ``expect`` at all. The candidate's title splits
      into the FACT it seeks and the SOURCE it seeks it in, and a race is same fact + different
      sources — so the two halves are compared against different bars instead of being mashed
      into one symmetric similarity score. Tier 2 stays out of merge consumption regardless
      (see ``SimpleMergePolicy._race_registry``) and keeps its own entry in
      :data:`RACE_GROUPS_INFERRED_TIERS` so its hit rate is measurable on its own.

      This replaced a symmetric whole-title Jaccard measured at 50% precision live: its two
      false positives were both a search paired with its OWN following visit, which is a chain
      step pair, and a threshold sweep found no bar that separated them — same-route chain
      pairs scored 0.65-0.73, genuine cross-route pairs 0.46-0.50, so the chain pairs ranked
      ABOVE the races the signal existed to find. Decomposing fixes that by construction: a
      search and its own visit share a source (that is what makes them one route) and differ in
      target (one seeks a fact, the other seeks "the top search result").

    Either tier's clustering groups by TARGET only, so a cluster can mix a real race with
    the chain steps feeding it and fail :func:`race_route_evidence` as a batch. Such a cluster
    is handed to :func:`_recover_subsets` rather than dropped, which registers the largest
    strict subsets that pass the gate on their own.

    Writes to :data:`RACE_GROUPS_INFERRED`, never to :data:`RACE_GROUPS`, and never to
    ``DetailKey.RACE_GROUP``: with the merge-side flag off (the default) the whole pass is
    instrumentation, and ``merge()`` behaves exactly as it does today. Singleton groups are
    dropped for the same reason :func:`link_race_groups` drops them. Already-tagged, parked
    and non-leaf candidates are skipped — an authored group is better evidence than this
    inference, and a parked A->B fallback is sequential by construction.

    Leafness is read as "carries a non-merge action" rather than off ``details.is_leaf``,
    because the engine stamps ``is_leaf`` in a LATER loop than this pass's call site.

    :returns: ``{label: tier}`` for the groups registered on ``parent`` (empty when none).
    """
    with_expect: List[Any] = []
    without_expect: List[Any] = []
    for node in nodes:
        details = getattr(node, "details", None)
        if not isinstance(details, dict) or not _is_leaf_candidate(details):
            continue
        if details.get(DetailKey.RACE_GROUP.value) or details.get(ALTERNATIVE_PENDING):
            continue
        if details.get(DetailKey.ALTERNATIVE_OF_NODE.value):
            continue
        expect = details.get(DetailKey.EXPECT.value)
        if isinstance(expect, str) and expect.strip():
            with_expect.append(node)
        else:
            without_expect.append(node)

    registry: Dict[str, List[str]] = {}
    tiers: Dict[str, int] = {}
    expect_of = lambda n: _tokens(n.details.get(DetailKey.EXPECT.value))  # noqa: E731
    route_of = lambda n: _route_of(getattr(n, "title", ""))  # noqa: E731
    target_of = lambda n: route_of(n).target  # noqa: E731. One-line signature accessor
    for tier, group_nodes, signature, key, compatible in (
        (1, with_expect, expect_of, expect_of, _same_target),
        (2, _routed_candidates(without_expect), target_of, route_of, _same_target_other_source),
    ):
        for cluster in _cluster(group_nodes, key, compatible):
            groups = (
                [list(cluster)] if race_route_evidence(list(cluster))
                else _recover_subsets(cluster, key, compatible)
            )
            for members in groups:
                label = _inferred_label(members, signature, taken=set(registry))
                registry[label] = [m.node_id for m in members]
                tiers[label] = tier
                for member in members:
                    member.details[RACE_GROUP_INFERRED] = label

    if registry:
        _merge_into(parent, RACE_GROUPS_INFERRED, registry)
        _merge_into(parent, RACE_GROUPS_INFERRED_TIERS, tiers)
    return tiers


def _cluster(nodes: Sequence[Any], key, compatible) -> List[List[Any]]:
    """Greedy seed-and-absorb clustering of ``nodes`` into same-target groups of 2+.

    Each unassigned node in declared order seeds a group and absorbs every later unassigned
    node ``compatible`` with THAT SEED (not with the growing group), so membership never
    depends on absorption order and a chain of pairwise-compatible-but-collectively-unrelated
    candidates cannot drift into one group.

    ``key`` maps a node to whatever ``compatible`` compares — an ``expect`` token set for
    tier 1, a :class:`_Route` for tier 2 — and a falsy key never seeds (an empty ``expect``
    is evidence of nothing).
    """
    keyed = [(node, key(node)) for node in nodes]
    assigned: set = set()
    clusters: List[List[Any]] = []
    for index, (seed, seed_key) in enumerate(keyed):
        if index in assigned or not seed_key:
            continue
        members = [seed]
        taken = [index]
        for other_index in range(index + 1, len(keyed)):
            if other_index in assigned:
                continue
            other, other_key = keyed[other_index]
            if compatible(seed_key, other_key):
                members.append(other)
                taken.append(other_index)
        if len(members) < 2:
            continue
        assigned.update(taken)
        clusters.append(members)
    return clusters


def _same_target(seed_key: set, other_key: set) -> bool:
    """Tier 1 compatibility: two ``expect`` contracts promise the same measurable datum."""
    return _jaccard(seed_key, other_key) >= RACE_INFERENCE_SIMILARITY


def _same_target_other_source(seed_key: "_Route", other_key: "_Route") -> bool:
    """Tier 2 compatibility: same fact sought, named source demonstrably different.

    The race definition's two halves applied to the two halves of the title decomposition,
    compared against their own bars instead of mashed into one symmetric score. Requiring
    both is what excludes a search and its own following visit, which agree on source and
    disagree on target — the exact false positive that retired the old symmetric signal.
    """
    if _jaccard(seed_key.target, other_key.target) < RACE_INFERENCE_SIMILARITY:
        return False
    return _jaccard(seed_key.source, other_key.source) < RACE_ROUTE_SOURCE_OVERLAP


def _recover_subsets(cluster: Sequence[Any], key, compatible) -> List[List[Any]]:
    """Rescue the genuine races hiding inside a cluster that failed the route gate whole.

    Clustering groups candidates by TARGET alone, so one cluster routinely mixes a real race
    with the chain steps that feed it: the 2026-08-21 14b probe emitted four leaves sharing
    one ``expect`` — two searches of different sources plus each search's own follow-up visit
    — and :func:`race_route_evidence` rightly rejects that foursome, which used to discard the
    two-search race sitting inside it as well.

    So a rejected cluster is re-examined subset-wise rather than dropped: the LARGEST strict
    subset size at which any subset is both internally ``compatible`` (re-checked against the
    subset's own earliest-declared member, since the seed that vouched for the others may not
    be in the subset) and passes :func:`race_route_evidence` wins, and every disjoint valid
    subset of exactly that size is registered. Precedence is therefore size first, then
    declared order, and a member registered once is never offered to a later subset — two
    overlapping readings of the same candidates would double-count it at merge time.

    Only rejection triggers this, so the pass can never be less permissive than before, and
    only sizes strictly below the cluster's own can be registered, so it can never re-admit
    the batch that was just rejected. Clusters of 2 have no smaller valid subset and return
    immediately. Enumeration is brute force because observed clusters run to single-digit
    membership.
    """
    if len(cluster) < 3:
        return []
    keys = {id(node): key(node) for node in cluster}
    for size in range(len(cluster) - 1, 1, -1):
        accepted: List[List[Any]] = []
        claimed: set = set()
        for subset in combinations(cluster, size):
            if claimed.intersection(id(node) for node in subset):
                continue
            first = keys[id(subset[0])]
            if not all(compatible(first, keys[id(other)]) for other in subset[1:]):
                continue
            if not race_route_evidence(list(subset)):
                continue
            accepted.append(list(subset))
            claimed.update(id(node) for node in subset)
        if accepted:
            return accepted
    return []


class _Route(NamedTuple):
    """A title decomposed into the fact it seeks and the source it seeks it in."""

    target: set
    source: set


def _route_of(title: Any) -> _Route:
    """Split ``title`` into ``(target, source)`` token sets around its source preposition.

    "Search for Hardanger Bridge main span length on English Wikipedia" becomes target
    ``{hardanger, bridge, main, span, length}`` and source ``{english, wikipedia}``.

    Two deterministic steps, no classifier and no extra spend:

    1. drop the leading imperative verb (:data:`_TITLE_VERBS`) — it names the ACTION, and two
       routes to one fact routinely use different verbs;
    2. split at the FIRST source-introducing preposition (:data:`_SOURCE_PREPOSITIONS`, plus
       the two-word :data:`_SOURCE_PREPOSITION_PAIRS`). First, not last: later prepositions
       belong to the source phrase itself ("in the infobox on English Wikipedia").

    A title with no such preposition gets an empty ``source``, which
    :func:`_routed_candidates` reads as "no route evidence" and drops. That is the intended
    strictness: a candidate that never says where it is looking cannot be shown to be looking
    somewhere ELSE than its sibling.
    """
    words = "".join(char if char.isalnum() else " " for char in str(title or "").lower()).split()
    if words and words[0] in _TITLE_VERBS:
        words = words[1:]

    split_at = len(words)
    source_start = len(words)
    for index, word in enumerate(words):
        pair = next(
            (p for p in _SOURCE_PREPOSITION_PAIRS
             if p[0] == word and words[index + 1:index + 2] == [p[1]]),
            None,
        )
        if pair is not None:
            split_at, source_start = index, index + len(pair)
            break
        if word in _SOURCE_PREPOSITIONS:
            split_at, source_start = index, index + 1
            break

    return _Route(
        target=_tokens(" ".join(words[:split_at])),
        source=_tokens(" ".join(words[source_start:])),
    )


def _routed_candidates(nodes: Sequence[Any]) -> List[Any]:
    """``nodes`` whose title names both a target and a source, in declared order."""
    routed = []
    for node in nodes:
        route = _route_of(getattr(node, "title", ""))
        if route.target and route.source:
            routed.append(node)
    return routed


def race_route_evidence(nodes: Sequence[Any]) -> Optional[str]:
    """Do these candidates take different routes *in the sense a RACE needs*? Which evidence?

    A race-specific fork of :func:`idea_sequencing.disjoint_approach_reason`, which that
    function must not become: its job is dispatch-parallelism safety, other call sites depend
    on its current verdicts, and a wrong dispatch decision costs a serialized step while a
    wrong race group costs a discarded correct finding once merge consumption is on. So the
    stricter rules live here.

    Two additions on top of that function's verdict:

    * ``mixed_search_visit`` is REJECTED. "A search and its own resulting visit" is the
      definition of a chain step pair, and it was the actual driver of both false positives the
      2026-08-21 probe audit found. Sound evidence for dispatch (both can be issued at once),
      no evidence at all for a race.
    * every VISIT member must carry a concrete URL individually, not merely as part of a batch
      that happens to pass ``concrete_urls``. The task-122 tier-1 near-miss was only saved
      because its visit nodes had empty URLs at capture time; filling them in would have
      registered a false race, so the invariant is stated rather than left to luck.

    :returns: the surviving reason string, or ``None`` when this batch is no race.
    """
    from agent.app.idea_policies.action_constants import NodeDetailsExtractor
    from agent.app.idea_policies.base import IdeaActionType
    from agent.app.idea_sequencing import concrete_url, disjoint_approach_reason

    reason = disjoint_approach_reason(list(nodes))
    if not reason or reason == "mixed_search_visit":
        return None
    for node in nodes:
        details = getattr(node, "details", None)
        if not isinstance(details, dict):
            return None
        if NodeDetailsExtractor.get_action(details) != IdeaActionType.VISIT.value:
            continue
        if not concrete_url(details):
            return None
    return reason


#: :func:`race_value_agreement` verdicts. Four states, not a bool: "no member returned a
#: comparable value" is emphatically NOT agreement, and treating it as one would turn the
#: silence of a failed race into positive confirmation, which is the exact hallucination
#: shape this check exists to catch.
RACE_VALUE_AGREE = "agree"
RACE_VALUE_DISAGREE = "disagree"
RACE_VALUE_SINGLE = "single"
RACE_VALUE_UNKNOWN = "unknown"

#: Ancestor-side ``{label: verdict}`` record of the value-agreement check, written by
#: ``SimpleMergePolicy`` for EVERY known race group whatever the merge flags say. Detection is
#: unconditional so the failure mode is measurable in report captures before it is acted on.
RACE_VALUE_AGREEMENT = "race_value_agreement"

#: Number tokens for VALUE extraction (as opposed to ``contract_satisfaction._NUMBER_RE``'s
#: presence test): also accepts space/NBSP-separated thousands groups, because the same figure
#: is written "1,310", "1310" and "1 310" across the very sources a race compares.
_RACE_NUMBER_RE = re.compile("\\d{1,3}(?:[ \u00a0\u202f,]\\d{3})+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?")

#: Unit word (lowercased, punctuation-stripped) -> canonical unit. Only the unit token sitting
#: immediately after the number is read. Two members whose units are BOTH known and different
#: are not comparable at all (1,310 m against 4,298 ft is one agreeing fact written twice), so
#: that pair reports ``unknown`` rather than a false ``disagree``; unit-free numbers stay
#: comparable with anything, since "1310" beside "1,310 m" is the ordinary case.
_RACE_UNITS: Dict[str, str] = {
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "km": "km", "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "cm": "cm", "centimetre": "cm", "centimetres": "cm", "centimeter": "cm",
    "mm": "mm", "ft": "ft", "foot": "ft", "feet": "ft",
    "mi": "mi", "mile": "mi", "miles": "mi",
    # No bare "in"/"t": both are far commoner as an English preposition and a stray letter
    # than as inches/tonnes ("1310 in 2013"), and a misread unit is exactly what turns a
    # comparable pair into a spurious unit mismatch.
    "inch": "in", "inches": "in",
    "yd": "yd", "yard": "yd", "yards": "yd",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "%": "%", "percent": "%",
}

#: How far past a number token the unit word is looked for.
_RACE_UNIT_WINDOW = 16


class _RaceValue(NamedTuple):
    """One member's answer to the asked-for quantity: normalized value plus its unit."""

    value: float
    unit: str

    def render(self) -> str:
        text = f"{self.value:.6f}".rstrip("0").rstrip(".")
        return f"{text} {self.unit}".strip()


class RaceValueEvidence(NamedTuple):
    """Full verdict of :func:`race_value_agreement` — the datum compared and what was found."""

    verdict: str
    datum: str = ""
    values: Dict[str, str] = {}


def race_value_agreement(nodes: Sequence[Any], mandate: str = "") -> str:
    """Did these race members' routes return the SAME VALUE for the asked-for quantity?

    The missing half of :func:`race_route_evidence`, which establishes only that the members
    took DIFFERENT ROUTES and never looks at what those routes came back with. Today "all
    three routes confirm X" is pure narration by the merge/finalize model — a real 2026-08-21
    completion asserted exactly that about a number appearing in no fetched page anywhere in
    the run. Nothing computed it, so nothing could contradict it. This does.

    Deterministic, no LLM call, no network. Four verdicts (see :data:`RACE_VALUE_AGREE` and
    friends): ``agree``, ``disagree``, ``single`` (one member carried a comparable value, so
    there is nothing to cross-check) and ``unknown`` (none did — insufficient data, which is
    neither confirmation nor contradiction).

    **Scoping is the load-bearing part**, along two axes. Only the datums the mandate (else the
    members' own contracts) actually asks for are compared, via
    ``contract_satisfaction._required_datums`` and its ``_find_number_near`` cue-proximity
    search; comparing every number found anywhere would fire on incidental drift — the
    strong-agent trace's own example is a ranked-list position reading 20 on one page and 21 on
    another while both agree on the keystone figure — and the careful researcher treats that as
    noise precisely because it is not the thing being asked. And the reading is scoped to the
    TARGET ENTITY's own mention where the page names it (:func:`_member_value`), because "near
    the word span" is a document-wide test that a 60k-character ranked list answers with some
    other bridge's row. Both exclusions are structural, not special cases bolted on.

    Cue classes with no stable page wording (year/count/price, an empty ``variants`` tuple)
    are deliberately NOT compared: their extractor falls back to "the first number anywhere in
    the evidence", which across two different pages compares two arbitrary numbers and would
    manufacture disagreement. They fall through to ``unknown``.
    """
    return race_value_evidence(nodes, mandate).verdict


def race_value_evidence(nodes: Sequence[Any], mandate: str = "") -> RaceValueEvidence:
    """:func:`race_value_agreement` plus the datum compared and each member's value.

    Callers that report or log the verdict want to say WHICH quantity disagreed and with what
    readings; the plain-string function above is the predicate view of this same computation.

    EVERY scoped datum is compared and the verdicts are combined by precedence — ``agree``
    first, then ``disagree``, then ``single``, then ``unknown`` — rather than reporting whatever
    the first datum with any value at all happened to say. Both halves are live-driven fixes
    (2026-08-21 ollama probe, which measured this check killing 7 of 8 CORRECT answers):

    * a mandate routinely names more than one measurable quantity, and the extra one is often
      the DECOY. Task 150 asks for a bridge's main span and warns in the same breath that its
      total length is the wrong answer sitting one infobox line above; ``_required_datums``
      yields ``length`` before ``span``, so the check locked onto the trap and reported the two
      routes disagreeing (1,380 against the list page's rank number) while both stated 1,310 m
      for the span they were actually racing for;
    * precedence is asymmetric on purpose. Agreement on ANY asked-for quantity is
      cross-confirmation — that is the whole signal this check produces — while a contradiction
      matters only when NO asked-for quantity agrees. The two error directions are not
      symmetric either: a missed ``disagree`` leaves the merge exactly as it was before this
      mechanism existed, a false ``disagree`` destroys a correct, genuinely confirmed answer.
    """
    members = [node for node in nodes if node is not None]
    anchors = _entity_anchors(mandate)
    best: Optional[RaceValueEvidence] = None
    for index, (canonical, variants) in enumerate(_scoped_datums(members, mandate)):
        values: Dict[str, _RaceValue] = {}
        for member in members:
            evidence = _member_evidence(member)
            if not evidence:
                continue
            evidence_low = evidence.lower()
            parsed = _member_value(evidence_low, variants, anchors, entity_fallback=index == 0)
            if parsed is not None:
                values[str(getattr(member, "node_id", len(values)))] = parsed
        if not values:
            continue
        found = RaceValueEvidence(
            _verdict_for(values), canonical,
            {node_id: value.render() for node_id, value in values.items()},
        )
        if found.verdict == RACE_VALUE_AGREE:
            return found
        if best is None or _VERDICT_PRECEDENCE.index(found.verdict) < _VERDICT_PRECEDENCE.index(
            best.verdict
        ):
            best = found
    return best or RaceValueEvidence(RACE_VALUE_UNKNOWN, "", {})


#: Which verdict a multi-datum comparison reports, best first. See :func:`race_value_evidence`.
_VERDICT_PRECEDENCE = (RACE_VALUE_AGREE, RACE_VALUE_DISAGREE, RACE_VALUE_SINGLE,
                       RACE_VALUE_UNKNOWN)


def _verdict_for(values: Dict[str, _RaceValue]) -> str:
    """One datum's verdict from the members that returned a value for it."""
    if len(values) < 2:
        return RACE_VALUE_SINGLE
    if len({value.unit for value in values.values() if value.unit}) > 1:
        return RACE_VALUE_UNKNOWN
    distinct = {round(value.value, 6) for value in values.values()}
    return RACE_VALUE_AGREE if len(distinct) == 1 else RACE_VALUE_DISAGREE


def _member_value(evidence_low: str, variants: Sequence[str], anchors: Sequence[str],
                  entity_fallback: bool = False) -> Optional["_RaceValue"]:
    """This member's reading of one datum, scoped to the target entity where possible.

    Cue proximity first (``length ... 1,380 metres``), narrowed to cue occurrences beside a
    mention of the entity. When the entity IS named but no cue occurrence sits beside it,
    ``entity_fallback`` reads the nearest measurement to the entity's own mention instead: a
    ranked table prints ``Hardanger Bridge | 1,310 m | 2013`` with the quantity named once in a
    header row thousands of characters away, so demanding a cue beside the row would discard
    the very route the mandate sent the run down. That fallback accepts only numbers carrying a
    RECOGNIZED UNIT, which is what keeps it from reading the row's rank, year or footnote
    marker as the answer.

    The fallback is offered for the mandate's PRIMARY datum only (see :func:`_scoped_datums`),
    because an unlabelled measurement beside an entity answers the question that was asked and
    nothing else. Offering it to a secondary datum turns the same figure into an answer for
    every quantity at once, which manufactures agreement on the decoy datum and hides a real
    disagreement on the asked-for one.

    Never falls back to the unanchored search once an anchor is present: reading a number from
    somewhere else on the page is the failure this scoping exists to remove.
    """
    from agent.app.idea_policies.contract_satisfaction import _anchor_spans, _find_number_near

    present = _present_anchors(evidence_low, anchors)
    match = _find_number_near(evidence_low, variants, _RACE_NUMBER_RE, anchors=present)
    parsed = _parse_race_value(evidence_low, match)
    if parsed is not None or not entity_fallback:
        return parsed
    return _measurement_near_anchor(evidence_low, _anchor_spans(evidence_low, present))


def _measurement_near_anchor(evidence_low: str, spans: Sequence[Any]) -> Optional["_RaceValue"]:
    """The unit-carrying number nearest an entity mention, within the anchor window."""
    from agent.app.idea_policies.contract_satisfaction import _ANCHOR_WINDOW

    best: Optional[_RaceValue] = None
    best_distance: Optional[float] = None
    for start, end in spans:
        anchor_center = (start + end) / 2
        win_start = max(0, start - _ANCHOR_WINDOW)
        win_end = min(len(evidence_low), end + _ANCHOR_WINDOW)
        for num in _RACE_NUMBER_RE.finditer(evidence_low, win_start, win_end):
            value = _parse_race_value(evidence_low, num)
            if value is None or not value.unit:
                continue
            distance = abs((num.start() + num.end()) / 2 - anchor_center)
            if best_distance is None or distance < best_distance:
                best, best_distance = value, distance
    return best


#: An anchor token shorter than this identifies nothing ("one", "bu"), and one occurring more
#: often than the hit cap identifies nothing either — "bridge" appears 82 times on the ranked
#: list of bridge spans, so anchoring on it would scope the search to the whole page.
_RACE_ANCHOR_MIN_LEN = 5
_RACE_ANCHOR_MAX_HITS = 12


#: A capitalized word right after one of these (or at the very start) is capitalized by
#: SENTENCE POSITION, which says nothing about it being a name.
_SENTENCE_END = frozenset(".!?:;—-\n\r•")


def _entity_anchors(mandate: str) -> List[str]:
    """Proper-noun tokens naming what the race is ABOUT, longest (most distinctive) first.

    Reuses ``contract_satisfaction``'s subject vocabulary (capitalized, minus plan boilerplate
    and quantity words) rather than its ``_subject_tokens``, whose top-3-longest cap is tuned
    for a one-line sub-goal contract: on task 150's full mandate that cap keeps
    ``norwegian-language`` and drops ``hardanger``, the one token the target row on the ranked
    list actually contains.

    Two rejections a sub-goal contract never needs but a full mandate does, because a mandate
    is prose with its own conventions: ALL-CAPS words are this corpus's EMPHASIS marker
    ("Three INDEPENDENT routes", "ANY ONE OF THEM IS SUFFICIENT"), and a word capitalized only
    by sentence position is capitalized by grammar. Both are ordinary English that matches
    ordinary English anywhere on any page — anchoring on them scopes a search to a random
    paragraph of whatever page is being read, which is the opposite of scoping.
    """
    from agent.app.idea_policies.contract_satisfaction import (
        _CASED_TOKEN_RE, _NOISE_TOKENS, _QUANTITY_WORDS,
    )

    text = str(mandate or "")
    anchors: List[str] = []
    seen = set()
    for match in _CASED_TOKEN_RE.finditer(text):
        raw = match.group()
        if not raw[:1].isupper() or raw.isupper():
            continue
        before = text[:match.start()].rstrip()
        if not before or before[-1] in _SENTENCE_END:
            continue
        token = raw.lower().strip("'’-")
        if len(token) < _RACE_ANCHOR_MIN_LEN or token in seen:
            continue
        if token in _NOISE_TOKENS or token in _QUANTITY_WORDS:
            continue
        seen.add(token)
        anchors.append(token)
    anchors.sort(key=len, reverse=True)
    return anchors


def _present_anchors(evidence_low: str, anchors: Sequence[str]) -> List[str]:
    """The anchors this evidence names distinctively enough to scope a search with."""
    return [a for a in anchors if 0 < evidence_low.count(a) <= _RACE_ANCHOR_MAX_HITS]


#: Non-English wordings of a datum, appended to that datum's own cue variants for race value
#: extraction only. Deliberately NOT a multilingual table: the one cross-language route this
#: repo's corpus actually contains is task 150's Norwegian Wikipedia article, whose infobox
#: labels the raced-for quantity ``hovedspenn``/``største spenn``. Without them a genuinely
#: cross-confirming route contributes nothing and the verdict degrades to ``single`` — quiet
#: for the wrong reason. Extend from attested corpus wordings, not speculatively.
_RACE_CUE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "span": ("hovedspenn", "største spenn"),
}


def _scoped_datums(nodes: Sequence[Any], mandate: str) -> List[Any]:
    """The measurable datums worth comparing: the mandate's ask, else the members' contracts.

    The mandate is preferred because it is the ONE thing every member of a race is a route to;
    a member's own ``expect``/goal is the fallback for a sub-goal whose parent mandate phrases
    the quantity elsewhere. Either way the empty-``variants`` cue classes are dropped (see
    :func:`race_value_agreement`), so an unscopeable ask yields no datum and the verdict is
    ``unknown`` rather than a comparison of arbitrary numbers.

    The mandate's datums come back ordered by the LONGEST cue phrase that matched, most
    specific first, so "main span" (9 characters, the ask) is compared before "longest"
    (7, the decoy the same mandate warns about). Ordering only decides which datum a
    multi-datum verdict is REPORTED against — :func:`race_value_evidence` compares all of
    them — but reporting the trap datum as the compared quantity is exactly how the live
    probe's false disagreements read.
    """
    from agent.app.idea_policies.contract_satisfaction import (
        _QUANTITY_CUES,
        _required_datums,
        derive_step_contract,
    )

    text = (mandate or "").lower()
    scoped = [(c, v + _RACE_CUE_ALIASES.get(c, ())) for c, v in _required_datums(text) if v]
    if scoped:
        specificity = {}
        for cue, canonical, _variants in _QUANTITY_CUES:
            if cue in text:
                specificity[canonical] = max(specificity.get(canonical, 0), len(cue))
        scoped.sort(key=lambda datum: -specificity.get(datum[0], 0))
        return scoped
    seen = set()
    for node in nodes:
        try:
            contract = derive_step_contract(node)
        except Exception:  # noqa: BLE001. A merge-time check must never crash a run
            continue
        for canonical, variants in contract.datums:
            if variants and canonical not in seen:
                seen.add(canonical)
                scoped.append((canonical, variants + _RACE_CUE_ALIASES.get(canonical, ())))
    return scoped


def _member_evidence(node: Any) -> str:
    """What this member's route actually RETURNED, as text to read the datum out of.

    A search member's snippets count: a race member is whatever route was run, and refusing to
    read one member's payload would silently turn a disagreement into ``single``. Model-authored
    text (a think/merge member's own content) counts too, and deliberately so — an unsourced
    claim contradicting two fetched pages is the very case this check exists to surface.
    """
    from agent.app.idea_policies.base import IdeaActionType
    from agent.app.idea_policies.contract_satisfaction import _search_evidence, _visit_body

    details = getattr(node, "details", None)
    if not isinstance(details, dict):
        return ""
    result = details.get(DetailKey.ACTION_RESULT.value)
    if not isinstance(result, dict):
        return ""
    if details.get(DetailKey.ACTION.value) == IdeaActionType.SEARCH.value:
        return _without_urls(_search_evidence(result))
    body = _visit_body(result)
    if body:
        return _without_urls(body)
    for key in ("summary", "answer", "text"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return _without_urls(val)
    return ""


#: URLs are blanked out of the evidence before any number is read from it. A URL is dense with
#: digits that are not measurements (ids, years, ``/2023/``, a bare ``e/0``) and it sits right
#: beside the snippet text carrying the cue, so it routinely wins the nearest-number search
#: against the figure the snippet actually states.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _without_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def _parse_race_value(evidence: str, match: Any) -> Optional[_RaceValue]:
    """Normalize a matched number plus the unit word that follows it."""
    if match is None:
        return None
    raw = match.group()
    for separator in (",", " ", " ", " "):
        raw = raw.replace(separator, "")
    try:
        value = float(raw)
    except ValueError:
        return None
    tail = evidence[match.end():match.end() + _RACE_UNIT_WINDOW].lower()
    unit = ""
    for token in "".join(c if (c.isalnum() or c == "%") else " " for c in tail).split()[:1]:
        unit = _RACE_UNITS.get(token, "")
    return _RaceValue(value=value, unit=unit)


def _inferred_label(members: Sequence[Any], signature, taken: set) -> str:
    """A short, human-readable label for a group, from the vocabulary its members share."""
    shared: set = set()
    for index, member in enumerate(members):
        tokens = signature(member)
        shared = tokens if index == 0 else (shared & tokens)
    base = "inferred:" + ("-".join(sorted(shared)[:3]) or str(len(taken) + 1))
    label = base
    suffix = 2
    while label in taken:
        label = f"{base}-{suffix}"
        suffix += 1
    return label


def _merge_into(parent: "IdeaNode", key: str, values: Dict[str, Any]) -> None:
    existing = parent.details.get(key)
    if not isinstance(existing, dict):
        existing = {}
        parent.details[key] = existing
    existing.update(values)


def _is_leaf_candidate(details: Dict[str, Any]) -> bool:
    from agent.app.idea_policies.action_constants import NodeDetailsExtractor

    if NodeDetailsExtractor.is_merge_action(details):
        return False
    return bool(NodeDetailsExtractor.get_action(details)) or bool(
        details.get(DetailKey.IS_LEAF.value)
    )


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def _tokens(text: Any) -> set:
    """Comparable content words of ``text``: lowercased, punctuation-split, stopwords dropped."""
    raw = "".join(char if char.isalnum() else " " for char in str(text or "").lower())
    return {token for token in raw.split() if token and token not in _STOPWORDS}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()
