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

from typing import TYPE_CHECKING, Any, Dict, List, Sequence

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

#: Dropped before every similarity comparison. Function words are boilerplate shared by every
#: sibling of every plan, so leaving them in inflates the score of unrelated candidates.
_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "this", "to", "via", "with",
})


def infer_race_groups(parent: "IdeaNode", nodes: Sequence["IdeaNode"]) -> Dict[str, int]:
    """Register race groups read off PLAN SHAPE, with no authored tag of any kind.

    :func:`link_race_groups` can only see a race the model explicitly labelled, and the live
    emission probe (``scripts/alt_branch_emission_probe.py``, 2026-08-21) found that tag never
    emitted at the 0.5b/7b tiers this repo targets — prompt text alone had hit its ceiling. So
    the same relationship is recovered structurally instead, from the module docstring's own
    definition of a race: **the same target reached by different routes**. Both halves are
    required, because either half alone is a well-known false positive:

    * different routes alone — :func:`idea_sequencing.disjoint_approach_reason` — is exactly
      what a six-way breadth fan-out over six unrelated entities also looks like;
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
    * **tier 2**: title-token overlap AFTER the parent's own title vocabulary is stripped from
      both sides, for plans with no ``expect`` at all. Stripping is what stops every sibling of
      a narrow parent from looking alike, and it makes this tier strict enough that it is
      expected to fire rarely — hence :data:`RACE_GROUPS_INFERRED_TIERS`, which keeps tier 2's
      hit rate (and false-positive rate) measurable separately from tier 1's before either is
      allowed to influence a merge.

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
    from agent.app.idea_sequencing import disjoint_approach_reason

    parent_vocab = _tokens(getattr(parent, "title", ""))

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
    for tier, group_nodes, signature in (
        (1, with_expect, lambda n: _tokens(n.details.get(DetailKey.EXPECT.value))),
        (2, without_expect, lambda n: _tokens(n.title) - parent_vocab),
    ):
        for members in _cluster_by_similarity(group_nodes, signature):
            if not disjoint_approach_reason(list(members)):
                continue
            label = _inferred_label(members, signature, taken=set(registry))
            registry[label] = [m.node_id for m in members]
            tiers[label] = tier
            for member in members:
                member.details[RACE_GROUP_INFERRED] = label

    if registry:
        _merge_into(parent, RACE_GROUPS_INFERRED, registry)
        _merge_into(parent, RACE_GROUPS_INFERRED_TIERS, tiers)
    return tiers


def _cluster_by_similarity(nodes: Sequence[Any], signature) -> List[List[Any]]:
    """Greedy seed-and-absorb clustering of ``nodes`` into same-target groups of 2+.

    Each unassigned node in declared order seeds a group and absorbs every later unassigned
    node similar enough to THAT SEED (not to the growing group), so membership never depends
    on absorption order and a chain of pairwise-similar-but-collectively-unrelated candidates
    cannot drift into one group.
    """
    signatures = [(node, signature(node)) for node in nodes]
    assigned: set = set()
    clusters: List[List[Any]] = []
    for index, (seed, seed_tokens) in enumerate(signatures):
        if index in assigned or not seed_tokens:
            continue
        members = [seed]
        taken = [index]
        for other_index in range(index + 1, len(signatures)):
            if other_index in assigned:
                continue
            other, other_tokens = signatures[other_index]
            if _jaccard(seed_tokens, other_tokens) >= RACE_INFERENCE_SIMILARITY:
                members.append(other)
                taken.append(other_index)
        if len(members) < 2:
            continue
        assigned.update(taken)
        clusters.append(members)
    return clusters


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
