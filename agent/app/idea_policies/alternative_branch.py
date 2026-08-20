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


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()
