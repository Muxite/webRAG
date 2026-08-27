"""Run-scoped novelty / churn guard (opt-in).

Phase 0's ``graph_no_reexpand`` ablation (docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md section 3)
falsifies "churn is the main failure". The engine has no notion of novelty today: a node whose
action produced nothing can be re-authored under a freshly worded sub-goal and re-executed
indefinitely, which is exactly what task 123 showed (43 visits, the same sub-goals re-issued 5-8
times, 1/4 sub-entities resolved). ``IdeaDag.has_executed_action`` does not catch this -- it only
records an action once it SUCCEEDED, so the repeated-failure loop is invisible to it.

This module supplies the two pieces that were missing:

* :func:`novelty_key` -- a stable, normalized identity for a proposed action, over
  ``(action_type, canonical_target, unresolved_requirement_ids)``. The target is the ARGUMENT
  (URL / query), because churn is argument-level: the title changes, the call does not.
* :class:`NoveltyGuard` -- attempt counting per key, reset by PROGRESS rather than by time. An
  attempt only counts against the budget if no new evidence appeared since the previous attempt
  of that same key; a key that keeps producing evidence is never blocked.
* :func:`evidence_watermark` -- what "progress" means, measured over the BRANCH the action sits
  on (:func:`branch_scope_id`) rather than run-wide. Run-wide, the healthy branches of a
  multi-branch mandate hold the number up for the stuck one and nothing is ever blocked.

The threshold (``max_attempts=2``, i.e. the THIRD identical no-progress attempt is blocked) is a
FIRST GUESS, not a measured value. It should be revisited against the mechanism suite's dead-end
retry-cap task (``agent/app/idea_tests/test_305_mech_dead_end_retry_cap.py``), which is the first
fixture able to say whether it is too tight (killing a legitimate second look at a flaky page) or
too loose (still paying for 5-8 repeats).

That revision is :func:`sub_goal_scope_id`. Armed against 305 the guard blocked NOTHING, twice
over: the per-TARGET key is too fine for the failure it is meant to catch. One dead-end sub-goal
is pursued through several textually distinct targets (305 ships two trap URLs on purpose, and
the model re-words the query each try), so every attempt opens a fresh budget and no key reaches
the threshold. Attempts are therefore counted a SECOND time over ``(sub_goal_scope_id,
action_type)`` -- the target dimension dropped, the action type kept so a stuck VISIT loop and a
stuck SEARCH loop stay separate budgets -- and either key striking vetoes the action. The
progress reset does the safety work: a scope whose actions keep yielding evidence never blocks,
so the coarse key only ever fires on a scope that is genuinely standing still.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

_WHITESPACE = re.compile(r"\s+")

#: "argument not given", so that ``scope_id=None`` can mean the explicit whole-graph scope.
_UNSET = object()


def _normalize(text: Any) -> str:
    """Lowercased, whitespace-collapsed text; ``""`` for anything unusable."""
    if text is None:
        return ""
    return _WHITESPACE.sub(" ", str(text)).strip().lower()


def canonical_target(action_type: str, details: Optional[Mapping[str, Any]]) -> str:
    """The concrete thing an action addresses: its URL, its query, else its goal/title.

    URLs drop a trailing slash and any fragment, which are the two variations that make the same
    page look like two targets. Deliberately NOT a full URL canonicalization (no query-parameter
    reordering, no host normalization): a guard that over-merges targets would block genuinely
    distinct work, and this key only ever *blocks*, so it fails open by staying conservative.
    """
    details = details if isinstance(details, Mapping) else {}
    from agent.app.idea_policies.action_constants import NodeDetailsExtractor
    from agent.app.idea_policies.base import DetailKey, IdeaActionType

    action = _normalize(action_type)
    if action == IdeaActionType.VISIT.value:
        url = _normalize(NodeDetailsExtractor.get_url(details))
        if url:
            url = url.split("#", 1)[0]
            if url.endswith("/"):
                url = url[:-1]
            return url
    if action == IdeaActionType.SEARCH.value:
        query = _normalize(
            details.get(DetailKey.QUERY.value) or details.get(DetailKey.PROMPT.value)
        )
        if query:
            return query
    return _normalize(
        details.get(DetailKey.GOAL.value)
        or details.get(DetailKey.ORIGINAL_GOAL.value)
        or details.get("title")
    )


def novelty_key(
    action_type: str,
    details: Optional[Mapping[str, Any]],
    unresolved_requirement_ids: Iterable[Any] = (),
) -> str:
    """Stable identity for a proposed action: ``"<action>|<target>|<unresolved ids>"``.

    The unresolved-requirement set is part of the key on purpose: re-issuing the same search
    after the run's open requirements changed is a DIFFERENT step (the surrounding state moved),
    while re-issuing it against an unchanged deficit is the churn this guards. Requirement ids
    come from the task ledger when it is running and are simply empty otherwise, which makes the
    key coarser (stricter), never wrong.
    """
    ids = sorted({_normalize(item) for item in (unresolved_requirement_ids or ()) if _normalize(item)})
    return f"{_normalize(action_type)}|{canonical_target(action_type, details)}|{','.join(ids)}"


def branch_scope_id(graph, node_id: Any) -> Optional[str]:
    """The id of the top-level branch ``node_id`` sits on (the root's own child on its path).

    This is the unit the watermark is measured over. Requirement ids are entity NAMES from the
    task ledger (``agent/app/task_ledger.py``) and nothing tags an ``Evidence``/``Claim`` record
    or an ``action_result`` with the requirement it served, so "evidence for the requirement this
    action addresses" has no direct encoding; the branch a node hangs off is the reachability
    proxy the graph does carry, and re-expansion mints a repeat under the SAME branch, which is
    exactly the churn being counted.

    ``None`` for the root itself, for an unknown node, or on any error -- callers read that as
    "no narrower scope than the whole graph".
    """
    try:
        path = graph.path_to_root(node_id)
    except Exception:  # noqa: BLE001. A guard must never break the run it observes.
        return None
    if not path:
        return None
    if len(path) < 2:
        return None      # the node IS the root
    return str(path[-2].node_id)


def sub_goal_scope_id(graph, node_id: Any) -> Optional[str]:
    """The id of the SUB-GOAL ``node_id`` serves: its nearest non-action ancestor.

    Coarser than :func:`branch_scope_id`, and for a different question. ``branch_scope_id``
    answers "whose evidence counts as progress for THIS target"; this answers "which open
    sub-goal is this attempt spending budget on", so that N textually-distinct targets pursued
    for one sub-goal are one budget rather than N fresh ones. Task 305 is the case: two
    distinct trap URLs plus several phrasings of one query each mint their own strict key, so no
    key ever reaches the strike threshold while the run burns its budget on one dead end.

    The requirement-owning ancestor has no explicit marker on the node (requirement ids are
    entity NAMES from the task ledger and nothing tags a node with the one it serves), so the
    structural proxy is "the nearest STRICT ancestor that is not itself an executable action" --
    a planner/sub-goal node. On the flat plans the engine actually produces (every action node a
    direct child of the root) that resolves to the root, which is the correct reading: those
    action nodes are the sub-goals, and their retries all belong to one budget. On a nested plan
    it resolves to the sub-goal node that owns the subtree, so two genuinely different sub-goals
    keep separate budgets.

    ``None`` for the root itself, for an unknown node, or on any error.
    """
    from agent.app.idea_policies.base import DetailKey

    try:
        path = graph.path_to_root(node_id)
    except Exception:  # noqa: BLE001. A guard must never break the run it observes.
        return None
    if not path or len(path) < 2:
        return None      # unknown node, or the node IS the root
    for ancestor in path[1:]:
        details = ancestor.details if isinstance(ancestor.details, dict) else {}
        if not details.get(DetailKey.ACTION.value):
            return str(ancestor.node_id)
    return str(path[-1].node_id)     # every ancestor is an action node; fall back to the root


def semantic_cluster_anchor(action_type: str, details: Optional[Mapping[str, Any]]) -> str:
    """A deterministic, cross-node-stable label for the semantic cluster ``details`` belongs to.

    Used ONLY as (half of) the flat-plan fallback coarse KEY -- not the watermark scope, which is
    :func:`sub_goal_cluster_ids`. The lexicographically smallest salient token
    (:func:`_entity_tokens`) is a pure function of this one node's own target, so it names the
    same cluster regardless of which node computes it first or how many later attempts join that
    cluster -- unlike a key built from cluster MEMBERSHIP (a node-id list), which would change,
    and therefore reset the attempt count, every time a new attempt joined. Falls back to the
    full canonical target when there is no salient token, which degrades to the strict per-node
    key rather than merging unrelated targets on a false signal.
    """
    tokens = _entity_tokens(canonical_target(action_type, details))
    return min(tokens) if tokens else canonical_target(action_type, details)


#: Short/common words dropped from :func:`_entity_tokens` -- not a linguistic stopword list, just
#: enough to keep the salient-token overlap from firing on shared connective words rather than a
#: shared subject.
_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whose", "with", "from", "into", "about",
    "find", "search", "visit", "page", "pages", "source", "sources", "data", "information",
    "does", "this", "that", "these", "those", "have", "been", "were", "will", "would",
    "could", "should", "there", "their", "please", "need", "look", "looking",
})


def _entity_tokens(text: Any) -> "frozenset[str]":
    """Normalized token set of ``text``, filtered to what plausibly names a subject.

    Used only to group DIFFERENT ``canonical_target`` strings that plausibly address the same
    underlying entity -- two phrasings of a search query, or two candidate URLs, for one sub-goal.
    Tokens shorter than 4 characters and the connective words in :data:`_STOPWORDS` are dropped,
    since those are the ones two unrelated targets are likeliest to share by accident.
    """
    normalized = _normalize(text)
    if not normalized:
        return frozenset()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return frozenset(token for token in tokens if len(token) >= 4 and token not in _STOPWORDS)


def sub_goal_cluster_ids(
    graph, node_id: Any, action_type: str, overlap_threshold: float = 0.3,
) -> "frozenset[str]":
    """Node ids of the flat-plan siblings that plausibly serve the same sub-goal as ``node_id``.

    Only meaningful -- and only ever called -- once :func:`sub_goal_scope_id` has already
    degraded to the root (a flat plan): the structural signal that distinguishes sub-goals from
    each other has nothing left to say there, so this substitutes a semantic one. An
    ``action_type``-matching action node elsewhere in the graph joins ``node_id``'s cluster when
    its canonical-target entity tokens overlap ``node_id``'s by at least ``overlap_threshold``
    (Jaccard). Always includes ``node_id`` itself.

    A node with no salient tokens (an empty or purely-connective canonical target) clusters with
    nothing else -- falls open to the strict, per-node case rather than merging unrelated targets
    on a false signal, matching this module's fail-open-by-staying-conservative design (see
    :func:`canonical_target`).
    """
    from agent.app.idea_policies.base import DetailKey

    node = graph.get_node(node_id)
    if node is None:
        return frozenset({str(node_id)})
    details = node.details if isinstance(node.details, dict) else {}
    action = _normalize(action_type)
    self_tokens = _entity_tokens(canonical_target(action_type, details))
    cluster = {str(node_id)}
    if not self_tokens:
        return frozenset(cluster)
    try:
        candidates = list(graph.iter_breadth_first(graph.root_id()))
    except Exception:  # noqa: BLE001. A guard must never break the run it observes.
        return frozenset(cluster)
    seen: set = set()
    for candidate in candidates:
        if candidate.node_id in seen:
            continue
        seen.add(candidate.node_id)
        if str(candidate.node_id) == str(node_id):
            continue
        cand_details = candidate.details if isinstance(candidate.details, dict) else {}
        if _normalize(cand_details.get(DetailKey.ACTION.value)) != action:
            continue
        cand_tokens = _entity_tokens(canonical_target(action_type, cand_details))
        if not cand_tokens:
            continue
        overlap = len(self_tokens & cand_tokens) / len(self_tokens | cand_tokens)
        if overlap >= overlap_threshold:
            cluster.add(str(candidate.node_id))
    return frozenset(cluster)


def evidence_watermark(
    graph, node_id: Any = None, scope_id: Any = _UNSET, scope_ids: Optional[Iterable[Any]] = None,
) -> int:
    """A monotone count of the evidence accumulated inside ``node_id``'s BRANCH so far.

    Counts the ``Evidence``/``Claim`` sidecars that ``IdeaEngine._maybe_record_evidence`` writes
    (``evidence_store.Evidence`` / ``.Claim``) -- the records the plan names as the progress
    signal. Those are written only in ``run_policy_evidence_store_mode == "observe"``, so this
    ALSO counts successful action results, which are what "new evidence appeared" means when that
    observer is off. Without the fallback the watermark would be a constant zero on the default
    configuration and every key would look like no-progress, which would turn a churn guard into
    a blanket retry cap.

    SCOPE is the whole point. Counted run-wide, a multi-branch mandate -- which is the shape this
    guard exists for -- keeps the number climbing off whichever branches are healthy, and the ONE
    stuck branch reads that unrelated progress as "new evidence appeared" and is never blocked.
    So the count is taken over the subtree of :func:`branch_scope_id`, and only degrades to the
    whole graph when there is no narrower scope (a root-level action, an unknown node).

    Never decreases within a branch (nothing removes a result or a sidecar), so a strictly
    greater value between two attempts of the same key really does mean that branch learned
    something.

    ``scope_id`` overrides the derived branch scope (``None`` meaning the whole graph). A budget
    counted over one scope must have its progress signal counted over the SAME scope, so the
    sub-goal-scoped key (:func:`sub_goal_scope_id`) passes its own scope in rather than reusing
    the narrower branch watermark, which would read a sibling's real progress as no-progress.

    ``scope_ids``, when given, overrides ``scope_id`` entirely: the count is taken over exactly
    this set of node ids rather than a single node's subtree (``graph.iter_breadth_first``
    assumes one). This is what lets a scope be a semantic CLUSTER
    (:func:`sub_goal_cluster_ids`) instead of a structural subtree -- the flat-plan case where
    the sub-goal-scoped key has no ancestor left to distinguish sub-goals with.
    """
    from agent.app.idea_policies.action_constants import ActionResultExtractor
    from agent.app.idea_policies.base import DetailKey

    total = 0
    if scope_ids is not None:
        try:
            nodes = [n for n in (graph.get_node(sid) for sid in scope_ids) if n is not None]
        except Exception:  # noqa: BLE001. A guard must never break the run it observes.
            return 0
    else:
        if scope_id is _UNSET:
            scope_id = branch_scope_id(graph, node_id) if node_id is not None else None
        try:
            nodes = list(graph.iter_breadth_first(scope_id))
        except Exception:  # noqa: BLE001. A guard must never break the run it observes.
            return 0
    seen: set = set()
    for node in nodes:
        # A node with several parents is yielded once per parent; count it once.
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        details = node.details if isinstance(node.details, dict) else {}
        if isinstance(details.get(DetailKey.EVIDENCE.value), dict):
            total += 1
        claims = details.get(DetailKey.CLAIMS.value)
        if isinstance(claims, list):
            total += len(claims)
        result = details.get(DetailKey.ACTION_RESULT.value)
        if isinstance(result, dict) and ActionResultExtractor.is_success(result):
            total += 1
    return total


@dataclass
class NoveltyGuard:
    """Per-key attempt counter with progress-based reset. Run-scoped; reset between runs.

    :param max_attempts: How many NO-PROGRESS attempts a key may spend before the next one is
        blocked (2 -> the third identical attempt is blocked). A first guess; see the module
        docstring for what should revise it.
    """

    max_attempts: int = 2
    #: key -> (no-progress attempts so far, evidence watermark at the last attempt)
    _state: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    #: the SUB-GOAL-scoped budget, same semantics over a coarser key (see the class docstring).
    _coarse_state: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    #: observe-only: coarse key -> {strict key -> attempts}, for keys that never blocked.
    _near_miss: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: observe-only counters for the coarse budget itself (see :meth:`record_coarse_attempt`).
    _coarse_stats: Dict[str, int] = field(
        default_factory=lambda: {"records": 0, "resets": 0, "max_attempts": 0}
    )

    def reset(self) -> None:
        self._state.clear()
        self._coarse_state.clear()
        self._near_miss.clear()
        self._coarse_stats.update({"records": 0, "resets": 0, "max_attempts": 0})

    @staticmethod
    def _attempts(state: Dict[str, Tuple[int, int]], key: str) -> int:
        return state.get(key, (0, 0))[0]

    def _is_blocked(self, state: Dict[str, Tuple[int, int]], key: str, watermark: int) -> bool:
        if self.max_attempts <= 0:
            return False
        count, last_watermark = state.get(key, (0, 0))
        if count < self.max_attempts:
            return False
        return watermark <= last_watermark

    def _record(self, state: Dict[str, Tuple[int, int]], key: str, watermark: int) -> int:
        count, last_watermark = state.get(key, (0, 0))
        count = 1 if watermark > last_watermark else count + 1
        state[key] = (count, watermark)
        return count

    def attempts(self, key: str) -> int:
        """No-progress attempts recorded for ``key`` (0 if it has never been attempted)."""
        return self._attempts(self._state, key)

    def is_blocked(self, key: str, watermark: int) -> bool:
        """Whether this key has burned its budget with nothing to show for it.

        False whenever ``watermark`` is above the one recorded at the key's last attempt: new
        evidence has appeared since, so this attempt is a step in a moving run rather than a
        repeat of a dead one.
        """
        return self._is_blocked(self._state, key, watermark)

    def record_attempt(self, key: str, watermark: int) -> int:
        """Register an attempt of ``key`` at ``watermark``; returns the new no-progress count.

        Progress since the previous attempt RESETS the count to 1 rather than decrementing it:
        the budget is "two consecutive fruitless tries", not "two tries ever".
        """
        return self._record(self._state, key, watermark)

    def coarse_attempts(self, key: str) -> int:
        """No-progress attempts recorded for a SUB-GOAL-scoped ``key``."""
        return self._attempts(self._coarse_state, key)

    def is_coarse_blocked(self, key: str, watermark: int) -> bool:
        """:meth:`is_blocked` over the sub-goal-scoped budget."""
        return self._is_blocked(self._coarse_state, key, watermark)

    def record_coarse_attempt(self, key: str, watermark: int) -> int:
        """:meth:`record_attempt` over the sub-goal-scoped budget."""
        previous = self._attempts(self._coarse_state, key)
        count = self._record(self._coarse_state, key, watermark)
        self._coarse_stats["records"] += 1
        # A reset means the scope's watermark MOVED between two attempts of this key. Counted
        # because it is the other half of "why did nothing block": a coarse key that keeps being
        # reset never reaches the threshold no matter how fine or coarse the key is.
        if previous >= 1 and count == 1:
            self._coarse_stats["resets"] += 1
        self._coarse_stats["max_attempts"] = max(self._coarse_stats["max_attempts"], count)
        return count

    def observe_near_miss(self, coarse_key: str, state_key: str, attempts: int) -> None:
        """Record that ``state_key`` spent ``attempts`` under ``coarse_key`` without blocking.

        Observe-only. This is the measurement that says whether the strict key FANS OUT: several
        distinct targets each stopping short of the strike threshold while one sub-goal is
        retried, which is invisible in a per-target counter by construction.
        """
        self._near_miss.setdefault(str(coarse_key), {})[str(state_key)] = int(attempts)

    def forget_near_miss(self, coarse_key: str, state_key: str) -> None:
        """Drop a key that went on to block; a blocked key is not a near miss."""
        self._near_miss.get(str(coarse_key), {}).pop(str(state_key), None)

    def near_miss_summary(self) -> Dict[str, Any]:
        """Observe-only counters for ``final_payload["novelty_guard"]``.

        ``near_miss_keys`` is the WORST sub-goal scope's fan-out (distinct strict keys that spent
        attempts there without any one of them blocking), because the question is whether ONE
        sub-goal's retries scatter across keys -- summed run-wide, a healthy run with many
        one-shot sub-goals would look identical.
        """
        per_scope = {scope: dict(keys) for scope, keys in self._near_miss.items() if keys}
        return {
            "near_miss_keys": max((len(keys) for keys in per_scope.values()), default=0),
            "near_miss_keys_total": sum(len(keys) for keys in per_scope.values()),
            "near_miss_total_attempts": max(
                (sum(keys.values()) for keys in per_scope.values()), default=0
            ),
            "near_miss_by_scope": {
                scope: {"keys": len(keys), "attempts": sum(keys.values())}
                for scope, keys in per_scope.items()
            },
            "sub_goal_attempts_recorded": self._coarse_stats["records"],
            "sub_goal_progress_resets": self._coarse_stats["resets"],
            "sub_goal_max_attempts": self._coarse_stats["max_attempts"],
        }
