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

The threshold (``max_attempts=2``, i.e. the THIRD identical no-progress attempt is blocked) is a
FIRST GUESS, not a measured value. It should be revisited against the mechanism suite's dead-end
retry-cap task (``agent/app/idea_tests/test_305_mech_dead_end_retry_cap.py``), which is the first
fixture able to say whether it is too tight (killing a legitimate second look at a flaky page) or
too loose (still paying for 5-8 repeats).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

_WHITESPACE = re.compile(r"\s+")


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


def evidence_watermark(graph) -> int:
    """A monotone count of the evidence this run has accumulated so far.

    Counts the ``Evidence``/``Claim`` sidecars that ``IdeaEngine._maybe_record_evidence`` writes
    (``evidence_store.Evidence`` / ``.Claim``) -- the records the plan names as the progress
    signal. Those are written only in ``run_policy_evidence_store_mode == "observe"``, so this
    ALSO counts successful action results, which are what "new evidence appeared" means when that
    observer is off. Without the fallback the watermark would be a constant zero on the default
    configuration and every key would look like no-progress, which would turn a churn guard into
    a blanket retry cap.

    Never decreases within a run (nothing removes a result or a sidecar), so a strictly greater
    value between two attempts of the same key really does mean something new was learned.
    """
    from agent.app.idea_policies.action_constants import ActionResultExtractor
    from agent.app.idea_policies.base import DetailKey

    total = 0
    try:
        nodes = list(graph.iter_breadth_first())
    except Exception:  # noqa: BLE001. A guard must never break the run it observes.
        return 0
    for node in nodes:
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

    def reset(self) -> None:
        self._state.clear()

    def attempts(self, key: str) -> int:
        """No-progress attempts recorded for ``key`` (0 if it has never been attempted)."""
        return self._state.get(key, (0, 0))[0]

    def is_blocked(self, key: str, watermark: int) -> bool:
        """Whether this key has burned its budget with nothing to show for it.

        False whenever ``watermark`` is above the one recorded at the key's last attempt: new
        evidence has appeared since, so this attempt is a step in a moving run rather than a
        repeat of a dead one.
        """
        if self.max_attempts <= 0:
            return False
        count, last_watermark = self._state.get(key, (0, 0))
        if count < self.max_attempts:
            return False
        return watermark <= last_watermark

    def record_attempt(self, key: str, watermark: int) -> int:
        """Register an attempt of ``key`` at ``watermark``; returns the new no-progress count.

        Progress since the previous attempt RESETS the count to 1 rather than decrementing it:
        the budget is "two consecutive fruitless tries", not "two tries ever".
        """
        count, last_watermark = self._state.get(key, (0, 0))
        count = 1 if watermark > last_watermark else count + 1
        self._state[key] = (count, watermark)
        return count
