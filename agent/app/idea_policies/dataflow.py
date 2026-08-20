"""Deterministic (no LLM) detection of unfilled dataflow slots in leaf-action candidates.

A "slot" is a template/placeholder or an explicit back-reference to a prior plan step still
sitting in a `search`/`visit` candidate's own scheduling fields (``optional_url``, ``url``,
``link``, ``link_idea``, ``query``, ``prompt``) -- evidence the model wrote the candidate
anticipating a SIBLING's not-yet-produced output rather than a value it already has.

Motivated by a read-only census over 108 stored chain-task results (see
``scripts/measure_dataflow_slots.py``): 19 of 418 search/visit leaf nodes executed with a
literal unfilled placeholder still in one of these fields (``<to be determined after
previous visit>``, ``<URL from previous search>``, ``[engineer's name]``, ...). None of the
19 "ran fine" despite the placeholder -- 3 failed outright with a missing/invalid-URL error,
and the other 16 were "silently repaired" by ``VisitLeafAction``'s unscoped sibling-URL
scavenging fallback (or, for `search`, produced a query result the node's own confidence
judge scored low/insufficient). That script also confirmed a 0% false-positive rate,
including on fan-out/parallel-shaped sibling batches -- the regression risk this module
exists to keep at bay.

Restricted to ``action in {search, visit}``: excluding ``think``/``save``/``merge`` is
LOAD-BEARING. A prior heuristic that also scanned ``think`` candidates produced false
positives on `parallel_merge`-shaped tasks, where a `think` node legitimately combines two
already-completed chains' results -- any back-reference language in it describes work that is
already DONE and on the path, not a dependency on a sibling that hasn't run yet.

A ``search`` candidate also checks its node ``title`` as a slot source, but ONLY when
neither ``query`` nor ``prompt`` is present -- the exact condition under which
``SearchLeafAction`` itself falls back to ``node.title`` at execution time (``actions.py``'s
``get_query(node.details, fallback_title=node.title)``). Without this a placeholder living
only in the title -- never in ``query``/``prompt`` -- would execute unflagged even though
it is exactly the value the search runs with.

Explicitly NOT a slot: bare pronouns; "the first result" (a search-result ordinal the action
itself resolves at execution time, not a sibling dependency); and any bracketed/braced span
that itself contains a concrete ``http(s)`` URL (e.g. ``<https://en.wikipedia.org/wiki/X>`` --
a badly-formatted but perfectly usable URL, not an unfilled template).

This module is purely additive: nothing here changes existing scheduling behavior by itself.
It becomes load-bearing only through ``idea_sequencing.siblings_are_independent`` and
``idea_sequencing.defer_unresolved_slot_candidates``, both gated behind
``EngineConfig.parallel_requires_evidence`` / ``EngineConfig.defer_unresolved_slots``
(default OFF).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent.app.idea_policies.base import DetailKey, IdeaActionType

#: Fields VisitLeafAction actually resolves a candidate from at execution time, in the same
#: order of appearance as ``actions.py``'s own ``optional_url``/``link_idea`` handling
#: (``visit_url`` is deliberately excluded -- ``idea_engine.py`` writes it back AFTER a
#: successful visit; it is an output, not a scheduling input).
_VISIT_FIELDS: tuple = ("optional_url", DetailKey.URL.value, DetailKey.LINK.value, "link_idea")

#: Fields ``NodeDetailsExtractor.get_query`` (and SearchLeafAction, transitively) resolve a
#: search candidate's query from: ``query`` -> ``prompt``.
_SEARCH_FIELDS: tuple = (DetailKey.QUERY.value, DetailKey.PROMPT.value)

_URL_RE = re.compile(r"https?://")
_TEMPLATE_RE = re.compile(r"\{[^{}]*\}|<[^<>]*>|\[[^\[\]]*\]")
_ORDINAL_RE = re.compile(
    r"from (?:the )?previous (?:step|search|visit|page|result|hop)"
    r"|previous (?:step|search|visit|page|result|hop)"
    r"|(?:identified|found|mentioned|described|determined|selected|discovered) (?:in|at) step \d+"
    r"|found above"
    r"|completed before"
    r"|the previous hop",
    re.IGNORECASE,
)


def unresolved_slots(details: Dict[str, Any], title: Optional[str] = None) -> List[str]:
    """Field names in ``details`` that still carry an unresolved dataflow slot.

    Restricted to ``action in {search, visit}`` -- every other action (notably
    ``think``/``save``/``merge``) returns ``[]`` unconditionally; see module docstring.

    :param details: A leaf candidate's ``details`` dict (a graph node's ``details``, or an
        about-to-be-created candidate's).
    :param title: The candidate's node title. Checked as a slot source ONLY for a
        ``search`` candidate, and ONLY when neither ``query`` nor ``prompt`` is present in
        ``details`` -- i.e. exactly the condition under which
        ``NodeDetailsExtractor.get_query(details, fallback_title=...)`` (and, transitively,
        ``SearchLeafAction.execute``, see ``actions.py``'s
        ``get_query(node.details, fallback_title=node.title)``) falls back to it at
        execution time. Without this, a placeholder living only in the title -- never in
        ``query``/``prompt`` -- would execute unflagged even though it is exactly the value
        the search actually runs with.
    :returns: The (possibly empty) list of field names that matched a slot pattern
        (``"title"`` included when the title fallback above matched). Field order follows
        the action's own field-resolution priority.
    """
    action = details.get(DetailKey.ACTION.value)
    if action == IdeaActionType.VISIT.value:
        return _scan_fields(details, _VISIT_FIELDS)
    if action == IdeaActionType.SEARCH.value:
        slots = _scan_fields(details, _SEARCH_FIELDS)
        if not details.get(DetailKey.QUERY.value) and not details.get(DetailKey.PROMPT.value):
            if isinstance(title, str) and title.strip() and _has_slot(title):
                slots.append("title")
        return slots
    return []


def _scan_fields(details: Dict[str, Any], fields: tuple) -> List[str]:
    slots: List[str] = []
    for field in fields:
        value = details.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if _has_slot(value):
            slots.append(field)
    return slots


def _has_slot(value: str) -> bool:
    """A template placeholder with no URL inside it, or an explicit back-reference."""
    for match in _TEMPLATE_RE.finditer(value):
        if not _URL_RE.search(match.group(0)):
            return True
    return bool(_ORDINAL_RE.search(value))
