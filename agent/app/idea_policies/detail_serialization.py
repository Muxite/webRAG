"""Fit a node's `details` into a prompt budget without breaking the JSON.

Both prompt sites that show `details` to a model used to apply their budget as
`json.dumps(details)[:budget]`, a character cut through whatever string the budget landed in.
The evaluator got the fix first (see `_serialize_details_for_prompt`); the planner in
`expansion.py` cut the same way after its own compaction pass, so the helpers live here and
both import them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agent.app.idea_policies.base import DetailKey


def _safe_serialize_details(details: Dict[str, Any]) -> str:
    try:
        return json.dumps(details, ensure_ascii=True, default=str)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=True)


#: How many entries of any one list or dict survive clipping, once `limit` is large enough to
#: allow it. A page's `links_full` runs to hundreds of items and `link_contexts` to hundreds of
#: URL *keys*, so neither fits under a per-string bound alone. Below this many characters the
#: bound tightens with `limit`, which is what gives the bisection a floor it can always reach.
_MAX_CONTAINER_ENTRIES = 20


def _clip_leaves(value: Any, limit: int) -> Any:
    """Shorten every string leaf to `limit` and every container, structure left intact."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return f"{value[:limit]}... [+{len(value) - limit} chars truncated]"
    keep = max(1, min(_MAX_CONTAINER_ENTRIES, limit))
    if isinstance(value, dict):
        items = list(value.items())
        clipped = {
            (_clip_leaves(key, limit) if isinstance(key, str) else key): _clip_leaves(item, limit)
            for key, item in items[:keep]
        }
        if len(items) > keep:
            clipped["..."] = f"[+{len(items) - keep} keys truncated]"
        return clipped
    if isinstance(value, list):
        clipped_items = [_clip_leaves(item, limit) for item in value[:keep]]
        if len(value) > keep:
            clipped_items.append(f"... [+{len(value) - keep} items truncated]")
        return clipped_items
    return value


#: `action_result` fields that duplicate `content` verbatim (checked: `content_full` starts with
#: `content` on 2103/2103 recorded visit results) or that carry link bookkeeping a model cannot
#: use in that form. Dropped only when the blob is over budget, so an under-budget candidate is
#: unaffected. The planner's own `_compact_details_for_expansion` already drops the two `content`
#: duplicates before this runs; the link fields are what it leaves behind.
_BULK_RESULT_FIELDS = ("content_full", "content_with_links", "links_full", "link_contexts",
                       "vector_context")


def _compact_details_for_prompt(details: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the bulk duplicates from `action_result` so the budget can go to `content`."""
    action_result = details.get(DetailKey.ACTION_RESULT.value)
    if not isinstance(action_result, dict):
        return details
    dropped = [field for field in _BULK_RESULT_FIELDS if field in action_result]
    if not dropped:
        return details
    compact_result = {k: v for k, v in action_result.items() if k not in dropped}
    compact_result["_omitted_fields"] = dropped
    compact = dict(details)
    compact[DetailKey.ACTION_RESULT.value] = compact_result
    return compact


def _serialize_details_for_prompt(details: Dict[str, Any], max_chars: int) -> str:
    """Serialize `details` into at most `max_chars` of *valid* JSON.

    This used to be `json.dumps(details)[:max_chars]`, which cuts wherever the budget lands --
    almost always inside `action_result`'s page text. Every one of the 2888 recorded executed
    candidates that hit the evaluator's cap reached the judge as an unterminated blob, and every
    field serialized after the cut was gone with it: `visit_url` and `visit_content_length` on
    1982/1987 truncated visit nodes, `provides_data` on 76%. Four siblings that had all fetched
    the same URL were therefore indistinguishable in the prompt (ASSUMPTION_AUDIT.md T1-3c).

    Clipping the leaves instead keeps every key visible and the blob parseable at the same
    budget: the duplicated bulk fields go first, then the largest per-leaf allowance that still
    fits is found by bisection, so the page text keeps most of the space it had. Under budget
    the output is byte-identical to `_safe_serialize_details`, so the candidates that fit are
    unaffected.
    """
    text = _safe_serialize_details(details)
    if len(text) <= max_chars:
        return text
    compact = _compact_details_for_prompt(details)
    low, high = 0, max_chars
    best: Optional[str] = None
    while low <= high:
        mid = (low + high) // 2
        candidate = _safe_serialize_details(_clip_leaves(compact, mid))
        if len(candidate) <= max_chars:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    if best is not None:
        return best
    # A budget too small for even a fully clipped object (a handful of characters) still has to
    # come back parseable, so wrap the old prefix as a JSON *string value* and shrink it until
    # the escaped envelope fits.
    room = max(0, max_chars - 32)
    while True:
        fallback = json.dumps({"details_truncated": text[:room]}, ensure_ascii=True)
        if len(fallback) <= max_chars or room == 0:
            return fallback
        room //= 2
