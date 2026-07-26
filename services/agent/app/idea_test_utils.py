"""
Shared utilities for idea test system.
"""

import json
import re
import inspect
from typing import Dict, Any, Optional


def extract_final_text(result: Dict[str, Any]) -> str:
    """
    Extract final text from test result output.
    :param result: Test result dict.
    :return: Final text string.
    """
    output = result.get("output", {})
    if not isinstance(output, dict):
        return str(output)
    
    final_deliverable = output.get("final_deliverable", "")
    if isinstance(final_deliverable, str):
        return final_deliverable
    elif isinstance(final_deliverable, dict):
        return json.dumps(final_deliverable, ensure_ascii=True)
    elif isinstance(final_deliverable, list):
        return json.dumps(final_deliverable, ensure_ascii=True)
    return str(final_deliverable)


async def call_validation_function(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Call validation function (async or sync).
    :param func: Validation function.
    :param args: Positional arguments.
    :param kwargs: Keyword arguments.
    :return: Validation result dict.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


def normalize_url(u: str) -> str:
    """
    Normalize a URL for chain/adjacency matching: lowercase, drop scheme, strip
    trailing slash and any #fragment. Robust enough for Wikipedia-style links.
    :param u: URL string.
    :return: Normalized key.
    """
    s = str(u or "").strip().lower()
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    s = s.split("#", 1)[0]
    return s.rstrip("/")


def build_visit_link_graph(result: Dict[str, Any]):
    """
    Reconstruct the agent's traversal as an objective link graph from its visit
    action results. Each successfully visited page maps to the set of outgoing
    links it actually contained (``links_full``), enabling true adjacency checks
    for navigation / wiki-race tests (unlike a model's self-reported claims).
    :param result: Test result payload.
    :return: ``(link_map, visited_order)`` where ``link_map`` is
             ``{normalized_url: set(normalized_outgoing_links)}`` and
             ``visited_order`` is the list of visited URLs in execution order.
    """
    link_map: Dict[str, set] = {}
    visited_order = []
    graph = result.get("graph") or {}
    nodes = graph.get("nodes") or {}
    node_items = nodes.values() if isinstance(nodes, dict) else (nodes if isinstance(nodes, list) else [])
    for node in node_items:
        if not isinstance(node, dict):
            continue
        ar = (node.get("details") or {}).get("action_result") or {}
        if not (isinstance(ar, dict) and ar.get("action") == "visit" and ar.get("success")):
            continue
        urls = ar.get("urls_visited") or ([ar.get("url")] if ar.get("url") else [])
        links = ar.get("links_full") or ar.get("links") or []
        norm_links = {normalize_url(x) for x in links if isinstance(x, str)}
        for u in urls:
            if not u:
                continue
            key = normalize_url(u)
            visited_order.append(key)
            link_map.setdefault(key, set()).update(norm_links)
    return link_map, visited_order


def count_words(text: str) -> int:
    """
    Count words in text.
    :param text: Input text.
    :return: Word count.
    """
    return len(str(text).split())


def count_chars(text: str) -> int:
    """
    Count characters in text.
    :param text: Input text.
    :return: Character count.
    """
    return len(str(text))


# ---------------------------------------------------------------------------
# Tolerant numeric-keystone matchers.
#
# Several task validators used a brittle fixed-literal regex for a numeric keystone
# (e.g. ``re.compile(r"\b13\.51\b")`` or ``re.compile(r"\b300\s*m\b")``). That false-fails a
# CORRECT, grounded answer that is merely rounded ("13.5" instead of "13.51"), given extra
# precision ("13.510"), or phrased with a spelled-out unit ("300 metres" instead of "300 m") —
# which systematically penalizes an agent that normalizes its answer instead of echoing the
# page's exact substring. These helpers centralize the fix so each task validator can opt into
# tolerant matching without duplicating the regex/parsing logic per file.
# ---------------------------------------------------------------------------

_NUM_RX = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)(?![\d])")


def numeric_value_matches(text: str, value: float, rel_tol: float = 0.01) -> bool:
    """
    Tolerant numeric-keystone matcher: True if ``text`` contains a bare or comma-grouped
    number within a relative tolerance band of ``value``. Accepts standard roundings and
    extra decimal precision (e.g. "13.5" or "13.510" both satisfy a canonical 13.51; "129"
    satisfies a canonical 128.9) instead of requiring an exact fixed-decimal literal match,
    which false-fails a correct-but-differently-rounded grounded answer. A near-miss number
    that is genuinely a different value (e.g. "113.51" vs. "13.51") is still correctly
    rejected, because it is parsed and compared as its own numeric value, not a substring.
    :param text: Reported answer text to scan for numeric mentions.
    :param value: The canonical/ground-truth numeric value.
    :param rel_tol: Relative tolerance band around ``value`` (default 1%).
    :return: True if any number in ``text`` falls within [value*(1-rel_tol), value*(1+rel_tol)].
    """
    lo, hi = value * (1 - rel_tol), value * (1 + rel_tol)
    for raw in _NUM_RX.findall(str(text or "")):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if lo <= v <= hi:
            return True
    return False


def unit_value_matches(text: str, value: float, unit_rx: str, rel_tol: float = 0.0) -> bool:
    """
    Tolerant unit-anchored numeric matcher: True if ``text`` contains a number within
    ``rel_tol`` of ``value`` immediately followed (allowing a hyphen/space) by one of the unit
    spellings in ``unit_rx`` (a regex fragment, e.g. ``r"m\\b|met(?:er|re)s?"``). Fixes the
    "abbreviation-only" false-negative where a keystone regex like ``r"\\b300\\s*m\\b"`` rejects
    a correctly grounded "300 metres" / "300-meter" answer, because the word-boundary right
    after "m" blocks the spelled-out unit from ever matching.
    :param text: Reported answer text to scan.
    :param value: The canonical numeric value that must precede the unit.
    :param unit_rx: Regex fragment of acceptable unit spellings following the number.
    :param rel_tol: Relative tolerance band around ``value`` (default 0 = exact value only).
    :return: True if a qualifying "<value><unit>" mention is found.
    """
    lo, hi = value * (1 - rel_tol), value * (1 + rel_tol)
    pat = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)[\s-]*(?:" + unit_rx + r")", re.IGNORECASE)
    for raw in pat.findall(str(text or "")):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if lo <= v <= hi:
            return True
    return False


def dimension_value_matches(text: str, a: int, b: int, joiner_rx: str = r"[x×*]|by") -> bool:
    """
    Tolerant "A x B" dimension matcher: True if ``text`` contains ``a`` and ``b`` joined by one
    of several acceptable joiners/operators (the default accepts "x", the unicode multiplication
    sign "×", "*", or the spelled-out word "by"), so "16 by 16" is accepted alongside
    "16x16" / "16 x 16" / "16×16" instead of only the single literal operator form.
    :param text: Reported answer text to scan.
    :param a: First dimension value.
    :param b: Second dimension value.
    :param joiner_rx: Regex fragment of acceptable joiners between the two numbers.
    :return: True if a qualifying "<a><joiner><b>" mention is found.
    """
    pat = re.compile(rf"\b{a}\s*(?:{joiner_rx})\s*{b}\b", re.IGNORECASE)
    return bool(pat.search(str(text or "")))
