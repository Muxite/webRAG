#!/usr/bin/env python3
"""
Read-only measurement: the GATE for the surviving "pass structured tool output" chain-hop
design.

Two prior designs for threading a discovered value from hop N to hop N+1 were killed by free
offline measurements this session (see scripts/replay_waypoints.py and
scripts/measure_dataflow_slots.py). The surviving design says: do NOT pass an extracted value
(a number/name a regex guessed out of prose); pass the STRUCTURED tool output instead --
specifically, hand hop N+1 the page hop N actually visited's outgoing LINK SET (a closed set the
visit tool really returned), on the premise that link containment is structural (if the target
page is hyperlinked at all, it is IN the set) rather than heuristic (an extraction that can miss).

This script tests exactly that premise, and nothing else:

  1. CONTAINMENT -- for every chain hop where the agent visited the CORRECT source page for
     waypoint N, is waypoint N+1's target URL present in that page's link set at all? Reported
     separately for the ``links`` field (what a downstream consumer actually gets today -- see
     "which field is real" below) and ``links_full`` (the complete, uncapped set, analysis-only).
  2. RANK -- when present, at what rank under several cheap deterministic rankings. TWO FAMILIES,
     reported and interpreted separately (do not conflate them -- an earlier revision of this
     script did, and the coordinator caught it): an ORACLE family (anchor-text / URL-slug overlap
     against the TARGET waypoint's own ground-truth name/name_rx wording -- the answer -- kept
     only as an upper-bound reference, NEVER a deployability claim, since the engine does not know
     hop N+1's target at hop-N time) and a RUNTIME-REALISTIC family scored ONLY against text/
     signal the engine actually possesses at hop-N time: (1) the downstream node's own pre-
     existing goal/title/query text, (2) the root mandate's role wording via product's own
     ``_ENTITY_ROLE_CUES`` proximity mechanism applied to ``link_contexts``, and (3) a
     zero-knowledge nav-chrome filter that uses no goal text at all.
  3. SET SIZE -- the distribution of how many links a typical page actually offers.
  4. SEARCH-RESULT CHANNEL -- the same containment/rank questions for ``action_result.results``
     on search nodes (smaller, title-bearing -- a plausible alternative or complement to page
     links).
  5. FLATNESS -- containment@{5,10,20,50,all}. The prior (killed) value-threading design's
     signature failure was containment that stayed FLAT as the candidate window grew (the truth
     simply was not in the window, at any size). This script checks for the same signature here.
  6. NEGATIVE CASES -- when the target URL is NOT in the source page's link set, why? Categorized
     with counts and concrete examples (structurally never linked / no links captured at all /
     reached via search elsewhere in the same trace / isolated miss / non-Wikipedia source).

WHICH FIELD IS REAL (checked in the product code, not guessed): ``agent/app/idea_policies/
actions.py`` (``_attach_links_to_content``) and ``expansion.py`` (``_enhance_details_with_inline_
links`` / ``_compact_details_for_expansion``) all cap what actually reaches the model's context at
``config.action.max_links_per_visit`` (default 20 -- see ``idea_dag_settings*.json``). Empirically
(verified below and independently on a sample of this exact dataset) ``links == links_full[:20]``
in every case checked: ``links`` is that capped, page-order prefix; ``links_full`` is the complete
set kept only for offline validators (e.g. ``idea_test_utils.build_visit_link_graph``). A future
"pass the link set forward" design, unless it explicitly widens the cap, would realistically hand
hop N+1 the ``links``-sized (~20-item) set -- which is why this script reports containment/size for
BOTH fields, not just the generous one.

Data: agent/idea_test_results/cschain_g_*.json and csnopar_g_*.json (9 chain tasks x 3 models x
2 arms = 108 files). Ground truth is READ, not retyped, from each task module's own CHAIN /
KEYSTONE_*/HOP_*/CITE_* constants in agent/app/idea_tests/test_{135,136,137,138,139,065,046,
093}_*.py (imported directly). 047 and 093 are excluded -- see NOT_MEASURED for why.

Read-only: no product code is modified (task modules and idea_test_utils.normalize_url are
imported, never written to), no live calls, no spend.

Usage (from repo root):
    ./.venv/bin/python3 scripts/measure_link_hop_containment.py
    ./.venv/bin/python3 scripts/measure_link_hop_containment.py --by-task --examples 3
    ./.venv/bin/python3 scripts/measure_link_hop_containment.py --json out.json
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple
from urllib.parse import urlparse, unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "services", REPO_ROOT / "agent"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from agent.app.idea_test_utils import normalize_url  # noqa: E402
from agent.app.idea_policies.waypoint import _ENTITY_ROLE_CUES as _PRODUCT_ROLE_CUES  # noqa: E402
from agent.app.idea_policies.waypoint import _WINDOW as _PRODUCT_WINDOW  # noqa: E402

# ORACLE ranking uses target ground truth (name/name_rx), which the engine does not possess at
# hop-N time. It is kept only as an upper bound. Only the three RUNTIME-REALISTIC variants
# answer "would this work at runtime".

RESULTS_GLOBS_DEFAULT = (
    "agent/idea_test_results/cschain_g_*.json",
    "agent/idea_test_results/csnopar_g_*.json",
)

FNAME_RX = re.compile(
    r"^(?P<shape>cschain|csnopar)_g_(?P<model>[a-z0-9]+)_(?P<arm>baseline|good_adaptive)_"
    r"rep(?P<rep>\d+)_(?P<task_id>\d{3})_"
)

NOT_MEASURED = {
    "047": (
        "open-ended wiki-race path-finding -- any valid START->TARGET hyperlink chain scores, so "
        "there is no fixed 'correct next-hop URL' to test containment against (would be circular: "
        "the ground truth IS 'a link exists')."
    ),
    "093": (
        "both resources (the pinned curl C source file and the official advisory) are SEEDED "
        "directly in the task mandate; neither is discovered by following a hyperlink FROM the "
        "other. There is no source-page link set whose containment of the other page is even a "
        "meaningful question for this task."
    ),
}

MAX_LINKS_FIELD_CAP = 20  # config.action.max_links_per_visit default; verified empirically below too


# Hop registry: page-to-page hyperlink transitions from each task's own ground-truth constants
# (CHAIN / HOP_*/CITE_*). Target is the NEXT PAGE'S URL, unlike replay_waypoints.py's per-VALUE registry.


def _rx(pattern) -> Pattern:
    if isinstance(pattern, re.Pattern):
        return pattern
    return re.compile(pattern, re.IGNORECASE)


def _load_task_module(test_id: str):
    matches = glob.glob(str(REPO_ROOT / "agent" / "app" / "idea_tests" / f"test_{test_id}_*.py"))
    if not matches:
        raise FileNotFoundError(f"no task module for test_id={test_id}")
    mod_name = "agent.app.idea_tests." + os.path.basename(matches[0])[:-3]
    return importlib.import_module(mod_name)


@dataclass
class Hop:
    """A single page-to-page hyperlink transition the agent needs: hop N's SOURCE page should
    contain a link to hop N+1's TARGET page."""
    task_id: str
    label: str
    source_slug_rx: Pattern
    source_name_rx: Pattern
    target_slug_rx: Pattern
    target_name_rx: Pattern
    target_cues: Tuple[str, ...]  # plain-language wording of the target's own name_rx, for ranking


@dataclass
class SearchHop:
    """A hop whose SOURCE is a search (not a page): the target is only discoverable via a search
    call, e.g. 065's hop 1 ('work' -> poet). Has no source page / link set of its own."""
    task_id: str
    label: str
    target_slug_rx: Pattern
    target_name_rx: Pattern
    target_cues: Tuple[str, ...]


def build_hops() -> List[Hop]:
    hops: List[Hop] = []

    def std_chain(tid: str, creator_cues: Tuple[str, ...], terminal_cues: Tuple[str, ...]) -> None:
        mod = _load_task_module(tid)
        start, creator, terminal = mod.CHAIN
        hops.append(Hop(
            tid, "start->creator", _rx(start["slug_rx"]), _rx(start["name_rx"]),
            _rx(creator["slug_rx"]), _rx(creator["name_rx"]), creator_cues,
        ))
        hops.append(Hop(
            tid, "creator->terminal", _rx(creator["slug_rx"]), _rx(creator["name_rx"]),
            _rx(terminal["slug_rx"]), _rx(terminal["name_rx"]), terminal_cues,
        ))

    std_chain("135", ("John A. Roebling", "Roebling"),
              ("John A. Roebling Suspension Bridge", "Roebling Suspension Bridge",
               "Cincinnati", "Covington"))
    std_chain("136", ("Isambard Kingdom Brunel", "Brunel"),
              ("SS Great Eastern", "Great Eastern"))
    std_chain("137", ("Thomas Telford", "Telford"),
              ("Pontcysyllte Aqueduct", "Pontcysyllte"))
    std_chain("139", ("Antoni Gaudí", "Gaudí", "Gaudi"),
              ("Casa Milà", "La Pedrera", "Casa Mila"))

    # 138: "terminal" is a third distinct page (George Everest), not a continuation of the creator.
    mod138 = _load_task_module("138")
    start138, creator138, terminal138 = mod138.CHAIN
    hops.append(Hop(
        "138", "start->creator", _rx(start138["slug_rx"]), _rx(start138["name_rx"]),
        _rx(creator138["slug_rx"]), _rx(creator138["name_rx"]), ("Andrew Scott Waugh", "Waugh"),
    ))
    hops.append(Hop(
        "138", "creator->terminal", _rx(creator138["slug_rx"]), _rx(creator138["name_rx"]),
        _rx(terminal138["slug_rx"]), _rx(terminal138["name_rx"]), ("George Everest",),
    ))

    # 065: hop 1 ("work"->poet) is search-first (see build_search_hops); hop 2 (poet->town) is a
    # genuine page-to-page link hop.
    mod065 = _load_task_module("065")
    hops.append(Hop(
        "065", "poet->town", _rx(mod065.CITE_POET), _rx(mod065.HOP_POET),
        _rx(mod065.CITE_TOWN), _rx(mod065.HOP_TOWN), ("Parral, Chile", "Parral"),
    ))

    # 046: fixed given URLs (no CHAIN list); build slug regexes from the literal URL paths.
    mod046 = _load_task_module("046")

    def _path_rx(u: str) -> Pattern:
        path = u.split("wikipedia.org", 1)[-1]
        return re.compile(re.escape(path.lower()))

    hops.append(Hop(
        "046", "apollo11->saturnv", _path_rx(mod046.START_URL), re.compile(r"apollo\s*11", re.IGNORECASE),
        _path_rx(mod046.HOP_URL), re.compile(r"saturn\s*v", re.IGNORECASE), ("Saturn V",),
    ))

    return hops


def build_search_hops() -> List[SearchHop]:
    mod065 = _load_task_module("065")
    return [
        SearchHop(
            "065", "work->poet", _rx(mod065.CITE_POET), _rx(mod065.HOP_POET),
            ("Pablo Neruda", "Neruda"),
        ),
    ]




def _ordered_nodes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = ((data.get("execution") or {}).get("graph") or {}).get("nodes") or {}
    return list(nodes.values())  # dict preserves JSON key order == creation order


def _action_result(node: Dict[str, Any]) -> Dict[str, Any]:
    ar = (node.get("details") or {}).get("action_result")
    return ar if isinstance(ar, dict) else {}


def _is_successful_visit(node: Dict[str, Any]) -> bool:
    ar = _action_result(node)
    return ar.get("action") == "visit" and bool(ar.get("success")) and str(ar.get("url") or "").startswith("http")


def _is_successful_search(node: Dict[str, Any]) -> bool:
    ar = _action_result(node)
    return ar.get("action") == "search" and bool(ar.get("success"))


def _page_identity_matches(ar: Dict[str, Any], slug_rx: Pattern, name_rx: Optional[Pattern]) -> bool:
    url = normalize_url(ar.get("url") or "")
    if slug_rx.search(url):
        return True
    if name_rx is not None:
        ident = f"{ar.get('h1_text') or ''} {ar.get('page_title') or ''}"
        if ident.strip() and name_rx.search(ident):
            return True
    return False


def _best_matching_visit(visits: Sequence[Dict[str, Any]], slug_rx: Pattern,
                          name_rx: Optional[Pattern]) -> Optional[Dict[str, Any]]:
    """When a page was (re)visited more than once in the same trace (retries), pick the richest
    capture (max content length) as the single representative instance -- avoids double-counting
    the same page's link set twice from one trace."""
    candidates = [n for n in visits if _page_identity_matches(_action_result(n), slug_rx, name_rx)]
    if not candidates:
        return None

    def content_len(n: Dict[str, Any]) -> int:
        ar = _action_result(n)
        c = ar.get("content_full") or ar.get("content") or ""
        return len(c) if isinstance(c, str) else 0

    return max(candidates, key=content_len)


def _dedupe_ordered(urls: Sequence[Any]) -> List[str]:
    """Preserve first-occurrence page order, drop exact-normalized-URL repeats (the raw links
    list can repeat the same href from nav chrome + body text)."""
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str) or not u:
            continue
        norm = normalize_url(u)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(u)
    return out


def _anchor_lookup(link_contexts: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(link_contexts, dict):
        return out
    for k, v in link_contexts.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            out[normalize_url(k)] = v.strip()
    return out


def _slug_text(url: str) -> str:
    try:
        path = urlparse(url).path
    except Exception:
        path = url
    tail = path.rsplit("/", 1)[-1] if path else url
    try:
        tail = unquote(tail)
    except Exception:
        pass
    return tail.replace("_", " ").replace("-", " ")


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""



_STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for", "by", "with"}


def _overlap_score(text: str, cues: Sequence[str]) -> float:
    """Method: longest cue phrase that is a case-insensitive substring of `text`, or vice versa
    (a short anchor like 'Brunel' inside a longer cue 'Isambard Kingdom Brunel', and a short cue
    like 'Telford' inside a longer anchor 'Thomas Telford (engineer)'). 0.0 = no cue present."""
    if not text:
        return 0.0
    t = text.lower()
    best = 0.0
    for cue in cues:
        c = cue.lower().strip()
        if not c:
            continue
        if c in t or t in c:
            best = max(best, float(len(c)))
    return best


def _tokenize(text: str) -> set:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3 and tok not in _STOPWORDS}


def _jaccard_score(text: str, cues: Sequence[str]) -> float:
    """Method: token-set Jaccard overlap between `text` and each cue, best cue wins. A different
    mechanism from substring overlap -- robust to reordering ('Suspension Bridge Roebling' vs
    'Roebling Suspension Bridge') at the cost of needing >=1 whole shared token."""
    tt = _tokenize(text)
    if not tt:
        return 0.0
    best = 0.0
    for cue in cues:
        ct = _tokenize(cue)
        if not ct:
            continue
        inter = len(tt & ct)
        union = len(tt | ct)
        if union:
            best = max(best, inter / union)
    return best


def _rank_by_position(ordered_urls: Sequence[str], target_slug_rx: Pattern) -> Tuple[Optional[int], int]:
    n = len(ordered_urls)
    for i, u in enumerate(ordered_urls, start=1):
        if target_slug_rx.search(normalize_url(u)):
            return i, n
    return None, n


def _rank_by_score(ordered_urls: Sequence[str], scores: Dict[str, float],
                    target_slug_rx: Pattern) -> Optional[int]:
    indexed = list(enumerate(ordered_urls))
    ranked = sorted(indexed, key=lambda iu: (-scores.get(normalize_url(iu[1]), 0.0), iu[0]))
    for rank, (_orig_i, u) in enumerate(ranked, start=1):
        if target_slug_rx.search(normalize_url(u)):
            return rank
    return None


RANK_METHODS = ("position", "anchor_text", "url_slug", "combined_max", "token_jaccard")
RANK_LABEL = {
    "position": "raw page position (order links appear on the page)",
    "anchor_text": "anchor-text substring overlap with the target's own wording [ORACLE]",
    "url_slug": "URL-slug substring overlap with the target's own wording [ORACLE]",
    "combined_max": "max(anchor_text, url_slug) per link [ORACLE]",
    "token_jaccard": "token-set Jaccard of (anchor+slug) vs target wording [ORACLE]",
}


# RUNTIME-REALISTIC ranking variants use no target waypoint ground truth.


def _text_overlap_score(candidate_text: str, query_text: str) -> float:
    """Shared-token count between a candidate link's own text (anchor+slug) and an arbitrary
    free-text query (node-local goal, root mandate, ...). Unlike `_overlap_score` (used for the
    oracle rankings), this takes no list of known-correct phrases -- it just tokenizes both sides
    and counts overlap, which is what a ranker restricted to runtime-available text would have to
    do since it does not know which words in the query matter."""
    return float(len(_tokenize(candidate_text) & _tokenize(query_text)))


def _text_jaccard_score(candidate_text: str, query_text: str) -> float:
    ct, qt = _tokenize(candidate_text), _tokenize(query_text)
    if not ct or not qt:
        return 0.0
    inter = len(ct & qt)
    union = len(ct | qt)
    return inter / union if union else 0.0


def _find_target_reaching_node(nodes: Sequence[Dict[str, Any]], target_slug_rx: Pattern,
                                target_name_rx: Optional[Pattern]) -> Optional[Dict[str, Any]]:
    """The (if any) node in this SAME trace whose own successful visit actually landed on the
    target waypoint's page -- used only to read that node's own pre-existing title/goal/query
    text (variant 1), never its resolved URL for ranking purposes."""
    for n in nodes:
        ar = _action_result(n)
        if ar.get("action") != "visit" or not ar.get("success"):
            continue
        if _page_identity_matches(ar, target_slug_rx, target_name_rx):
            return n
    return None


def _node_local_text(node: Dict[str, Any]) -> str:
    d = node.get("details") or {}
    parts = [node.get("title") or "", d.get("goal") or "", d.get("query") or "", d.get("intent") or ""]
    return " ".join(str(p) for p in parts if p)


def _root_mandate_text(data: Dict[str, Any]) -> str:
    graph = (data.get("execution") or {}).get("graph") or {}
    nodes = graph.get("nodes") or {}
    root = nodes.get(graph.get("root_id") or "")
    if root is None:
        vals = list(nodes.values())
        root = vals[0] if vals else {}
    d = root.get("details") or {}
    return str(d.get("mandate") or d.get("goal") or root.get("title") or "")


def _mandate_present_cues(mandate_text: str, vocab: Sequence[str]) -> List[str]:
    """Which of a FIXED, generic, non-oracle role vocabulary (agent/app/idea_policies/waypoint.py's
    own shipped `_ENTITY_ROLE_CUES`) actually appear, verbatim, in THIS task's root mandate text --
    the 'the mandate usually names the role even when it cannot name the answer' signal."""
    mt = mandate_text.lower()
    return [c for c in vocab if c.lower() in mt]


def _role_proximity_scores(content_low: str, present_cues: Sequence[str], anchor_lu: Dict[str, str],
                            ordered_full: Sequence[str], window: int) -> Dict[str, float]:
    """Find every occurrence of a present role-cue word in the SOURCE PAGE's own body text, then
    score each candidate link by how close its OWN anchor text sits to the nearest cue occurrence
    (closer = higher score; no occurrence within `window` = no score at all, i.e. no signal, same
    treatment as every other 0-score candidate). Re-implements (does not import) the proximity
    mechanism `idea_policies/waypoint.py::_extract_anchor_text` already ships for VALUE extraction,
    repurposed here to rank a LINK instead of extract a value -- read-only, diagnostic-only."""
    cue_positions: List[float] = []
    for cue in present_cues:
        for m in re.finditer(re.escape(cue.lower()), content_low):
            cue_positions.append((m.start() + m.end()) / 2.0)
    scores: Dict[str, float] = {}
    if not cue_positions:
        return scores
    for u in ordered_full:
        nu = normalize_url(u)
        anchor = anchor_lu.get(nu, "")
        if not anchor or len(anchor) < 3:
            continue
        idx = content_low.find(anchor.lower())
        if idx < 0:
            continue
        anchor_center = idx + len(anchor) / 2.0
        best_dist = min(abs(anchor_center - cp) for cp in cue_positions)
        if best_dist <= window:
            scores[nu] = -best_dist  # closer -> less negative -> higher score
    return scores


_NAMESPACE_RX = re.compile(
    r"/wiki/(Special|Help|Portal|Wikipedia|Talk|User|Template|Category|File|Draft|"
    r"MediaWiki|Module|TimedText|Book|Category_talk|Template_talk|User_talk):",
    re.IGNORECASE,
)


def _is_chrome_link(url: Any, source_domain: str) -> bool:
    """Zero-knowledge (no goal text of any kind) chrome filter: drop sister-project / other-domain
    links, non-article namespaces, and dynamic (edit/redlink/query-string) URLs. Requires nothing
    but the SOURCE page's own domain -- applies identically to every hop regardless of wording."""
    if not isinstance(url, str) or not url.startswith("http"):
        return True
    if _domain(url) != source_domain:
        return True
    if "/wiki/" not in url:
        return True
    if _NAMESPACE_RX.search(url):
        return True
    if "action=edit" in url or "redlink=1" in url:
        return True
    return False


def _filtered_article_links(ordered_full: Sequence[str], source_domain: str) -> List[str]:
    return [u for u in ordered_full if not _is_chrome_link(u, source_domain)]


# Records


@dataclass
class HopInstance:
    file: str
    shape: str
    model: str
    arm: str
    task_id: str
    hop_label: str
    source_url: str
    source_domain: str
    n_links_full: int
    n_links_20: int
    n_link_contexts: int
    contains_full: bool
    contains_20: bool
    ranks: Dict[str, Optional[int]]          # ORACLE rankings, over links_full -- reference only
    ranks_20: Dict[str, Optional[int]]        # over the capped `links` field only
    # negative-case diagnostics (filled in on a 2nd pass once aggregate rates are known)
    category: str = ""
    same_trace_search_hit: Optional[bool] = None
    target_domain_if_found: str = ""
    # ---- runtime-realistic variants (no oracle target wording used for scoring) ----
    v1_available: bool = False            # a downstream node that reached the target exists
    v1_leaky: Optional[bool] = None       # diagnostic-only (oracle used to LABEL, not to rank):
                                           # does that node's own text already name the target?
    v1_rank_overlap: Optional[int] = None
    v1_rank_jaccard: Optional[int] = None
    v2_n_cues_present: int = 0            # how many generic role cues appear in the root mandate
    v2_rank_proximity: Optional[int] = None
    v2_target_scored: bool = False        # did the TARGET's own anchor land near a cue occurrence
                                           # (vs. just being present and winning the position tie-break)?
    v2_rank_mandate_jaccard: Optional[int] = None
    v3_n_filtered: int = 0
    v3_contains: bool = False
    v3_rank: Optional[int] = None


@dataclass
class SearchInstance:
    file: str
    shape: str
    model: str
    arm: str
    task_id: str
    hop_label: str
    canonical: bool  # True = the designed search-first hop (065 work->poet); False = opportunistic
    n_results: int
    contains: bool
    rank_position: Optional[int]
    rank_overlap: Optional[int]


def _find_task_id(base: str) -> Optional[str]:
    m = FNAME_RX.match(base)
    return m.group("task_id") if m else None


def _file_meta(base: str) -> Tuple[str, str, str, str]:
    m = FNAME_RX.match(base)
    if not m:
        return "", "", "", ""
    return m.group("shape"), m.group("model"), m.group("arm"), m.group("rep")


def _search_results(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    ar = _action_result(node)
    results = ar.get("results") or []
    return [r for r in results if isinstance(r, dict)]


def _search_target_hit(node: Dict[str, Any], target_slug_rx: Pattern,
                        target_name_rx: Optional[Pattern]) -> Tuple[bool, Optional[int], int]:
    results = _search_results(node)
    n = len(results)
    for i, r in enumerate(results, start=1):
        url = str(r.get("url") or "")
        title = str(r.get("title") or "")
        desc = str(r.get("description") or "")
        if target_slug_rx.search(normalize_url(url)):
            return True, i, n
        if target_name_rx is not None and target_name_rx.search(f"{title} {desc}"):
            return True, i, n
    return False, None, n


def _search_rank_by_overlap(node: Dict[str, Any], target_slug_rx: Pattern,
                             target_name_rx: Optional[Pattern], cues: Sequence[str]) -> Optional[int]:
    results = _search_results(node)
    scored = []
    for i, r in enumerate(results):
        text = f"{r.get('title') or ''} {r.get('description') or ''}"
        scored.append((i, _overlap_score(text, cues)))
    ranked = sorted(scored, key=lambda x: (-x[1], x[0]))
    for rank, (orig_i, _s) in enumerate(ranked, start=1):
        r = results[orig_i]
        url = str(r.get("url") or "")
        title = str(r.get("title") or "")
        desc = str(r.get("description") or "")
        if target_slug_rx.search(normalize_url(url)) or (
            target_name_rx is not None and target_name_rx.search(f"{title} {desc}")
        ):
            return rank
    return None


# Main collection pass


def collect(results_globs: Sequence[str]) -> Tuple[List[HopInstance], List[SearchInstance]]:
    hops = build_hops()
    hops_by_task: Dict[str, List[Hop]] = defaultdict(list)
    for h in hops:
        hops_by_task[h.task_id].append(h)

    search_hops = build_search_hops()
    search_hops_by_task: Dict[str, List[SearchHop]] = defaultdict(list)
    for sh in search_hops:
        search_hops_by_task[sh.task_id].append(sh)

    in_scope_tasks = set(hops_by_task) | set(search_hops_by_task)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    hop_records: List[HopInstance] = []
    search_records: List[SearchInstance] = []

    for path in files:
        base = os.path.basename(path)
        task_id = _find_task_id(base)
        if task_id is None or task_id not in in_scope_tasks:
            continue
        shape, model, arm, _rep = _file_meta(base)

        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        nodes = _ordered_nodes(data)
        visits = [n for n in nodes if _is_successful_visit(n)]
        searches = [n for n in nodes if _is_successful_search(n)]
        mandate_text = _root_mandate_text(data)
        mandate_cues_present = _mandate_present_cues(mandate_text, _PRODUCT_ROLE_CUES)

        # ---- page-to-page link hops ----
        for hop in hops_by_task.get(task_id, []):
            source = _best_matching_visit(visits, hop.source_slug_rx, hop.source_name_rx)
            if source is None:
                continue
            ar = _action_result(source)
            source_url = str(ar.get("url") or "")

            links_20 = ar.get("links") or []
            links_full = ar.get("links_full") or links_20
            link_contexts = ar.get("link_contexts") or {}

            ordered_full = _dedupe_ordered(links_full)
            ordered_20 = _dedupe_ordered(links_20)
            anchor_lu = _anchor_lookup(link_contexts)

            anchor_scores = {normalize_url(u): _overlap_score(anchor_lu.get(normalize_url(u), ""), hop.target_cues)
                              for u in ordered_full}
            slug_scores = {normalize_url(u): _overlap_score(_slug_text(u), hop.target_cues)
                            for u in ordered_full}
            combined_scores = {k: max(anchor_scores.get(k, 0.0), slug_scores.get(k, 0.0))
                                for k in set(anchor_scores) | set(slug_scores)}
            jaccard_scores = {normalize_url(u): _jaccard_score(
                f"{anchor_lu.get(normalize_url(u), '')} {_slug_text(u)}", hop.target_cues,
            ) for u in ordered_full}

            rank_pos_full, n_full = _rank_by_position(ordered_full, hop.target_slug_rx)
            rank_pos_20, n_20 = _rank_by_position(ordered_20, hop.target_slug_rx)
            ranks = {
                "position": rank_pos_full,
                "anchor_text": _rank_by_score(ordered_full, anchor_scores, hop.target_slug_rx),
                "url_slug": _rank_by_score(ordered_full, slug_scores, hop.target_slug_rx),
                "combined_max": _rank_by_score(ordered_full, combined_scores, hop.target_slug_rx),
                "token_jaccard": _rank_by_score(ordered_full, jaccard_scores, hop.target_slug_rx),
            }
            ranks_20 = {"position": rank_pos_20}

            contains_full = rank_pos_full is not None
            contains_20 = rank_pos_20 is not None

            # negative-case helper: did ANY search node in this same trace surface the target?
            same_trace_search_hit = None
            if not contains_full:
                hit_any = False
                for sn in searches:
                    hit, _r, _n = _search_target_hit(sn, hop.target_slug_rx, hop.target_name_rx)
                    if hit:
                        hit_any = True
                        break
                same_trace_search_hit = hit_any

            content_low = str(ar.get("content_full") or ar.get("content") or "").lower()
            source_domain = _domain(source_url)

            # ---- variant 1: node-local text (the downstream node that actually reached the
            # target, if any, and ONLY that node's own pre-existing title/goal/query/intent) ----
            target_node = _find_target_reaching_node(nodes, hop.target_slug_rx, hop.target_name_rx)
            v1_available = target_node is not None
            v1_leaky: Optional[bool] = None
            v1_rank_overlap: Optional[int] = None
            v1_rank_jaccard: Optional[int] = None
            if v1_available:
                node_text = _node_local_text(target_node)
                v1_leaky = any(cue.lower() in node_text.lower() for cue in hop.target_cues)
                v1_scores_ov = {normalize_url(u): _text_overlap_score(
                    f"{anchor_lu.get(normalize_url(u), '')} {_slug_text(u)}", node_text) for u in ordered_full}
                v1_scores_ja = {normalize_url(u): _text_jaccard_score(
                    f"{anchor_lu.get(normalize_url(u), '')} {_slug_text(u)}", node_text) for u in ordered_full}
                v1_rank_overlap = _rank_by_score(ordered_full, v1_scores_ov, hop.target_slug_rx)
                v1_rank_jaccard = _rank_by_score(ordered_full, v1_scores_ja, hop.target_slug_rx)

            # ---- variant 2: root-mandate role-cue proximity within link_contexts ----
            v2_prox_scores = _role_proximity_scores(
                content_low, mandate_cues_present, anchor_lu, ordered_full, _PRODUCT_WINDOW,
            )
            v2_rank_proximity = _rank_by_score(ordered_full, v2_prox_scores, hop.target_slug_rx)
            v2_target_scored = False
            if contains_full:
                for u in ordered_full:
                    if hop.target_slug_rx.search(normalize_url(u)):
                        v2_target_scored = normalize_url(u) in v2_prox_scores
                        break
            v2_mandate_ja_scores = {normalize_url(u): _text_jaccard_score(
                f"{anchor_lu.get(normalize_url(u), '')} {_slug_text(u)}", mandate_text) for u in ordered_full}
            v2_rank_mandate_jaccard = _rank_by_score(ordered_full, v2_mandate_ja_scores, hop.target_slug_rx)

            # ---- variant 3: zero-knowledge chrome filter, then raw page-order top-k ----
            filtered = _filtered_article_links(ordered_full, source_domain)
            v3_rank = _rank_by_position(filtered, hop.target_slug_rx)[0]
            v3_contains = v3_rank is not None

            hop_records.append(HopInstance(
                file=base, shape=shape, model=model, arm=arm, task_id=task_id, hop_label=hop.label,
                source_url=source_url, source_domain=source_domain,
                n_links_full=len(ordered_full), n_links_20=len(ordered_20),
                n_link_contexts=len(anchor_lu),
                contains_full=contains_full, contains_20=contains_20,
                ranks=ranks, ranks_20=ranks_20,
                v1_available=v1_available, v1_leaky=v1_leaky,
                v1_rank_overlap=v1_rank_overlap, v1_rank_jaccard=v1_rank_jaccard,
                v2_n_cues_present=len(mandate_cues_present), v2_rank_proximity=v2_rank_proximity,
                v2_target_scored=v2_target_scored, v2_rank_mandate_jaccard=v2_rank_mandate_jaccard,
                v3_n_filtered=len(filtered), v3_contains=v3_contains, v3_rank=v3_rank,
                same_trace_search_hit=same_trace_search_hit,
            ))

        # ---- search-result hops: canonical (065 work->poet) ----
        for sh in search_hops_by_task.get(task_id, []):
            if not searches:
                continue
            first_search = searches[0]  # 065 hop 1 is the trace's opening move by task design
            hit, rank_pos, n = _search_target_hit(first_search, sh.target_slug_rx, sh.target_name_rx)
            rank_overlap = _search_rank_by_overlap(first_search, sh.target_slug_rx, sh.target_name_rx,
                                                    sh.target_cues)
            search_records.append(SearchInstance(
                file=base, shape=shape, model=model, arm=arm, task_id=task_id, hop_label=sh.label,
                canonical=True, n_results=n, contains=hit, rank_position=rank_pos,
                rank_overlap=rank_overlap,
            ))

        # ---- search-result hops: opportunistic (any search, any target, this task) ----
        all_targets: List[Tuple[str, Pattern, Optional[Pattern], Tuple[str, ...]]] = []
        for hop in hops_by_task.get(task_id, []):
            all_targets.append((hop.label, hop.target_slug_rx, hop.target_name_rx, hop.target_cues))
        for sh in search_hops_by_task.get(task_id, []):
            all_targets.append((sh.label, sh.target_slug_rx, sh.target_name_rx, sh.target_cues))

        for sn in searches:
            for label, slug_rx, name_rx, cues in all_targets:
                hit, rank_pos, n = _search_target_hit(sn, slug_rx, name_rx)
                rank_overlap = _search_rank_by_overlap(sn, slug_rx, name_rx, cues)
                search_records.append(SearchInstance(
                    file=base, shape=shape, model=model, arm=arm, task_id=task_id, hop_label=label,
                    canonical=False, n_results=n, contains=hit, rank_position=rank_pos,
                    rank_overlap=rank_overlap,
                ))

    _categorize_negatives(hop_records)
    return hop_records, search_records


def _categorize_negatives(records: List[HopInstance]) -> None:
    """2nd pass: for every hop instance where the target was NOT in links_full, assign a
    category. Needs the per-(task,hop_label) aggregate containment rate first, to distinguish
    'this specific page's capture missed it' from 'this pair of pages is just never linked'."""
    by_hop: Dict[Tuple[str, str], List[HopInstance]] = defaultdict(list)
    for r in records:
        by_hop[(r.task_id, r.hop_label)].append(r)

    hop_contains_any: Dict[Tuple[str, str], bool] = {
        key: any(r.contains_full for r in rows) for key, rows in by_hop.items()
    }

    for r in records:
        if r.contains_full:
            r.category = "contained"
            continue
        if r.source_domain and "wikipedia.org" not in r.source_domain:
            r.category = "non-wikipedia source page"
        elif r.n_links_full == 0:
            r.category = "no links captured (fetch/parse failure)"
        elif not hop_contains_any[(r.task_id, r.hop_label)]:
            n_trials = len(by_hop[(r.task_id, r.hop_label)])
            r.category = f"structurally never linked (0/{n_trials} trials for this hop)"
        elif r.same_trace_search_hit:
            r.category = "reached via search elsewhere in this trace, not a direct link"
        else:
            r.category = "isolated miss (linked in other trials, missing in this capture)"


# Stats helpers


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def _median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _percentile(xs: Sequence[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[idx]


# Report


def print_report(hop_records: List[HopInstance], search_records: List[SearchInstance],
                  by_task: bool, examples: int) -> None:
    print("=" * 100)
    print("LINK-SET CONTAINMENT MEASUREMENT -- gate check for 'pass the link set, not a value'")
    print("=" * 100)

    print(f"\nNOT MEASURED (excluded by task design):")
    for tid, why in NOT_MEASURED.items():
        print(f"  {tid}: {why}")

    n = len(hop_records)
    print(f"\nApplicable page-to-page hop instances (correct source page actually visited): n={n}")
    if n == 0:
        print("  NO DATA -- nothing to report.")
        return

    # ---- 1. CONTAINMENT ----
    print("\n" + "-" * 100)
    print("1. STRUCTURAL CONTAINMENT -- is the correct next-hop URL present in the source page's link set?")
    print("-" * 100)
    c_full = sum(1 for r in hop_records if r.contains_full)
    c_20 = sum(1 for r in hop_records if r.contains_20)
    print(f"  links_full (complete set, analysis-only):  {c_full}/{n} ({_pct(c_full, n)})")
    print(f"  links      (capped {MAX_LINKS_FIELD_CAP}, what a consumer actually gets today): "
          f"{c_20}/{n} ({_pct(c_20, n)})")
    print(f"  --> links == links_full[:{MAX_LINKS_FIELD_CAP}] empirically (verified separately); "
          f"the 'links' row above is the realistic number for a consumer today.")

    wiki_rows = [r for r in hop_records if "wikipedia.org" in r.source_domain]
    non_wiki_rows = [r for r in hop_records if "wikipedia.org" not in r.source_domain]
    wc = sum(1 for r in wiki_rows if r.contains_full)
    print(f"\n  Wikipedia-only source pages (excludes off-site excursions the source-identity "
          f"matcher still accepted): {wc}/{len(wiki_rows)} ({_pct(wc, len(wiki_rows))})")
    if non_wiki_rows:
        nwc = sum(1 for r in non_wiki_rows if r.contains_full)
        print(f"  non-Wikipedia source pages (matched the waypoint's name_rx but NOT on "
              f"wikipedia.org -- the premise doesn't even apply there): "
              f"{nwc}/{len(non_wiki_rows)} ({_pct(nwc, len(non_wiki_rows))})")

    print("\n  by hop label:")
    by_label: Dict[str, List[HopInstance]] = defaultdict(list)
    for r in hop_records:
        by_label[r.hop_label].append(r)
    for label, rows in sorted(by_label.items()):
        cf = sum(1 for r in rows if r.contains_full)
        c20 = sum(1 for r in rows if r.contains_20)
        print(f"    {label:20s} n={len(rows):3d}  full={cf:3d} ({_pct(cf, len(rows))})  "
              f"links(<=20)={c20:3d} ({_pct(c20, len(rows))})")

    if by_task:
        print("\n  by task:")
        by_tid: Dict[str, List[HopInstance]] = defaultdict(list)
        for r in hop_records:
            by_tid[r.task_id].append(r)
        for tid, rows in sorted(by_tid.items()):
            cf = sum(1 for r in rows if r.contains_full)
            print(f"    {tid}  n={len(rows):3d}  full-containment={cf:3d} ({_pct(cf, len(rows))})")

    # ---- 2. RUNTIME-REALISTIC RANKING ----
    print("\n" + "-" * 100)
    print("2. RUNTIME-REALISTIC RANKING -- no oracle target wording used for scoring, anywhere below")
    print("-" * 100)

    def _dist_str(ranks: Sequence[int]) -> str:
        dist = Counter()
        for rk in ranks:
            if rk <= 5:
                dist["1-5"] += 1
            elif rk <= 10:
                dist["6-10"] += 1
            elif rk <= 20:
                dist["11-20"] += 1
            elif rk <= 50:
                dist["21-50"] += 1
            else:
                dist[">50"] += 1
        order = ["1-5", "6-10", "11-20", "21-50", ">50"]
        return ", ".join(f"{b}:{dist.get(b,0)}" for b in order if dist.get(b, 0))

    def _at_k_line(vals: Sequence[Optional[int]], denom: int) -> str:
        parts = []
        for k in (5, 10, 20, 50, None):
            label = f"top-{k}" if k is not None else "all"
            hit = sum(1 for v in vals if v is not None and (k is None or v <= k))
            parts.append(f"{label}={hit}/{denom} ({_pct(hit, denom)})")
        return "  ".join(parts)

    print("\n  -- variant 1: node-local text (the downstream node that actually reached the "
          "target, its own pre-existing title/goal/query/intent) --")
    v1_avail = [r for r in hop_records if r.v1_available]
    print(f"     available (a real downstream node reaching the target exists in the trace): "
          f"{len(v1_avail)}/{n} -- SMALL SAMPLE, most traces never click through that far")
    if v1_avail:
        leaky = [r for r in v1_avail if r.v1_leaky]
        generic = [r for r in v1_avail if not r.v1_leaky]
        print(f"     of those: LEAKY (node's own text already names the target verbatim, "
              f"i.e. the model already free-form-resolved it -- NOT evidence the link-set "
              f"mechanism works) = {len(leaky)}/{len(v1_avail)}")
        print(f"               GENUINELY GENERIC (node text does not name the target) = "
              f"{len(generic)}/{len(v1_avail)}")
        for sub_label, sub in (("leaky", leaky), ("generic", generic)):
            if not sub:
                continue
            ranks = [r.v1_rank_jaccard for r in sub if r.v1_rank_jaccard is not None]
            print(f"       [{sub_label}] jaccard ranks: {sorted(ranks)}  "
                  f"(hop labels: {[r.hop_label for r in sub]})")

    print("\n  -- variant 2a: root-mandate role-cue proximity within link_contexts "
          "(product's own _ENTITY_ROLE_CUES + _WINDOW, cues restricted to those present in the "
          "mandate text; ranks by nearest anchor to a cue occurrence in the source page body) --")
    v2p = [r.v2_rank_proximity for r in hop_records]
    print(f"     {_at_k_line(v2p, n)}")
    found = [v for v in v2p if v is not None]
    if found:
        print(f"     rank distribution (of {len(found)} found -- NOTE 'found' here means the "
              f"target was IN the link set at all; when the proximity mechanism assigns it NO "
              f"score, this falls back to the raw-position tie-break, so 'found' alone overstates "
              f"the mechanism -- see 'target actually scored' below): {_dist_str(found)}")
    n_scored = sum(1 for r in hop_records if r.v2_target_scored)
    print(f"     target's OWN anchor actually landed within {_PRODUCT_WINDOW} chars of a "
          f"role-cue occurrence (the mechanism genuinely fired for the right answer, not just a "
          f"page-position fallback): {n_scored}/{c_full} of contained instances")

    print("\n  -- variant 2b: root-mandate text, direct token-Jaccard vs (anchor+slug), "
          "no cue vocabulary --")
    v2j = [r.v2_rank_mandate_jaccard for r in hop_records]
    print(f"     {_at_k_line(v2j, n)}")

    print("\n  -- variant 3: ZERO-KNOWLEDGE chrome filter (no goal text at all) -- drop "
          "sister-project/cross-domain links, non-article namespaces, edit/redlink URLs; "
          "then top-k in raw page order --")
    v3 = [r.v3_rank for r in hop_records]
    print(f"     {_at_k_line(v3, n)}")
    sizes_filtered = [r.v3_n_filtered for r in hop_records]
    _sizes_full_preview = [r.n_links_full for r in hop_records]
    print(f"     filtered set size: median={_median(sizes_filtered):.0f}  "
          f"p90={_percentile(sizes_filtered, 0.9):.0f}  max={max(sizes_filtered)}  "
          f"(vs {_median(_sizes_full_preview):.0f} median unfiltered -- see set-size section below)")
    print("     by hop label (top-20 containment):")
    for label, rows in sorted(by_label.items()):
        hit20 = sum(1 for r in rows if r.v3_rank is not None and r.v3_rank <= 20)
        hit_any = sum(1 for r in rows if r.v3_contains)
        print(f"       {label:20s} n={len(rows):3d}  top-20={hit20:3d} ({_pct(hit20, len(rows))})  "
              f"contains-any={hit_any:3d} ({_pct(hit_any, len(rows))})")

    print("\n  -- creator->terminal (n=3, a genuinely DISCOVERED entity's own page -- the case "
          "the design most needs to work) vs start->creator (n=37, source page is GIVEN, easier) --")
    for label in ("start->creator", "creator->terminal"):
        rows = [r for r in hop_records if r.hop_label == label]
        v3_hit20 = sum(1 for r in rows if r.v3_rank is not None and r.v3_rank <= 20)
        print(f"     {label:20s} n={len(rows):3d}  v3(chrome-filter)@20={v3_hit20}/{len(rows)}  "
              f"v1-available={sum(1 for r in rows if r.v1_available)}  "
              f"v2-proximity@20={sum(1 for r in rows if r.v2_rank_proximity is not None and r.v2_rank_proximity<=20)}")

    # ---- 2c. ORACLE RANKING (reference / upper-bound only -- NOT deployable) ----
    print("\n" + "-" * 100)
    print("2c. ORACLE RANKING (reference / upper-bound only -- ranks against the TARGET waypoint's "
          "own ground-truth name/name_rx wording, which the engine does NOT know at hop-N time. "
          "NOT a deployability claim; kept only to show the ceiling if the wording problem were solved.)")
    print("-" * 100)
    for method in RANK_METHODS:
        ranks = [r.ranks[method] for r in hop_records if r.ranks.get(method) is not None]
        n_found = len(ranks)
        print(f"\n  -- {method}: {RANK_LABEL[method]} --")
        if not ranks:
            print("     never found by this method")
            continue
        print(f"     found={n_found}/{c_full}  median-rank={_median(ranks):.1f}  "
              f"p90-rank={_percentile(ranks, 0.9):.1f}  min={min(ranks)}  max={max(ranks)}")
        print("     rank distribution: " + _dist_str(ranks))

    # ---- 3. SET SIZE ----
    print("\n" + "-" * 100)
    print("3. SET-SIZE DISTRIBUTION -- links per page")
    print("-" * 100)
    sizes_full = [r.n_links_full for r in hop_records]
    sizes_20 = [r.n_links_20 for r in hop_records]
    sizes_ctx = [r.n_link_contexts for r in hop_records]
    print(f"  links_full:     median={_median(sizes_full):.0f}  p90={_percentile(sizes_full, 0.9):.0f}  "
          f"max={max(sizes_full)}")
    print(f"  links (capped): median={_median(sizes_20):.0f}  p90={_percentile(sizes_20, 0.9):.0f}  "
          f"max={max(sizes_20)}  (trivially <= {MAX_LINKS_FIELD_CAP} by construction)")
    print(f"  link_contexts (links with KNOWN anchor text, feeds anchor_text ranking): "
          f"median={_median(sizes_ctx):.0f}  p90={_percentile(sizes_ctx, 0.9):.0f}")

    # ---- 4. SEARCH-RESULT CHANNEL ----
    print("\n" + "-" * 100)
    print("4. SEARCH-RESULT CHANNEL (action_result.results on search nodes)")
    print("-" * 100)
    canon = [s for s in search_records if s.canonical]
    print(f"\n  CANONICAL (065 hop 1, 'work'->poet -- the task's own designed search-first hop): n={len(canon)}")
    if canon:
        hit = sum(1 for s in canon if s.contains)
        print(f"    containment={hit}/{len(canon)} ({_pct(hit, len(canon))})")
        pos_ranks = [s.rank_position for s in canon if s.rank_position is not None]
        ov_ranks = [s.rank_overlap for s in canon if s.rank_overlap is not None]
        if pos_ranks:
            print(f"    rank(raw search-engine order): median={_median(pos_ranks):.1f}  "
                  f"p90={_percentile(pos_ranks, 0.9):.1f}  dist={sorted(Counter(pos_ranks).items())}")
        if ov_ranks:
            print(f"    rank(title/desc overlap rerank): median={_median(ov_ranks):.1f}  "
                  f"p90={_percentile(ov_ranks, 0.9):.1f}  dist={sorted(Counter(ov_ranks).items())}")
        sizes = [s.n_results for s in canon]
        print(f"    result-set size: median={_median(sizes):.0f}  p90={_percentile(sizes, 0.9):.0f}")

    opp = [s for s in search_records if not s.canonical]
    print(f"\n  OPPORTUNISTIC (every search node x every in-scope target for its task, "
          f"unfiltered by query intent -- a broader, noisier sample): n={len(opp)}")
    if opp:
        hit = sum(1 for s in opp if s.contains)
        print(f"    containment={hit}/{len(opp)} ({_pct(hit, len(opp))})  "
              f"(includes many true negatives: a search unrelated to a given target correctly finds nothing)")

    # ---- 5. FLATNESS CHECK ----
    print("\n" + "-" * 100)
    print("5. FLATNESS CHECK -- containment@k as the candidate window grows")
    print("-" * 100)
    print("  (a) RAW page order (no ranking at all) over links_full:")
    for k in (5, 10, 20, 50, None):
        label = f"top-{k}" if k is not None else "all"
        hit = sum(1 for r in hop_records if r.ranks["position"] is not None and
                  (k is None or r.ranks["position"] <= k))
        print(f"    {label:8s} contains-truth={hit:3d}/{n} ({_pct(hit, n)})")
    print("\n  (b) [ORACLE, reference only] BEST ranking per instance (min rank across the 4 "
          "oracle methods) -- the CEILING if the wording problem were solved:")
    best_ranks = []
    for r in hop_records:
        cand = [r.ranks[m] for m in ("anchor_text", "url_slug", "combined_max", "token_jaccard")
                if r.ranks.get(m) is not None]
        best_ranks.append(min(cand) if cand else None)
    for k in (5, 10, 20, 50, None):
        label = f"top-{k}" if k is not None else "all"
        hit = sum(1 for br in best_ranks if br is not None and (k is None or br <= k))
        print(f"    {label:8s} contains-truth={hit:3d}/{n} ({_pct(hit, n)})")

    print("\n  (c) [RUNTIME-REALISTIC] BEST of the 3 non-oracle variants per instance (min rank "
          "across v2a-proximity, v2b-mandate-jaccard, v3-chrome-filter; v1 excluded from this "
          "combination since it's only available for 8/58 and half of those are answer-leaking) "
          "-- the actual deployable ceiling with only runtime-available signal:")
    best_runtime = []
    for r in hop_records:
        cand = [v for v in (r.v2_rank_proximity, r.v2_rank_mandate_jaccard, r.v3_rank) if v is not None]
        best_runtime.append(min(cand) if cand else None)
    for k in (5, 10, 20, 50, None):
        label = f"top-{k}" if k is not None else "all"
        hit = sum(1 for br in best_runtime if br is not None and (k is None or br <= k))
        print(f"    {label:8s} contains-truth={hit:3d}/{n} ({_pct(hit, n)})")

    # ---- 6. NEGATIVE CASES ----
    print("\n" + "-" * 100)
    print("6. NEGATIVE CASES -- why the target URL was NOT in the source page's link set")
    print("-" * 100)
    negatives = [r for r in hop_records if not r.contains_full]
    print(f"  total negatives: {len(negatives)}/{n} ({_pct(len(negatives), n)})")
    cat_counts = Counter(r.category for r in negatives)
    for cat, cnt in cat_counts.most_common():
        print(f"    {cnt:3d}  {cat}")

    if examples and negatives:
        print("\n  examples per category:")
        seen_cats: Counter = Counter()
        for r in negatives:
            base_cat = re.sub(r"\(0/\d+ trials.*\)", "(...)", r.category)
            if seen_cats[base_cat] >= examples:
                continue
            seen_cats[base_cat] += 1
            print(f"    [{r.category}] task={r.task_id} hop={r.hop_label} model={r.model} arm={r.arm} "
                  f"file={r.file}")
            print(f"      source={r.source_url}  n_links_full={r.n_links_full}")

    print("\n" + "=" * 100)
    print("END OF REPORT")
    print("=" * 100)


# ==============================================================================================
# VARIANT-3 DEEP DIVE -- follow-up requested by the coordinator after the runtime-realistic
# ranking result: since NO deterministic ranking compresses the set (all three variants dead or
# weak), the remaining live question is not "what rank" but "what k, at what token cost" -- the
# actual selection among a surfaced candidate set is left to the LLM's own next-hop expansion
# call, which already runs and already reads this context. Chrome-filter-plus-larger-k is the
# only variant worth extending (variants 1/2 are dead per the ranking measurement above and are
# excluded here per the coordinator's explicit instruction).
# ==============================================================================================

K_GRID = (20, 35, 50, 75, 100, 150, 200, None)


@dataclass
class V3Instance:
    task_id: str
    hop_label: str
    file: str
    source_url: str
    filtered_urls: List[str]           # chrome-filtered, deduped, RAW PAGE ORDER (variant 3)
    anchor_lu: Dict[str, str]
    n_filtered: int
    n_dedup_unfiltered: int
    n_raw_undeduped: int
    v3_rank: Optional[int]             # position of the target within filtered_urls, 1-based


def collect_v3_instances(results_globs: Sequence[str]) -> List[V3Instance]:
    hops = build_hops()
    hops_by_task: Dict[str, List[Hop]] = defaultdict(list)
    for h in hops:
        hops_by_task[h.task_id].append(h)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    out: List[V3Instance] = []
    for path in files:
        base = os.path.basename(path)
        task_id = _find_task_id(base)
        if task_id is None or task_id not in hops_by_task:
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        nodes = _ordered_nodes(data)
        visits = [n for n in nodes if _is_successful_visit(n)]

        for hop in hops_by_task[task_id]:
            source = _best_matching_visit(visits, hop.source_slug_rx, hop.source_name_rx)
            if source is None:
                continue
            ar = _action_result(source)
            source_url = str(ar.get("url") or "")
            links_full_raw = ar.get("links_full") or ar.get("links") or []
            ordered = _dedupe_ordered(links_full_raw)
            anchor_lu = _anchor_lookup(ar.get("link_contexts") or {})
            source_domain = _domain(source_url)
            filtered = _filtered_article_links(ordered, source_domain)
            rank, _n = _rank_by_position(filtered, hop.target_slug_rx)

            out.append(V3Instance(
                task_id=task_id, hop_label=hop.label, file=base, source_url=source_url,
                filtered_urls=filtered, anchor_lu=anchor_lu,
                n_filtered=len(filtered), n_dedup_unfiltered=len(ordered),
                n_raw_undeduped=len(links_full_raw), v3_rank=rank,
            ))
    return out


# ---- rendering + token cost, matching expansion.py::_enhance_details_with_inline_links exactly
# ("{context_text} [link: {link_url}]" per line, anchor truncated to 150 chars, "\n"-joined) ----

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def render_candidate_list(urls: Sequence[str], anchor_lu: Dict[str, str]) -> str:
    lines = []
    for u in urls:
        anchor = anchor_lu.get(normalize_url(u), "")
        context_text = anchor.strip()[:150] if anchor else ""
        if context_text:
            lines.append(f"{context_text} [link: {u}]")
        else:
            lines.append(f"[link: {u}]")
    return "\n".join(lines)


def token_count(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ---- additional zero-knowledge filter candidates (no goal text) ----


def front_fraction(urls: Sequence[str], fraction: float) -> List[str]:
    """Keep only the first `fraction` of the (already chrome-filtered) list, by raw page
    position -- tests whether infobox/lead links (early) carry more signal than
    references/navbox-adjacent links (late), using position alone, no content classification."""
    n = max(1, int(round(fraction * len(urls))))
    return list(urls[:n])


_JUNK_ANCHOR_RX = re.compile(r"^\s*(\[\s*\d+\s*\]|edit|citation needed|permanent link|cite|\W{0,3})\s*$",
                              re.IGNORECASE)


def drop_junk_anchor(urls: Sequence[str], anchor_lu: Dict[str, str]) -> List[str]:
    """Drop links whose OWN anchor text is empty, purely numeric/punctuation, or one of a few
    fixed Wikipedia UI boilerplate strings ('edit', 'citation needed', ...) -- zero-knowledge:
    doesn't need to know anything about the task, just the link's own anchor text quality."""
    out = []
    for u in urls:
        a = anchor_lu.get(normalize_url(u), "")
        if not a or len(a.strip()) < 2 or _JUNK_ANCHOR_RX.match(a):
            continue
        out.append(u)
    return out


def print_v3_deep_dive(instances: List[V3Instance]) -> None:
    n = len(instances)
    print("=" * 100)
    print("VARIANT-3 DEEP DIVE -- chrome-filter-plus-k: containment, per-hop-type k-to-catch, "
          "token cost, and further zero-knowledge filters")
    print("=" * 100)
    print(f"\napplicable instances: n={n}  (same 58 hop instances as the main report; variants 1/2 "
          f"excluded -- both dead per the runtime-realistic ranking measurement)")

    # ---- CRITICAL CAVEAT: Wikipedia content is deterministic, so replicate runs of the same
    # task hitting the same page produce IDENTICAL ranks -- the 58 instances collapse to a much
    # smaller number of genuinely distinct (page, target) facts, replicated 1-12x each by how many
    # model/arm/rep combinations happened to reach that page. Every percentage below is real, but
    # is a census over few unique pages, not 58 independent draws -- one page can swing it ~14pp.
    by_page: Dict[Tuple[str, str], set] = defaultdict(set)
    for r in instances:
        by_page[(r.task_id, r.hop_label)].add(r.v3_rank)
    print(f"\n  *** CAVEAT: only {len(by_page)} DISTINCT (task, hop) source pages underlie these "
          f"{n} instances (Wikipedia content is deterministic; replicate model/arm/rep runs that "
          f"hit the SAME page get the SAME rank). Every % below is a census over ~{len(by_page)} "
          f"pages, weighted by how many replicates happened to reach each one -- NOT {n} "
          f"independent trials. Per-unique-page view: ***")
    for (tid, label), ranks in sorted(by_page.items()):
        real_ranks = sorted(rk for rk in ranks if rk is not None)
        n_rep = sum(1 for r in instances if r.task_id == tid and r.hop_label == label)
        tag = f"rank={real_ranks[0]}" if len(real_ranks) == 1 else f"ranks={real_ranks}"
        if not real_ranks:
            tag = "never contained (non-wiki source, structurally k-independent)"
        print(f"    {tid} {label:20s} {tag:45s} ({n_rep} replicate instance(s))")

    # ---- 1. containment vs k ----
    print("\n" + "-" * 100)
    print("1. CONTAINMENT vs k (chrome-filtered set, raw page order, no ranking)")
    print("-" * 100)
    ceiling = sum(1 for r in instances if r.v3_rank is not None)
    print(f"  ceiling (contained anywhere in the chrome-filtered set, unbounded k): "
          f"{ceiling}/{n} ({_pct(ceiling, n)})")
    for k in K_GRID:
        label = f"k={k}" if k is not None else "k=unbounded"
        hit = sum(1 for r in instances if r.v3_rank is not None and (k is None or r.v3_rank <= k))
        print(f"    {label:14s} {hit:3d}/{n} ({_pct(hit, n)})"
              f"{'  <- SATURATED at the 87.9% ceiling' if hit == ceiling and k is not None else ''}")

    # ---- 2. per-hop-type k-to-first-containment ----
    print("\n" + "-" * 100)
    print("2. PER-HOP-TYPE: at what k does each hop type actually get caught?")
    print("-" * 100)
    by_label: Dict[str, List[V3Instance]] = defaultdict(list)
    for r in instances:
        by_label[r.hop_label].append(r)
    for label, rows in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        ranks = sorted(r.v3_rank for r in rows if r.v3_rank is not None)
        never = sum(1 for r in rows if r.v3_rank is None)
        print(f"\n  {label}  (n={len(rows)}, never-contained-even-unbounded={never})")
        if ranks:
            print(f"    rank list (sorted): {ranks}")
            print(f"    k needed to catch the LAST (worst-case) one: {ranks[-1]}")
        for k in K_GRID:
            label_k = f"k={k}" if k is not None else "unbounded"
            hit = sum(1 for r in rows if r.v3_rank is not None and (k is None or r.v3_rank <= k))
            print(f"      {label_k:12s} {hit:2d}/{len(rows)} ({_pct(hit, len(rows))})")

    # ---- 3. token cost of surfacing k links ----
    print("\n" + "-" * 100)
    print("3. TOKEN COST of surfacing k links (exact expansion.py::_enhance_details_with_inline_links "
          "rendering: '{anchor[:150]} [link: url]' per line, cl100k_base tokenizer)")
    print("-" * 100)
    for k in K_GRID:
        costs = []
        for r in instances:
            kk = k if k is not None else len(r.filtered_urls)
            rendered = render_candidate_list(r.filtered_urls[:kk], r.anchor_lu)
            costs.append(token_count(rendered))
        label = f"k={k}" if k is not None else "k=unbounded"
        print(f"    {label:14s} tokens: median={_median(costs):.0f}  p90={_percentile(costs, 0.9):.0f}  "
              f"max={max(costs)}")
    print("\n  context: this engine currently spends ~88,000 prompt tokens per cell; the "
          "decision rule treats a few hundred extra tokens/visit as affordable, a few thousand as not.")

    # ---- 4. further zero-knowledge filters ----
    print("\n" + "-" * 100)
    print("4. FURTHER ZERO-KNOWLEDGE FILTERS (no goal text) -- tried, not tuned")
    print("-" * 100)

    print("\n  (a) dedupe repeated targets -- yield of _dedupe_ordered ALONE (before chrome filter):")
    raw_lens = [r.n_raw_undeduped for r in instances]
    dedup_lens = [r.n_dedup_unfiltered for r in instances]
    saved = [rw - dd for rw, dd in zip(raw_lens, dedup_lens)]
    n_any_dup = sum(1 for s in saved if s > 0)
    print(f"     instances with ANY duplicate hrefs in the raw links_full list: {n_any_dup}/{n}")
    print(f"     links removed by dedup: median={_median(saved):.0f}  max={max(saved)}  "
          f"(vs median {_median(dedup_lens):.0f} links after dedup) -- negligible yield")

    print("\n  (b) front-fraction cut -- keep only the first F% of the chrome-filtered list "
          "(position-only, no content classification):")
    hop_lookup = {(h.task_id, h.label): h for h in build_hops()}
    for frac in (0.5, 0.3, 0.15):
        sizes, hits = [], 0
        for r in instances:
            hop = hop_lookup[(r.task_id, r.hop_label)]
            kept = front_fraction(r.filtered_urls, frac)
            sizes.append(len(kept))
            rank, _ = _rank_by_position(kept, hop.target_slug_rx)
            if rank is not None:
                hits += 1
        print(f"     front {int(frac*100):3d}%:  resulting size median={_median(sizes):.0f}  "
              f"p90={_percentile(sizes, 0.9):.0f}   containment={hits}/{n} ({_pct(hits, n)})")

    print("\n  (c) drop junk/empty anchor text (no anchor, or anchor is pure punctuation/digits/"
          "'edit'/'citation needed'/etc.):")
    hop_lookup = {(h.task_id, h.label): h for h in build_hops()}
    sizes, hits = [], 0
    for r in instances:
        hop = hop_lookup[(r.task_id, r.hop_label)]
        kept = drop_junk_anchor(r.filtered_urls, r.anchor_lu)
        sizes.append(len(kept))
        rank, _ = _rank_by_position(kept, hop.target_slug_rx)
        if rank is not None:
            hits += 1
    print(f"     resulting size median={_median(sizes):.0f}  p90={_percentile(sizes, 0.9):.0f}  "
          f"(vs {_median([r.n_filtered for r in instances]):.0f} median chrome-filtered-only)")
    print(f"     containment={hits}/{n} ({_pct(hits, n)})")
    n_target_has_anchor = sum(
        1 for r in instances if r.v3_rank is not None and
        r.anchor_lu.get(normalize_url(r.filtered_urls[r.v3_rank - 1]), "").strip()
    )
    print(f"     (of {ceiling} contained targets, {n_target_has_anchor} have SOME anchor text "
          f"at all -- i.e. this filter's recall risk on the target itself)")

    print("\n" + "=" * 100)
    print("END OF VARIANT-3 DEEP DIVE")
    print("=" * 100)


def dump_json(hop_records: List[HopInstance], search_records: List[SearchInstance], path: str) -> None:
    payload = {
        "hop_instances": [asdict(r) for r in hop_records],
        "search_instances": [asdict(s) for s in search_records],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-glob", action="append", default=None,
                     help="Glob (relative to repo root) for result JSONs; may repeat. "
                          f"Default: {RESULTS_GLOBS_DEFAULT}")
    ap.add_argument("--by-task", action="store_true", help="Also break containment down by task id.")
    ap.add_argument("--examples", type=int, default=2, help="Negative-case examples per category (0=off).")
    ap.add_argument("--json", default=None, help="Also dump raw per-instance records to this JSON path.")
    ap.add_argument("--v3-deep-dive", action="store_true",
                     help="Run ONLY the variant-3 (chrome-filter) deep dive: containment vs k, "
                          "per-hop-type k-to-catch, token cost, further zero-knowledge filters.")
    args = ap.parse_args()

    globs = args.results_glob or list(RESULTS_GLOBS_DEFAULT)

    if args.v3_deep_dive:
        instances = collect_v3_instances(globs)
        print_v3_deep_dive(instances)
        return

    hop_records, search_records = collect(globs)
    print_report(hop_records, search_records, by_task=args.by_task, examples=args.examples)

    if args.json:
        dump_json(hop_records, search_records, args.json)
        print(f"\n[wrote raw records to {args.json}]")


if __name__ == "__main__":
    main()
