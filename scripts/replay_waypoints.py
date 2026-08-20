#!/usr/bin/env python3
"""
Read-only measurement: how often is the VALUE that hop N discovered (and hop N+1 needs)
deterministically recoverable from the page content hop N actually fetched, WITHOUT an LLM
call?

Answers the design question for "value threading" on sequential multi-hop chain tasks. Today
the engine passes hop N+1 only visit *metadata* ("Visited https://... page='X'. 21066 chars,
340 links") -- never the datum a downstream hop needs (a name, a place, a number). This script
replays saved chain-task results, locates -- for every waypoint of every chain task -- the node
that actually fetched the page the value should live on, and tests several deterministic
(non-LLM) extraction methods against the task's own ground-truth regexes.

Data: agent/idea_test_results/cschain_g_*.json and csnopar_g_*.json (9 chain tasks x 3 models x
2 arms = 108 files). Ground truth: the KEYSTONE_*/CHAIN/HOP_* constants already authored in each
agent/app/idea_tests/test_{135,136,137,138,139,046,047,065,093}_*.py module (imported directly,
not retyped). Test 047 (open-ended wiki-race path-finding, no fixed intermediate ground truth)
is intentionally excluded -- see NOT_MEASURED below.

Extractor prototypes here are built ON TOP OF agent/app/idea_policies/contract_satisfaction.py
(derive_step_contract, _number_near, _QUANTITY_CUES) via a tiny NodeView shim, but that module
is never modified -- this script is read-only w.r.t. product code.

``--precision`` measures the OTHER half of the question the recall pass above cannot answer:
what either extractor does when pointed at a page that is NOT a waypoint's correct source (the
real risk at the production write site) -- using the same blind, ground-truth-agnostic
prototypes. It gated the design of ``agent/app/idea_policies/waypoint.py``'s page-identity
guard. ``--validate-product`` then re-runs BOTH the recall and precision questions a third time,
against the actual shipped ``waypoint.build_waypoint`` (not a prototype) -- the strongest form
of validation, and confirms the guard the precision numbers demanded is doing its job.

Usage (from repo root):
    ./.venv/bin/python3 scripts/replay_waypoints.py --by-hop-type
    ./.venv/bin/python3 scripts/replay_waypoints.py --by-hop-type --by-task --examples 2
    ./.venv/bin/python3 scripts/replay_waypoints.py --precision --examples 3
    ./.venv/bin/python3 scripts/replay_waypoints.py --validate-product
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "services", REPO_ROOT / "agent"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from agent.app.idea_policies.contract_satisfaction import (  # noqa: E402
    _number_near,
    derive_step_contract,
)
from agent.app.idea_test_utils import normalize_url  # noqa: E402

NOT_MEASURED = {
    "047": (
        "open-ended wiki-race path-finding -- any valid START->TARGET hyperlink chain scores, "
        "so there is no fixed sequence of intermediate VALUES to check extraction against "
        "(the 'value' at each hop is just 'does a link exist', which is definitionally what "
        "method 4 (link_contexts) checks -- testing it here would be circular)."
    ),
}

WINDOW = 200        # chars either side of a cue word, matches contract_satisfaction._NUMBER_WINDOW
LEAD_CHARS = 500     # "first sentence(s) of page" heuristic window


# Waypoint registry -- one entry per (task, hop-that-must-be-extracted), built from each task
# module's OWN ground-truth constants (CHAIN / KEYSTONE_* / HOP_*), not retyped by hand.


@dataclass
class Waypoint:
    task_id: str
    key: str
    hop_type: str  # "quantity" | "entity" | "place" | "date" | "other"
    value_rx: Pattern
    source_kind: str  # "visit" | "search_first"
    source_matchers: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    cue_variants: Tuple[str, ...] = ()
    desc: str = ""


def _rx(pattern) -> Pattern:
    """Accept either an already-compiled Pattern (reuse as-is) or a raw regex string."""
    if isinstance(pattern, re.Pattern):
        return pattern
    return re.compile(pattern, re.IGNORECASE)


def _load_task_module(test_id: str):
    matches = glob.glob(str(REPO_ROOT / "agent" / "app" / "idea_tests" / f"test_{test_id}_*.py"))
    if not matches:
        raise FileNotFoundError(f"no task module for test_id={test_id}")
    mod_name = "agent.app.idea_tests." + os.path.basename(matches[0])[:-3]
    return importlib.import_module(mod_name)


def _visit_source(*, slug_rx: str = "", name_rx: str = "") -> Callable[[Dict[str, Any]], bool]:
    """A visit node matches a chain waypoint's PAGE when its URL slug or its h1/title text
    matches that waypoint's own regex (both authored, ground-truth regexes from the task)."""

    def matcher(node: Dict[str, Any]) -> bool:
        ar = (node.get("details") or {}).get("action_result") or {}
        url = normalize_url(ar.get("url") or "")
        ident = f"{ar.get('h1_text') or ''} {ar.get('page_title') or ''}"
        if slug_rx and re.search(slug_rx, url, re.IGNORECASE):
            return True
        if name_rx and re.search(name_rx, ident, re.IGNORECASE):
            return True
        return False

    return matcher


def _visit_source_url(target_url: str) -> Callable[[Dict[str, Any]], bool]:
    target = normalize_url(target_url)

    def matcher(node: Dict[str, Any]) -> bool:
        ar = (node.get("details") or {}).get("action_result") or {}
        return normalize_url(ar.get("url") or "") == target

    return matcher


def _standard_chain_waypoints(test_id: str, quantity_cues: Tuple[str, ...]) -> List[Waypoint]:
    """The 4 identically-shaped tasks (135/136/137/139): start(given) -> creator(entity) ->
    terminal(place/artifact) -> keystone(quantity). Each waypoint's value is what the PREVIOUS
    hop's page must reveal."""
    mod = _load_task_module(test_id)
    start, creator, terminal = mod.CHAIN
    return [
        Waypoint(
            test_id, "creator", "entity", _rx(creator["name_rx"]),
            "visit", [_visit_source(slug_rx=start["slug_rx"], name_rx=start["name_rx"])],
            desc=f"{creator['name']} -- discoverable on the {start['name']} page",
        ),
        Waypoint(
            test_id, "terminal", "place", _rx(terminal["name_rx"]),
            "visit", [_visit_source(slug_rx=creator["slug_rx"], name_rx=creator["name_rx"])],
            desc=f"{terminal['name']} -- discoverable on the {creator['name']} page",
        ),
        Waypoint(
            test_id, "keystone", "quantity", mod.KEYSTONE_RX,
            "visit", [_visit_source(slug_rx=terminal["slug_rx"], name_rx=terminal["name_rx"])],
            cue_variants=quantity_cues,
            desc=f"keystone quantity -- discoverable on the {terminal['name']} page",
        ),
    ]


def _waypoints_138() -> List[Waypoint]:
    """138 deviates from the template: the 'terminal' is a fact (the 1856 declared survey), not
    a distinct page -- ground truth says it may be confirmed on either Waugh's page or the
    George Everest page, so the keystone source is deliberately OR'd across both candidates."""
    mod = _load_task_module("138")
    start, creator, terminal = mod.CHAIN
    keystone_src = [
        _visit_source(slug_rx=terminal["slug_rx"], name_rx=terminal["name_rx"]),
        _visit_source(slug_rx=creator["slug_rx"], name_rx=creator["name_rx"]),
    ]
    return [
        Waypoint(
            "138", "creator", "entity", _rx(creator["name_rx"]),
            "visit", [_visit_source(slug_rx=start["slug_rx"], name_rx=start["name_rx"])],
            desc=f"{creator['name']} -- discoverable on the Mount Everest page",
        ),
        Waypoint(
            "138", "announce_year", "date", re.compile(r"\b1856\b"),
            "visit", [
                _visit_source(slug_rx=start["slug_rx"], name_rx=start["name_rx"]),
                _visit_source(slug_rx=creator["slug_rx"], name_rx=creator["name_rx"]),
            ],
            cue_variants=("survey", "announced", "declared", "trigonometrical"),
            desc="1856 survey/announcement year -- discoverable on the Everest or Waugh page",
        ),
        Waypoint(
            "138", "keystone", "quantity", mod.KEYSTONE_RX,
            "visit", keystone_src,
            cue_variants=("height", "declared", "feet", "announced"),
            desc="1856 declared height (29,002 ft) -- Waugh or George Everest page",
        ),
    ]


def _waypoints_065() -> List[Waypoint]:
    """URL-free chain: hop 1 (poet) is SEARCH-sourced, not a fixed given page -- this is the
    exact example the task brief itself uses ('hop 1 finds Pablo Neruda')."""
    mod = _load_task_module("065")
    return [
        Waypoint(
            "065", "poet", "entity", _rx(mod.HOP_POET),
            "search_first", [],
            desc="poet (Pablo Neruda) -- discoverable in the first search's results",
        ),
        Waypoint(
            "065", "town", "place", _rx(mod.HOP_TOWN),
            "visit", [_visit_source(slug_rx=mod.CITE_POET)],
            cue_variants=("born", "birthplace", "birth"),
            desc="birthplace town (Parral) -- discoverable on the Pablo Neruda page",
        ),
        Waypoint(
            "065", "elevation", "quantity", _rx(mod.KEYSTONE_ELEVATION),
            "visit", [_visit_source(slug_rx=mod.CITE_TOWN)],
            cue_variants=("elevation", "altitude", "above sea level"),
            desc="town elevation (162 m) -- discoverable on the Parral, Chile page",
        ),
    ]


def _waypoints_046() -> List[Waypoint]:
    """Given a fixed START_URL; hop 1 is a pure hyperlink-follow (Apollo 11 -> Saturn V)."""
    mod = _load_task_module("046")
    start_matcher = _visit_source_url(mod.START_URL)
    hop_matcher = _visit_source_url(mod.HOP_URL)
    return [
        Waypoint(
            "046", "rocket", "place", re.compile(r"saturn\s*v", re.IGNORECASE),
            "visit", [start_matcher],
            cue_variants=("rocket", "launch vehicle", "launched by", "launch"),
            desc="Saturn V (the rocket) -- discoverable on the Apollo 11 page",
        ),
        Waypoint(
            "046", "height", "quantity", re.compile(r"\b363\b|\b1(10|11)(\.\d+)?\s*m"),
            "visit", [hop_matcher],
            cue_variants=("height", "tall", "total height"),
            desc="rocket total height -- discoverable on the Saturn V page",
        ),
        Waypoint(
            "046", "company", "entity", _rx(mod.TARGET_COMPANY),
            "visit", [hop_matcher],
            cue_variants=("first stage", "contractor", "built by", "manufacturer"),
            desc="first-stage contractor (Boeing) -- discoverable on the Saturn V page",
        ),
    ]


def _waypoints_093() -> List[Waypoint]:
    """Cross-domain chain: advisory (prose) -> C source. 'entity'/'quantity' don't quite fit a
    source-code identifier or a byte threshold read out of code -- tagged 'other'/'quantity'."""
    mod = _load_task_module("093")
    source_matcher = _visit_source_url(mod.SOURCE_URL)
    advisory_matcher = _visit_source_url(mod.ADVISORY_URL)
    return [
        Waypoint(
            "093", "flag_variable", "other", mod.KEYSTONE_VARIABLE,
            "visit", [source_matcher],
            cue_variants=("resolve", "local", "socks5", "bool", "flag"),
            desc="exact C flag identifier -- only in the seeded source file",
        ),
        Waypoint(
            "093", "function_name", "other", mod._FUNC,
            "visit", [source_matcher],
            cue_variants=("static", "int", "socks5"),
            desc="vulnerable C function name -- in the source file",
        ),
        Waypoint(
            "093", "threshold", "quantity", mod._THRESHOLD,
            "visit", [source_matcher],
            cue_variants=("length", "hostname", "host name", "max", "buffer"),
            desc="255-byte host-name threshold -- in the source file",
        ),
        Waypoint(
            "093", "fix_version", "other", mod._FIX,
            "visit", [advisory_matcher],
            cue_variants=("fixed in", "fixed", "version"),
            desc="fix version (curl 8.4.0) -- on the advisory page",
        ),
    ]


def build_all_waypoints() -> List[Waypoint]:
    wps: List[Waypoint] = []
    wps += _standard_chain_waypoints("135", ("main span", "span", "longest span"))
    wps += _standard_chain_waypoints("136", ("length", "long"))
    wps += _standard_chain_waypoints("137", ("length", "long", "total length"))
    wps += _standard_chain_waypoints("139", ("area", "floor area", "per floor", "square"))
    wps += _waypoints_138()
    wps += _waypoints_065()
    wps += _waypoints_046()
    wps += _waypoints_093()
    return wps


# Evidence extraction from a raw result node -- the "stored page content" hop N actually fetched.


def _ordered_nodes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = ((data.get("execution") or {}).get("graph") or {}).get("nodes") or {}
    return list(nodes.values())  # dict preserves JSON key order == creation order


def extract_visit_evidence(node: Dict[str, Any]) -> Dict[str, Any]:
    ar = (node.get("details") or {}).get("action_result") or {}
    content = ar.get("content_full") or ar.get("content") or ""
    h1 = ar.get("h1_text") or ""
    title = ar.get("page_title") or ""
    url = ar.get("url") or ""
    link_contexts = ar.get("link_contexts") or {}
    if not isinstance(link_contexts, dict):
        link_contexts = {}
    anchor_blob = "\n".join(f"{v} {k}" for k, v in link_contexts.items())
    blob = "\n".join([h1, title, url, content, anchor_blob])
    return {
        "kind": "visit",
        "node": node,
        "url": url,
        "h1": h1,
        "title": title,
        "content": content,
        "content_low": content.lower(),
        "link_contexts": link_contexts,
        "anchor_blob": anchor_blob,
        "lead": content[:LEAD_CHARS],
        "blob": blob,
        "char_len": len(content),
    }


def extract_search_evidence(node: Dict[str, Any]) -> Dict[str, Any]:
    ar = (node.get("details") or {}).get("action_result") or {}
    results = ar.get("results") or []
    titles = []
    parts = []
    for r in results:
        if isinstance(r, dict):
            titles.append(str(r.get("title") or ""))
            parts.append(" ".join(str(r.get(k) or "") for k in ("title", "url", "description")))
        elif isinstance(r, str):
            parts.append(r)
    blob = "\n".join(parts)
    return {
        "kind": "search",
        "node": node,
        "top_title": titles[0] if titles else "",
        "content_low": blob.lower(),
        "lead": parts[0] if parts else "",
        "blob": blob,
        "char_len": len(blob),
    }


class _NodeView:
    """Minimal shim so contract_satisfaction.derive_step_contract (written for the live engine's
    node objects, which expose .details/.title as ATTRIBUTES) can run unmodified against our
    raw result-JSON dicts."""

    __slots__ = ("details", "title")

    def __init__(self, node: Dict[str, Any]):
        self.details = node.get("details") or {}
        self.title = node.get("title") or ""


def find_best_source(nodes: Sequence[Dict[str, Any]], wp: Waypoint) -> Optional[Dict[str, Any]]:
    if wp.source_kind == "search_first":
        for node in nodes:
            if (node.get("details") or {}).get("action") != "search":
                continue
            ar = (node.get("details") or {}).get("action_result") or {}
            if ar.get("results"):
                return node
        return None

    candidates = []
    for node in nodes:
        if (node.get("details") or {}).get("action") != "visit":
            continue
        ar = (node.get("details") or {}).get("action_result") or {}
        url = str(ar.get("url") or "")
        if not url.startswith("http"):
            continue  # a failed visit stores its GOAL text in "url", not a real page
        if any(m(node) for m in wp.source_matchers):
            candidates.append(node)
    if not candidates:
        return None

    def content_len(n: Dict[str, Any]) -> int:
        ar = (n.get("details") or {}).get("action_result") or {}
        c = ar.get("content_full") or ar.get("content") or ""
        return len(c)

    return max(candidates, key=content_len)


# Deterministic extraction methods under test.


def _cue_window_hit(content_low: str, cues: Tuple[str, ...], value_rx: Pattern, window: int) -> bool:
    """Method 1 (cue-window): a regex for the ground-truth VALUE within `window` chars of any
    of the datum's own wording variants. Same mechanism as contract_satisfaction._number_near,
    but checks the CORRECT value instead of "any digit" -- _number_near discards the match and
    returns only a bool; here we keep the match and compare it to ground truth."""
    for cue in cues:
        for m in re.finditer(re.escape(cue.lower()), content_low):
            start = max(0, m.start() - window)
            end = min(len(content_low), m.end() + window)
            if value_rx.search(content_low[start:end]):
                return True
    return False


def _prod_number_near(node: Dict[str, Any], content_low: str) -> Optional[bool]:
    """Literal, unmodified reuse of production's derive_step_contract + _number_near, run
    against the node's OWN goal/expect text. Returns None when no quantity cue is derivable
    from that text at all (contract not checkable) -- distinct from a False "miss"."""
    contract = derive_step_contract(_NodeView(node))
    if not contract.datums:
        return None
    return any(_number_near(content_low, variants) for _canon, variants in contract.datums)


def evaluate_waypoint(wp: Waypoint, ev: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    out: Dict[str, Optional[bool]] = {}
    out["ceiling"] = bool(wp.value_rx.search(ev["blob"]))

    out["cue_window"] = (
        _cue_window_hit(ev["content_low"], wp.cue_variants, wp.value_rx, WINDOW)
        if wp.cue_variants else None
    )

    out["prod_number_near"] = (
        _prod_number_near(ev["node"], ev["content_low"]) if ev["kind"] == "visit" else None
    )

    if ev["kind"] == "visit":
        ident = f"{ev['h1']} {ev['title']}"
        out["h1_title"] = bool(wp.value_rx.search(ident)) if ident.strip() else False
    else:
        out["h1_title"] = bool(wp.value_rx.search(ev["top_title"])) if ev["top_title"] else False

    out["lead"] = bool(wp.value_rx.search(ev["lead"])) if ev["lead"] else False

    out["link_context"] = (
        bool(wp.value_rx.search(ev["anchor_blob"])) if ev["kind"] == "visit" and ev["anchor_blob"] else (
            None if ev["kind"] == "visit" else None
        )
    )
    return out


METHOD_ORDER = ["cue_window", "prod_number_near", "h1_title", "lead", "link_context"]
METHOD_LABEL = {
    "cue_window": "1: cue-window value match",
    "prod_number_near": "1b: prod _number_near (reused as-is)",
    "h1_title": "2: h1_text/title",
    "lead": "3: first ~500 chars (lead)",
    "link_context": "4: link_contexts anchor text",
}


# PRECISION -- the recall pass above only asks "when applied to the CORRECT source page, does a
# deterministic method find the value". It says nothing about what happens when the SAME
# extractor is pointed at a page that is NOT the correct source for that waypoint -- the risk a
# production write-site actually runs into (a leaf's own contract text fires on whatever page it
# happened to fetch, right or wrong). These two functions are BLIND: unlike `_cue_window_hit`
# above (which cheats by checking for the ground-truth `value_rx` inside the window), they have
# NO knowledge of the correct answer -- they extract a NUMBER / anchor exactly as a production
# extractor would, and only afterwards is the extracted candidate compared against ground truth
# to classify it as right, wrong, or "abstained" (no candidate found).

_BLIND_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Role/creation words that plausibly sit next to the hyperlinked VALUE of an entity or place
# waypoint (a designer, an engineer, a birthplace, a namesake work). Not curated per-waypoint
# like `Waypoint.cue_variants` -- this is the generic vocabulary a production extractor would
# have to fall back on for a hop type `_QUANTITY_CUES` was never meant to cover.
_ENTITY_ROLE_CUES: Tuple[str, ...] = (
    "designer", "designed by", "chief engineer", "engineer", "engineered by",
    "architect", "architected", "founder", "founded by", "discovered by",
    "invented by", "author", "written by", "poet", "born", "birthplace",
    "born in", "creator", "created by", "built by", "constructed by",
    "developed by", "completed", "record-breaking", "launch vehicle",
    "rocket", "aqueduct", "steamship", "location", "located",
)


def _extract_number_near_blind(content_low: str, variants: Sequence[str], window: int) -> Optional[str]:
    """The FIRST number found within `window` chars of the FIRST occurrence of any cue variant.

    Prototype for `contract_satisfaction._find_number_near` (Step 2) -- same "cue word then a
    nearby number" shape as production's own `_number_near`, except it keeps the matched digits
    instead of discarding them into a bool.
    """
    for variant in variants:
        m = re.search(re.escape(variant.lower()), content_low)
        if not m:
            continue
        start = max(0, m.start() - window)
        end = min(len(content_low), m.end() + window)
        num = _BLIND_NUM_RE.search(content_low[start:end])
        if num:
            return num.group(0)
    return None


def _extract_anchor_near_blind(
    content_low: str, link_contexts: Dict[str, str], cues: Sequence[str], window: int
) -> Optional[Tuple[str, str]]:
    """The FIRST known hyperlink anchor text that lands within `window` chars of a role-cue
    word's FIRST occurrence in the page's own body text.

    Cross-references body-text proximity (a role word) against the confirmed hyperlink
    structure (`link_contexts` -- the anchor is a REAL link on the page, not just a capitalized
    word in prose), which is what makes this different from -- and stronger than -- a bare
    body-text regex (the audit's h1/lead methods, which barely worked at all).
    """
    if not link_contexts:
        return None
    anchors = sorted(
        (
            (v.strip(), k)
            for k, v in link_contexts.items()
            if isinstance(v, str) and isinstance(k, str) and len(v.strip()) >= 3
        ),
        key=lambda kv: -len(kv[0]),
    )
    if not anchors:
        return None
    for cue in cues:
        m = re.search(re.escape(cue.lower()), content_low)
        if not m:
            continue
        start = max(0, m.start() - window)
        end = min(len(content_low), m.end() + window)
        window_text = content_low[start:end]
        for anchor_text, url in anchors:
            if anchor_text.lower() in window_text:
                return anchor_text, url
    return None


@dataclass
class PrecisionRecord:
    file: str
    task_id: str
    key: str
    hop_type: str
    method: str  # "cue_window" | "anchor_text"
    correct_url: str
    wrong_url: str
    extracted: str
    is_wrong: bool  # extracted a candidate that does NOT satisfy ground truth
    snippet: str


def _wrong_pages(nodes: Sequence[Dict[str, Any]], wp: Waypoint, correct_url: str) -> List[Dict[str, Any]]:
    """Every OTHER visited page in the same run -- same file, same task, definitely fetched --
    that is not this waypoint's own correct source. Deduplicated by URL (a page hit-refreshed
    across retries should not multiply-count)."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for node in nodes:
        if (node.get("details") or {}).get("action") != "visit":
            continue
        ar = (node.get("details") or {}).get("action_result") or {}
        url = str(ar.get("url") or "")
        if not url.startswith("http"):
            continue
        norm = normalize_url(url)
        if norm == normalize_url(correct_url) or norm in seen:
            continue
        content = ar.get("content_full") or ar.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        seen.add(norm)
        out.append(node)
    return out


def run_precision(results_globs: Sequence[str]) -> List[PrecisionRecord]:
    waypoints_by_task: Dict[str, List[Waypoint]] = {}
    for wp in build_all_waypoints():
        waypoints_by_task.setdefault(wp.task_id, []).append(wp)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    records: List[PrecisionRecord] = []
    for path in files:
        base = os.path.basename(path)
        m = FNAME_RX.match(base)
        if not m:
            continue
        task_id = m.group("task_id")
        if task_id not in waypoints_by_task:
            continue

        with open(path) as fh:
            data = json.load(fh)
        nodes = _ordered_nodes(data)

        for wp in waypoints_by_task[task_id]:
            if wp.source_kind != "visit":
                continue  # search-first waypoints have no fixed "page" to be wrong about
            source = find_best_source(nodes, wp)
            if source is None:
                continue
            correct_ar = (source.get("details") or {}).get("action_result") or {}
            correct_url = str(correct_ar.get("url") or "")

            for wrong in _wrong_pages(nodes, wp, correct_url):
                ar = (wrong.get("details") or {}).get("action_result") or {}
                wrong_url = str(ar.get("url") or "")
                content = ar.get("content_full") or ar.get("content") or ""
                content_low = content.lower()

                if wp.cue_variants:
                    extracted = _extract_number_near_blind(content_low, wp.cue_variants, WINDOW)
                    if extracted is not None:
                        is_wrong = not bool(wp.value_rx.search(extracted))
                        idx = content_low.find(extracted.lower())
                        snippet = content[max(0, idx - 80): idx + len(extracted) + 80].replace("\n", " ")
                        records.append(PrecisionRecord(
                            file=base, task_id=task_id, key=wp.key, hop_type=wp.hop_type,
                            method="cue_window", correct_url=correct_url, wrong_url=wrong_url,
                            extracted=extracted, is_wrong=is_wrong, snippet=snippet,
                        ))

                if wp.hop_type in ("entity", "place"):
                    link_contexts = ar.get("link_contexts") or {}
                    if not isinstance(link_contexts, dict):
                        link_contexts = {}
                    hit = _extract_anchor_near_blind(content_low, link_contexts, _ENTITY_ROLE_CUES, WINDOW)
                    if hit is not None:
                        anchor_text, anchor_url = hit
                        is_wrong = not bool(wp.value_rx.search(anchor_text))
                        records.append(PrecisionRecord(
                            file=base, task_id=task_id, key=wp.key, hop_type=wp.hop_type,
                            method="anchor_text", correct_url=correct_url, wrong_url=wrong_url,
                            extracted=anchor_text, is_wrong=is_wrong, snippet=anchor_url,
                        ))

    return records


def print_precision(records: List[PrecisionRecord], examples: int = 0) -> None:
    print("\n=== PRECISION (extractor applied to a page that is NOT the waypoint's correct source) ===")
    for method in ("cue_window", "anchor_text"):
        rows = [r for r in records if r.method == method]
        n = len(rows)
        wrong = [r for r in rows if r.is_wrong]
        print(
            f"\n-- {method} -- wrong-page trials={n}  emitted-a-value={n}  "
            f"WRONG={len(wrong)} ({_pct(len(wrong), n)})  "
            f"coincidentally-correct={n - len(wrong)} ({_pct(n - len(wrong), n)})"
        )
        by_hop: Dict[str, List[PrecisionRecord]] = {}
        for r in rows:
            by_hop.setdefault(r.hop_type, []).append(r)
        for ht, hrows in sorted(by_hop.items()):
            hwrong = [r for r in hrows if r.is_wrong]
            print(
                f"     {ht:8s} trials={len(hrows):4d}  WRONG={len(hwrong):4d} "
                f"({_pct(len(hwrong), len(hrows))})"
            )
        if examples:
            for r in wrong[:examples]:
                print(
                    f"     WRONG-EXAMPLE {r.task_id}/{r.key} extracted={r.extracted!r} "
                    f"from {r.wrong_url} (correct source: {r.correct_url})"
                )
                print(f"       ...{r.snippet}...")


# VALIDATE-PRODUCT -- re-runs both the recall and precision questions against the actual shipped
# `waypoint.build_waypoint`, not a hand-rolled prototype. Confirms the page-identity guard the
# `--precision` numbers demanded is doing its job: recall on the CORRECT source should survive
# (the guard must not reject the pages it's supposed to accept), and the wrong-page false-positive
# rate that was ~100% for the blind prototypes above should collapse.
#
# The recall row alone was not enough to greenlight surfacing a value into the next hop's
# prompt: "emitted" is not "correct". The breakdown below answers, offline, over the same
# corpus: which method/hop-type (if any) is trustworthy on the CORRECT page, how much of the
# "wrong" bucket is really a near-miss (fine for a next-hop search query, not for a final
# answer), and -- for the genuine misses -- whether the extractor was confused by 2+ plausible
# candidates (ambiguity, an argument for a narrow disambiguation call) or looked in the wrong
# part of the page entirely with only one candidate on offer (a deterministic targeting bug).

# Canonical ground-truth STRING per (task_id, waypoint key) -- hand-read from each task module's
# own CHAIN/KEYSTONE_*/HOP_* constants (the same source `build_all_waypoints()` compiles
# `value_rx` from), used only for near-miss/containment diagnostics. `value_rx` itself remains
# the sole source of truth for "strict correct" everywhere in this script.
TRUTH_LITERAL: Dict[Tuple[str, str], str] = {
    ("135", "creator"): "John A. Roebling",
    ("135", "terminal"): "John A. Roebling Suspension Bridge",
    ("135", "keystone"): "1,057 ft (322 m)",
    ("136", "creator"): "Isambard Kingdom Brunel",
    ("136", "terminal"): "SS Great Eastern",
    ("136", "keystone"): "692 ft (211 m)",
    ("137", "creator"): "Thomas Telford",
    ("137", "terminal"): "Pontcysyllte Aqueduct",
    ("137", "keystone"): "307 m (336 yd)",
    ("139", "creator"): "Antoni Gaudí",
    ("139", "terminal"): "Casa Milà (La Pedrera)",
    ("139", "keystone"): "1,323 m2",
    ("138", "creator"): "Andrew Scott Waugh",
    ("138", "announce_year"): "1856",
    ("138", "keystone"): "29,002 ft",
    ("065", "poet"): "Pablo Neruda",
    ("065", "town"): "Parral",
    ("065", "elevation"): "162 m",
    ("046", "rocket"): "Saturn V",
    ("046", "height"): "363 ft (111 m)",
    ("046", "company"): "Boeing",
    ("093", "flag_variable"): "socks5_resolve_local",
    ("093", "function_name"): "do_SOCKS5",
    ("093", "threshold"): "255",
    ("093", "fix_version"): "curl 8.4.0",
}


def _norm_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _numeric_tokens(s: str) -> set:
    # >=2 digits only: a bare single digit ("3") trivially substring-matches almost any
    # longer truth string that happens to contain it ("1,323 m2") without meaning anything --
    # caught empirically (see the module's own near-miss diagnostic run) and excluded here so
    # the near-miss count isn't inflated by an artifact of the check itself.
    return {m.group(0).replace(",", "") for m in re.finditer(r"\d[\d,]*", s) if len(m.group(0)) >= 2}


def _is_near_miss(value: str, truth: str) -> bool:
    """Fine for a next-hop SEARCH query, not for a final answer: string containment either
    direction (``"Roebling"`` vs ``"John A. Roebling"``), or a shared numeric token
    (``"322"`` vs ``"322 m"`` vs ``"1,057 ft (322 m)"``). Requires >=3 normalized chars before
    doing containment at all -- a short/bare extracted value (a single digit, a 1-2 letter
    fragment) trivially "contains"/"is contained by" almost anything and would otherwise
    inflate this count with meaningless matches."""
    v, t = _norm_alnum(value), _norm_alnum(truth)
    if len(v) >= 3 and t and (v in t or t in v):
        return True
    return bool(_numeric_tokens(value) & _numeric_tokens(truth))


def _diagnose_cue_window(contract, evidence_low: str):
    """Read-only clone of ``waypoint._extract_cue_window`` (never imported as private API
    across modules -- duplicated here on purpose, diagnostic-only) that ALSO returns the
    winning window text and every DISTINCT number found in it, not just the first. Mirrors
    ``_find_number_near``'s exact selection order (variant, then occurrence, then first
    window with any number) so "which window fired" is never guessed at."""
    from agent.app.idea_policies.contract_satisfaction import _NUMBER_RE as _NUM_RE
    for canonical, variants in contract.datums:
        if not variants:
            m = _NUM_RE.search(evidence_low)
            if m:
                candidates = sorted(set(_NUM_RE.findall(evidence_low)))
                return canonical, "(no stable wording -- any number)", evidence_low, candidates
            continue
        for variant in variants:
            for vm in re.finditer(re.escape(variant), evidence_low):
                start = max(0, vm.start() - WINDOW)
                end = min(len(evidence_low), vm.end() + WINDOW)
                window_text = evidence_low[start:end]
                if _NUM_RE.search(window_text):
                    candidates = sorted(set(_NUM_RE.findall(window_text)))
                    return canonical, variant, window_text, candidates
    return None


def _diagnose_anchor_text(evidence_low: str, link_contexts: Dict[str, Any]):
    """Read-only clone of ``waypoint._extract_anchor_text`` that ALSO returns every candidate
    anchor found in the winning window, not just the nearest one."""
    from agent.app.idea_policies.waypoint import _ENTITY_ROLE_CUES, _WINDOW as _WP_WINDOW
    if not isinstance(link_contexts, dict) or not link_contexts:
        return None
    anchors = [
        (v.strip(), k) for k, v in link_contexts.items()
        if isinstance(v, str) and isinstance(k, str) and len(v.strip()) >= 3
    ]
    if not anchors:
        return None
    for cue in _ENTITY_ROLE_CUES:
        match = re.search(re.escape(cue), evidence_low)
        if not match:
            continue
        start = max(0, match.start() - _WP_WINDOW)
        end = min(len(evidence_low), match.end() + _WP_WINDOW)
        window_text = evidence_low[start:end]
        candidates = sorted({a for a, _u in anchors if a.lower() in window_text})
        if candidates:
            return cue, window_text, candidates
    return None


# CONTAINMENT -- "picking one is the only part it does badly; does the truth even reach the
# candidate set at all?" Ranked variants of the two diagnose functions above (nearest-to-cue
# first, ties broken by first occurrence), run over ALL correct-page trials regardless of
# whether the guard/selection would currently emit anything -- the ceiling question, not "what
# does build_waypoint do today". Read-only, no product code involved.


def _ranked_cue_window_candidates(contract, evidence_low: str):
    """Like `_diagnose_cue_window`, but returns (cue_label, [(number, distance), ...]) ranked
    nearest-to-cue first -- the order a "top-K candidate set" would present."""
    from agent.app.idea_policies.contract_satisfaction import _NUMBER_RE as _NUM_RE
    for canonical, variants in contract.datums:
        if not variants:
            seen: Dict[str, float] = {}
            for m in _NUM_RE.finditer(evidence_low):
                seen.setdefault(m.group(0), float(m.start()))
            if seen:
                ranked = sorted(seen.items(), key=lambda kv: kv[1])
                return f"{canonical}/(no stable wording)", ranked
            continue
        for variant in variants:
            for vm in re.finditer(re.escape(variant), evidence_low):
                start = max(0, vm.start() - WINDOW)
                end = min(len(evidence_low), vm.end() + WINDOW)
                window_text = evidence_low[start:end]
                cue_center = (vm.start() + vm.end()) / 2
                found = list(_NUM_RE.finditer(window_text))
                if not found:
                    continue
                seen = {}
                for fm in found:
                    num = fm.group(0)
                    num_center = start + fm.start() + len(num) / 2
                    dist = abs(num_center - cue_center)
                    if num not in seen or dist < seen[num]:
                        seen[num] = dist
                ranked = sorted(seen.items(), key=lambda kv: kv[1])
                return f"{canonical}/{variant}", ranked
    return None


def _ranked_anchor_candidates(evidence_low: str, link_contexts: Dict[str, Any]):
    """Like `_diagnose_anchor_text`, but returns (cue, [(anchor_text, distance), ...]) ranked
    nearest-to-cue first -- the SAME proximity heuristic `_extract_anchor_text` already uses
    to pick a single winner, just not truncated to one."""
    from agent.app.idea_policies.waypoint import _ENTITY_ROLE_CUES, _WINDOW as _WP_WINDOW
    if not isinstance(link_contexts, dict) or not link_contexts:
        return None
    anchors = [
        (v.strip(), k) for k, v in link_contexts.items()
        if isinstance(v, str) and isinstance(k, str) and len(v.strip()) >= 3
    ]
    if not anchors:
        return None
    for cue in _ENTITY_ROLE_CUES:
        match = re.search(re.escape(cue), evidence_low)
        if not match:
            continue
        start = max(0, match.start() - _WP_WINDOW)
        end = min(len(evidence_low), match.end() + _WP_WINDOW)
        window_text = evidence_low[start:end]
        cue_center = (match.start() + match.end()) / 2
        seen: Dict[str, float] = {}
        for anchor_text, _url in anchors:
            idx = window_text.find(anchor_text.lower())
            if idx < 0:
                continue
            anchor_center = start + idx + len(anchor_text) / 2
            dist = abs(anchor_center - cue_center)
            if anchor_text not in seen or dist < seen[anchor_text]:
                seen[anchor_text] = dist
        if seen:
            ranked = sorted(seen.items(), key=lambda kv: kv[1])
            return cue, ranked
    return None


@dataclass
class ContainmentRecord:
    task_id: str
    key: str
    hop_type: str
    method: str
    n_candidates: int
    truth_rank: Optional[int]  # 1-based rank of the first candidate satisfying value_rx, or None


CONTAINMENT_KS = (1, 3, 5, 10, None)  # None = "all candidates in the window"


def run_containment_analysis(results_globs: Sequence[str]) -> List[ContainmentRecord]:
    from agent.app.idea_policies.contract_satisfaction import derive_step_contract as _derive

    waypoints_by_task: Dict[str, List[Waypoint]] = {}
    for wp in build_all_waypoints():
        waypoints_by_task.setdefault(wp.task_id, []).append(wp)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    records: List[ContainmentRecord] = []
    for path in files:
        base = os.path.basename(path)
        m = FNAME_RX.match(base)
        if not m:
            continue
        task_id = m.group("task_id")
        if task_id not in waypoints_by_task:
            continue
        with open(path) as fh:
            data = json.load(fh)
        nodes = _ordered_nodes(data)

        for wp in waypoints_by_task[task_id]:
            if wp.source_kind != "visit":
                continue
            source = find_best_source(nodes, wp)
            if source is None:
                continue
            source_view = _NodeView(source)
            ar = (source.get("details") or {}).get("action_result") or {}
            content = ar.get("content_full") or ar.get("content") or ""
            if not content.strip():
                continue
            from agent.app.idea_policies.waypoint import _visit_evidence as _wp_visit_evidence
            evidence_low = _wp_visit_evidence(ar).lower()
            contract = _derive(source_view)

            cue_diag = _ranked_cue_window_candidates(contract, evidence_low)
            if cue_diag is not None:
                _label, ranked = cue_diag
                rank = next(
                    (i for i, (cand, _d) in enumerate(ranked, 1) if wp.value_rx.search(cand)),
                    None,
                )
                records.append(ContainmentRecord(
                    task_id=task_id, key=wp.key, hop_type=wp.hop_type, method="cue_window",
                    n_candidates=len(ranked), truth_rank=rank,
                ))

            anchor_diag = _ranked_anchor_candidates(evidence_low, ar.get("link_contexts") or {})
            if anchor_diag is not None:
                _cue, ranked = anchor_diag
                rank = next(
                    (i for i, (cand, _d) in enumerate(ranked, 1) if wp.value_rx.search(cand)),
                    None,
                )
                records.append(ContainmentRecord(
                    task_id=task_id, key=wp.key, hop_type=wp.hop_type, method="anchor_text",
                    n_candidates=len(ranked), truth_rank=rank,
                ))

    return records


def _containment_at_k(records: List[ContainmentRecord], k: Optional[int]) -> str:
    n = len(records)
    if n == 0:
        return "  n/a (0 applicable trials)"
    hit = sum(1 for r in records if r.truth_rank is not None and (k is None or r.truth_rank <= k))
    return f"n={n:3d}  contains-truth={hit:3d} ({_pct(hit, n)})"


def _median(xs: List[int]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def _percentile(xs: List[int], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return xs[idx]


def print_containment(records: List[ContainmentRecord]) -> None:
    print(f"\n=== CONTAINMENT (over {len(records)} applicable trials -- every correct-page "
          f"trial where that method's window/cue fired at all, whether or not the guard would "
          f"currently let it emit) ===")

    print("\n-- containment@K, overall by method --")
    for method in ("cue_window", "anchor_text"):
        rows = [r for r in records if r.method == method]
        print(f"  {method}:")
        for k in CONTAINMENT_KS:
            label = f"top-{k}" if k is not None else "all-in-window"
            print(f"    {label:14s} {_containment_at_k(rows, k)}")

    print("\n-- containment@K by method x hop type --")
    for method in ("cue_window", "anchor_text"):
        for ht in sorted({r.hop_type for r in records}):
            rows = [r for r in records if r.method == method and r.hop_type == ht]
            if not rows:
                continue
            print(f"  {method:12s} x {ht:10s}")
            for k in CONTAINMENT_KS:
                label = f"top-{k}" if k is not None else "all"
                print(f"      {label:14s} {_containment_at_k(rows, k)}")

    print("\n-- candidate-set size distribution (n candidates per applicable trial) --")
    for method in ("cue_window", "anchor_text"):
        sizes = [r.n_candidates for r in records if r.method == method]
        if not sizes:
            continue
        print(
            f"  {method:12s} n={len(sizes):3d}  median={_median(sizes):.1f}  "
            f"p90={_percentile(sizes, 0.9):.1f}  max={max(sizes)}"
        )

    print("\n-- rank of truth, when present in the window (1 = nearest to the cue) --")
    for method in ("cue_window", "anchor_text"):
        ranks = [r.truth_rank for r in records if r.method == method and r.truth_rank is not None]
        if not ranks:
            print(f"  {method:12s} (truth never found in-window)")
            continue
        from collections import Counter
        counts = Counter(ranks)
        dist = ", ".join(f"rank {r}: {c}" for r, c in sorted(counts.items())[:8])
        print(
            f"  {method:12s} n={len(ranks):3d}  median-rank={_median(ranks):.1f}  "
            f"p90-rank={_percentile(ranks, 0.9):.1f}  [{dist}]"
        )


@dataclass
class RecallRecord:
    file: str
    task_id: str
    key: str
    hop_type: str
    method: str
    value: str
    truth: str
    strict_correct: bool
    near_miss: bool
    source_url: str
    n_candidates: Optional[int]
    window_snippet: str
    cue: str


def run_guarded_validation(results_globs: Sequence[str]) -> List[RecallRecord]:
    from agent.app.idea_policies import waypoint as _waypoint
    from agent.app.idea_policies.contract_satisfaction import derive_step_contract as _derive

    waypoints_by_task: Dict[str, List[Waypoint]] = {}
    for wp in build_all_waypoints():
        waypoints_by_task.setdefault(wp.task_id, []).append(wp)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    recall_trials = recall_emitted = recall_correct = 0
    precision_trials = precision_emitted = precision_wrong = 0
    records: List[RecallRecord] = []

    for path in files:
        base = os.path.basename(path)
        m = FNAME_RX.match(base)
        if not m:
            continue
        task_id = m.group("task_id")
        if task_id not in waypoints_by_task:
            continue

        with open(path) as fh:
            data = json.load(fh)
        nodes = _ordered_nodes(data)

        for wp in waypoints_by_task[task_id]:
            if wp.source_kind != "visit":
                continue  # build_waypoint only extracts from VISIT leaves (see its docstring)
            source = find_best_source(nodes, wp)
            if source is None:
                continue
            source_view = _NodeView(source)
            correct_ar = (source.get("details") or {}).get("action_result") or {}

            recall_trials += 1
            rec = _waypoint.build_waypoint(source_view, correct_ar, "visit", "x", 0)
            if rec is not None and rec.value is not None:
                recall_emitted += 1
                value = str(rec.value)
                strict = bool(wp.value_rx.search(value))
                truth = TRUTH_LITERAL.get((task_id, wp.key), "")
                near = (not strict) and truth and _is_near_miss(value, truth)
                if strict:
                    recall_correct += 1

                n_candidates: Optional[int] = None
                window_snippet = ""
                cue = ""
                if not strict:
                    content = correct_ar.get("content_full") or correct_ar.get("content") or ""
                    evidence_low = _waypoint._visit_evidence(correct_ar).lower()  # noqa: SLF001
                    if rec.method == "cue_window":
                        diag = _diagnose_cue_window(_derive(source_view), evidence_low)
                        if diag:
                            canon, cue, window_text, candidates = diag
                            n_candidates = len(candidates)
                            window_snippet = window_text[:280]
                            cue = f"{canon}/{cue}"
                    elif rec.method == "anchor_text":
                        diag = _diagnose_anchor_text(evidence_low, correct_ar.get("link_contexts") or {})
                        if diag:
                            cue, window_text, candidates = diag
                            n_candidates = len(candidates)
                            window_snippet = window_text[:280]

                records.append(RecallRecord(
                    file=base, task_id=task_id, key=wp.key, hop_type=wp.hop_type,
                    method=rec.method, value=value, truth=truth,
                    strict_correct=strict, near_miss=bool(near),
                    source_url=str(correct_ar.get("url") or ""),
                    n_candidates=n_candidates, window_snippet=window_snippet, cue=cue,
                ))

            correct_url = str(correct_ar.get("url") or "")
            for wrong in _wrong_pages(nodes, wp, correct_url):
                wrong_ar = (wrong.get("details") or {}).get("action_result") or {}
                # SAME leaf/contract (the waypoint's own source node), but evidence from a
                # DIFFERENT, wrong page -- exactly the production risk: a leaf's own contract
                # text fires against whatever page it actually fetched, right or wrong.
                precision_trials += 1
                wrec = _waypoint.build_waypoint(source_view, wrong_ar, "visit", "x", 0)
                if wrec is not None and wrec.value is not None:
                    precision_emitted += 1
                    if not wp.value_rx.search(str(wrec.value)):
                        precision_wrong += 1

    print("\n=== VALIDATE-PRODUCT (the actual shipped waypoint.build_waypoint, guard included) ===")
    print(
        f"\nRECALL on the CORRECT source page: trials={recall_trials}  "
        f"emitted-a-value={recall_emitted} ({_pct(recall_emitted, recall_trials)})  "
        f"of those, correct={recall_correct} ({_pct(recall_correct, recall_emitted)})"
    )
    print(
        f"PRECISION on a WRONG source page:   trials={precision_trials}  "
        f"emitted-a-value={precision_emitted} ({_pct(precision_emitted, precision_trials)})  "
        f"of those, WRONG={precision_wrong} ({_pct(precision_wrong, precision_emitted)})"
    )
    return records


def print_recall_breakdown(records: List[RecallRecord], examples: int = 0) -> None:
    n = len(records)
    strict = sum(1 for r in records if r.strict_correct)
    near = sum(1 for r in records if r.near_miss)
    wrong = n - strict - near
    print(
        f"\n=== RECALL BREAKDOWN (of {n} emitted-on-the-correct-page values) ===\n"
        f"strict-correct={strict} ({_pct(strict, n)})  "
        f"near-miss={near} ({_pct(near, n)})  "
        f"genuinely-wrong={wrong} ({_pct(wrong, n)})"
    )

    print("\n-- by method --")
    for method in ("cue_window", "anchor_text"):
        rows = [r for r in records if r.method == method]
        if not rows:
            continue
        s = sum(1 for r in rows if r.strict_correct)
        nm = sum(1 for r in rows if r.near_miss)
        w = len(rows) - s - nm
        print(
            f"  {method:12s} n={len(rows):3d}  strict={s:3d} ({_pct(s, len(rows))})  "
            f"near-miss={nm:3d} ({_pct(nm, len(rows))})  wrong={w:3d} ({_pct(w, len(rows))})"
        )

    print("\n-- by hop type --")
    for ht in sorted({r.hop_type for r in records}):
        rows = [r for r in records if r.hop_type == ht]
        s = sum(1 for r in rows if r.strict_correct)
        nm = sum(1 for r in rows if r.near_miss)
        w = len(rows) - s - nm
        print(
            f"  {ht:10s} n={len(rows):3d}  strict={s:3d} ({_pct(s, len(rows))})  "
            f"near-miss={nm:3d} ({_pct(nm, len(rows))})  wrong={w:3d} ({_pct(w, len(rows))})"
        )

    print("\n-- by method x hop type (strict-correct rate) --")
    for method in ("cue_window", "anchor_text"):
        for ht in sorted({r.hop_type for r in records}):
            rows = [r for r in records if r.method == method and r.hop_type == ht]
            if not rows:
                continue
            s = sum(1 for r in rows if r.strict_correct)
            print(f"  {method:12s} x {ht:10s} n={len(rows):3d}  strict={s:3d} ({_pct(s, len(rows))})")

    wrong_rows = [r for r in records if not r.strict_correct and not r.near_miss]
    diagnosable = [r for r in wrong_rows if r.n_candidates is not None]
    ambiguity = [r for r in diagnosable if r.n_candidates >= 2]
    targeting = [r for r in diagnosable if r.n_candidates == 1]
    undiagnosed = [r for r in wrong_rows if r.n_candidates is None]
    print(
        f"\n-- wrong-on-right-page diagnosis (n={len(wrong_rows)}) --\n"
        f"  ambiguity (2+ candidates in the window, picked wrong): "
        f"{len(ambiguity)} ({_pct(len(ambiguity), len(wrong_rows))})\n"
        f"  targeting (only 1 candidate in the window, and it's wrong): "
        f"{len(targeting)} ({_pct(len(targeting), len(wrong_rows))})\n"
        f"  undiagnosed (no window could be recovered): {len(undiagnosed)}"
    )

    if examples:
        for label, rows in (("AMBIGUITY", ambiguity), ("TARGETING", targeting)):
            print(f"\n-- {label} examples --")
            for r in rows[:examples]:
                print(
                    f"  {r.task_id}/{r.key} [{r.method}] cue={r.cue!r} candidates={r.n_candidates} "
                    f"extracted={r.value!r} truth={r.truth!r} url={r.source_url}"
                )
                print(f"    window: ...{r.window_snippet}...")
            if not rows:
                print("  (none)")


# ORACLE -- the free measurement that GATES the ``target_datum`` design (see the design brief):
# what would extraction accuracy be if the datum the expansion call is going to fetch were
# NAMED ahead of time, instead of derived (today) from goal text that routinely never mentions
# it at all (task 046's leaf goal is literally "Visit the Saturn V page" -- the word "height"
# appears nowhere)? ``ORACLE_TARGET_DATUM`` hand-authors a ``(kind, label)`` per waypoint from
# each task module's OWN ground truth -- the LABEL only ("main span length"), never the answer
# ("1,057 ft"); ``cue_variants`` are wording (not values) for the quantity/date waypoints only,
# reused verbatim from ``build_all_waypoints()``'s own hand-authored cue lists above (never the
# entity/place "role cue" lists, which are about WHERE an anchor sits, not a number to extract).
#
# Two independent levers, measured separately (4-way ablation) so it is clear which HALF of the
# idea is doing the work:
#   * ``use_oracle_label``  -- swap the DATUM cue (what ``_extract_cue_window`` searches for)
#     from today's goal-text derivation to the oracle label's wording. Subject tokens (WHICH page
#     is expected) are always left untouched -- ``target_datum`` narrows WHAT, never WHERE.
#   * ``use_kind_dispatch`` -- swap the cue_window/anchor_text DISPATCH decision from today's
#     "the leaf's own contract carries no datum cue" heuristic to the oracle's explicit ``kind``
#     (quantity/date -> cue_window only, entity/place -> anchor_text only; ``kind=None`` defers
#     to the old heuristic, unchanged).

from agent.app.idea_policies.contract_satisfaction import StepContract as _StepContract  # noqa: E402


@dataclass
class OracleDatum:
    kind: Optional[str]              # "quantity" | "date" | "entity" | "place" | None
    label: str                        # generic description of the sought value -- NEVER the answer
    cue_variants: Tuple[str, ...] = ()  # wording variants for a cue-window search (quantity/date
                                          # only) -- hand-authored, non-leaking, reused verbatim
                                          # from this same file's ``build_all_waypoints()`` cues


ORACLE_TARGET_DATUM: Dict[Tuple[str, str], OracleDatum] = {
    ("135", "creator"): OracleDatum("entity", "the bridge's designer / chief engineer"),
    ("135", "terminal"): OracleDatum("place", "the suspension bridge that engineer completed over another river"),
    ("135", "keystone"): OracleDatum("quantity", "the terminal bridge's main span", ("main span", "span", "longest span")),

    ("136", "creator"): OracleDatum("entity", "the ship's designer / engineer"),
    ("136", "terminal"): OracleDatum("place", "the ship that engineer designed"),
    ("136", "keystone"): OracleDatum("quantity", "the ship's length", ("length", "long")),

    ("137", "creator"): OracleDatum("entity", "the aqueduct's designer / engineer"),
    ("137", "terminal"): OracleDatum("place", "the aqueduct that engineer designed"),
    ("137", "keystone"): OracleDatum("quantity", "the aqueduct's total length", ("length", "long", "total length")),

    ("139", "creator"): OracleDatum("entity", "the building's architect"),
    ("139", "terminal"): OracleDatum("place", "the other building that architect designed"),
    ("139", "keystone"): OracleDatum("quantity", "the building's floor area", ("area", "floor area", "per floor", "square")),

    ("138", "creator"): OracleDatum("entity", "the person who proposed the peak's name"),
    ("138", "announce_year"): OracleDatum(
        "date", "the year that person's survey publicly announced the peak's height",
        ("survey", "announced", "declared", "trigonometrical"),
    ),
    ("138", "keystone"): OracleDatum(
        "quantity", "the height that survey publicly declared",
        ("height", "declared", "feet", "announced"),
    ),

    ("065", "poet"): OracleDatum("entity", "the poet who wrote the given collection"),
    ("065", "town"): OracleDatum("place", "the poet's birthplace town"),
    ("065", "elevation"): OracleDatum(
        "quantity", "the town's elevation above sea level",
        ("elevation", "altitude", "above sea level"),
    ),

    ("046", "rocket"): OracleDatum("place", "the rocket that launched the mission"),
    ("046", "height"): OracleDatum(
        "quantity", "the rocket's total height", ("height", "tall", "total height"),
    ),
    ("046", "company"): OracleDatum("entity", "the company that built the rocket's first stage"),

    # 093's three "other"-typed waypoints ask for a code identifier / version string -- neither
    # extraction mechanism (a NUMBER near a cue, or a hyperlink anchor near a role cue) is built
    # to recover one, so ``kind=None`` (no dispatch opinion) and no cue variants: this is the
    # "target_datum can't help every hop type" case the design brief expects, not hidden here.
    ("093", "flag_variable"): OracleDatum(None, "the exact C identifier of the flag that decides who resolves the host name"),
    ("093", "function_name"): OracleDatum(None, "the C function that contains that flag"),
    ("093", "threshold"): OracleDatum(
        "quantity", "the maximum host-name length the code checks",
        ("length", "hostname", "host name", "max", "buffer"),
    ),
    ("093", "fix_version"): OracleDatum(None, "the curl version that fixed the bug"),
}


def _oracle_contract(node_view: Any, oracle: Optional[OracleDatum], use_oracle_label: bool) -> _StepContract:
    """Today's goal-text-derived contract, with its DATUM cue optionally swapped for the oracle
    label's wording. Subject tokens always come from the goal text unchanged -- see module note."""
    from agent.app.idea_policies.contract_satisfaction import derive_step_contract as _derive

    own = _derive(node_view)
    if not use_oracle_label or oracle is None:
        return own
    canonical = re.sub(r"[^a-z0-9]+", "_", oracle.label.lower()).strip("_") or "datum"
    datums = [(canonical, oracle.cue_variants)] if oracle.cue_variants else []
    return _StepContract(text=own.text, source=own.source, datums=datums, subject_tokens=own.subject_tokens)


def _oracle_extract(
    node_view: Any, result: Dict[str, Any], oracle: Optional[OracleDatum],
    use_oracle_label: bool, use_kind_dispatch: bool,
) -> Tuple[Optional[str], str]:
    """Mirrors ``waypoint.build_waypoint``'s control flow exactly (same guard, same two
    extractors, imported not reimplemented) with two swappable levers -- see module note.
    Returns ``(value, method)``; ``method`` is ``"none"`` when nothing was emitted."""
    from agent.app.idea_policies import waypoint as _wp

    contract = _oracle_contract(node_view, oracle, use_oracle_label)
    contract = _wp._widen_subject_tokens_with_parent_goal(node_view, contract)  # noqa: SLF001
    if not _wp._page_identity_ok(contract, result):  # noqa: SLF001
        return None, "none"

    evidence = _wp._visit_evidence(result)  # noqa: SLF001
    if not evidence.strip():
        return None, "none"
    evidence_low = evidence.lower()

    value, _kind = _wp._extract_cue_window(contract, evidence_low)  # noqa: SLF001
    method = "cue_window" if value is not None else None

    if use_kind_dispatch and oracle is not None and oracle.kind is not None:
        allow_anchor = oracle.kind in ("entity", "place")
    else:
        allow_anchor = not contract.datums

    if value is None and allow_anchor:
        link_contexts = result.get("link_contexts")
        anchor_value = _wp._extract_anchor_text(  # noqa: SLF001
            evidence_low, link_contexts if isinstance(link_contexts, dict) else {}
        )
        if anchor_value is not None:
            value, method = anchor_value, "anchor_text"

    if value is None or method is None:
        return None, "none"
    return value, method


@dataclass
class OracleRecord:
    task_id: str
    key: str
    hop_type: str
    kind: Optional[str]
    method: str
    value: str
    truth: str
    strict_correct: bool
    near_miss: bool


def run_oracle_mode(
    results_globs: Sequence[str], use_oracle_label: bool, use_kind_dispatch: bool,
) -> Tuple[List[OracleRecord], Dict[str, int]]:
    """Replays the SAME (correct-page, wrong-page) trial set as ``run_guarded_validation`` --
    same waypoint registry, same ``find_best_source``/``_wrong_pages`` -- swapping only the
    extraction call for ``_oracle_extract``. With both levers off this must reproduce
    ``--validate-product`` exactly (asserted by ``--oracle --self-check``)."""
    waypoints_by_task: Dict[str, List[Waypoint]] = {}
    for wp in build_all_waypoints():
        waypoints_by_task.setdefault(wp.task_id, []).append(wp)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    recall_trials = recall_emitted = recall_correct = 0
    precision_trials = precision_emitted = precision_wrong = 0
    records: List[OracleRecord] = []

    for path in files:
        base = os.path.basename(path)
        m = FNAME_RX.match(base)
        if not m:
            continue
        task_id = m.group("task_id")
        if task_id not in waypoints_by_task:
            continue

        with open(path) as fh:
            data = json.load(fh)
        nodes = _ordered_nodes(data)

        for wp in waypoints_by_task[task_id]:
            if wp.source_kind != "visit":
                continue
            source = find_best_source(nodes, wp)
            if source is None:
                continue
            node_view = _NodeView(source)
            oracle = ORACLE_TARGET_DATUM.get((task_id, wp.key))
            correct_ar = (source.get("details") or {}).get("action_result") or {}

            recall_trials += 1
            value, method = _oracle_extract(node_view, correct_ar, oracle, use_oracle_label, use_kind_dispatch)
            if value is not None:
                recall_emitted += 1
                value = str(value)
                strict = bool(wp.value_rx.search(value))
                truth = TRUTH_LITERAL.get((task_id, wp.key), "")
                near = (not strict) and bool(truth) and _is_near_miss(value, truth)
                if strict:
                    recall_correct += 1
                records.append(OracleRecord(
                    task_id=task_id, key=wp.key, hop_type=wp.hop_type,
                    kind=(oracle.kind if oracle else None), method=method,
                    value=value, truth=truth, strict_correct=strict, near_miss=bool(near),
                ))

            correct_url = str(correct_ar.get("url") or "")
            for wrong in _wrong_pages(nodes, wp, correct_url):
                wrong_ar = (wrong.get("details") or {}).get("action_result") or {}
                precision_trials += 1
                wvalue, _wmethod = _oracle_extract(
                    node_view, wrong_ar, oracle, use_oracle_label, use_kind_dispatch
                )
                if wvalue is not None:
                    precision_emitted += 1
                    if not wp.value_rx.search(str(wvalue)):
                        precision_wrong += 1

    summary = dict(
        recall_trials=recall_trials, recall_emitted=recall_emitted, recall_correct=recall_correct,
        precision_trials=precision_trials, precision_emitted=precision_emitted, precision_wrong=precision_wrong,
    )
    return records, summary


def print_oracle_mode(label: str, records: List[OracleRecord], summary: Dict[str, int]) -> None:
    print(f"\n=== ORACLE: {label} ===")
    print(
        f"RECALL on the CORRECT source page: trials={summary['recall_trials']}  "
        f"emitted-a-value={summary['recall_emitted']} ({_pct(summary['recall_emitted'], summary['recall_trials'])})  "
        f"of those, correct={summary['recall_correct']} ({_pct(summary['recall_correct'], summary['recall_emitted'])})"
    )
    print(
        f"PRECISION on a WRONG source page:   trials={summary['precision_trials']}  "
        f"emitted-a-value={summary['precision_emitted']} ({_pct(summary['precision_emitted'], summary['precision_trials'])})  "
        f"of those, WRONG={summary['precision_wrong']} ({_pct(summary['precision_wrong'], summary['precision_emitted'])})"
    )

    n = len(records)
    if n == 0:
        return
    strict = sum(1 for r in records if r.strict_correct)
    near = sum(1 for r in records if r.near_miss)
    wrong = n - strict - near
    print(
        f"of {n} emitted-on-the-correct-page values: strict-correct={strict} ({_pct(strict, n)})  "
        f"near-miss={near} ({_pct(near, n)})  genuinely-wrong={wrong} ({_pct(wrong, n)})"
    )

    print("-- by method --")
    for method in ("cue_window", "anchor_text"):
        rows = [r for r in records if r.method == method]
        if not rows:
            continue
        s = sum(1 for r in rows if r.strict_correct)
        print(f"  {method:12s} n={len(rows):3d}  strict={s:3d} ({_pct(s, len(rows))})")

    print("-- by hop type --")
    for ht in sorted({r.hop_type for r in records}):
        rows = [r for r in records if r.hop_type == ht]
        s = sum(1 for r in rows if r.strict_correct)
        print(f"  {ht:10s} n={len(rows):3d}  strict={s:3d} ({_pct(s, len(rows))})")

    print("-- by method x hop type --")
    for method in ("cue_window", "anchor_text"):
        for ht in sorted({r.hop_type for r in records}):
            rows = [r for r in records if r.method == method and r.hop_type == ht]
            if not rows:
                continue
            s = sum(1 for r in rows if r.strict_correct)
            print(f"  {method:12s} x {ht:10s} n={len(rows):3d}  strict={s:3d} ({_pct(s, len(rows))})")


# Driver

FNAME_RX = re.compile(
    r"^(?P<shape>cschain|csnopar)_g_(?P<model>[a-z0-9]+)_(?P<arm>baseline|good_adaptive)_"
    r"rep(?P<rep>\d+)_(?P<task_id>\d{3})_"
)


@dataclass
class Record:
    file: str
    shape: str
    model: str
    arm: str
    task_id: str
    key: str
    hop_type: str
    desc: str
    methods: Dict[str, Optional[bool]]
    evidence_kind: str
    evidence_len: int
    snippet: str


def run(results_globs: Sequence[str]) -> List[Record]:
    waypoints_by_task: Dict[str, List[Waypoint]] = {}
    for wp in build_all_waypoints():
        waypoints_by_task.setdefault(wp.task_id, []).append(wp)

    files: List[str] = []
    for g in results_globs:
        files.extend(glob.glob(str(REPO_ROOT / g)))
    files = sorted(set(files))

    records: List[Record] = []
    for path in files:
        base = os.path.basename(path)
        m = FNAME_RX.match(base)
        if not m:
            continue
        task_id = m.group("task_id")
        if task_id not in waypoints_by_task:
            continue  # e.g. 047 -- intentionally not measured

        with open(path) as fh:
            data = json.load(fh)
        nodes = _ordered_nodes(data)

        for wp in waypoints_by_task[task_id]:
            source = find_best_source(nodes, wp)
            if source is None:
                continue  # hop never actually reached its source page in this run
            ev = (
                extract_search_evidence(source)
                if wp.source_kind == "search_first"
                else extract_visit_evidence(source)
            )
            if ev["char_len"] == 0:
                continue
            methods = evaluate_waypoint(wp, ev)

            snippet = ""
            if methods["ceiling"]:
                idx = wp.value_rx.search(ev["blob"])
                if idx:
                    s, e = idx.span()
                    snippet = ev["blob"][max(0, s - 80): e + 80].replace("\n", " ")

            records.append(Record(
                file=base, shape=m.group("shape"), model=m.group("model"), arm=m.group("arm"),
                task_id=task_id, key=wp.key, hop_type=wp.hop_type, desc=wp.desc,
                methods=methods, evidence_kind=ev["kind"], evidence_len=ev["char_len"],
                snippet=snippet,
            ))

    return records


# Reporting


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "   n/a"


def print_overall(records: List[Record]) -> None:
    n = len(records)
    ceiling = sum(1 for r in records if r.methods["ceiling"])
    print(f"\n=== OVERALL ({n} waypoint instances reached across all tasks/models/arms/shapes) ===")
    print(f"ceiling (answer present anywhere in fetched payload): {ceiling}/{n} ({_pct(ceiling, n)})")
    for meth in METHOD_ORDER:
        applicable = [r for r in records if r.methods[meth] is not None]
        hits = [r for r in applicable if r.methods[meth]]
        print(
            f"  {METHOD_LABEL[meth]:38s} applicable={len(applicable):3d}  "
            f"hit={len(hits):3d}  recall/reached={_pct(len(hits), n)}  "
            f"recall/applicable={_pct(len(hits), len(applicable))}  "
            f"recall/ceiling={_pct(len(hits), ceiling)}"
        )


def print_by_hop_type(records: List[Record]) -> None:
    print("\n=== BY HOP TYPE ===")
    hop_types = sorted({r.hop_type for r in records}, key=lambda h: (h != "quantity", h))
    for ht in hop_types:
        rows = [r for r in records if r.hop_type == ht]
        n = len(rows)
        ceiling = sum(1 for r in rows if r.methods["ceiling"])
        print(f"\n-- {ht}  (N={n} waypoints reached, ceiling={ceiling}/{n} = {_pct(ceiling, n)}) --")
        for meth in METHOD_ORDER:
            applicable = [r for r in rows if r.methods[meth] is not None]
            if not applicable:
                print(f"  {METHOD_LABEL[meth]:38s} n/a for this hop type")
                continue
            hits = [r for r in applicable if r.methods[meth]]
            print(
                f"  {METHOD_LABEL[meth]:38s} applicable={len(applicable):3d}  hit={len(hits):3d}  "
                f"recall/reached={_pct(len(hits), n)}  recall/ceiling={_pct(len(hits), ceiling)}"
            )


def print_by_task(records: List[Record]) -> None:
    print("\n=== BY TASK / WAYPOINT ===")
    keys = sorted({(r.task_id, r.key) for r in records})
    for task_id, key in keys:
        rows = [r for r in records if r.task_id == task_id and r.key == key]
        n = len(rows)
        ceiling = sum(1 for r in rows if r.methods["ceiling"])
        ht = rows[0].hop_type
        desc = rows[0].desc
        best = max(
            (sum(1 for r in rows if r.methods[m]) for m in METHOD_ORDER),
            default=0,
        )
        print(f"  {task_id}/{key:14s} [{ht:8s}] N={n:3d} ceiling={ceiling:3d} best-method-hit={best:3d}  {desc}")


def print_examples(records: List[Record], per_type: int) -> None:
    print("\n=== EXAMPLES ===")
    hop_types = sorted({r.hop_type for r in records}, key=lambda h: (h != "quantity", h))
    for ht in hop_types:
        rows = [r for r in records if r.hop_type == ht]
        hits = [r for r in rows if r.methods["ceiling"] and any(
            r.methods[m] for m in METHOD_ORDER if r.methods[m] is not None
        )]
        misses = [r for r in rows if r.methods["ceiling"] and not any(
            r.methods[m] for m in METHOD_ORDER if r.methods[m] is not None
        )]
        print(f"\n-- {ht} --")
        for r in hits[:per_type]:
            won_by = [m for m in METHOD_ORDER if r.methods[m]]
            print(f"  HIT  {r.task_id}/{r.key} [{r.model}/{r.arm}/{r.shape}] via {won_by}")
            print(f"       ...{r.snippet}...")
        if not hits:
            print("  (no clean hits)")
        for r in misses[:per_type]:
            print(f"  MISS {r.task_id}/{r.key} [{r.model}/{r.arm}/{r.shape}] "
                  f"(present on page, len={r.evidence_len}, but no deterministic method found it)")
            print(f"       ...{r.snippet}...")
        if not misses:
            print("  (no clean misses)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results-glob", action="append", default=None,
        help="glob (repo-root-relative) for result files; repeatable. "
             "Default: agent/idea_test_results/{cschain_g,csnopar_g}_*.json",
    )
    ap.add_argument("--by-hop-type", action="store_true", help="print the per-hop-type recall table")
    ap.add_argument("--by-task", action="store_true", help="print the per-task/waypoint breakdown")
    ap.add_argument("--examples", type=int, default=0, metavar="N", help="print N hit + N miss examples per hop type")
    ap.add_argument(
        "--precision", action="store_true",
        help="measure false-positive rate: extractor applied to a page that is NOT the "
             "waypoint's correct source (gates the Step 2 design -- see module docstring)",
    )
    ap.add_argument(
        "--validate-product", action="store_true",
        help="re-run recall + precision against the actual shipped "
             "idea_policies.waypoint.build_waypoint (not a prototype)",
    )
    ap.add_argument(
        "--containment", action="store_true",
        help="does the ground truth reach the deterministic CANDIDATE SET (top-K, ranked by "
             "proximity to the cue), as opposed to being the single value picked -- the "
             "decisive number for a 'surface the candidate set, not a single value' design",
    )
    ap.add_argument(
        "--oracle", action="store_true",
        help="Step 1's free gate measurement: re-run recall/precision with a hand-authored, "
             "non-leaking target_datum (kind+label) substituted for the goal-text-derived cue, "
             "as a 4-way ablation (baseline / label-only / kind-only / full) so it's clear which "
             "half of the target_datum idea would carry the improvement -- see module docstring",
    )
    args = ap.parse_args()

    globs = args.results_glob or [
        "agent/idea_test_results/cschain_g_*.json",
        "agent/idea_test_results/csnopar_g_*.json",
    ]

    if args.oracle:
        modes = [
            ("baseline (both levers off -- must match --validate-product exactly)", False, False),
            ("label-only (oracle wording, today's anchor-fallback heuristic)", True, False),
            ("kind-only (today's wording, oracle-gated anchor dispatch)", False, True),
            ("full (oracle wording + oracle-gated dispatch)", True, True),
        ]
        for label, use_label, use_kind in modes:
            records, summary = run_oracle_mode(globs, use_label, use_kind)
            print_oracle_mode(label, records, summary)
        return 0

    if args.containment:
        containment_records = run_containment_analysis(globs)
        if not containment_records:
            print("No applicable trials found -- check --results-glob and repo layout.", file=sys.stderr)
            return 1
        print_containment(containment_records)
        return 0

    if args.validate_product:
        records = run_guarded_validation(globs)
        print_recall_breakdown(records, examples=args.examples)
        return 0

    if args.precision:
        precision_records = run_precision(globs)
        if not precision_records:
            print("No wrong-page trials found -- check --results-glob and repo layout.", file=sys.stderr)
            return 1
        print_precision(precision_records, examples=args.examples)
        return 0

    records = run(globs)

    if not records:
        print("No waypoint instances found -- check --results-glob and repo layout.", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} (run, waypoint) instances where the source page was actually fetched.")
    if NOT_MEASURED:
        print("\nExcluded tasks:")
        for tid, why in NOT_MEASURED.items():
            print(f"  {tid}: {why}")

    print_overall(records)
    if args.by_hop_type:
        print_by_hop_type(records)
    if args.by_task:
        print_by_task(records)
    if args.examples:
        print_examples(records, args.examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
