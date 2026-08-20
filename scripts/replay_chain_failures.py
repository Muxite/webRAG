#!/usr/bin/env python3
"""Failure-mode census for the cschain_g / csnopar_g chain-task benchmark cells.

Read-only diagnostic. Loads the 108 (54 + 54) result JSONs for the 9 sequential "chain" tasks
(135,136,137,138,139,046,047,065,093) x 3 models x 2 arms x {cschain_g, csnopar_g}, plus their
per-cell driver logs, and classifies each cell into a failure mode -- independently of the
validator's own ``overall_score`` / ``chain_coverage``, which this script's own findings show
can be gamed (a shallow "at least one page visited" grounding gate lets the finalize step
confabulate the rest of a chain from parametric memory).

Two independent signals feed the classification:
  1. The validator's OWN checks, already computed and stored under ``validation.grep_validations``
     (keystone_*, chain_coverage/coverage/chain_progress/path_adjacency).
  2. An INDEPENDENT re-derivation of "which page did the agent actually visit, in what order,
     and does it match the task's real waypoint sequence" -- built from
     ``execution.graph.nodes[*].details.action_result`` (the actual visit URLs/content) ordered
     via ``execution.observability.decisions.trace`` (real per-step timest<-ordered log), and
     matched against hand-verified ``slug_rx``/``name_rx`` waypoint regexes lifted from the task
     definitions in ``agent/app/idea_tests/test_{046,047,065,093,135..139}_*.py`` (read-only
     reference -- this script does not import or modify that package).

Usage:
    ./.venv/bin/python3 scripts/replay_chain_failures.py
    ./.venv/bin/python3 scripts/replay_chain_failures.py --run cschain_g --model flash
    ./.venv/bin/python3 scripts/replay_chain_failures.py --json-out /tmp/cells.json --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_DEFAULT = REPO_ROOT / "agent" / "idea_test_results"

FNAME_RX = re.compile(
    r"^(?P<runset>cschain_g|csnopar_g)_(?P<modelkey>[a-z0-9]+)_(?P<arm>baseline|good_adaptive)_"
    r"rep(?P<rep>\d+)_(?P<testid>\d{3})_(?P<modelslug>.+)_graph_r(?P<rnum>\d+)\.json$"
)

MODEL_LABELS = {
    "flash": "google/gemini-2.5-flash-lite",
    "nano": "openai/gpt-4.1-nano",
    "l1b": "meta-llama/llama-3.2-1b-instruct",
}

TASK_NAMES = {
    "046": "navigation_traverse (Apollo11->SaturnV)",
    "047": "graph_wikirace (Pizza->RomanEmpire)",
    "065": "leak_resistant_chain (poet->town elevation)",
    "093": "cve_chain_b (curl SOCKS5 advisory->source)",
    "135": "roebling_ohio_bridge_chain",
    "136": "brunel_great_eastern_chain",
    "137": "telford_pontcysyllte_chain",
    "138": "everest_waugh_survey_chain",
    "139": "gaudi_pedrera_chain",
}

# Waypoint specs, hand-transcribed (read-only reference, never imported) from the CHAIN / HOP_*
# regex objects that actually live in agent/app/idea_tests/test_{tid}_*.py. Used to independently
# re-derive "was the right page visited, in the right order" -- NOT the validator's own
# (gameable) chain_coverage/coverage score.
WAYPOINTS: Dict[str, Dict[str, Any]] = {
    "135": {"start_given": True, "chain": [
        {"key": "start", "name": "Brooklyn Bridge",
         "slug_rx": r"wiki/brooklyn_bridge", "name_rx": r"brooklyn\s+bridge"},
        {"key": "creator", "name": "John A. Roebling",
         "slug_rx": r"wiki/john_a\.?_roebling(?!_suspension)", "name_rx": r"john\s+a\.?\s+roebling|\broebling\b"},
        {"key": "terminal", "name": "Roebling Suspension Bridge (Cincinnati)",
         "slug_rx": r"roebling_suspension_bridge|wiki/[^ ]*cincinnati",
         "name_rx": r"cincinnati|covington|roebling_suspension|roebling\s+suspension"},
    ]},
    "136": {"start_given": True, "chain": [
        {"key": "start", "name": "Clifton Suspension Bridge",
         "slug_rx": r"wiki/clifton_suspension_bridge", "name_rx": r"clifton\s+suspension|clifton\s+bridge"},
        {"key": "creator", "name": "Isambard Kingdom Brunel",
         "slug_rx": r"wiki/isambard_kingdom_brunel", "name_rx": r"\bbrunel\b"},
        {"key": "terminal", "name": "SS Great Eastern",
         "slug_rx": r"wiki/ss_great_eastern|wiki/great_eastern", "name_rx": r"great\s+eastern"},
    ]},
    "137": {"start_given": True, "chain": [
        {"key": "start", "name": "Menai Suspension Bridge",
         "slug_rx": r"wiki/menai_suspension_bridge", "name_rx": r"menai\s+suspension|menai\s+bridge"},
        {"key": "creator", "name": "Thomas Telford",
         "slug_rx": r"wiki/thomas_telford", "name_rx": r"\btelford\b"},
        {"key": "terminal", "name": "Pontcysyllte Aqueduct",
         "slug_rx": r"wiki/pontcysyllte", "name_rx": r"pontcysyllte"},
    ]},
    "138": {"start_given": True, "chain": [
        {"key": "start", "name": "Mount Everest",
         "slug_rx": r"wiki/mount_everest", "name_rx": r"mount\s+everest|\beverest\b"},
        {"key": "creator", "name": "Andrew Scott Waugh",
         "slug_rx": r"wiki/andrew_scott_waugh", "name_rx": r"\bwaugh\b"},
        {"key": "terminal", "name": "1856 declared survey height",
         "slug_rx": r"wiki/george_everest", "name_rx": r"\b1856\b|great\s+trigonometric|declared"},
    ]},
    "139": {"start_given": True, "chain": [
        {"key": "start", "name": "Sagrada Familia",
         "slug_rx": r"wiki/sagrada_fam", "name_rx": r"sagrada\s+fam[ií]lia"},
        {"key": "creator", "name": "Antoni Gaudi",
         "slug_rx": r"wiki/antoni_gaud", "name_rx": r"gaud[ií]"},
        {"key": "terminal", "name": "Casa Mila (La Pedrera)",
         "slug_rx": r"wiki/casa_mil", "name_rx": r"casa\s+mil[àa]|pedrera"},
    ]},
    "065": {"start_given": False, "chain": [
        {"key": "poet", "name": "Pablo Neruda",
         "slug_rx": r"wiki/pablo_neruda", "name_rx": r"\bneruda\b|\breyes basoalto\b"},
        {"key": "town", "name": "Parral, Chile",
         "slug_rx": r"wiki/parral", "name_rx": r"\bparral\b"},
    ]},
    "046": {"start_given": True, "chain": [
        {"key": "start", "name": "Apollo 11", "slug_rx": r"wiki/apollo_11", "name_rx": r"apollo\s*11"},
        {"key": "target", "name": "Saturn V", "slug_rx": r"wiki/saturn_v", "name_rx": r"saturn\s*v"},
    ]},
}
# 093: both required URLs are SEEDED directly in the mandate (no navigation/derivation needed) --
# handled specially, not via the ordered-waypoint machinery above.
SOURCE_093_RX = re.compile(r"raw\.githubusercontent\.com/curl/curl.*socks\.c", re.IGNORECASE)
ADVISORY_093_RX = re.compile(r"curl\.se/docs/cve-2023-38545", re.IGNORECASE)
# 047: open-ended pathfinding (no fixed intermediate waypoint) -- relies on the validator's own
# keystone_verified_chain / chain_progress checks rather than a hand-authored waypoint sequence.

KNOWN_ACTIONS = {"visit", "search", "merge", "think", "verify"}
PLACEHOLDER_RX = re.compile(
    r"\{[^}]{0,80}\}|<[^<>]{0,80}>|\[[^\[\]]{0,80}\]|from the previous step|"
    r"identified in (?:the )?(?:previous )?step|in step \d+|to be determined|previous step",
    re.IGNORECASE,
)
INFRA_ERROR_RX = re.compile(
    r"HTTP fetch failed|timeout|timed out|All URL visits failed|connection (?:refused|reset)|"
    r"blocked|status=(?:4\d\d|5\d\d)",
    re.IGNORECASE,
)


# Loading / filename parsing

def normalize_url(u: str) -> str:
    s = str(u or "").strip().lower()
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    s = s.split("#", 1)[0]
    return s.rstrip("/")


def discover_files(results_dir: Path, run_filter: Optional[str], model_filter: Optional[str],
                    task_filter: Optional[List[str]]) -> List[Tuple[Path, Dict[str, str]]]:
    out = []
    for p in sorted(results_dir.glob("*_graph_r*.json")):
        m = FNAME_RX.match(p.name)
        if not m:
            continue
        g = m.groupdict()
        if run_filter and g["runset"] != run_filter:
            continue
        if model_filter and g["modelkey"] != model_filter:
            continue
        if task_filter and g["testid"] not in task_filter:
            continue
        out.append((p, g))
    return out


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARN: could not load {p}: {e}", file=sys.stderr)
        return None


def get_checks(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    val = d.get("validation") or {}
    return {c.get("check"): c for c in (val.get("grep_validations") or [])}


def keystone_check_name(checks: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for name in checks:
        if name and name.startswith("keystone"):
            return name
    return None


def traversal_check_name(checks: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for name in ("chain_coverage", "coverage", "chain_progress", "path_adjacency"):
        if name in checks:
            return name
    return None


# Node / visit / decision-trace analysis

def build_action_positions(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    """node_id-prefix -> chronological position, from decisions.trace 'action' stage entries."""
    pos = {}
    i = 0
    for t in trace:
        if t.get("stage") != "action":
            continue
        i += 1
        prefix = t.get("node_id") or ""
        if prefix and prefix not in pos:
            pos[prefix] = i
    return pos


def pos_for_node(node_id: str, positions: Dict[str, int]) -> Optional[int]:
    for prefix, i in positions.items():
        if node_id.startswith(prefix):
            return i
    return None


def collect_visit_events(nodes: Dict[str, Any], positions: Dict[str, int]) -> List[Dict[str, Any]]:
    events = []
    for nid, n in nodes.items():
        det = n.get("details") or {}
        ar = det.get("action_result")
        if not isinstance(ar, dict) or ar.get("action") != "visit":
            continue
        urls = ar.get("urls_visited") or ([ar.get("url")] if ar.get("url") else [])
        for u in urls:
            if not u:
                continue
            events.append({
                "node_id": nid,
                "pos": pos_for_node(nid, positions),
                "url": u,
                "success": bool(ar.get("success")),
                "title": str(ar.get("page_title") or ar.get("h1_text") or ""),
                "content": str(ar.get("content_full") or ar.get("content") or "")[:3000],
            })
    events.sort(key=lambda e: (e["pos"] is None, e["pos"] if e["pos"] is not None else 0))
    return events


def match_waypoint(url: str, title: str, content: str, chain: List[Dict[str, Any]]) -> Optional[int]:
    low_url = (url or "").lower()
    for i, wp in enumerate(chain):
        if re.search(wp["slug_rx"], low_url, re.IGNORECASE):
            return i
    text = f"{title} {content}".lower()
    for i, wp in enumerate(chain):
        if re.search(wp["name_rx"], text, re.IGNORECASE):
            return i
    return None


def classify_traversal(test_id: str, visit_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Independent ground-truth traversal classification (NOT the validator's chain_coverage)."""
    if test_id == "093":
        urls_low = " ".join(normalize_url(e["url"]) for e in visit_events if e["success"])
        return {
            "applicable": True, "mode": "given_urls",
            "source_visited": bool(SOURCE_093_RX.search(urls_low)),
            "advisory_visited": bool(ADVISORY_093_RX.search(urls_low)),
            "n_distinct_visits": len({normalize_url(e["url"]) for e in visit_events if e["success"]}),
        }
    spec = WAYPOINTS.get(test_id)
    if not spec:
        return {"applicable": False}
    chain = spec["chain"]
    start_given = spec["start_given"]
    seen: List[Dict[str, Any]] = []
    seen_urls = set()
    for e in visit_events:
        if not e["success"]:
            continue
        nu = normalize_url(e["url"])
        if nu in seen_urls:
            continue
        seen_urls.add(nu)
        seen.append(e)
    matched = [match_waypoint(e["url"], e["title"], e["content"], chain) for e in seen]
    n = len(chain)
    terminal_idx = n - 1
    reached_terminal = terminal_idx in matched
    reached_max_idx = max([m for m in matched if m is not None], default=-1)
    hop1_slot = 1 if start_given else 0          # visit-sequence slot that SHOULD carry hop-1
    hop1_target_idx = 1 if start_given else 0    # CHAIN index of the hop-1 waypoint
    if len(matched) > hop1_slot:
        hop1_status = "correct" if matched[hop1_slot] == hop1_target_idx else "wrong"
    else:
        hop1_status = "not_attempted"
    return {
        "applicable": True, "mode": "chain",
        "n_waypoints": n, "n_distinct_visits": len(seen),
        "matched_sequence": matched, "reached_terminal": reached_terminal,
        "reached_max_idx": reached_max_idx, "hop1_status": hop1_status,
        "visited_urls": [normalize_url(e["url"]) for e in seen],
    }


def collect_node_errors(nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs = []
    for nid, n in nodes.items():
        if n.get("status") != "failed":
            continue
        det = n.get("details") or {}
        ar = det.get("action_result")
        msg = ""
        if isinstance(ar, dict):
            msg = str(ar.get("error") or ar.get("reason") or "")
        errs.append({"node_id": nid, "action": det.get("action"), "message": msg})
    return errs


def collect_placeholder_nodes(nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for nid, n in nodes.items():
        det = n.get("details") or {}
        for key in ("query", "link_idea", "optional_url"):
            v = det.get(key)
            if isinstance(v, str) and PLACEHOLDER_RX.search(v):
                ar = det.get("action_result")
                executed = isinstance(ar, dict)
                success = bool(ar.get("success")) if executed else None
                out.append({
                    "node_id": nid, "field": key, "value": v[:160],
                    "status": n.get("status"), "action": det.get("action"),
                    "executed": executed, "success": success,
                })
    return out


def collect_malformed_actions(nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for nid, n in nodes.items():
        det = n.get("details") or {}
        a = det.get("action")
        if a and a not in KNOWN_ACTIONS:
            ar = det.get("action_result")
            fell_back_to_think = isinstance(ar, dict) and ar.get("action") == "think"
            out.append({"node_id": nid, "declared_action": a, "fell_back_to_think": fell_back_to_think,
                         "goal": str(det.get("goal") or "")[:160]})
    return out


def count_think_nodes(nodes: Dict[str, Any]) -> Tuple[int, int]:
    declared = sum(1 for n in nodes.values() if (n.get("details") or {}).get("action") == "think")
    executed = sum(1 for n in nodes.values()
                   if isinstance((n.get("details") or {}).get("action_result"), dict)
                   and (n.get("details") or {}).get("action_result", {}).get("action") == "think")
    return declared, executed


def repeated_url_revisits(visit_events: List[Dict[str, Any]]) -> Dict[str, int]:
    """Same normalized URL fetched by >=2 DIFFERENT nodes -- evidence of loop-like re-fetching
    with no new evidence gathered (a variant of the 're-expansion loop' failure the task asks
    about, manifesting across sibling nodes rather than on a single re-expanded node)."""
    by_url_nodes: Dict[str, set] = defaultdict(set)
    for e in visit_events:
        if e["success"]:
            by_url_nodes[normalize_url(e["url"])].add(e["node_id"])
    return {u: len(ns) for u, ns in by_url_nodes.items() if len(ns) >= 2}


def grounding_gate_summary(decisions: Dict[str, Any]) -> Dict[str, Any]:
    trace = decisions.get("trace") or []
    has_finalize = any(t.get("stage") == "finalize" for t in trace)
    grounded_at_dv = None
    missing_seen = []
    grounding_chosen_seq = []
    for t in trace:
        if t.get("stage") == "grounding":
            grounding_chosen_seq.append(t.get("chosen"))
            miss = t.get("metadata", {}).get("missing") or []
            if miss:
                missing_seen.append(tuple(miss))
            if t.get("chosen") == "grounded" and grounded_at_dv is None:
                grounded_at_dv = t.get("metadata", {}).get("distinct_visits")
    replans = decisions.get("by_stage", {}).get("grounding")
    fin = next((t for t in reversed(trace) if t.get("stage") == "finalize"), None)
    # Did the engine ever actually reach "grounded", or did it dead-end (give up on replanning
    # with the gate still unsatisfied) and finalize anyway? Both paths reach `finalize`, but
    # only the first genuinely satisfied the gate -- distinct failure mechanisms.
    reached_grounded = "grounded" in grounding_chosen_seq
    dead_end_no_followup = (not reached_grounded) and any(
        c == "ungrounded-no-followup" for c in grounding_chosen_seq
    )
    return {
        "has_finalize": has_finalize,
        "distinct_visits_at_first_grounded": grounded_at_dv,
        "reached_grounded": reached_grounded,
        "dead_end_no_followup": dead_end_no_followup,
        "grounding_chosen_sequence": grounding_chosen_seq,
        "missing_phrases": missing_seen,
        "finalize_replans": (fin or {}).get("metadata", {}).get("replans"),
        "finalize_missing": (fin or {}).get("metadata", {}).get("missing"),
    }


# Cell-log (AUTO-PARALLEL / DEPENDENCY DETECTED) analysis

AUTO_PARALLEL_RX = re.compile(r"\[STEP (\d+)\] AUTO-PARALLEL: (\d+) independent leaf siblings")
DEPENDENCY_RX = re.compile(r"\[STEP (\d+)\] DEPENDENCY DETECTED: Forcing sequential")
SIBLING_DEP_EVIDENCE_RX = re.compile(
    r"Found URL from sibling results|Extracted URL from sibling results|"
    r"Extracted URLs? from search node|extract_urls_from_search"
)


def cell_log_path(log_dir: Path, run_prefix: str, meta: Dict[str, str]) -> Path:
    return log_dir / f"{run_prefix}_{meta['modelkey']}_{meta['arm']}_rep{meta['rep']}_{meta['testid']}.log"


def analyze_cell_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"found": False}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    auto_parallel_events = []
    dependency_forced = 0
    for i, line in enumerate(lines):
        m = AUTO_PARALLEL_RX.search(line)
        if m:
            step, n_sib = int(m.group(1)), int(m.group(2))
            # look ahead a generous window (until the next STEP N/ boundary) for evidence that a
            # sibling actually depended on another sibling's output despite being scheduled
            # "independent" and concurrent.
            window = []
            for j in range(i + 1, min(i + 60, len(lines))):
                if re.search(r"=== STEP \d+/\d+ ===", lines[j]):
                    break
                window.append(lines[j])
            suspicious = any(SIBLING_DEP_EVIDENCE_RX.search(w) for w in window)
            auto_parallel_events.append({"step": step, "n_siblings": n_sib, "suspicious_dependency": suspicious})
        if DEPENDENCY_RX.search(line):
            dependency_forced += 1
    return {
        "found": True,
        "auto_parallel_count": len(auto_parallel_events),
        "auto_parallel_suspicious_count": sum(1 for e in auto_parallel_events if e["suspicious_dependency"]),
        "auto_parallel_events": auto_parallel_events,
        "dependency_forced_count": dependency_forced,
    }


# Per-cell record + primary failure-mode classification

def build_cell_record(path: Path, meta: Dict[str, str], log_dir: Path) -> Optional[Dict[str, Any]]:
    d = load_json(path)
    if d is None:
        return None
    test_id = meta["testid"]
    checks = get_checks(d)
    ks_name = keystone_check_name(checks)
    tr_name = traversal_check_name(checks)
    ks = checks.get(ks_name, {}) if ks_name else {}
    tr = checks.get(tr_name, {}) if tr_name else {}
    val = d.get("validation") or {}
    ex = d.get("execution") or {}
    obs = ex.get("observability") or {}
    graph = ex.get("graph") or {}
    nodes = graph.get("nodes") or {}
    decisions = obs.get("decisions") or {}
    trace = decisions.get("trace") or []
    positions = build_action_positions(trace)
    visit_events = collect_visit_events(nodes, positions)
    trav = classify_traversal(test_id, visit_events)
    node_errors = collect_node_errors(nodes)
    placeholders = collect_placeholder_nodes(nodes)
    malformed = collect_malformed_actions(nodes)
    think_declared, think_executed = count_think_nodes(nodes)
    revisits = repeated_url_revisits(visit_events)
    grounding = grounding_gate_summary(decisions)

    infra_failed_flag = bool(d.get("infra_failed"))
    n_visits_final = int((obs.get("visit") or {}).get("count", 0) or 0)
    n_search = int((obs.get("search") or {}).get("count", 0) or 0)
    usd = ((obs.get("cost") or {}).get("usd"))
    overall_score = val.get("overall_score")
    keystone_passed = bool(ks.get("passed")) if ks else None

    # ---- primary failure-mode classification -------------------------------------------------
    mode = None
    detail = ""
    infra_error_nodes = [e for e in node_errors if INFRA_ERROR_RX.search(e["message"] or "")]

    if keystone_passed:
        mode = "WON"
        detail = "keystone check passed"
    elif infra_failed_flag:
        mode = "TOOL_FAILURE_INFRA"
        detail = "top-level infra_failed=true"
    elif not grounding.get("has_finalize"):
        mode = "BUDGET_EXHAUSTION_SILENT"
        detail = "run has no finalize decision -- cut off externally, no error logged"
    elif trav.get("applicable") and trav.get("mode") == "chain":
        if trav["reached_terminal"]:
            mode = "TRAVERSAL_RIGHT_ANSWER_WRONG"
            detail = (f"visited the terminal page ({trav['visited_urls'][-1] if trav['visited_urls'] else '?'}) "
                      f"but keystone extraction/synthesis was wrong")
        elif trav["hop1_status"] == "wrong":
            mode = "WRONG_HOP1"
            detail = f"2nd required hop landed off-chain; visited={trav['visited_urls']}"
        elif trav["n_distinct_visits"] <= 1:
            mode = "NEVER_REACHED_TERMINUS"
            if grounding.get("dead_end_no_followup"):
                detail = (f"DEAD-END: grounding never satisfied (missing="
                          f"{grounding.get('finalize_missing')}) but engine gave up replanning and "
                          f"finalized anyway with {trav['n_distinct_visits']} visit(s)")
            else:
                detail = (f"only {trav['n_distinct_visits']} distinct page(s) visited before engine "
                          f"finalized (grounding gate satisfied at distinct_visits="
                          f"{grounding.get('distinct_visits_at_first_grounded')})")
        else:
            mode = "NEVER_REACHED_TERMINUS"
            detail = f"reached waypoint idx {trav['reached_max_idx']}/{trav['n_waypoints']-1}, stopped short"
    elif trav.get("applicable") and trav.get("mode") == "given_urls":
        both = trav["source_visited"] and trav["advisory_visited"]
        if both:
            mode = "TRAVERSAL_RIGHT_ANSWER_WRONG"
            detail = "both seeded URLs visited but exact identifier extraction wrong"
        else:
            mode = "NEVER_REACHED_TERMINUS"
            detail = f"source_visited={trav['source_visited']}, advisory_visited={trav['advisory_visited']}"
    else:
        # 047 (open pathfinding) and any other non-applicable case: fall back to the validator's
        # own traversal-proxy check.
        if tr and tr.get("score", 0) and tr.get("score") >= 0.99:
            mode = "TRAVERSAL_RIGHT_ANSWER_WRONG"
            detail = f"{tr_name}={tr.get('reason')}"
        elif infra_error_nodes:
            mode = "TOOL_FAILURE_INFRA"
            detail = f"{len(infra_error_nodes)} node(s) with infra-flavored errors"
        else:
            mode = "NEVER_REACHED_TERMINUS"
            detail = f"{ks_name}={ks.get('reason') if ks else 'n/a'} | {tr_name}={tr.get('reason') if tr else 'n/a'}"

    if mode != "WON" and infra_error_nodes and mode != "TOOL_FAILURE_INFRA":
        # keep the primary mode but tag that partial tool failures were also present
        detail += f" [+{len(infra_error_nodes)} partial node error(s), run still finalized]"

    return {
        "path": str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path),
        "runset": meta["runset"], "modelkey": meta["modelkey"], "model": MODEL_LABELS.get(meta["modelkey"], meta["modelkey"]),
        "arm": meta["arm"], "test_id": test_id, "task_name": TASK_NAMES.get(test_id, test_id),
        "overall_score": overall_score, "keystone_check": ks_name, "keystone_passed": keystone_passed,
        "traversal_check": tr_name, "traversal_score": tr.get("score"), "traversal_reason": tr.get("reason"),
        "mode": mode, "mode_detail": detail,
        "n_visits_final": n_visits_final, "n_search": n_search, "usd": usd,
        "infra_failed_flag": infra_failed_flag,
        "node_error_count": len(node_errors), "node_errors": node_errors,
        "infra_error_node_count": len(infra_error_nodes),
        "placeholder_nodes": placeholders,
        "malformed_action_nodes": malformed,
        "think_declared": think_declared, "think_executed": think_executed,
        "repeated_url_revisits": revisits,
        "grounding": grounding,
        "traversal": trav,
        "log": analyze_cell_log(cell_log_path(log_dir, meta["runset"], meta)),
    }


# Reporting

def fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0*n/d:.0f}%)" if d else f"{n}/0"


def print_census(cells: List[Dict[str, Any]]) -> None:
    modes = ["WON", "TRAVERSAL_RIGHT_ANSWER_WRONG", "WRONG_HOP1", "NEVER_REACHED_TERMINUS",
             "TOOL_FAILURE_INFRA", "BUDGET_EXHAUSTION_SILENT"]

    print("=" * 100)
    print("FAILURE-MODE CENSUS -- overall")
    print("=" * 100)
    c = Counter(x["mode"] for x in cells)
    for m in modes:
        print(f"  {m:32s} {fmt_pct(c.get(m, 0), len(cells))}")
    other = set(c) - set(modes)
    for m in other:
        print(f"  {m:32s} {fmt_pct(c[m], len(cells))}")
    print()

    def table(key_fn, label):
        print("-" * 100)
        print(f"By {label}")
        print("-" * 100)
        keys = sorted({key_fn(x) for x in cells})
        header = f"{'':20s}" + "".join(f"{m[:14]:>16s}" for m in modes) + f"{'TOTAL':>8s}"
        print(header)
        for k in keys:
            sub = [x for x in cells if key_fn(x) == k]
            cc = Counter(x["mode"] for x in sub)
            row = f"{str(k):20s}" + "".join(f"{cc.get(m,0):>16d}" for m in modes) + f"{len(sub):>8d}"
            print(row)
        print()

    table(lambda x: x["runset"], "run (cschain_g vs csnopar_g)")
    table(lambda x: x["arm"], "arm")
    table(lambda x: x["modelkey"], "model")
    table(lambda x: x["test_id"], "task")
    print("-" * 100)
    print("By arm x model (rows) -- WON count / total")
    print("-" * 100)
    for runset in sorted({x["runset"] for x in cells}):
        for arm in sorted({x["arm"] for x in cells}):
            for mk in sorted({x["modelkey"] for x in cells}):
                sub = [x for x in cells if x["runset"] == runset and x["arm"] == arm and x["modelkey"] == mk]
                if not sub:
                    continue
                won = sum(1 for x in sub if x["mode"] == "WON")
                print(f"  {runset:10s} {arm:14s} {mk:6s} won={fmt_pct(won, len(sub))}")
    print()


def print_unintended_behaviors(cells: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print("UNINTENDED BEHAVIOR: unfilled query-slot placeholders that EXECUTED anyway")
    print("=" * 100)
    total = 0
    executed_ok = 0
    executed_fail = 0
    skipped = 0
    for x in cells:
        for ph in x["placeholder_nodes"]:
            total += 1
            if ph["executed"] and ph["success"]:
                executed_ok += 1
            elif ph["executed"] and ph["success"] is False:
                executed_fail += 1
            else:
                skipped += 1
    print(f"  total placeholder-bearing nodes found: {total}")
    print(f"    executed and SUCCEEDED (repaired via sibling/search URL-extraction fallback): {executed_ok}")
    print(f"    executed and FAILED outright ('missing valid URL or link_idea' etc.): {executed_fail}")
    print(f"    never executed (skipped/pending): {skipped}")
    print("  examples:")
    shown = 0
    for x in cells:
        for ph in x["placeholder_nodes"]:
            print(f"    {x['path']}  node={ph['node_id']}  field={ph['field']}  "
                  f"status={ph['status']} executed={ph['executed']} success={ph['success']}")
            print(f"        value: {ph['value']!r}")
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break
    print()

    print("=" * 100)
    print("UNINTENDED BEHAVIOR: 'think' nodes (ThinkLeafAction -- no LLM call)")
    print("=" * 100)
    declared_total = sum(x["think_declared"] for x in cells)
    executed_total = sum(x["think_executed"] for x in cells)
    print(f"  declared as think (details.action=='think'): {declared_total}")
    print(f"  actually executed as think (action_result.action=='think'): {executed_total}")
    print(f"    (delta = {executed_total - declared_total} nodes whose DECLARED action was something else "
          f"entirely -- a malformed/hallucinated action name that silently fell back to a no-op think)")
    by_model = Counter()
    for x in cells:
        if x["think_executed"]:
            by_model[x["modelkey"]] += x["think_executed"]
    print(f"  think-executions by model: {dict(by_model)}")
    print()

    print("=" * 100)
    print("UNINTENDED BEHAVIOR: malformed / out-of-vocabulary action names")
    print("=" * 100)
    all_malformed = [(x, m) for x in cells for m in x["malformed_action_nodes"]]
    print(f"  total nodes with a declared action outside {sorted(KNOWN_ACTIONS)}: {len(all_malformed)}")
    for x, m in all_malformed:
        print(f"    {x['path']}  node={m['node_id']}  declared_action={m['declared_action']!r}  "
              f"fell_back_to_think={m['fell_back_to_think']}  goal={m['goal']!r}")
    print()

    print("=" * 100)
    print("UNINTENDED BEHAVIOR: AUTO-PARALLEL firing on siblings with runtime dependency evidence")
    print("=" * 100)
    ap_total = 0
    ap_susp = 0
    dep_forced_total = 0
    examples = []
    for x in cells:
        log = x["log"]
        if not log.get("found"):
            continue
        ap_total += log["auto_parallel_count"]
        ap_susp += log["auto_parallel_suspicious_count"]
        dep_forced_total += log["dependency_forced_count"]
        for ev in log["auto_parallel_events"]:
            if ev["suspicious_dependency"]:
                examples.append((x["path"], ev["step"], ev["n_siblings"]))
    print(f"  total AUTO-PARALLEL firings across all cell logs: {ap_total}")
    print(f"  AUTO-PARALLEL firings with sibling-dependency evidence in the surrounding log window "
          f"(a batched sibling actually needed another sibling's URL/search output): {ap_susp}")
    print(f"  DEPENDENCY DETECTED / Forcing sequential firings (the engine's own guard catching a "
          f"real dependency and NOT parallelizing): {dep_forced_total}")
    print("  suspicious AUTO-PARALLEL examples (file, step, n_siblings):")
    for e in examples[:15]:
        print(f"    {e}")
    print()

    print("=" * 100)
    print("UNINTENDED BEHAVIOR: repeated re-fetch of the identical URL by different nodes")
    print("=" * 100)
    revisit_cells = [x for x in cells if x["repeated_url_revisits"]]
    print(f"  cells with >=1 URL fetched by >=2 different nodes: {len(revisit_cells)}/{len(cells)}")
    for x in revisit_cells[:10]:
        print(f"    {x['path']}: {x['repeated_url_revisits']}")
    print()

    print("=" * 100)
    print("ROOT-CAUSE MECHANISM: shallow grounding gate (checks 'any visit', not 'the right visit')")
    print("=" * 100)
    dvs = Counter()
    leak_examples = []
    for x in cells:
        dv = x["grounding"].get("distinct_visits_at_first_grounded")
        if dv is not None:
            dvs[dv] += 1
        if dv == 1 and x["keystone_passed"] and x["test_id"] in WAYPOINTS and len(WAYPOINTS[x["test_id"]]["chain"]) >= 3:
            leak_examples.append(x)
    print(f"  distribution of distinct_visits at the moment the grounding gate first fired 'grounded': {dict(sorted(dvs.items(), key=lambda kv: (kv[0] is None, kv[0])))}")
    print(f"  cells grounded at distinct_visits==1 on a >=3-hop chain task that STILL passed the keystone "
          f"(parametric-memory leak through the shallow gate): {len(leak_examples)}")
    for x in leak_examples[:10]:
        print(f"    {x['path']}  overall_score={x['overall_score']}")
    print()

    print("=" * 100)
    print("ROOT-CAUSE MECHANISM: dead-end finalize (grounding gate never satisfied, but the engine")
    print("gives up on replanning and finalizes anyway -- 0/1 visits, pure parametric confabulation)")
    print("=" * 100)
    dead_end = [x for x in cells if x["grounding"].get("dead_end_no_followup")]
    print(f"  {len(dead_end)}/{len(cells)} cells hit 'ungrounded-no-followup' and finalized anyway")
    for x in dead_end[:15]:
        print(f"    {x['path']}  n_visits_final={x['n_visits_final']}  keystone_passed={x['keystone_passed']}  overall={x['overall_score']}")
    print()

    print("=" * 100)
    print("Runs cut off with no 'finalize' decision at all (true silent budget/step exhaustion)")
    print("=" * 100)
    no_fin = [x for x in cells if not x["grounding"].get("has_finalize")]
    print(f"  {len(no_fin)}/{len(cells)} cells never reached a finalize decision")
    for x in no_fin:
        print(f"    {x['path']}")
    print()

    print("=" * 100)
    print("Top-level infra_failed=true cells")
    print("=" * 100)
    inf = [x for x in cells if x["infra_failed_flag"]]
    print(f"  {len(inf)}/{len(cells)} cells")
    for x in inf:
        print(f"    {x['path']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--run", choices=["cschain_g", "csnopar_g"], default=None,
                     help="restrict to one run (default: both)")
    ap.add_argument("--model", choices=sorted(MODEL_LABELS), default=None,
                     help="restrict to one model key (default: all)")
    ap.add_argument("--tasks", type=str, default=None,
                     help="comma-separated test_ids to restrict to (default: all 9)")
    ap.add_argument("--json-out", type=Path, default=None, help="dump full per-cell records as JSON")
    ap.add_argument("--verbose", action="store_true", help="print every cell's classification")
    args = ap.parse_args()

    task_filter = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    files = discover_files(args.results_dir, args.run, args.model, task_filter)
    if not files:
        print("No matching result files found.", file=sys.stderr)
        sys.exit(1)

    cells = []
    for p, meta in files:
        log_dir = args.results_dir / f"_{meta['runset']}" / "cell_logs"
        rec = build_cell_record(p, meta, log_dir)
        if rec is not None:
            cells.append(rec)

    print(f"Loaded {len(cells)} cells from {len(files)} matched files under {args.results_dir}\n")

    if args.verbose:
        print("=" * 100)
        print("PER-CELL CLASSIFICATION")
        print("=" * 100)
        for x in sorted(cells, key=lambda x: (x["test_id"], x["runset"], x["modelkey"], x["arm"])):
            print(f"{x['test_id']} {x['runset']:10s} {x['modelkey']:6s} {x['arm']:14s} "
                  f"score={x['overall_score']!s:>20s} mode={x['mode']:28s} {x['mode_detail']}")
        print()

    print_census(cells)
    print_unintended_behaviors(cells)

    if args.json_out:
        args.json_out.write_text(json.dumps(cells, indent=2, default=str), encoding="utf-8")
        print(f"Wrote full per-cell records to {args.json_out}")


if __name__ == "__main__":
    main()
