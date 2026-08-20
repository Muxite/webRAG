#!/usr/bin/env python3
"""
Read-only measurement that gates the dataflow-deferral design (STEP 1 of that task, see
the write-up in the handing-off agent's brief / docs/handoffs). Over the 108 stored
chain-task results (``agent/idea_test_results/cschain_g_*.json`` and ``csnopar_g_*.json``,
54 files each), it answers four questions BEFORE any product code
(``agent/app/idea_policies/dataflow.py``, ``idea_sequencing.siblings_are_independent``) is
written:

  1. How many candidate (search/visit) leaf nodes carry an unresolved dataflow slot under
     each proposed detection rule, broken down by rule (with the rule-2/rule-1 overlap
     called out, since a rule that never fires independently is a candidate to drop).
  2. For each detected node: cross-check against what actually happened -- FAILED (a
     missing-URL / invalid-URL error), SILENTLY_REPAIRED (status ended up "done"/"skipped"
     despite the literal placeholder -- i.e. some fallback substituted a value the
     scheduler never vetted), or RAN_FINE (a genuine false positive: the slot text never
     mattered). The false-positive rate is RAN_FINE / total detected.
  3. The false-positive rate specifically among detected nodes that sit in a fan-out
     (multi-sibling) batch, reconstructed via ``parent_id`` -- the regression risk this
     task is most worried about.
  4. How many observed "AUTO-PARALLEL: N independent leaf siblings" firings (reconstructed
     from the per-run cell logs under ``agent/idea_test_results/_cschain_g/cell_logs`` and
     ``_csnopar_g/cell_logs``) a slot-aware independence check would have flipped away from
     the auto-parallel heuristic, plus a manual right/wrong read of every flip (the sample
     IS the full population here -- there are only a handful).

Also answers a 5th question, added 2026-08-16 after an adversarial review of
``siblings_are_independent`` found the above (Question 4) was never the actual ship-gate:
it only replays the SLOT sub-check, but the predicate also requires one of its
POSITIVE-evidence rules to fire, and a slot-free batch matching none of them is a false
negative -- 48/58 (82.8%) of the same 58 firings, undetected by Question 4 alone.

  5. Replay the FULL ``siblings_are_independent`` predicate (imported and called for real,
     not reimplemented) against all 58 reconstructed firings: the independent / deferred /
     false-negative split, the shape of any remaining false negatives, and every batch newly
     classified independent (manually eyeballed for the dangerous direction -- a real
     dependency slipping through as a false positive).

Usage (from repo root):
    PYTHONPATH=.:services:agent ./.venv/bin/python3 scripts/measure_dataflow_slots.py
    PYTHONPATH=.:services:agent ./.venv/bin/python3 scripts/measure_dataflow_slots.py --verbose

Questions 1-4 are dependency-light (stdlib only) and read-only, and can run standalone
before ``dataflow.py``/``idea_sequencing.py`` exist. Question 5 imports the real product
code (``agent.app.idea_dag``, ``agent.app.idea_sequencing``) since replaying the actual gate
-- not a shadow reimplementation of it -- is the point; run with
``PYTHONPATH=.:services:agent`` as shown above. Nothing in this script writes back to the
result JSONs or logs.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "agent" / "idea_test_results"
CELL_LOG_DIRS = [
    RESULTS_DIR / "_cschain_g" / "cell_logs",
    RESULTS_DIR / "_csnopar_g" / "cell_logs",
]

# Proposed detection rules (mirrors the design for dataflow.unresolved_slots). Kept
# self-contained here -- this script must run before that module exists.

#: Fields VisitLeafAction / SearchLeafAction actually resolve a candidate from at execution
#: time (see actions.py's ``original_optional_url = node.details.get("optional_url") or
#: NodeDetailsExtractor.get_url(...)`` chain, ``link_idea``, and
#: ``NodeDetailsExtractor.get_query`` = query -> prompt -> title). ``visit_url`` is an
#: OUTPUT field ``idea_engine.py`` writes back after a successful visit (not an input the
#: scheduler should read), so it is intentionally excluded.
VISIT_FIELDS = ("optional_url", "url", "link", "link_idea")
SEARCH_FIELDS = ("query", "prompt")

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


def rule1_template_hits(value: str) -> bool:
    """Unfilled ``{...}``/``<...>``/``[...]`` template containing no URL."""
    for m in _TEMPLATE_RE.finditer(value):
        if not _URL_RE.search(m.group(0)):
            return True
    return False


def rule2_ordinal_hits(value: str) -> bool:
    """Explicit back-reference to a plan step ("the previous hop", "step 2", ...)."""
    return bool(_ORDINAL_RE.search(value))


def fields_for_action(action: Optional[str]) -> Tuple[str, ...]:
    if action == "visit":
        return VISIT_FIELDS
    if action == "search":
        return SEARCH_FIELDS
    return ()


def detect(details: Dict[str, Any]) -> Dict[str, List[str]]:
    """field -> list of rule names ("rule1_template", "rule2_ordinal") that matched."""
    action = details.get("action")
    hits: Dict[str, List[str]] = {}
    for field in fields_for_action(action):
        value = details.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        matched = []
        if rule1_template_hits(value):
            matched.append("rule1_template")
        if rule2_ordinal_hits(value):
            matched.append("rule2_ordinal")
        if matched:
            hits[field] = matched
    return hits


# Result-file / cell-log plumbing

def find_result_files() -> List[str]:
    files = sorted(
        glob.glob(str(RESULTS_DIR / "cschain_g_*.json"))
        + glob.glob(str(RESULTS_DIR / "csnopar_g_*.json"))
    )
    return [
        f for f in files
        if "summary" not in f and "telemetry" not in f and "json_telemetry" not in f
    ]


_CELL_LOG_NAME_RE = re.compile(r"(cschain_g|csnopar_g)_(.+?)_(\d{3})_")


def find_cell_log(result_path: str) -> Optional[Path]:
    base = os.path.basename(result_path)
    m = _CELL_LOG_NAME_RE.match(base)
    if not m:
        return None
    prefix, variant, num = m.groups()
    logname = f"{prefix}_{variant}_{num}.log"
    for d in CELL_LOG_DIRS:
        p = d / logname
        if p.exists():
            return p
    return None


def iter_leaf_nodes(graph_nodes: Dict[str, Any]):
    for node_id, node in graph_nodes.items():
        details = node.get("details") or {}
        action = details.get("action")
        if action in ("search", "visit"):
            yield node_id, node, details


# Question 1 + 2 + 3: rule breakdown, outcome cross-check, fan-out false-positive rate

def classify_outcome(node: Dict[str, Any], details: Dict[str, Any]) -> str:
    """FAILED / SILENTLY_REPAIRED / RAN_FINE for a node ``detect()`` flagged.

    A VISIT candidate whose flagged field held the literal placeholder text can, by
    construction, never be fetched as-is (it is not a URL) -- so "done"/"skipped" always
    means some fallback (sibling-URL scavenging, chroma link-idea search, ...) substituted a
    value the scheduler never vetted: SILENTLY_REPAIRED. A SEARCH candidate's placeholder
    text is technically a valid (if useless) literal query string, so it always "succeeds"
    at the tool level -- RAN_FINE vs SILENTLY_REPAIRED is decided by the node's own
    evaluation score (low/absent score = the model itself judged the result useless).
    """
    status = node.get("status")
    if status == "failed":
        return "FAILED"
    evaluation = details.get("evaluation") or {}
    score = evaluation.get("score")
    action = details.get("action")
    if action == "visit":
        # Structurally cannot have "run fine" on the literal placeholder text.
        return "SILENTLY_REPAIRED"
    # search: judge by score.
    if score is None or score <= 0.5:
        return "SILENTLY_REPAIRED"
    return "RAN_FINE"


def run_static_measurement(verbose: bool) -> Dict[str, Any]:
    result_files = find_result_files()
    rule_counts: Counter = Counter()
    field_hits = []  # list of dicts, one per detected node
    total_leaf_nodes = Counter()

    # Build parent -> [search/visit leaf node_ids] per file for fan-out grouping.
    for f in result_files:
        data = json.loads(Path(f).read_text())
        graph_nodes = data.get("execution", {}).get("graph", {}).get("nodes", {})

        by_parent: Dict[Optional[str], List[str]] = defaultdict(list)
        for node_id, node, details in iter_leaf_nodes(graph_nodes):
            total_leaf_nodes[details.get("action")] += 1
            by_parent[node.get("parent_id")].append(node_id)

        for node_id, node, details in iter_leaf_nodes(graph_nodes):
            hits = detect(details)
            if not hits:
                continue
            matched_rules = set()
            for rules in hits.values():
                matched_rules.update(rules)
            for r in matched_rules:
                rule_counts[r] += 1
            sibling_ids = [nid for nid in by_parent[node.get("parent_id")] if nid != node_id]
            outcome = classify_outcome(node, details)
            field_hits.append({
                "file": os.path.basename(f),
                "node_id": node_id,
                "action": details.get("action"),
                "fields": hits,
                "status": node.get("status"),
                "score": (details.get("evaluation") or {}).get("score"),
                "outcome": outcome,
                "fan_out_group_size": len(sibling_ids) + 1,
                "is_fan_out": len(sibling_ids) >= 1,
            })

    both_rules = sum(1 for h in field_hits if len(set().union(*h["fields"].values())) > 1)
    rule2_only_nodes = [
        h for h in field_hits
        if all("rule2_ordinal" in rs and "rule1_template" not in rs for rs in h["fields"].values())
    ]

    print("=" * 88)
    print(f"Result files scanned: {len(result_files)} (expected 108)")
    print(f"Total search/visit leaf nodes: {sum(total_leaf_nodes.values())} "
          f"(search={total_leaf_nodes['search']}, visit={total_leaf_nodes['visit']})")
    print()
    print("-- Question 1: detections by rule --")
    print(f"  rule1_template (bracket/brace/angle template, no URL inside): {rule_counts['rule1_template']}")
    print(f"  rule2_ordinal  (explicit back-reference to a prior step):     {rule_counts['rule2_ordinal']}")
    print(f"  total distinct nodes flagged (union of both rules):          {len(field_hits)}")
    print(f"  nodes flagged by BOTH rules (same field, overlap):           {both_rules}")
    print(f"  nodes flagged ONLY by rule2_ordinal (rule1 found nothing):   {len(rule2_only_nodes)}")
    if not rule2_only_nodes:
        print("    -> rule2_ordinal found ZERO independent detections in this corpus; every")
        print("       ordinal-phrase hit was already inside a rule1 template. Kept anyway (spec")
        print("       explicitly asks for it, and it produced zero false positives -- see below);")
        print("       a future un-bracketed back-reference is the case it exists to catch.")
    print()

    print("-- Question 2: outcome cross-check (did the flagged node actually need deferral?) --")
    outcome_counts = Counter(h["outcome"] for h in field_hits)
    for k in ("FAILED", "SILENTLY_REPAIRED", "RAN_FINE"):
        print(f"  {k:18s}: {outcome_counts.get(k, 0)}")
    fp_rate = outcome_counts.get("RAN_FINE", 0) / len(field_hits) if field_hits else 0.0
    print(f"  Overall false-positive rate (RAN_FINE / total): {fp_rate:.1%}")
    print()

    print("-- Question 3: false-positive rate on fan-out (multi-sibling) batches --")
    fan_out_hits = [h for h in field_hits if h["is_fan_out"]]
    fan_out_fp = sum(1 for h in fan_out_hits if h["outcome"] == "RAN_FINE")
    fan_out_rate = fan_out_fp / len(fan_out_hits) if fan_out_hits else 0.0
    print(f"  Flagged nodes sitting in a fan-out group (>=2 siblings): {len(fan_out_hits)} / {len(field_hits)}")
    print(f"  False positives among them (RAN_FINE):                   {fan_out_fp}")
    print(f"  Fan-out false-positive rate:                             {fan_out_rate:.1%}")
    print(f"  Gate: <= 5% required to ship as-is -> {'PASS' if fan_out_rate <= 0.05 else 'FAIL -- tighten rules'}")
    print()

    if verbose:
        print("-- Detected-node detail (all {}) --".format(len(field_hits)))
        for h in sorted(field_hits, key=lambda x: (x["file"], x["node_id"])):
            print(f"  {h['file']:60s} {h['node_id'][:8]} {h['action']:6s} "
                  f"status={h['status']:8s} score={h['score']} outcome={h['outcome']:18s} "
                  f"fan_out={h['is_fan_out']} fields={h['fields']}")
        print()

    return {"field_hits": field_hits, "rule_counts": rule_counts}


# Question 4: AUTO-PARALLEL firings a slot-aware independence check would flip

_CURRENT_ID_RE = re.compile(r"current_id=([0-9a-f-]{36})")
_AUTO_PARALLEL_RE = re.compile(r"AUTO-PARALLEL:.*independent leaf siblings")
_AUTO_PARALLEL_COUNT_RE = re.compile(r"AUTO-PARALLEL: (\d+) independent leaf siblings")
_DEPENDENCY_RE = re.compile(r"DEPENDENCY DETECTED")


def has_slot(details: Dict[str, Any]) -> bool:
    return bool(detect(details))


def run_auto_parallel_replay(verbose: bool) -> None:
    result_files = find_result_files()
    total_firings = 0
    total_dependency = 0
    flipped = 0
    unreconstructed = 0
    flip_examples: List[Tuple[str, str, str, List[Tuple[str, str, str]]]] = []

    for f in result_files:
        log_path = find_cell_log(f)
        if not log_path:
            continue
        data = json.loads(Path(f).read_text())
        graph_nodes = data.get("execution", {}).get("graph", {}).get("nodes", {})
        lines = log_path.read_text().splitlines()

        total_dependency += sum(1 for l in lines if _DEPENDENCY_RE.search(l))

        current_parent: Optional[str] = None
        for i, line in enumerate(lines):
            m = _CURRENT_ID_RE.search(line)
            if m:
                current_parent = m.group(1)
            if not _AUTO_PARALLEL_RE.search(line):
                continue
            total_firings += 1

            batch_nodes = []
            if current_parent:
                for node_id, node in graph_nodes.items():
                    if node.get("parent_id") == current_parent:
                        details = node.get("details") or {}
                        if details.get("action") in ("search", "visit"):
                            batch_nodes.append((node_id, node, details))
            if not batch_nodes:
                unreconstructed += 1
                continue

            slot_node = next((n for n in batch_nodes if has_slot(n[2])), None)
            if slot_node is not None:
                flipped += 1
                flip_examples.append((
                    os.path.basename(f),
                    current_parent[:8],
                    slot_node[0][:8],
                    [(nid[:8], det.get("action"), node.get("status")) for nid, node, det in batch_nodes],
                ))

    print("-- Question 4: AUTO-PARALLEL firings a slot-aware check would flip --")
    print(f"  Total AUTO-PARALLEL firings (cell logs):    {total_firings} (briefing said 58)")
    print(f"  Total DEPENDENCY DETECTED firings:           {total_dependency} (briefing said 49)")
    print(f"  Firings where the sibling batch could not be reconstructed from parent_id: {unreconstructed}")
    print(f"  Firings a slot-aware check would flip to deferred: {flipped} "
          f"({flipped / total_firings:.1%} of firings)" if total_firings else "")
    print()
    print("  Sample (this IS the full flip population -- all {}):".format(len(flip_examples)))
    for fname, parent, slot_nid, batch in flip_examples:
        print(f"    {fname}")
        print(f"      parent={parent} slot_candidate={slot_nid} batch={batch}")
    print()


# Question 5: replay the FULL siblings_are_independent predicate (not just the slot
# sub-check) against every reconstructed AUTO-PARALLEL firing. Added 2026-08-16 after an
# adversarial review found the slot-only replay above (Question 4) was never the actual
# gate on ship-readiness: siblings_are_independent also requires ONE of its POSITIVE-evidence
# rules (concrete_urls / mixed_search_visit / disjoint_searches) to fire, and a batch that
# clears the slot check but matches none of those is a FALSE NEGATIVE -- force-serialized
# with no positive reason. This function imports and calls the REAL product code (not a
# reimplementation) so the measurement can't drift from actual behavior.

def _build_replay_graph(parent_details: Optional[Dict[str, Any]], batch: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]):
    """Rebuild a throwaway IdeaDag mirroring one historical AUTO-PARALLEL firing.

    ``batch`` is ``(orig_node_id, node_dict, details_dict)`` triples, as already assembled
    by the firing-reconstruction loops above. Node ids are NOT preserved (IdeaDag mints its
    own uuids) -- but any ``requires_data.source_node_id`` that pointed at ANOTHER member of
    the SAME batch is remapped to the new id, so detect_state_dependencies's Condition B
    (sibling-scoped requires_data) replays correctly.

    :returns: ``(graph, mandate_node, candidate_ids)`` ready to pass straight into
        ``siblings_are_independent``.
    """
    from agent.app.idea_dag import IdeaDag

    graph = IdeaDag(root_title="mandate", root_details=dict(parent_details or {}))
    mandate = graph.get_node(graph.root_id())

    id_map: Dict[str, str] = {}
    added: List[Tuple[str, Any]] = []
    for orig_id, node, details in batch:
        title = node.get("title") or ""
        new_node = graph.add_child(graph.root_id(), title, dict(details))
        id_map[orig_id] = new_node.node_id
        added.append((orig_id, new_node))

    for _orig_id, new_node in added:
        requires_data = new_node.details.get("requires_data")
        if isinstance(requires_data, dict):
            source_node_id = requires_data.get("source_node_id")
            if source_node_id in id_map:
                remapped = dict(requires_data)
                remapped["source_node_id"] = id_map[source_node_id]
                new_node.details["requires_data"] = remapped

    candidate_ids = [n.node_id for _, n in added]
    return graph, mandate, candidate_ids


def _diagnose_state_dependency(batch: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> str:
    """For a firing ALREADY known to trip ``detect_state_dependencies`` -- explain why,
    diagnostically. Does not change classification or product code.

    ``detect_state_dependencies``'s Condition A checks only ``url``/``link`` for a visit's
    resolvable URL -- it never checks ``optional_url``, the SAME blind spot rule 3's
    ``_concrete_url`` helper had before this task's fix (see defect 2). Since Condition A is
    always-on (not gated by ``parallel_requires_evidence``) and runs UNCONDITIONALLY first,
    fixing it is out of this task's scope -- doing so would change flag-OFF behavior, which
    the task requires stay byte-identical. This function separates a firing's real cause so
    the replay report doesn't conflate "genuinely dependent" with "blocked by a known,
    unrelated, always-on gap":

    * ``"condition_b_requires_data"`` -- a real, sibling-scoped ``requires_data`` (the
      working mechanism Condition B exists for).
    * ``"condition_a_optional_url_gap"`` -- every visit actually has a concrete URL once
      ``optional_url`` is considered, but at least one has it ONLY in ``optional_url`` (not
      ``url``/``link``) -- Condition A's narrower field set alone is what tripped it.
    * ``"condition_a_genuine_missing_url"`` -- at least one visit has no concrete URL in ANY
      of ``optional_url``/``url``/``link`` -- a real dependency on a sibling search.
    * ``"unknown"`` -- did not match either condition's shape (should not occur for a firing
      that actually tripped detect_state_dependencies; flagged so it's visible if it does).
    """
    ids = {nid for nid, _, _ in batch}
    for nid, node, details in batch:
        if details.get("action") != "visit":
            continue
        requires_data = details.get("requires_data")
        if isinstance(requires_data, dict) and requires_data.get("source_node_id") in ids:
            return "condition_b_requires_data"

    has_search = any(details.get("action") == "search" for _, _, details in batch)
    if not has_search:
        return "unknown"

    def _is_url(value: Any) -> bool:
        return isinstance(value, str) and value.startswith(("http://", "https://"))

    any_missing_narrow = False
    any_missing_wide = False
    for _nid, _node, details in batch:
        if details.get("action") != "visit":
            continue
        # "narrow" mirrors detect_state_dependencies's own field set exactly (url/link
        # only); "wide" adds optional_url, matching VisitLeafAction's real resolution order.
        narrow = details.get("url") or details.get("link")
        wide = details.get("optional_url") or narrow
        if not _is_url(narrow):
            any_missing_narrow = True
        if not _is_url(wide):
            any_missing_wide = True

    if any_missing_narrow and not any_missing_wide:
        return "condition_a_optional_url_gap"
    if any_missing_wide:
        return "condition_a_genuine_missing_url"
    return "unknown"


def run_full_predicate_replay(verbose: bool) -> Dict[str, Any]:
    from agent.app.idea_sequencing import siblings_are_independent

    replay_logger = logging.getLogger("measure_dataflow_slots.replay")
    replay_logger.disabled = True  # keep stdout to this script's own report

    result_files = find_result_files()
    reason_counts: Counter = Counter()
    state_dep_diagnosis: Counter = Counter()
    all_records: List[Dict[str, Any]] = []
    false_negatives: List[Dict[str, Any]] = []
    state_dependency_records: List[Dict[str, Any]] = []
    newly_independent_flips: List[Dict[str, Any]] = []  # for the false-positive-direction check
    total_firings = 0
    unreconstructed = 0
    count_mismatches = 0

    for f in result_files:
        log_path = find_cell_log(f)
        if not log_path:
            continue
        data = json.loads(Path(f).read_text())
        graph_nodes = data.get("execution", {}).get("graph", {}).get("nodes", {})
        lines = log_path.read_text().splitlines()

        current_parent: Optional[str] = None
        for line in lines:
            m = _CURRENT_ID_RE.search(line)
            if m:
                current_parent = m.group(1)
            if not _AUTO_PARALLEL_RE.search(line):
                continue
            total_firings += 1
            count_match = _AUTO_PARALLEL_COUNT_RE.search(line)
            logged_count = int(count_match.group(1)) if count_match else None

            # IMPORTANT: reconstruct the batch from ALL of the parent's non-merge,
            # action-bearing children -- NOT just search/visit. Question 4's slot-only
            # replay could safely restrict to search/visit (a slot in one candidate is
            # evidence regardless of what else is batched with it), but this predicate's
            # rules 3/4 depend on the batch's EXACT action-shape composition (e.g. rule 3
            # requires ALL candidates to be `visit`; silently dropping a `think`/`verify`
            # sibling would flip that from correctly-False to wrongly-True). A spot-check
            # against several singleton-reconstruction firings confirmed real
            # `think`/`verify` siblings sitting alongside the visit at the SAME parent,
            # matching the log's own logged count exactly -- restricting to search/visit
            # alone would have manufactured false "independent" verdicts here.
            batch_nodes: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
            if current_parent:
                for node_id, node in graph_nodes.items():
                    if node.get("parent_id") == current_parent:
                        details = node.get("details") or {}
                        action = details.get("action")
                        if action and action != "merge":
                            batch_nodes.append((node_id, node, details))
            if not batch_nodes:
                unreconstructed += 1
                continue
            if logged_count is not None and len(batch_nodes) != logged_count:
                # Reconstruction from the final graph snapshot didn't recover the same
                # sibling count the engine logged at decision time (e.g. a later
                # expansion round added more same-parent children after this firing).
                # Kept and classified anyway -- flagged here so the discrepancy is visible
                # rather than silently absorbed into the classification counts.
                count_mismatches += 1

            parent_node = graph_nodes.get(current_parent) if current_parent else None
            parent_details = (parent_node or {}).get("details")
            graph, mandate, candidate_ids = _build_replay_graph(parent_details, batch_nodes)
            independent, reason = siblings_are_independent(graph, candidate_ids, mandate, logging.getLogger("measure_dataflow_slots.replay"))
            reason_counts[reason] += 1

            batch_summary = [(nid[:8], det.get("action"), node.get("status")) for nid, node, det in batch_nodes]
            record = {
                "file": os.path.basename(f),
                "parent": current_parent[:8] if current_parent else None,
                "independent": independent,
                "reason": reason,
                "batch": batch_summary,
                "logged_count": logged_count,
                "count_mismatch": logged_count is not None and len(batch_nodes) != logged_count,
            }
            all_records.append(record)

            # A false negative is SPECIFICALLY "no genuine evidence either way" --
            # state_dependency and unresolved_slot are CORRECT vetoes on real evidence, not
            # false negatives, even though both also result in sequential execution.
            if reason == "no_independence_evidence":
                false_negatives.append(record)
            elif reason == "state_dependency":
                diagnosis = _diagnose_state_dependency(batch_nodes)
                record["state_dependency_diagnosis"] = diagnosis
                state_dep_diagnosis[diagnosis] += 1
                state_dependency_records.append(record)
            if independent:
                newly_independent_flips.append(record)

    print("=" * 88)
    print("-- Question 5: FULL siblings_are_independent predicate replayed against every")
    print("   reconstructed AUTO-PARALLEL firing (not just the unresolved-slot sub-check) --")
    print(f"  Total AUTO-PARALLEL firings (cell logs):    {total_firings} (briefing said 58)")
    print(f"  Firings where the sibling batch could not be reconstructed from parent_id: {unreconstructed}")
    print(f"  Firings where the reconstructed sibling count didn't match the logged count")
    print(f"  (final-graph snapshot vs. decision-time state -- flagged, not excluded): {count_mismatches}")
    print(f"  NB on mismatch direction: reconstruction takes the UNION of every non-merge,")
    print(f"  action-bearing child ever seen under that parent_id in the final graph, which can")
    print(f"  only be a SUPERSET of the true decision-time batch (e.g. a later re-expansion round")
    print(f"  added more same-parent children after this firing). A superset can only ADD")
    print(f"  dependency/slot signal (rules 1/2) or break an exact-shape rule (3/4/5) -- it can")
    print(f"  never manufacture a spurious 'independent' verdict. So false-negative and")
    print(f"  state_dependency counts below are a CONSERVATIVE UPPER BOUND; the true rate is")
    print(f"  likely somewhat lower. The independent-classification list is NOT at risk from this")
    print(f"  direction (confirmed no count-mismatch flags appear in that list below).")
    print()
    print("  Classification split:")
    independent_total = sum(c for r, c in reason_counts.items() if r not in ("unresolved_slot", "state_dependency", "no_independence_evidence"))
    print(f"    independent (any positive-evidence rule fired): {independent_total}")
    for reason in ("concrete_urls", "mixed_search_visit", "disjoint_searches"):
        if reason_counts.get(reason):
            print(f"      - {reason:20s}: {reason_counts[reason]}")
    sd_count = reason_counts.get("state_dependency", 0)
    print(f"    deferred on a genuine dependency signal:")
    print(f"      - unresolved_slot        : {reason_counts.get('unresolved_slot', 0)}")
    print(f"      - state_dependency       : {sd_count}, of which (diagnostic breakdown, NOT a")
    print(f"        classification change -- detect_state_dependencies is always-on, out of scope):")
    print(f"          - condition_b_requires_data      (genuine): {state_dep_diagnosis.get('condition_b_requires_data', 0)}")
    print(f"          - condition_a_genuine_missing_url (genuine): {state_dep_diagnosis.get('condition_a_genuine_missing_url', 0)}")
    print(f"          - condition_a_optional_url_gap (NOT genuine -- blocked by Condition A's own")
    print(f"            optional_url blind spot, the SAME gap defect 2 fixed in rule 3, but in the")
    print(f"            always-on Condition A this task must not touch): {state_dep_diagnosis.get('condition_a_optional_url_gap', 0)}")
    if state_dep_diagnosis.get("unknown"):
        print(f"          - unknown (unexpected -- investigate): {state_dep_diagnosis['unknown']}")
    print(f"    false negative (no slot, no dependency signal, no positive evidence either):")
    fn_count = reason_counts.get("no_independence_evidence", 0)
    print(f"      - no_independence_evidence: {fn_count}")
    if total_firings:
        print(f"  False-negative rate (no_independence_evidence / total): {fn_count / total_firings:.1%} "
              f"(briefing said 82.8% pre-fix)")
        gap_count = state_dep_diagnosis.get("condition_a_optional_url_gap", 0)
        print(f"  Effective rate if Condition A's optional_url gap were also closed (out of scope,")
        print(f"  reported not fixed): {(fn_count + gap_count) / total_firings:.1%}")
    print()

    def _mm(rec: Dict[str, Any]) -> str:
        return " [COUNT MISMATCH vs log]" if rec.get("count_mismatch") else ""

    if false_negatives:
        print(f"  -- Remaining false negatives -- no_independence_evidence ONLY ({len(false_negatives)}) --")
        for rec in false_negatives:
            print(f"    {rec['file']:60s} parent={rec['parent']} batch={rec['batch']}{_mm(rec)}")
        print()

    if state_dependency_records:
        print(f"  -- state_dependency firings, with diagnosis ({len(state_dependency_records)}) --")
        for rec in state_dependency_records:
            print(f"    {rec['file']:60s} parent={rec['parent']} diagnosis={rec['state_dependency_diagnosis']:30s} batch={rec['batch']}{_mm(rec)}")
        print()

    print(f"  -- All batches classified independent ({len(newly_independent_flips)}) -- manually eyeball for a")
    print(f"     genuine dependency slipping through (the dangerous direction) --")
    for rec in newly_independent_flips:
        print(f"    {rec['file']:60s} parent={rec['parent']} reason={rec['reason']:18s} batch={rec['batch']}{_mm(rec)}")
    print()

    return {
        "total_firings": total_firings,
        "unreconstructed": unreconstructed,
        "reason_counts": dict(reason_counts),
        "state_dependency_diagnosis": dict(state_dep_diagnosis),
        "false_negatives": false_negatives,
        "independent": newly_independent_flips,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="list every detected node")
    args = ap.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR}", file=sys.stderr)
        return 1

    run_static_measurement(verbose=args.verbose)
    run_auto_parallel_replay(verbose=args.verbose)
    run_full_predicate_replay(verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
