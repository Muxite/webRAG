#!/usr/bin/env python3
"""Mechanical, source-only shape registry for the suite59 benchmark tasks.

Why this exists
---------------
`docs/AGGREGATION_SHAPE_FINDING_2026-08-30.md` §3 reports a **-0.461** graph-minus-
sequential_react delta on "23 aggregation-shaped tasks" produced by a hand
re-classification that no longer exists in the repo. An adversarial review flagged that
any re-derivation which is tuned until the count lands near 23 is contaminated: the count
would then be an input, not a result.

So this module states the classification rule **in code**, as named predicates over what a
task module's *source* exposes -- `get_validation_functions()` names,
`get_required_deliverables()`, and the structure of `get_task_statement()`. Nothing here
reads a benchmark result, a score, or a shape label written anywhere else. The in-comment
shape labels next to `TASK_SETS["core_long24"]` in `scripts/adaptive_ladder_run.py` are
loaded only as a **cross-check diagnostic** (`--agreement`); they are never an input to a
predicate, and the predicates must not be tuned to agree with them or to reach any count.

Definitions (fixed before the predicates were written)
-----------------------------------------------------
aggregation
    The deliverable combines >= 2 independently-retrievable entity facts: argmax/argmin
    over a roster, count-with-threshold, AND-filter over entities, coverage/list-all,
    survivor-by-elimination, or cross-source comparison/reconciliation.
chain
    Each hop's input is the previous hop's output -- a single dependent thread, including
    navigation/link-following and disambiguate-then-read re-expansion.
other
    Neither: a single-target lookup, or a multi-part report over one artifact.
breadth (sub-flag, aggregation only)
    >= 3 arms that could be fetched in parallel: an enumerated entity roster of >= 3 items
    under a fan-out cue, with no statement cue that gates one arm behind another.

Precedence: aggregation predicates win over chain predicates, because "each hop's input is
the previous hop's output" is falsified as soon as two facts are independently retrievable.

Usage
-----
    PYTHONPATH=.:services:agent python scripts/task_shapes.py            # write + report
    PYTHONPATH=.:services:agent python scripts/task_shapes.py --flat f.json
    PYTHONPATH=.:services:agent python scripts/task_shapes.py --stdout --agreement
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Where the registry lands. Consumers: ``scripts/compare_arms.py --shapes``.
REGISTRY_PATH = _REPO_ROOT / "agent" / "app" / "testing" / "task_shapes.json"

SHAPES = ("aggregation", "chain", "other")

#: One-paragraph statement of the rule, embedded in the JSON so the artifact carries its own
#: provenance and a reader never has to guess which revision of the predicates produced it.
RULE = (
    "Shape is decided by named predicates over each task module's SOURCE only "
    "(get_validation_functions() names, get_required_deliverables(), get_task_statement() "
    "structure) -- never over any benchmark result, score, or externally written label. "
    "A task is 'aggregation' if any aggregation predicate fires: an extremum keystone "
    "(argmax/argmin/earliest/latest/longest/closest/median/k-th, or a 'winner_*' check); a "
    "count/sum-with-threshold keystone ('keystone_count', 'passing_*', 'inrange_*', "
    "'subset_sum', 'keystone_difference'); an AND-filter keystone ('keystone_filter', "
    "'winner_attributes'); roster coverage (a 'validate_coverage'/'breadth_*'/'span_values'/"
    "'validate_table' validator TOGETHER WITH an enumerated >=3-item entity roster under a "
    "fan-out cue in the statement); survivor-by-elimination ('survivor', "
    "'branch_exploration', 'elimination_coverage'); cross-source comparison "
    "('reconciliation_coverage', 'contradiction_coverage', 'identifies_correct_source', "
    "'identifies_wrong_value', 'true_claims'); or an explicit two-entity combination "
    "('keystone_combination', 'visited_both'). Otherwise it is 'chain' if a chain predicate "
    "fires: a single-thread chain validator ('chain_coverage', 'keystone_chain', "
    "'chain_intermediate', 'chain_urls', 'chain_progress', 'terminal_resolution', "
    "'path_adjacency'), a disambiguate-then-read re-expansion validator "
    "('reexpansion_coverage', 'target_resolution'), or a statement dependent-hop cue "
    "('from step N', 'the previous stage', 'follow the link'). Otherwise 'other'. Aggregation "
    "wins ties because two independently-retrievable facts falsify 'each hop's input is the "
    "previous hop's output'. The 'breadth' sub-flag is set only on aggregation tasks whose "
    "statement enumerates >=3 roster items under a fan-out cue with no dependency-gate cue, "
    "i.e. >=3 arms that could be fetched in parallel."
)


# --------------------------------------------------------------------------------------
# Source loading (reuses the repo's own enumeration + real-package import pattern)
# --------------------------------------------------------------------------------------

def suite59_ids() -> List[str]:
    """The suite59 id list, imported the way ``task_discrimination._task_sets()`` does."""
    import adaptive_ladder_run as alr  # noqa: WPS433 -- deliberate late import
    return list(alr.TASK_SETS["suite59"])


def task_modules() -> Dict[str, Any]:
    """Map test_id -> imported task module for every discoverable task file.

    Uses ``runner.discover_test_modules`` + ``config.extract_test_id`` for enumeration and
    the real-package import path (``agent.app.idea_tests.<stem>``) so module-level constants
    keep their identity, per ``scripts/rescore_results.py``.
    """
    from agent.app.testing.runner import discover_test_modules
    from agent.app.testing.config import extract_test_id

    out: Dict[str, Any] = {}
    for path in discover_test_modules():
        test_id = extract_test_id(path)
        if not test_id:
            continue
        out[test_id] = importlib.import_module("agent.app.idea_tests." + path.stem)
    return out


def _validator_names(module: Any) -> List[str]:
    fns = []
    getter = getattr(module, "get_validation_functions", None)
    if callable(getter):
        try:
            fns = list(getter())
        except Exception:  # pragma: no cover - a module that cannot expose its validators
            fns = []
    names = [getattr(f, "__name__", "") for f in fns]
    return [n for n in names if n]


def _call_str(module: Any, name: str) -> str:
    fn = getattr(module, name, None)
    if not callable(fn):
        return ""
    try:
        value = fn()
    except Exception:  # pragma: no cover
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


class TaskSource:
    """The only three things a predicate is allowed to look at."""

    def __init__(self, test_id: str, module: Any) -> None:
        self.test_id = test_id
        self.module_name = getattr(module, "__name__", "").rsplit(".", 1)[-1]
        self.validators = _validator_names(module)
        self.validator_blob = " ".join(self.validators).lower()
        self.deliverables = _call_str(module, "get_required_deliverables")
        self.statement = _call_str(module, "get_task_statement")
        self.text = (self.statement + "\n" + self.deliverables).lower()


# --------------------------------------------------------------------------------------
# Statement-structure helpers
# --------------------------------------------------------------------------------------

#: An enumerated roster line: "  1. Sognefjord", "  C1: <claim>", "  3) Foo".
_ENUM_LINE = re.compile(r"^[ \t]{0,6}(?:\d{1,2}[.)]|[A-Z]\d{1,2}[.:)])[ \t]+\S", re.MULTILINE)

#: A fan-out cue: the statement asks for the SAME operation over EVERY roster member.
_FANOUT_CUE = re.compile(
    r"for each\b|for every\b|each of the (?:following|three|four|five|six|seven|eight)\b"
    r"|open each\b|read each\b|visit each\b|across all\b|each claim\b|each river\b"
    r"|all (?:three|four|five|six|seven|eight|nine|ten)\b|each of these\b",
    re.IGNORECASE,
)

#: A dependency gate: one arm cannot be fetched until another has been resolved.
_DEP_GATE_CUE = re.compile(
    r"unknown until the previous\b|previous stage\b|from step \d\b|the surviving\b"
    r"|the survivor\b|stage 2\b|then use (?:that|the)\b|follow the link\b",
    re.IGNORECASE,
)


def enumerated_item_count(source: TaskSource) -> int:
    return len(_ENUM_LINE.findall(source.statement))


def has_fanout_cue(source: TaskSource) -> bool:
    return bool(_FANOUT_CUE.search(source.statement))


def has_dependency_gate(source: TaskSource) -> bool:
    return bool(_DEP_GATE_CUE.search(source.statement))


def has_entity_roster(source: TaskSource) -> bool:
    """>= 3 enumerated items AND a fan-out cue.

    The fan-out cue is load-bearing: several single-target tasks (044/093) enumerate the
    *items of the report they want back*, not entities to fetch, and would otherwise be
    misread as rosters.
    """
    return enumerated_item_count(source) >= 3 and has_fanout_cue(source)


def _any_validator(source: TaskSource, *needles: str) -> bool:
    return any(n in source.validator_blob for n in needles)


# --------------------------------------------------------------------------------------
# Aggregation predicates
# --------------------------------------------------------------------------------------

def p_extremum_keystone(source: TaskSource) -> bool:
    """argmax / argmin / ordinal selection over a roster of candidates."""
    return _any_validator(
        source,
        "argmax", "argmin", "earliest", "latest", "oldest", "newest",
        "longest", "shortest", "closest", "median", "keystone_kth",
        "keystone_highest", "keystone_lowest", "keystone_largest", "keystone_smallest",
        "winner_", "keystone_percent_winner",
    )


def p_count_or_sum_threshold(source: TaskSource) -> bool:
    """count-with-threshold, in-range count, or a bounded sum/difference over entities."""
    return _any_validator(
        source,
        "keystone_count", "passing_", "inrange_", "subset_sum", "keystone_difference",
    )


def p_and_filter(source: TaskSource) -> bool:
    """Set intersection: the unique entity satisfying two independent constraints."""
    return _any_validator(source, "keystone_filter", "winner_attributes")


def p_roster_coverage(source: TaskSource) -> bool:
    """A coverage/breadth validator over an actual enumerated entity roster."""
    coverage_validator = (
        any(n == "validate_coverage" for n in source.validators)
        or _any_validator(source, "breadth_", "span_values", "validate_table")
    )
    return coverage_validator and has_entity_roster(source)


def p_survivor_elimination(source: TaskSource) -> bool:
    """Branch-and-eliminate: N candidates are each checked, one survives."""
    return _any_validator(source, "survivor", "branch_exploration", "elimination_coverage")


def p_cross_source_comparison(source: TaskSource) -> bool:
    """Two or more independently-read sources for the same fact are compared."""
    return _any_validator(
        source,
        "reconciliation_coverage", "contradiction_coverage", "identifies_correct_source",
        "identifies_wrong_value", "true_claims",
    )


def p_explicit_two_entity_combination(source: TaskSource) -> bool:
    """The keystone is explicitly a combination of two separately-visited entities."""
    return _any_validator(source, "keystone_combination", "visited_both")


AGGREGATION_PREDICATES: Tuple[Tuple[str, Callable[[TaskSource], bool]], ...] = (
    ("extremum_keystone", p_extremum_keystone),
    ("count_or_sum_threshold", p_count_or_sum_threshold),
    ("and_filter", p_and_filter),
    ("roster_coverage", p_roster_coverage),
    ("survivor_elimination", p_survivor_elimination),
    ("cross_source_comparison", p_cross_source_comparison),
    ("explicit_two_entity_combination", p_explicit_two_entity_combination),
)


# --------------------------------------------------------------------------------------
# Chain predicates
# --------------------------------------------------------------------------------------

def p_chain_validator(source: TaskSource) -> bool:
    """A single dependent thread: hop N's target is hop N-1's answer."""
    return _any_validator(
        source,
        "chain_coverage", "keystone_chain", "chain_intermediate", "chain_urls",
        "chain_progress", "terminal_resolution", "path_adjacency",
    )


def p_reexpansion_target(source: TaskSource) -> bool:
    """Disambiguate to the right page first, then read the attribute off it."""
    return _any_validator(source, "reexpansion_coverage", "target_resolution")


def p_statement_dependent_hop(source: TaskSource) -> bool:
    """Statement itself gates one hop behind another."""
    return has_dependency_gate(source)


CHAIN_PREDICATES: Tuple[Tuple[str, Callable[[TaskSource], bool]], ...] = (
    ("chain_validator", p_chain_validator),
    ("reexpansion_target", p_reexpansion_target),
    ("statement_dependent_hop", p_statement_dependent_hop),
)


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------

def classify(source: TaskSource) -> Dict[str, Any]:
    fired_agg = [name for name, fn in AGGREGATION_PREDICATES if fn(source)]
    fired_chain = [name for name, fn in CHAIN_PREDICATES if fn(source)]
    if fired_agg:
        shape = "aggregation"
    elif fired_chain:
        shape = "chain"
    else:
        shape = "other"

    breadth = bool(
        shape == "aggregation"
        and has_entity_roster(source)
        and not has_dependency_gate(source)
    )
    evidence = list(fired_agg) + [f"chain:{n}" for n in fired_chain]
    if breadth:
        evidence.append("breadth:parallel_entity_roster")
    return {"shape": shape, "breadth": breadth, "evidence": evidence}


def build_registry(ids: List[str] | None = None) -> Dict[str, Any]:
    """Classify every suite59 task. Deterministic: source in, dict out."""
    ids = list(ids) if ids is not None else suite59_ids()
    modules = task_modules()
    missing = [t for t in ids if t not in modules]
    if missing:
        raise RuntimeError(f"No task module discovered for ids: {missing}")

    registry: Dict[str, Any] = {
        "_rule": RULE,
        "_written": datetime.date.today().isoformat(),
    }
    for test_id in sorted(ids):
        registry[test_id] = classify(TaskSource(test_id, modules[test_id]))
    return registry


def load_registry(path: Path | str = REGISTRY_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def shape_map(registry: Dict[str, Any] | None = None) -> Dict[str, str]:
    """Flat ``{test_id: shape}`` with ``_``-prefixed metadata keys dropped.

    This is the form ``compare_arms.py --shapes`` expects.
    """
    registry = registry if registry is not None else load_registry()
    return {
        k: v["shape"] if isinstance(v, dict) else str(v)
        for k, v in registry.items()
        if not k.startswith("_")
    }


def aggregation_ids(registry: Dict[str, Any] | None = None) -> List[str]:
    """Sorted ids classified ``aggregation``."""
    return sorted(k for k, v in shape_map(registry).items() if v == "aggregation")


def breadth_ids(registry: Dict[str, Any] | None = None) -> List[str]:
    registry = registry if registry is not None else load_registry()
    return sorted(
        k for k, v in registry.items()
        if not k.startswith("_") and isinstance(v, dict) and v.get("breadth")
    )


# --------------------------------------------------------------------------------------
# Cross-check ONLY: the in-comment labels beside TASK_SETS["core_long24"]
# --------------------------------------------------------------------------------------

_AGG_COMMENT_WORDS = (
    "aggregation", "fan-out", "fanout", "count", "argmin", "argmax", "and-filter",
    "subset-sum", "survivor", "conflicting-source", "ratio", "roster", "breadth", "filter",
)
_CHAIN_COMMENT_WORDS = ("chain", "navigation", "wikirace", "link-following")


def ladder_comment_labels() -> Dict[str, str]:
    """Parse the shape words written in the comments of ``TASK_SETS["core_long24"]``.

    Diagnostic only -- never an input to a predicate.
    """
    text = (_SCRIPTS / "adaptive_ladder_run.py").read_text(encoding="utf-8")
    start = text.index('TASK_SETS["core_long24"] = [')
    body = text[start:]
    body = body[: body.index("\n]")]
    labels: Dict[str, str] = {}
    for line in body.splitlines():
        if "#" not in line:
            continue
        code, comment = line.split("#", 1)
        ids = re.findall(r'"(\d{3})"', code)
        if not ids:
            continue
        low = comment.lower()
        agg = any(w in low for w in _AGG_COMMENT_WORDS)
        chain = any(w in low for w in _CHAIN_COMMENT_WORDS)
        if agg and not chain:
            label = "aggregation"
        elif chain and not agg:
            label = "chain"
        elif agg and chain:
            label = "mixed-comment"
        else:
            label = "unlabelled"
        for test_id in ids:
            labels[test_id] = label
    return labels


def agreement_table(registry: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    labels = ladder_comment_labels()
    shapes = shape_map(registry)
    rows = []
    for test_id in sorted(labels):
        if test_id not in shapes:
            continue  # core_long24 carries ids (165-168) that are not in suite59
        mine, theirs = shapes[test_id], labels[test_id]
        if theirs in ("mixed-comment", "unlabelled"):
            verdict = "n/a"
        elif mine == theirs:
            verdict = "agree"
        else:
            verdict = "DISAGREE"
        rows.append((test_id, mine, theirs, verdict))
    return rows


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _report(registry: Dict[str, Any], show_agreement: bool) -> None:
    shapes = shape_map(registry)
    counts = {s: sum(1 for v in shapes.values() if v == s) for s in SHAPES}
    print(f"suite59 tasks classified: {len(shapes)}")
    for shape in SHAPES:
        ids = sorted(k for k, v in shapes.items() if v == shape)
        print(f"  {shape:12s} {counts[shape]:3d}  {' '.join(ids)}")
    b_ids = breadth_ids(registry)
    print(f"  {'breadth':12s} {len(b_ids):3d}  {' '.join(b_ids)}   (sub-flag of aggregation)")

    print("\nper-task evidence:")
    for test_id in sorted(shapes):
        entry = registry[test_id]
        print(f"  {test_id}  {entry['shape']:12s} breadth={int(entry['breadth'])}  "
              f"{','.join(entry['evidence'])}")

    if show_agreement:
        rows = agreement_table(registry)
        print("\ncross-check vs adaptive_ladder_run.py core_long24 in-comment labels "
              "(diagnostic, NOT an input):")
        print(f"  {'id':4s} {'registry':12s} {'comment':14s} verdict")
        for test_id, mine, theirs, verdict in rows:
            print(f"  {test_id:4s} {mine:12s} {theirs:14s} {verdict}")
        agree = sum(1 for r in rows if r[3] == "agree")
        dis = sum(1 for r in rows if r[3] == "DISAGREE")
        na = sum(1 for r in rows if r[3] == "n/a")
        print(f"  -> {agree} agree, {dis} disagree, {na} not comparable "
              f"(of {len(rows)} core_long24 ids in suite59)")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REGISTRY_PATH),
                        help="registry JSON destination (default: %(default)s)")
    parser.add_argument("--flat", default=None,
                        help="also write a flat {id: shape} map here, for compare_arms --shapes")
    parser.add_argument("--stdout", action="store_true",
                        help="print the registry JSON instead of writing it")
    parser.add_argument("--no-agreement", action="store_true",
                        help="suppress the cross-check table vs adaptive_ladder_run's comments")
    args = parser.parse_args(argv)

    registry = build_registry()
    blob = json.dumps(registry, indent=2, sort_keys=False) + "\n"
    if args.stdout:
        print(blob)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob, encoding="utf-8")
        print(f"wrote {out} ({len(shape_map(registry))} tasks)")

    if args.flat:
        flat = Path(args.flat)
        flat.parent.mkdir(parents=True, exist_ok=True)
        flat.write_text(json.dumps(shape_map(registry), indent=2) + "\n", encoding="utf-8")
        print(f"wrote flat shape map {flat}")

    _report(registry, show_agreement=not args.no_agreement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
