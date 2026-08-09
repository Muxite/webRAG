#!/usr/bin/env python3
"""
Bootstrap eval set for plan-library retrieval — day one, no production traffic, $0.

The retrieval-quality question ("does the library return the closest template, and is the
closest template actually the right one?") normally needs a reviewed corpus of real
retrievals. But a labelled set already exists in the repo: every reachable-tier task statement
IS a positive example with a KNOWN correct ``template_id`` (062 -> argmax_over_n_page_field,
064 -> computed_ratio_argmax, ...), and task statements from tiers with a different composition
shape are natural negatives that must stay below ``SUGGEST_THRESHOLD``.

That makes this the offline gate on ``retrieval.py``'s threshold constants, which are starting
values rather than asserted-correct ones: run it, and if a positive misses or a negative fires,
TUNE the constants (or the template's ``embedding_text``) rather than shipping known-bad
thresholds. ``--sweep`` scans a grid and reports which settings would separate the set.

NOT a pytest test: it needs a real Chroma + a real embedding model, and the offline suite must
stay free of live-service dependencies. Run it after ``scripts/sync_plan_library.py``, against
the same Chroma.

Usage::

    CHROMA_MODE=embedded CHROMA_EMBEDDED_PATH=./.chroma_plan_library \\
      PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/eval_plan_library_retrieval.py

    ... scripts/eval_plan_library_retrieval.py --sweep      # threshold grid search
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "services", _ROOT / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.connector_config import ConnectorConfig  # noqa: E402
from agent.app.connector_chroma import ConnectorChroma  # noqa: E402
from agent.app.plan_library.retrieval import (  # noqa: E402
    AUTO_APPLY_THRESHOLD,
    COLLECTION_NAME,
    DECISION_AUTO_APPLY,
    MIN_MARGIN,
    SUGGEST_THRESHOLD,
    PlanLibrary,
    RetrievalResult,
    archetype_hint_for,
)
from agent.app.testing.test_module import IdeaTestModule  # noqa: E402

# --- the labelled set -------------------------------------------------------------------
# Positives: the task each template was authored FROM (see each template's
# ``provenance.based_on_tasks``). 051/065 are tier-4/5 rather than reachable-tier, but they
# are the chain template's basis and are the only positives it has.
POSITIVES: Dict[str, str] = {
    "062": "argmax_over_n_page_field",
    "064": "computed_ratio_argmax",
    "069": "odd_one_out_negation",
    "070": "bounded_subset_sum",
    "072": "filtered_count_or_predicate",
    "076": "filtered_count_or_predicate",
    "078": "filtered_count_or_predicate",
    "051": "entity_chain_resolution",
    "065": "entity_chain_resolution",
}

# Negatives: tasks with a genuinely DIFFERENT composition shape — micro (one page, one fact)
# and format (one fact, demanded as typed JSON). Neither is a multi-entity composition, so a
# hit here would be a false positive.
NEGATIVES: List[str] = ["m01", "m02", "m03", "f01", "f02", "f03"]

# Reported but NOT asserted: hard-tier tasks that share a composition shape with the corpus
# (e.g. 073 counts over N entities, 075/077 are argmax over N page reads). Calling those
# "negatives" would be dishonest — a high similarity there is a CORRECT retrieval, and the
# library is deliberately kept general enough to serve them.
INFORMATIONAL: List[str] = ["063", "071", "073", "075", "077"]


def _load_statements(ids: List[str]) -> Dict[str, str]:
    tests_dir = _ROOT / "agent" / "app" / "idea_tests"
    wanted = set(ids)
    out: Dict[str, str] = {}
    for path in sorted(tests_dir.glob("test_*.py")):
        stem = path.stem[len("test_"):]
        test_id = stem.split("_")[0]
        if test_id not in wanted:
            continue
        try:
            out[test_id] = IdeaTestModule(path).get_task_statement()
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: could not load {path.name}: {exc}", file=sys.stderr)
    return out


async def _retrieve_all(
    library: PlanLibrary, chroma: ConnectorChroma, statements: Dict[str, str], top_k: int
) -> Dict[str, RetrievalResult]:
    results: Dict[str, RetrievalResult] = {}
    for test_id, statement in statements.items():
        results[test_id] = await library.retrieve(
            chroma,
            statement,
            top_k=top_k,
            archetype_hint=archetype_hint_for(statement),
            # Decisions are recomputed locally for the sweep; retrieve with the live
            # constants so the printed decision is the one production would take.
        )
    return results


def _top(result: RetrievalResult) -> Tuple[Optional[str], float, Optional[float]]:
    if not result.candidates:
        return None, 0.0, None
    top = result.candidates[0]
    runner_up = result.candidates[1] if len(result.candidates) > 1 else None
    margin = None if runner_up is None else top.similarity - runner_up.similarity
    return top.template_id, top.similarity, margin


def _decide(
    result: RetrievalResult, auto: float, suggest: float, margin_floor: float
) -> str:
    """Re-apply the decision policy with alternative constants (for ``--sweep``)."""
    if not result.candidates:
        return "fallthrough_no_match"
    top = result.candidates[0]
    runner_up = result.candidates[1] if len(result.candidates) > 1 else None
    margin = None if runner_up is None else top.similarity - runner_up.similarity
    if top.similarity < suggest:
        return "fallthrough_no_match"
    if top.similarity < auto:
        return "suggest"
    same = (
        runner_up is not None
        and bool(top.archetype)
        and top.archetype.lower() == runner_up.archetype.lower()
    )
    if margin is not None and margin < margin_floor and not same:
        return "fallthrough_low_margin"
    return DECISION_AUTO_APPLY


def _score(
    results: Dict[str, RetrievalResult], auto: float, suggest: float, margin_floor: float
) -> Dict[str, Any]:
    top1_ok = auto_ok = 0
    for test_id, expected in POSITIVES.items():
        result = results.get(test_id)
        if result is None:
            continue
        top_id, _, _ = _top(result)
        if top_id == expected:
            top1_ok += 1
            if _decide(result, auto, suggest, margin_floor) == DECISION_AUTO_APPLY:
                auto_ok += 1
    suppressed = sum(
        1
        for test_id in NEGATIVES
        if test_id in results
        and _decide(results[test_id], auto, suggest, margin_floor) == "fallthrough_no_match"
    )
    n_pos = sum(1 for t in POSITIVES if t in results)
    n_neg = sum(1 for t in NEGATIVES if t in results)
    return {
        "top1_ok": top1_ok, "n_pos": n_pos,
        "auto_ok": auto_ok,
        "neg_suppressed": suppressed, "n_neg": n_neg,
    }


def _print_table(
    results: Dict[str, RetrievalResult],
    ids: List[str],
    expected: Dict[str, str],
    *,
    assertive: bool = True,
) -> int:
    """Print one group; return how many ASSERTED cases failed (informational groups: 0)."""
    failures = 0
    for test_id in ids:
        result = results.get(test_id)
        if result is None:
            print(f"  {test_id}  (statement not found)")
            continue
        top_id, similarity, margin = _top(result)
        want = expected.get(test_id)
        if want is None:  # negative: must not be applicable
            ok = result.decision.startswith("fallthrough")
        else:
            ok = top_id == want
        verdict = ("ok " if ok else "FAIL") if assertive else "-- "
        failures += 0 if (ok or not assertive) else 1
        margin_text = "  n/a" if margin is None else f"{margin:+.3f}"
        runner_up = result.candidates[1].template_id if len(result.candidates) > 1 else "-"
        print(
            f"  {verdict} {test_id}  sim={similarity:.3f} margin={margin_text} "
            f"{result.decision:<24} top1={top_id or '-'}"
            + (f" (want {want})" if want and top_id != want else "")
            + f"  [hint={result.archetype_hint or '-'}, 2nd={runner_up}]"
        )
    return failures


async def main_async(args: argparse.Namespace) -> int:
    library = PlanLibrary(collection=args.collection, warn_on_drift=True)
    print(f"plan library: {len(library)} template(s); collection '{library.collection}'")
    if not library.templates:
        print("ERROR: no templates loaded", file=sys.stderr)
        return 1

    ids = list(POSITIVES) + NEGATIVES + INFORMATIONAL
    statements = _load_statements(ids)
    chroma = ConnectorChroma(ConnectorConfig())
    results = await _retrieve_all(library, chroma, statements, args.top_k)
    if not any(r.candidates for r in results.values()):
        print(
            "ERROR: every retrieval came back empty — is the collection synced? "
            "(run scripts/sync_plan_library.py against the same chroma)",
            file=sys.stderr,
        )
        return 1

    print(f"\nPOSITIVES (top-1 must equal the labelled template)  n={len(POSITIVES)}")
    failures = _print_table(results, list(POSITIVES), POSITIVES)
    print(f"\nNEGATIVES (must fall through, sim < {args.suggest:.2f})  n={len(NEGATIVES)}")
    failures += _print_table(results, NEGATIVES, {})
    print("\nINFORMATIONAL (same-shape hard-tier tasks; reported, not asserted)")
    _print_table(results, INFORMATIONAL, {}, assertive=False)

    score = _score(results, args.auto, args.suggest, args.margin)
    print(
        f"\nsummary @ auto={args.auto:.2f} suggest={args.suggest:.2f} margin={args.margin:.2f}: "
        f"top-1 {score['top1_ok']}/{score['n_pos']}, "
        f"auto-applied {score['auto_ok']}/{score['n_pos']}, "
        f"negatives suppressed {score['neg_suppressed']}/{score['n_neg']}"
    )

    if args.sweep:
        print("\nthreshold sweep (auto, suggest, margin) -> top1 / auto-applied / suppressed")
        rows = []
        for auto in [round(0.30 + 0.02 * i, 2) for i in range(21)]:
            for suggest in [round(0.20 + 0.02 * i, 2) for i in range(21)]:
                if suggest > auto:
                    continue
                for margin_floor in (0.0, 0.02, 0.05, 0.08):
                    s = _score(results, auto, suggest, margin_floor)
                    # Ties broken toward the STRICTEST thresholds that still score the same:
                    # among equally-good settings the conservative one is the safer default.
                    rows.append((s["auto_ok"], s["neg_suppressed"], auto, suggest, margin_floor, s))
        rows.sort(reverse=True)
        seen = set()
        for _, _, auto, suggest, margin_floor, s in rows:
            key = (s["auto_ok"], s["neg_suppressed"])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  auto={auto:.2f} suggest={suggest:.2f} margin={margin_floor:.2f} -> "
                f"top1 {s['top1_ok']}/{s['n_pos']}, auto {s['auto_ok']}/{s['n_pos']}, "
                f"neg {s['neg_suppressed']}/{s['n_neg']}"
            )
            if len(seen) >= 12:
                break

    if failures:
        print(f"\n{failures} labelled case(s) failed — tune the thresholds or the "
              f"template embedding_text, do not ship as is.", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--auto", type=float, default=AUTO_APPLY_THRESHOLD)
    parser.add_argument("--suggest", type=float, default=SUGGEST_THRESHOLD)
    parser.add_argument("--margin", type=float, default=MIN_MARGIN)
    parser.add_argument("--sweep", action="store_true", help="Grid-search the thresholds.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
