"""
Shared result-loading for benchmark analysis scripts (recovery_curve, level_ladder,
render_gallery). Centralizes:

  * where result JSONs live,
  * which run_id(s) make up the "final" combinable dataset (default: ``barrage24b`` --
    the closed-out cost-recovery run; 27+ curated tests x 3 models, R=3, tier 0),
  * how to flatten one result file into a canonical row carrying every field any report
    needs (cost, score, level, tooling, visits, tokens, groundedness, ...).

``idea_test_results/`` accumulates JSONs from many historical sessions -- stale pilots,
calibration probes, superseded timestamped runs. Everything here is scoped by run_id
*prefix* match so callers never silently pull in that noise; the default scope is just
the one final run. When later validation batches land (e.g. a rework of tests
073/080/081 -- ``073fix_validate``, ``080fix_validate``, ``081fix_validate``), pass them
via ``extra_run_ids`` to fold them into the combined dataset without touching the
default.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# The closed-out, final dataset for the cost-recovery study.
DEFAULT_RUN_IDS: List[str] = ["barrage24b"]

# Internal variant -> tooling-ablation rung (mirrors testing/tooling.py). Used to
# back-fill the tooling column for runs launched via the legacy variant axis.
_VARIANT_TO_TOOLING = {
    "minimal": "minimal", "naive_rag": "partial", "graph": "full",
    "sequential_react": "sequential", "graph_compiled": "compiled",
}


def results_dir() -> Path:
    for cand in (Path("services/agent/idea_test_results"), Path("agent/idea_test_results")):
        if cand.is_dir():
            return cand
    return Path(__file__).resolve().parent.parent / "services" / "agent" / "idea_test_results"


def discover_files(run_ids: Optional[Sequence[str]] = None, *, since: str = "",
                    files: Optional[Sequence[str]] = None) -> List[Path]:
    """Result files scoped to the given run-id prefixes.

    Defaults to :data:`DEFAULT_RUN_IDS` (the final ``barrage24b`` dataset). Pass
    ``run_ids=[]`` explicitly to disable run-id scoping entirely (matches every result
    file in the dir, filtered only by ``since``) -- used by callers that want the older
    "everything since a date" behavior.
    """
    rd = results_dir()
    if files:
        return [Path(p) for p in files]
    ids = list(DEFAULT_RUN_IDS) if run_ids is None else list(run_ids)
    prefixes = [f"{r}_" for r in ids] if ids else [""]
    return sorted(
        p for p in rd.glob("*_r*.json")
        if "_report_" not in p.name
        and (not since or p.name >= since)
        and any(p.name.startswith(pre) for pre in prefixes)
    )


def load_row(path: Path) -> Optional[Dict[str, Any]]:
    """Flatten one result JSON into a canonical row. ``None`` if unusable (no score)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    meta = d.get("test_metadata") or {}
    ex = d.get("execution") or {}
    val = d.get("validation") or {}
    obs = ex.get("observability") or {}
    cost = obs.get("cost") or {}
    score = val.get("overall_score")
    if score is None:
        return None
    variant = d.get("execution_variant")
    gflag = (obs.get("grounding") or {}).get("grounded")
    if gflag is not None:
        grounded = 1.0 if gflag else 0.0
    else:
        grounded = 0.0
        for c in val.get("grep_validations", []) or []:
            name = str(c.get("check", "")).lower()
            if any(t in name for t in ("citation", "grounding", "source", "adjacency", "url")):
                grounded = max(grounded, float(c.get("score", 0.0)))
    usd = cost.get("usd")
    return {
        "path": str(path),
        "model": d.get("model"),
        "origin": d.get("origin"),
        "variant": variant,
        "tooling": d.get("tooling_profile") or _VARIANT_TO_TOOLING.get(variant, variant),
        "tier": int(d.get("effort_tier") or 0),
        "test_id": meta.get("test_id"),
        "level": meta.get("level") or "legacy",
        "weight": meta.get("weight") or "-",
        "score": float(score),
        "usd": float(usd) if usd is not None else None,
        "cost_estimated": bool(cost.get("estimated")),
        "total_tokens": int((obs.get("llm") or {}).get("total_tokens") or 0),
        "visit_chars": int((obs.get("visit") or {}).get("chars") or 0),
        "visits": int((obs.get("visit") or {}).get("count", 0) or 0),
        # The three work channels exposed *individually* (llm inference calls, web
        # searches, page visits). ``tool_calls`` keeps their sum for backward compat.
        "llm_calls": int((obs.get("llm") or {}).get("calls", 0) or 0),
        "search_calls": int((obs.get("search") or {}).get("count", 0) or 0),
        "visit_calls": int((obs.get("visit") or {}).get("count", 0) or 0),
        "tool_calls": int((obs.get("llm") or {}).get("calls", 0) or 0)
                      + int((obs.get("search") or {}).get("count", 0) or 0)
                      + int((obs.get("visit") or {}).get("count", 0) or 0),
        "secs": float(ex.get("duration_seconds") or 0.0),
        "grounded": grounded,
    }


def load_rows(run_ids: Optional[Sequence[str]] = None, *, extra_run_ids: Optional[Sequence[str]] = None,
              since: str = "", files: Optional[Sequence[str]] = None,
              tests: Optional[Sequence[str]] = None, require_cost: bool = False) -> List[Dict[str, Any]]:
    """Load + flatten every result row for the combined dataset.

    Defaults to the final ``barrage24b`` run. ``extra_run_ids`` is additive -- pass the
    prefixes of later validation batches once they exist to merge them in without
    replacing the default scope.
    """
    ids = list(DEFAULT_RUN_IDS) if run_ids is None else list(run_ids)
    if extra_run_ids:
        ids = ids + list(extra_run_ids)
    paths = discover_files(ids, since=since, files=files)
    rows = [r for r in (load_row(p) for p in paths) if r and r.get("model") and r.get("variant")]
    if require_cost:
        rows = [r for r in rows if r.get("usd") is not None]
    if tests:
        tset = {t.strip() for t in tests if t and t.strip()}
        rows = [r for r in rows if r.get("test_id") in tset]
    return rows
