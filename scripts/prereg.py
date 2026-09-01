#!/usr/bin/env python3
"""Preregistration: state an experiment before running it, and audit it against its own design.

The trap this exists to close: **a cell that dies before writing output leaves no file.** Any
analysis that iterates the results directory therefore cannot see it. On block gpu0831,
``langgraph_react`` silently lost 6-7 of 48 cells and its mean was computed over the survivors,
which made it look like the best arm. The denominator must come from the experiment design, never
from the filesystem.

A prereg is also what makes an unattended run legible hours later: the hypothesis, the primary
endpoint and the abort conditions are fixed *before* any data exists, so a result cannot be
reinterpreted to fit whatever came out.

Usage::

    # write the design, before launching anything
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py write \
        --spec my_experiment.json

    # at any point during or after the run
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py audit \
        --run-id ledger001

Related machinery already in the repo, deliberately not duplicated here:

* ``scripts/adaptive_ladder_run.py:acquire_pid_lock`` already refuses to start a second driver
  against a held ``driver.lock`` -- the singleton gate. Use it; do not write another.
* ``IDEA_TEST_USD_CEILING`` (runner) and ``adaptive_ladder_run.py --budget`` already cap spend.
* ``LEDGER_MAX_LIVE_FALLBACKS`` caps live search calls under corpus replay.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

#: Fields without which a run is not an experiment. ``abort_conditions`` is required so an
#: unattended run has pre-declared stopping rules rather than a human watching it.
REQUIRED_FIELDS = ("run_id", "hypothesis", "tasks", "arms", "reps", "primary_endpoint",
                   "abort_conditions")
#: Where prereg manifests live, alongside the results they describe.
DEFAULT_PREREG_DIR = "agent/idea_test_results/prereg"
DEFAULT_RESULTS_DIR = "agent/idea_test_results"

#: The only ``abort_conditions`` keys ``audit()`` knows how to evaluate. A key outside this set
#: is not silently ignored -- ``validate()`` rejects it at ``write`` time (a typo'd key such as
#: ``max_infra_falied_rate`` would otherwise disable a gate without any signal that it happened),
#: and ``audit()`` reports it as an "unknown" gate defensively for specs loaded without going
#: through ``write``/``validate`` first.
KNOWN_ABORT_CONDITIONS = frozenset({
    "min_completion_rate", "max_infra_failed_rate", "max_live_fallbacks",
})


def validate(spec: Dict[str, Any]) -> List[str]:
    """Every reason ``spec`` is not a usable preregistration.

    :returns: human-readable error strings, empty when the spec is complete.
    """
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in spec:
            errors.append(f"missing required field: {field}")
    for field in ("tasks", "arms"):
        value = spec.get(field)
        if field in spec and (not isinstance(value, list) or not value):
            errors.append(f"{field} must be a non-empty list")
    reps = spec.get("reps")
    if "reps" in spec and (not isinstance(reps, int) or reps < 1):
        errors.append("reps must be a positive integer")
    abort_conditions = spec.get("abort_conditions")
    if isinstance(abort_conditions, dict):
        for key in abort_conditions:
            if key not in KNOWN_ABORT_CONDITIONS:
                errors.append(
                    f"unknown abort_conditions key: {key} (known: "
                    f"{', '.join(sorted(KNOWN_ABORT_CONDITIONS))})")
    return errors


def expected_cells(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The full task x arm x rep product -- the denominator, fixed before any data exists.

    :returns: one dict per planned cell, in deterministic order.
    """
    cells = []
    for task in spec.get("tasks", []):
        for arm in spec.get("arms", []):
            for rep in range(1, int(spec.get("reps", 1)) + 1):
                cells.append({"run_id": spec.get("run_id", ""), "task": str(task),
                              "arm": str(arm), "rep": rep})
    return cells


#: Result filenames end ``{variant}{tier_tag}{cfg_tag}_r{rep}.json`` where ``tier_tag`` is
#: ``_t<n>`` or empty (``idea_test_runner.py:1666``) and ``cfg_tag`` is ``_cfg<hex>`` (``:1667``).
#: Anchoring on these is what keeps one arm from matching another arm's prefix.
_TAG_SUFFIX = r"(?:_t\d+)?(?:_cfg[0-9a-fA-F]+)?"


def _cell_path(cell: Dict[str, Any], results_dir: str) -> Any:
    """The canonical result file for ``cell``, or ``None`` when it never landed.

    Arm matching is **anchored, not substring**. ``sequential_react`` is a prefix of
    ``sequential_react_extract``, and both contain underscores, so neither a substring test nor an
    underscore boundary can separate them -- a run where one arm died entirely would then report
    itself complete. The arm must be followed by the runner's own tag suffix and the rep marker.

    Excludes ``*_summary.json`` (which reflects only the last cell of a multi-invocation run) and
    ``*.jsonl`` traces; both inflate a naive glob, which once produced a throughput figure 2.1x
    too high.

    An unrecognised tag format causes a cell to be reported MISSING rather than found -- the safe
    direction, since it surfaces a discrepancy instead of hiding one.
    """
    pattern = re.compile(
        rf"^{re.escape(str(cell['run_id']))}.*_{re.escape(str(cell['task']))}_"
        rf".*_{re.escape(str(cell['arm']))}{_TAG_SUFFIX}_r{int(cell['rep'])}\.json$")
    for path in Path(results_dir).iterdir():
        if not path.is_file() or path.name.endswith("_summary.json"):
            continue
        if pattern.match(path.name):
            return path
    return None


def _cell_landed(cell: Dict[str, Any], results_dir: str) -> bool:
    """True when a canonical result file exists for ``cell``. See :func:`_cell_path`."""
    return _cell_path(cell, results_dir) is not None


def _infra_failed_rate(landed_paths: List[Path]) -> float:
    """Fraction of landed cells whose stored result carries a truthy top-level ``infra_failed``.

    Missing/unreadable files count as not-failed here (they were already excluded from ``landed``
    by :func:`_cell_path`, which only returns paths that matched the naming pattern); a file that
    exists but fails to parse is treated as no evidence of an infra failure rather than raising,
    since a gate function must not itself become a new way for an audit to crash.
    """
    if not landed_paths:
        return 0.0
    failed = 0
    for path in landed_paths:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(body, dict) and body.get("infra_failed"):
            failed += 1
    return failed / len(landed_paths)


def _evaluate_abort_conditions(spec: Dict[str, Any], completion_rate: float,
                                landed_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Evaluate each declared ``abort_conditions`` entry against the cells that actually landed.

    Three statuses, never conflated:

    * ``"pass"`` -- the condition was checked against real data and held.
    * ``"fail"`` -- the condition was checked against real data and was violated.
    * ``"unknown"`` -- the condition names something this module cannot recover from a stored
      cell (currently ``max_live_fallbacks``: the live-search-call count lives only on the
      in-memory ``ConnectorSearchCorpus`` instance for the run that produced a cell and is never
      written into the cell's JSON) or a key ``audit()`` does not recognise at all. Unknown must
      never be reported as, or folded into, a pass -- a gate that always appears to pass when it
      was never actually checked is worse than no gate.
    """
    gates: Dict[str, Dict[str, Any]] = {}
    for key, threshold in (spec.get("abort_conditions") or {}).items():
        if key == "min_completion_rate":
            passed = completion_rate >= threshold
            gates[key] = {
                "status": "pass" if passed else "fail",
                "threshold": threshold,
                "value": completion_rate,
                "detail": f"completion_rate={completion_rate:.3f} vs min {threshold}",
            }
        elif key == "max_infra_failed_rate":
            rate = _infra_failed_rate(landed_paths)
            passed = rate <= threshold
            gates[key] = {
                "status": "pass" if passed else "fail",
                "threshold": threshold,
                "value": rate,
                "detail": f"infra_failed_rate={rate:.3f} vs max {threshold}",
            }
        elif key == "max_live_fallbacks":
            gates[key] = {
                "status": "unknown",
                "threshold": threshold,
                "value": None,
                "detail": ("live-fallback count is not recoverable from a stored cell -- it "
                           "lives only on the ConnectorSearchCorpus instance of the run that "
                           "produced it and is never persisted into the cell's JSON"),
            }
        else:
            gates[key] = {
                "status": "unknown",
                "threshold": threshold,
                "value": None,
                "detail": f"unrecognised abort_conditions key: {key}",
            }
    return gates


def audit(spec: Dict[str, Any], results_dir: str = DEFAULT_RESULTS_DIR) -> Dict[str, Any]:
    """Compare what landed against what was designed, and gate on the declared abort conditions.

    A cell in ``missing`` is a **failure**, not an absence: it was planned, so its non-appearance
    is data. Reporting a mean over ``found`` alone is the survivor bias this function exists to
    prevent.

    ``abort_conditions`` used to be checked for presence only (by :func:`validate`) and never
    evaluated -- a spec could declare ``max_infra_failed_rate: 0.2`` and nothing would ever read
    it. This now evaluates every declared condition against the cells that actually landed; see
    :func:`_evaluate_abort_conditions` for the pass/fail/unknown contract.

    :returns: the original ``{expected, found, missing, complete, completion_rate}`` plus
        ``gates`` (per-condition ``{status, threshold, value, detail}``) and ``gates_passed``
        (``True`` iff every declared gate's status is ``"pass"`` -- unknown counts as not-passed).
    """
    cells = expected_cells(spec)
    cell_paths = [(cell, _cell_path(cell, results_dir)) for cell in cells]
    missing = [cell for cell, path in cell_paths if path is None]
    landed_paths = [path for _, path in cell_paths if path is not None]
    found = len(cells) - len(missing)
    completion_rate = (found / len(cells)) if cells else 0.0
    gates = _evaluate_abort_conditions(spec, completion_rate, landed_paths)
    return {
        "run_id": spec.get("run_id", ""),
        "expected": len(cells),
        "found": found,
        "missing": missing,
        "complete": not missing and bool(cells),
        "completion_rate": completion_rate,
        "gates": gates,
        "gates_passed": all(gate["status"] == "pass" for gate in gates.values()),
    }


def write(prereg_dir: str, spec: Dict[str, Any]) -> Path:
    """Validate and persist a preregistration.

    :raises ValueError: when the spec is incomplete -- a malformed prereg must fail here rather
        than silently yield a wrong denominator later.
    :returns: the path written.
    """
    errors = validate(spec)
    if errors:
        raise ValueError("invalid preregistration: " + "; ".join(errors))
    path = Path(prereg_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{spec['run_id']}.json"
    target.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load(path: str) -> Dict[str, Any]:
    """Read a preregistration manifest."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    writer = sub.add_parser("write", help="validate and store a preregistration")
    writer.add_argument("--spec", required=True, help="path to a spec JSON")
    writer.add_argument("--prereg-dir", default=DEFAULT_PREREG_DIR)

    auditor = sub.add_parser("audit", help="compare landed cells against the design")
    auditor.add_argument("--run-id", required=True)
    auditor.add_argument("--prereg-dir", default=DEFAULT_PREREG_DIR)
    auditor.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args()

    if args.command == "write":
        target = write(args.prereg_dir, load(args.spec))
        print(f"preregistered -> {target}")
        return 0

    spec = load(os.path.join(args.prereg_dir, f"{args.run_id}.json"))
    report = audit(spec, args.results_dir)
    print(f"run {report['run_id']}: {report['found']}/{report['expected']} cells "
          f"({report['completion_rate']:.1%})")
    if report["missing"]:
        print(f"MISSING {len(report['missing'])} cell(s) -- these are failures, not absences:")
        for cell in report["missing"]:
            print(f"  task={cell['task']} arm={cell['arm']} rep={cell['rep']}")
    else:
        print("complete: every designed cell landed")

    ok = True
    for name, gate in report["gates"].items():
        marker = {"pass": "OK", "fail": "FAIL", "unknown": "UNKNOWN"}[gate["status"]]
        print(f"  gate {name}: {marker} -- {gate['detail']}")
        if gate["status"] != "pass":
            ok = False

    if report["missing"] or not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
