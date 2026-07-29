"""Env-gated JSON parse-failure telemetry for the bad-model lab.

No-op unless ``IDEA_TEST_JSON_TELEMETRY`` is truthy. When enabled, every
JSON-mode decision in the compiled-react (`execution_compiled._run_leaf`) and
sequential-react (`execution_sequential._run_react`) leaves is classified and
appended as one JSONL line. The bad-model lab's ``analyze.py`` reads this to
tell *why* a weak model failed — fenced JSON vs prose vs truncation vs refusal
vs empty — which is exactly what selects the next mitigation to try.

Best-effort by construction: any error in here is swallowed so telemetry can
never perturb a benchmark run or a unit test. When the flag is off, ``record``
returns on the first line and costs nothing.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# testing/ -> app/ -> agent/ ; results live at services/agent/idea_test_results/
_RES = Path(__file__).resolve().parents[2] / "idea_test_results"

_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i am unable", "i'm unable", "as an ai",
    "i'm sorry", "sorry, but", "i apologize", "i am not able",
)

# Current task id, stamped once per task by runner.run_complete_test. Safe as a
# module global because benchmark runs are single-task-at-a-time
# (IDEA_TEST_CONCURRENCY=1). Lets analyze.py / the chart generator compute
# per-task parse-failure composition, not just per-run.
_CURRENT_TASK = None


def set_task(task_id) -> None:
    global _CURRENT_TASK
    _CURRENT_TASK = task_id


def current_task():
    """The task id stamped by ``set_task`` — read at call time, never cached.

    Public accessor so sibling telemetry modules (``contract_log``) can stamp the
    SAME task id without duplicating the global (one ``set_task`` call in
    ``runner.run_complete_test`` tags every telemetry stream).
    """
    return _CURRENT_TASK


def enabled() -> bool:
    return os.environ.get("IDEA_TEST_JSON_TELEMETRY", "") not in ("", "0", "false", "False")


def _path() -> Path:
    override = os.environ.get("IDEA_TEST_JSON_TELEMETRY_PATH")
    if override:
        return Path(override)
    run_id = os.environ.get("IDEA_TEST_RUN_ID", "adhoc")
    return _RES / f"{run_id}_json_telemetry.jsonl"


# --- schema-aware validity (format-stress tier) ------------------------------
# The 7 classes above bucket JSON *syntax*. valid_json only means json.loads()
# succeeded — it is blind to whether the required fields are present and typed, so
# {} scores valid_json. The format-stress tier needs a *schema*-aware check on top:
# schema_ok = parsed AND every required key present with the right JSON type. bool
# is handled explicitly (isinstance(True, int) is True in Python) so a boolean does
# not satisfy "number" and a number does not satisfy "boolean". See
# FORMAT_STRESS_TIER.md §1b; badmodel-lab/schema_check.py is the unit test.
_TYPE_PREDICATES = {
    "string":  lambda v: isinstance(v, str),
    "number":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array":   lambda v: isinstance(v, list),
    "object":  lambda v: isinstance(v, dict),
}


def schema_check(raw: str, required: dict, *, no_extra: bool = True) -> dict:
    """Classify a raw completion against a required typed schema.

    required: {key: json_type_name}. Returns a dict with parsed_ok, schema_ok,
    klass ("schema_valid" | "schema_partial" | "not_json"), and missing/mistyped/
    extra diagnostics. An empty/whitespace completion is not-parsed (mirrors the
    ``empty`` class); no fence-stripping (a fenced block is not-parsed here, and
    ``classify()`` buckets it ``fenced_json``).
    """
    out = {"parsed_ok": False, "schema_ok": False, "klass": "not_json",
           "missing": [], "mistyped": {}, "extra": []}
    s = (raw or "").strip()
    if not s:
        return out
    try:
        obj = json.loads(s)
        out["parsed_ok"] = True
    except Exception:
        return out
    if not isinstance(obj, dict):
        out["klass"] = "schema_partial"      # parsed, but not the required object
        out["missing"] = list(required)
        return out
    for key, tname in required.items():
        if key not in obj:
            out["missing"].append(key)
            continue
        pred = _TYPE_PREDICATES.get(tname)
        if pred is None:
            raise ValueError(f"unknown type name in schema: {tname!r}")
        if not pred(obj[key]):
            got = "boolean" if isinstance(obj[key], bool) else type(obj[key]).__name__
            out["mistyped"][key] = (tname, got)
    if no_extra:
        out["extra"] = [k for k in obj if k not in required]
    out["schema_ok"] = not (out["missing"] or out["mistyped"] or out["extra"])
    out["klass"] = "schema_valid" if out["schema_ok"] else "schema_partial"
    return out


def classify(raw: str, parsed_ok: bool) -> str:
    """Bucket a raw LLM completion into a JSON-capability failure class.

    Ordering matters: an all-whitespace reply that ``json.loads('{}')`` accepts
    is still an ``empty`` model output, not ``valid_json``.
    """
    s = (raw or "").strip()
    if not s:
        return "empty"
    if parsed_ok:
        return "valid_json"
    low = s.lower()
    if s.startswith("```") or "```json" in low[:24]:
        return "fenced_json"            # right idea, wrapped in markdown — cheap to repair
    if any(m in low[:200] for m in _REFUSAL_MARKERS):
        return "refusal"               # needs prompt simplification, not a format fix
    if "{" in s and s.count("{") > s.count("}"):
        return "truncated_json"        # started JSON, ran out of tokens — raise max_tokens
    if s[0] in "{[":
        return "malformed_json"        # tried JSON, botched syntax — grammar/repair
    return "prose"                     # ignored the format entirely — thin leaf / few-shot


def record(model_name: str, raw: str, json_mode: bool, parsed_ok: bool, *, phase: str = "leaf",
           schema_ok: bool | None = None) -> None:
    if not json_mode or not enabled():
        return
    try:
        entry = {
            "t": round(time.time(), 3),
            "model": model_name,
            "task_id": _CURRENT_TASK,
            "phase": phase,
            "class": classify(raw, parsed_ok),
            "parsed_ok": bool(parsed_ok),
            "raw_len": len(raw or ""),
            "raw_head": (raw or "")[:300],
        }
        # Format-stress tier: schema_ok distinguishes schema_valid (parsed AND all
        # required fields present+typed) from schema_partial (parsed but not). None
        # when the caller didn't run a schema check (all pre-existing call sites).
        if schema_ok is not None:
            entry["schema_ok"] = bool(schema_ok)
            entry["schema_class"] = "schema_valid" if schema_ok else "schema_partial"
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — telemetry must never affect a run
        pass
