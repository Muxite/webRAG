"""Deterministic verifier helpers — the doctrine's "make validators sound, not
subjective". Used by task validators (scoring) and later by best-of-N selection.

Everything here is pure and observable: a value is accepted only if it meets an
explicit rule (it literally occurs in the source, the file exists with the expected
content, the object matches the typed schema). No LLM judgement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Verdict:
    passed: bool
    reason: str = ""
    score: float = 0.0


def ok(reason: str = "") -> Verdict:
    return Verdict(True, reason, 1.0)


def fail(reason: str) -> Verdict:
    return Verdict(False, reason, 0.0)


# --- evidence-first: a claimed value must occur in the source we read -----------
def evidence_first(value: Any, source_text: str) -> bool:
    v = str(value or "").strip().lower()
    if not v:
        return False
    return v in (source_text or "").lower()


def contains_number(text: str, number: str) -> bool:
    """The exact number appears, not embedded in a longer number (511 not in 5110)."""
    return bool(re.search(rf"(?<!\d){re.escape(str(number))}(?!\d)", text or ""))


# --- filesystem assertions (confined to a workdir) ------------------------------
def file_exists(workdir: Path, rel: str) -> Verdict:
    p = Path(workdir) / rel
    return ok(f"{rel} exists") if p.is_file() else fail(f"{rel} missing")


def file_contains(workdir: Path, rel: str, needle: str) -> Verdict:
    p = Path(workdir) / rel
    if not p.is_file():
        return fail(f"{rel} missing")
    return ok(f"{rel} contains {needle!r}") if needle in p.read_text(errors="replace") \
        else fail(f"{rel} lacks {needle!r}")


def dir_has(workdir: Path, rel: str, name: str) -> Verdict:
    p = Path(workdir) / rel
    if not p.is_dir():
        return fail(f"{rel} not a dir")
    return ok(f"{name} present") if (p / name).exists() else fail(f"{name} not in {rel}")


# --- schema validity (reuse json_telemetry.schema_check; standalone fallback) ----
def _load_schema_check():
    try:
        from agent.app.testing.json_telemetry import schema_check  # services on path
        return schema_check
    except Exception:
        import sys
        root = Path(__file__).resolve().parents[2]     # webRAG/
        for p in (root, root / "services"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            from agent.app.testing.json_telemetry import schema_check
            return schema_check
        except Exception:
            return None


_SCHEMA_CHECK = _load_schema_check()


def _fallback_schema_check(raw: str, required: Dict[str, str], no_extra: bool = True) -> dict:
    import json as _json
    out = {"parsed_ok": False, "schema_ok": False, "klass": "not_json",
           "missing": [], "mistyped": {}, "extra": []}
    s = (raw or "").strip()
    if not s:
        return out
    try:
        obj = _json.loads(s)
        out["parsed_ok"] = True
    except Exception:
        return out
    if not isinstance(obj, dict):
        out["klass"] = "schema_partial"
        out["missing"] = list(required)
        return out
    preds = {"string": lambda v: isinstance(v, str),
             "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
             "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
             "boolean": lambda v: isinstance(v, bool),
             "array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict)}
    for k, t in required.items():
        if k not in obj:
            out["missing"].append(k)
        elif not preds[t](obj[k]):
            out["mistyped"][k] = (t, type(obj[k]).__name__)
    if no_extra:
        out["extra"] = [k for k in obj if k not in required]
    out["schema_ok"] = not (out["missing"] or out["mistyped"] or out["extra"])
    out["klass"] = "schema_valid" if out["schema_ok"] else "schema_partial"
    return out


def schema_valid(raw: str, required: Dict[str, str]) -> Verdict:
    fn = _SCHEMA_CHECK or _fallback_schema_check
    chk = fn(raw, required)
    if chk["schema_ok"]:
        return ok("schema_valid")
    if not chk["parsed_ok"]:
        return fail("not a parseable JSON object")
    return fail(f"schema_partial missing={chk['missing']} mistyped={chk['mistyped']} extra={chk['extra']}")
