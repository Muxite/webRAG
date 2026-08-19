"""Lift chain waypoints and keystones out of the benchmark task modules.

WHY THE TASK MODULES AND NOT THE RUN CORPUS
-------------------------------------------
The obvious source for benchmark items is ``agent/idea_test_results/`` -- 3.3 GB
of real runs. It does not work, and the reason is worth recording so nobody
re-attempts it: the corpus stores only *aggregates* of what a run saw.
``observability.visit`` is ``{"count": 1, "chars": 26142, "words": 4304}``.
There is no page text, no URL list, no link set anywhere in it, and the
``*_json_telemetry.jsonl`` sidecars hold a ~200-character completion prefix
with no prompt attached.

The task modules are a better source anyway: hand-authored, exact, committed
to git rather than gitignored, and small enough to freeze into a fixture. Each
chain task declares its waypoints as data (``CHAIN`` with ``name``/``name_rx``/
``slug_rx``) precisely so the statement, the validators and the compiled plan
cannot drift apart -- which makes them ground truth by construction.

Output is a plain JSON fixture consumed by the item builders. This module is
the ONLY thing that imports the task modules; everything downstream reads the
fixture.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
from pathlib import Path
from typing import Any, Dict, List

TASK_PACKAGE = "agent.app.idea_tests"


def _iter_task_modules():
    pkg = importlib.import_module(TASK_PACKAGE)
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if not info.name.startswith("test_"):
            continue
        try:
            yield info.name, importlib.import_module(f"{TASK_PACKAGE}.{info.name}")
        except Exception:
            continue  # a task module that cannot import is not a usable spec


def _keystone_pattern(mod) -> str:
    rx = getattr(mod, "KEYSTONE_RX", None)
    if rx is None:
        return ""
    return rx.pattern if hasattr(rx, "pattern") else str(rx)


def _literal_alternatives(pattern: str) -> List[str]:
    """Pull the concrete literals out of a keystone pattern.

    ``\\b565\\b|\\b165\\b`` -> ``["565", "165"]``. Alternatives that are not plain
    literals are dropped rather than guessed at.

    A literal must carry a DIGIT. Without that filter four modules contribute
    ``inch``, ``inches``, ``feet``, ``metre`` -- the unit half of a pattern like
    ``\\b(24|610)\\s*(inch|mm)\\b``, which is a word the claim builder would happily
    "corrupt" into another word while the actual datum stayed untouched.

    These alternatives are NOT a right/wrong pair. ``\\b565\\b|\\b165\\b`` is one
    value in two units (565 ft == 165 m), and four of the fifteen keystone modules
    have that shape, so anything downstream that treats alternative #2 as the false
    twin builds a family whose negatives are all true. The corrupted claim has to be
    generated and then checked against the pattern.
    """
    out: List[str] = []
    for alt in pattern.split("|"):
        cleaned = alt.replace(r"\b", "").replace(r"\s*", " ").strip()
        if not cleaned or not re.fullmatch(r"[A-Za-z0-9 .,'-]+", cleaned):
            continue
        if not re.search(r"\d", cleaned):
            continue
        out.append(cleaned)
    return out


def _evidence_excerpt(mod) -> str:
    """The module docstring, as the evidence block for a claim-checking item.

    Each tier-5 task module opens with a hand-authored walkthrough of the intended
    solution path, and it states the keystone datum in prose. That makes it the one
    place in this repo holding page-level evidence -- the run corpus does not (see
    this module's header).

    It is evidence for ``keystone_claim`` and POISON for ``select``: the same
    docstrings annotate the answer outright (``test_091``:
    ``Deriner Dam ... <- TALLEST of the six (argmax)``). Which is why this is stored
    as a named field for one family to opt into rather than folded into ``statement``.

    Line breaks are PRESERVED. The item builder strips the authoring banner
    ("Test 059: Tier 5 ...", "Level: ... Difficulty: ...") line by line, and a
    pre-collapsed docstring turns any line-anchored pattern into one that eats the
    whole text.
    """
    return (mod.__doc__ or "").strip()


def _candidate_set(mod) -> List[Dict[str, Any]]:
    """Constraint-satisfaction candidate sets, where exactly one entry survives.

    ``CANDIDATES``/``ENTITIES`` with a ``survivor`` (or ``winner``) flag is a
    ready-made selection item: N described options, one of which satisfies the
    stated constraint. The negative population is built in, which is exactly
    what a precision-side metric needs and what a hand-built distractor list
    would only approximate.
    """
    for attr in ("CANDIDATES", "ENTITIES"):
        raw = getattr(mod, attr, None)
        if not isinstance(raw, list) or len(raw) < 3:
            continue
        out: List[Dict[str, Any]] = []
        for c in raw:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            survives = bool(c.get("survivor", c.get("winner", False)))
            out.append({
                "key": c.get("key", ""),
                "name": c.get("name", ""),
                "desc": c.get("desc", ""),
                "slug_rx": c.get("slug_rx", ""),
                "survivor": survives,
            })
        if len(out) >= 3 and sum(1 for c in out if c["survivor"]) == 1:
            return out
    return []


def collect() -> Dict[str, Any]:
    specs: List[Dict[str, Any]] = []
    for name, mod in _iter_task_modules():
        chain = getattr(mod, "CHAIN", None)
        candidates = _candidate_set(mod)
        has_chain = isinstance(chain, list) and len(chain) >= 2
        pattern = _keystone_pattern(mod)
        literals = _literal_alternatives(pattern)
        # A keystone alone is enough to qualify. Requiring a chain or a candidate
        # set as well discarded modules whose only structure is a hard datum --
        # exactly the raw material the claim-checking family runs on.
        if not has_chain and not candidates and not literals:
            continue
        chain = chain if has_chain else []
        meta = {}
        if hasattr(mod, "get_test_metadata"):
            try:
                meta = mod.get_test_metadata() or {}
            except Exception:
                meta = {}
        statement = ""
        if hasattr(mod, "get_task_statement"):
            try:
                statement = mod.get_task_statement() or ""
            except Exception:
                statement = ""
        waypoints = [
            {
                "key": w.get("key", ""),
                "name": w.get("name", ""),
                "role": w.get("role", ""),
                "name_rx": w.get("name_rx", ""),
                "slug_rx": w.get("slug_rx", ""),
            }
            for w in chain
            if isinstance(w, dict) and w.get("name")
        ]
        if len(waypoints) < 2 and not candidates and not literals:
            continue

        # Evidence is only carried for keystone-bearing specs. Storing every
        # module's docstring would roughly triple the fixture for text no family
        # reads, and would put `select`'s answers in the fixture next to its items.
        evidence = _evidence_excerpt(mod) if literals else ""
        rx = getattr(mod, "KEYSTONE_RX", None)
        specs.append({
            "candidates": candidates,
            "module": name,
            "test_id": str(meta.get("test_id", "")),
            "test_name": meta.get("test_name", ""),
            "difficulty": meta.get("difficulty_level", ""),
            "statement": statement,
            "keystone_pattern": pattern,
            "keystone_literals": literals,
            "evidence": evidence,
            # Recorded, not acted on here: the item builder decides what to drop.
            # An item whose statement already contains the keystone measures reading
            # rather than judgement, and one whose evidence lacks it is unanswerable.
            "statement_leaks_keystone": bool(rx is not None and literals and rx.search(statement)),
            "keystone_in_evidence": bool(rx is not None and literals and rx.search(evidence)),
            "waypoints": waypoints,
        })
    return {"schema": 2, "source": TASK_PACKAGE, "specs": specs}


def main() -> int:
    data = collect()
    out = Path("agent/tests/fixtures/promptbench/task_specs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    n_way = sum(len(s["waypoints"]) for s in data["specs"])
    n_key = sum(1 for s in data["specs"] if s["keystone_literals"])
    n_cand = sum(1 for s in data["specs"] if s["candidates"])
    n_opts = sum(len(s["candidates"]) for s in data["specs"])
    n_leak = sum(1 for s in data["specs"] if s["statement_leaks_keystone"])
    n_eviq = sum(1 for s in data["specs"] if s["keystone_in_evidence"])
    print(f"wrote {out}")
    print(f"  specs: {len(data['specs'])}  waypoints: {n_way}  with keystone literals: {n_key}")
    print(f"  candidate sets: {n_cand}  options across them: {n_opts}")
    print(f"  keystone usable: {n_eviq} in evidence, {n_leak} leaking into the statement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
