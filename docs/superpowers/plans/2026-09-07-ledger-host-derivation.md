# Ledger Host Derivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ledger certify answers without depending on the model calling `derive`, by ranking operands with a small trained model and computing the demanded operation host-side before the answer is graded.

**Architecture:** Three new deterministic pieces sit beside the existing `LedgerToolkit`: an operand-attribution ranker (frozen MiniLM embeddings + logistic regression, weights stored as JSON), a `host_derive` method that uses the mandate's demanded operation, its entity roster and the ranker to mint SOURCE/DERIVED nodes tagged `host_derive` from pages only, and a sixth certify signal in `ledger_risk_coverage.py` that compares the answer to the host's value. A learned trust channel over mechanical cell features gives the full risk-coverage curve. No prompt, tool, or model call changes anywhere.

**Tech Stack:** Python 3.12, pytest, scikit-learn 1.9, sentence-transformers 5.6 (`all-MiniLM-L6-v2`, downloaded once), numpy. Corpus replay via `SEARCH_PROVIDER=corpus`. Campaign launch via `scripts/run_campaign.sh`.

**Spec:** `docs/superpowers/specs/2026-09-07-ledger-and-dag-next-gen-design.md` Part B (L1, L2, L3, L5).

## Global Constraints

- No LLM fine-tuning, no LoRA; completion endpoints untouched. Learned components are ≤ small encoders + classical models.
- Never convert units. `derive` refuses mismatched dimensions (spec Part B, `docs/LEDGER.md`).
- Holdout tasks 213, 217, 221 are SEALED: never used for fitting, threshold choice, or feature selection. Split by `test_id` always.
- Nodes minted by `host_derive` must carry `minted_by="host_derive"` and must be excluded from clauses 1–5 exactly as `answer_audit`/`shape_derive` nodes are (`scripts/ledger_risk_coverage.py:551`).
- Model identity and size are NOT features of the trust channel.
- Test invocation: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/<file>` (never bare `pytest` from the repo root: `.claude/worktrees/` holds copies).
- Commit style: single lowercase line, no punctuation, no body, no trailer.
- Do not edit `agent/app/**` while a campaign is running (`scripts/run_campaign.sh` lockfile).

---

## File Structure

| File | Responsibility |
|---|---|
| `agent/app/ledger_tools.py` (modify) | fix refusal recording; add `host_derive()`; add `HOST_DERIVE_TAG` |
| `agent/app/operand_attribution.py` (create) | feature extraction + ranker `rank_entries()`; loads `operand_attribution_model.json` |
| `agent/app/operand_attribution_model.json` (create, by fit script) | serialized logistic-regression weights + feature names + embedding model id |
| `scripts/operand_attribution_data.py` (create) | build weak-supervision pairs from stored pages; build the task eval set from modules 210–221 |
| `scripts/fit_operand_attribution.py` (create) | fit LR, evaluate on dev tasks, write the JSON artifact + report |
| `agent/app/testing/execution_sequential.py` (modify ~749-800) | `host_derive` token, call before `artifact()` |
| `agent/app/langgraph_solver.py` (modify ~1541, ~1918-1933) | same wiring |
| `scripts/ledger_risk_coverage.py` (modify) | `host_derive` summary + `host_agrees` signal + report section |
| `scripts/ledger_trust_channel.py` (create) | GBM trust channel, task-grouped CV, per-model curves |
| `agent/app/ledger_api.py` (modify) | supplied-source truncation + `clean_operation` offset defects |
| `agent/idea_test_results/prereg/mint03_*.json` (create) | preregistration for the confirmatory campaign |
| Tests: `agent/tests/ledger_tools_test.py`, `agent/tests/operand_attribution_test.py`, `agent/tests/ledger_risk_coverage_test.py`, `agent/tests/ledger_trust_channel_test.py`, `agent/tests/ledger_api_test.py` | one test file per unit |

---

### Task 1: Record refusals on the `LedgerToolkit` path

**Files:**
- Modify: `agent/app/ledger_tools.py:296-300` (the `except DerivationError` branch in `derive`)
- Test: `agent/tests/ledger_tools_test.py`

**Interfaces:**
- Consumes: `EvidenceGraph.record_refusal(operation, input_ids, error)` (`agent/app/testing/evidence_graph.py:1329`).
- Produces: `kit.artifact()["derivation_refusals"]` non-empty after a refused derive; `artifact()["counts"]` unchanged.

- [ ] **Step 1: Write the failing test**

Append to `agent/tests/ledger_tools_test.py`:

```python
def test_a_refused_derivation_is_recorded_on_the_artifact(kit):
    """`ledger_trace.query(kind="derive", status="refused")` reads `derivation_refusals`; on
    this host it was always empty, so "refused" and "never attempted" were indistinguishable."""
    observation = kit.derive("difference", ["419.7 metres", "380.0 metres"])
    assert observation.startswith("DERIVED")

    # Same units on the page would compute; force a unit mismatch through a second page.
    kit.register_page("https://example.com/b", "The bridge is 1,250 feet long.")
    refused = kit.derive("difference", ["419.7 metres", "1,250 feet"])
    assert refused.startswith("DERIVE REFUSED (UNIT_MISMATCH)")

    refusals = kit.artifact()["derivation_refusals"]
    assert len(refusals) == 1
    assert refusals[0]["operation"] == "difference"
    assert refusals[0]["code"] == "UNIT_MISMATCH"
    assert len(refusals[0]["input_ids"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_tools_test.py::test_a_refused_derivation_is_recorded_on_the_artifact -v`
Expected: FAIL with `assert 0 == 1` (refusals list empty).

- [ ] **Step 3: Record the refusal**

In `agent/app/ledger_tools.py`, replace the `except DerivationError` branch of `derive`:

```python
        try:
            node = self._graph.add_arith(name, input_ids, proposed_value=proposed_value)
        except DerivationError as exc:
            # Same contract as `execution_evidence_loop._handle_derive`: a refusal is an
            # observation for the model AND a countable row on the artifact, so
            # `ledger_trace.query(kind="derive", status="refused")` can see it on this host too.
            self._graph.record_refusal(name, input_ids, exc)
            return f"DERIVE REFUSED ({exc.code}): {exc}"
```

- [ ] **Step 4: Run the ledger test file**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_tools_test.py -v`
Expected: all PASS, including the new test.

- [ ] **Step 5: Commit**

```bash
git add agent/app/ledger_tools.py agent/tests/ledger_tools_test.py
git commit -m "record refused derivations on the ledger toolkit hosts so refused is distinguishable from never attempted"
```

---

### Task 2: Weak-supervision dataset for operand attribution

**Files:**
- Create: `scripts/operand_attribution_data.py`
- Test: `agent/tests/operand_attribution_data_test.py`

**Interfaces:**
- Consumes: `agent.app.quantity_index.build_index(page_text, limit=40) -> List[QuantityRef]` (fields `label, value, unit, start, end, source`); stored cells' `execution.output.pages[]` (`{page_id, url, text, ...}`).
- Produces:
  - `harvest_pairs(pages) -> List[dict]` rows: `{"entity", "field", "page_url", "page_title", "entry_index", "label", "value", "unit", "source", "context", "y"}`; `y=1` for the infobox entry whose label is the field phrase, `y=0` for every other entry on the same page.
  - `task_eval_set(test_ids) -> List[dict]` rows with the same keys built from modules 210–221's `OP_A`/`OP_B` (positives), `_DECOYS` values (hard negatives), over stored pages whose URL matches `OP_*["url"]`.
  - CLI writes `agent/idea_test_results/operand_attribution/{train.jsonl,eval_dev.jsonl,eval_holdout.jsonl}`.

- [ ] **Step 1: Write the failing test**

Create `agent/tests/operand_attribution_data_test.py`:

```python
"""Weak supervision for operand attribution: an infobox label/value pair on a stored page is a
self-labeled (field phrase -> entry) positive with no task involved, so the ranker can be trained
without touching tasks 210-221."""
from __future__ import annotations

import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "operand_attribution_data",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "operand_attribution_data.py")
oad = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oad)

PAGE_TEXT = (
    "Inco Superstack\n"
    "Height\n381 m\n"
    "Completed\n1972\n"
    "The stack served the Copper Cliff smelter and was 380 metres tall in some sources.\n"
)
PAGE = {"page_id": "p1", "url": "https://en.wikipedia.org/wiki/Inco_Superstack", "text": PAGE_TEXT}


def test_each_infobox_label_yields_one_positive_and_negatives_for_the_other_entries():
    rows = oad.harvest_pairs([PAGE])
    height_rows = [r for r in rows if r["field"] == "Height"]
    assert sum(r["y"] for r in height_rows) == 1
    positive = next(r for r in height_rows if r["y"] == 1)
    assert positive["value"] == "381" and positive["unit"] == "m"
    assert positive["entity"] == "Inco Superstack"
    assert any(r["y"] == 0 and r["value"] == "380" for r in height_rows)


def test_page_title_comes_from_the_url_slug_when_text_has_no_title():
    assert oad.entity_from_url("https://en.wikipedia.org/wiki/GRES-2_Power_Station") == "GRES-2 Power Station"


def test_task_eval_rows_mark_the_true_operand_and_the_decoys(monkeypatch):
    fake_module = type("M", (), {})()
    fake_module.OP_A = {"key": "a", "label": "GRES-2 height", "value": 419.7,
                        "url": "https://en.wikipedia.org/wiki/GRES-2_Power_Station",
                        "fact": "GRES-2 chimney height"}
    fake_module.OP_B = {"key": "b", "label": "Superstack height", "value": 381.0,
                        "url": "https://en.wikipedia.org/wiki/Inco_Superstack",
                        "fact": "Inco Superstack height"}
    fake_module._DECOYS = [("sum instead of difference", 800.7)]
    monkeypatch.setattr(oad, "_load_task_module", lambda test_id: fake_module)
    monkeypatch.setattr(oad, "_stored_pages_for_url", lambda url: [PAGE] if "Superstack" in url else [])
    rows = oad.task_eval_set(["210"])
    positives = [r for r in rows if r["y"] == 1]
    assert positives and all(abs(float(r["value"].replace(",", "")) - 381.0) < 1e-6 for r in positives)
    assert all(r["test_id"] == "210" for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/operand_attribution_data_test.py -v`
Expected: FAIL, `FileNotFoundError` (script does not exist).

- [ ] **Step 3: Write the data builder**

Create `scripts/operand_attribution_data.py`:

```python
#!/usr/bin/env python3
"""Build training and evaluation rows for the operand-attribution ranker.

Weak supervision (no task leakage): on every stored page, each infobox entry's LABEL is a
field phrase and that entry is the positive for it; every other entry on the page is a
negative for that phrase. The page's entity is its title (URL slug). Evaluation rows come from
tasks 210-221's `OP_A`/`OP_B` (positives) and `_DECOYS` (hard negatives) and are NEVER used
for fitting. Holdout 213/217/221 is written to its own file and never read by the fit script.

Usage::
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/operand_attribution_data.py \
        --out agent/idea_test_results/operand_attribution
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "agent"))

from agent.app.quantity_index import QuantityRef, build_index  # noqa: E402

RESULTS_DIR = ROOT / "agent" / "idea_test_results"
HOLDOUT = {"213", "217", "221"}
DEV = {"210", "211", "212", "214", "215", "216", "218", "219", "220"}
CONTEXT_CHARS = 160
VALUE_TOL = 0.02


def entity_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[_]+", " ", slug).split("#")[0].strip()


def _context(text: str, entry: QuantityRef) -> str:
    lo = max(0, entry.start - CONTEXT_CHARS)
    hi = min(len(text), entry.end + CONTEXT_CHARS)
    return text[lo:hi].replace("\n", " ")


def _row(entity: str, field: str, page: Dict[str, Any], index: int, entry: QuantityRef,
         text: str, y: int, **extra: Any) -> Dict[str, Any]:
    row = {
        "entity": entity, "field": field, "page_url": page.get("url", ""),
        "page_title": entity_from_url(page.get("url", "")), "entry_index": index,
        "label": entry.label, "value": entry.value, "unit": entry.unit, "source": entry.source,
        "context": _context(text, entry), "y": int(y),
    }
    row.update(extra)
    return row


def harvest_pairs(pages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in pages:
        text = str(page.get("text") or "")
        if not text:
            continue
        entries = build_index(text)
        entity = entity_from_url(str(page.get("url") or ""))
        labels = [(i, e) for i, e in enumerate(entries) if e.source == "infobox" and e.label]
        seen_fields = set()
        for pos_index, pos_entry in labels:
            field = pos_entry.label.strip()
            if field in seen_fields:
                continue  # one positive per field phrase; duplicate labels are ambiguous
            seen_fields.add(field)
            for i, entry in enumerate(entries):
                rows.append(_row(entity, field, page, i, entry, text, int(i == pos_index)))
    return rows


def _load_task_module(test_id: str):
    matches = glob.glob(str(ROOT / "agent" / "app" / "idea_tests" / f"test_{test_id}_*.py"))
    if not matches:
        raise FileNotFoundError(test_id)
    name = Path(matches[0]).stem
    return importlib.import_module(f"agent.app.idea_tests.{name}")


def _stored_pages_for_url(url: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_hashes = set()
    for path in glob.glob(str(RESULTS_DIR / "**" / "*_r1.json"), recursive=True):
        if "/smoke_archive/" in path:
            continue
        try:
            cell = json.load(open(path))
        except Exception:
            continue
        for page in ((cell.get("execution") or {}).get("output") or {}).get("pages") or []:
            if page.get("url") == url and page.get("text") and page.get("content_hash") not in seen_hashes:
                seen_hashes.add(page.get("content_hash"))
                out.append(page)
    return out


def _numeric(value: str) -> Optional[float]:
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def task_eval_set(test_ids: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for test_id in test_ids:
        module = _load_task_module(test_id)
        operands = [getattr(module, "OP_A", None), getattr(module, "OP_B", None)]
        decoys = [float(v) for _, v in getattr(module, "_DECOYS", [])]
        for operand in operands:
            if not operand:
                continue
            field = str(operand.get("fact") or operand.get("label") or "")
            for page in _stored_pages_for_url(str(operand.get("url") or "")):
                text = str(page.get("text") or "")
                entries = build_index(text)
                entity = entity_from_url(str(page.get("url") or ""))
                for i, entry in enumerate(entries):
                    number = _numeric(entry.value)
                    is_true = number is not None and abs(number - float(operand["value"])) <= VALUE_TOL * max(1.0, abs(float(operand["value"])))
                    is_decoy = number is not None and any(abs(number - d) <= VALUE_TOL * max(1.0, abs(d)) for d in decoys)
                    rows.append(_row(entity, field, page, i, entry, text, int(is_true),
                                     test_id=str(test_id), decoy=int(is_decoy and not is_true)))
    return rows


def _all_pages() -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    seen = set()
    for path in glob.glob(str(RESULTS_DIR / "**" / "*_r1.json"), recursive=True):
        if "/smoke_archive/" in path:
            continue
        try:
            cell = json.load(open(path))
        except Exception:
            continue
        for page in ((cell.get("execution") or {}).get("output") or {}).get("pages") or []:
            key = page.get("content_hash") or page.get("url")
            if page.get("text") and key not in seen:
                seen.add(key)
                pages.append(page)
    return pages


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = harvest_pairs(_all_pages())
    with open(out / "train.jsonl", "w") as fh:
        for row in train:
            fh.write(json.dumps(row) + "\n")
    for name, ids in (("eval_dev", sorted(DEV)), ("eval_holdout", sorted(HOLDOUT))):
        with open(out / f"{name}.jsonl", "w") as fh:
            for row in task_eval_set(ids):
                fh.write(json.dumps(row) + "\n")
    print(f"train rows={len(train)} positives={sum(r['y'] for r in train)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/operand_attribution_data_test.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Build the real dataset and record its size**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/operand_attribution_data.py --out agent/idea_test_results/operand_attribution`
Expected: prints `train rows=<N> positives=<P>` with N in the tens of thousands (4,359 stored pages, ~3,800 Wikipedia). Record N and P in the commit message body of Task 4's report, not here. `agent/idea_test_results/` is gitignored; do not commit the JSONL.

- [ ] **Step 6: Commit**

```bash
git add scripts/operand_attribution_data.py agent/tests/operand_attribution_data_test.py
git commit -m "build weakly supervised operand attribution rows from stored infobox pages and a sealed task eval set"
```

---

### Task 3: The operand-attribution ranker

**Files:**
- Create: `agent/app/operand_attribution.py`
- Create: `scripts/fit_operand_attribution.py`
- Test: `agent/tests/operand_attribution_test.py`

**Interfaces:**
- Consumes: rows from Task 2; `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")`.
- Produces:
  - `features(entity, field, entry: QuantityRef, page_title, page_text, embed) -> List[float]` with fixed `FEATURE_NAMES`.
  - `class OperandRanker` with `load(path=DEFAULT_MODEL_PATH) -> OperandRanker`, `score(entity, field, entries, page_title, page_text) -> List[float]`, `rank(...) -> List[Tuple[float, int]]` (score, entry index, descending).
  - `agent/app/operand_attribution_model.json`: `{"feature_names": [...], "coef": [...], "intercept": f, "embed_model": "all-MiniLM-L6-v2", "fit_report": {...}}`.
  - The embedding call is lazy and optional: with `embed=None` (no model available) the two cosine features are 0.0 and the lexical features still work, so the ranker never blocks a host.

- [ ] **Step 1: Write the failing test**

Create `agent/tests/operand_attribution_test.py`:

```python
"""The operand-attribution ranker: which quantity-index entry is the operand a (entity, field)
asks for. Lexical features must carry it even with no embedding model present."""
from __future__ import annotations

import json

import pytest

from agent.app import operand_attribution as oa
from agent.app.quantity_index import build_index

PAGE = ("Inco Superstack\nHeight\n381 m\nCompleted\n1972\nBase width\n35 m\n"
        "The Superstack is 381 metres tall.\n")


def test_feature_vector_has_the_declared_names_and_length():
    entries = build_index(PAGE)
    vec = oa.features("Inco Superstack", "height", entries[0], "Inco Superstack", PAGE, embed=None)
    assert len(vec) == len(oa.FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)


def test_label_overlap_feature_is_one_when_field_matches_label():
    entries = build_index(PAGE)
    height = next(e for e in entries if e.label.lower() == "height")
    width = next(e for e in entries if e.label.lower() == "base width")
    i_overlap = oa.FEATURE_NAMES.index("label_token_overlap")
    assert oa.features("x", "height", height, "x", PAGE, embed=None)[i_overlap] == 1.0
    assert oa.features("x", "height", width, "x", PAGE, embed=None)[i_overlap] < 1.0


def test_a_hand_built_model_ranks_the_matching_label_first(tmp_path):
    weights = {name: 0.0 for name in oa.FEATURE_NAMES}
    weights["label_token_overlap"] = 4.0
    weights["is_infobox"] = 1.0
    artifact = {"feature_names": oa.FEATURE_NAMES,
                "coef": [weights[n] for n in oa.FEATURE_NAMES],
                "intercept": -2.0, "embed_model": None, "fit_report": {}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact))
    ranker = oa.OperandRanker.load(path)
    entries = build_index(PAGE)
    ranked = ranker.rank("Inco Superstack", "height", entries, "Inco Superstack", PAGE)
    best_score, best_index = ranked[0]
    assert entries[best_index].value == "381"
    assert 0.0 < best_score < 1.0


def test_missing_model_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        oa.OperandRanker.load(tmp_path / "nope.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/operand_attribution_test.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError: agent.app.operand_attribution`.

- [ ] **Step 3: Write the ranker module**

Create `agent/app/operand_attribution.py`:

```python
"""Operand attribution: which quantity-index entry is the operand an (entity, field) asks for.

Replaces the first-match-wins / fixed-preference / candidate-cap heuristics in
``ledger_tools`` (`_locate`, `_find_backed_match`, `_best_explanation`,
``SHAPE_DERIVE_MAX_CANDIDATES``) with a RANKING. The ranker is a logistic regression over a
fixed feature vector; two of the features are cosine similarities from a frozen sentence
encoder and are 0.0 when no encoder is available, so the lexical features always work and a
host never blocks on a model download.

Weights live in ``operand_attribution_model.json`` next to this file, written by
``scripts/fit_operand_attribution.py``. No pickle: the artifact is plain JSON so it can be
diffed, reviewed and re-verified.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.app.quantity_index import QuantityRef

DEFAULT_MODEL_PATH = Path(__file__).with_name("operand_attribution_model.json")
EMBED_MODEL_ID = "all-MiniLM-L6-v2"

FEATURE_NAMES: List[str] = [
    "label_token_overlap",      # |tokens(field) ∩ tokens(label)| / |tokens(field)|
    "field_in_context",         # any field token appears within the ±160-char context
    "entity_in_context",        # any entity token (len>3) appears within the context
    "entity_is_page_title",     # page title == entity (normalized)
    "is_infobox",               # entry.source == "infobox"
    "has_unit",                 # entry.unit != ""
    "is_trivial_bare_int",      # unitless integer < 100 or a year 1900-2099
    "position_frac",            # entry index / n entries (document order)
    "label_cos",                # cosine(field, label) from the encoder, else 0.0
    "context_cos",              # cosine(field + entity, context) from the encoder, else 0.0
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CONTEXT_CHARS = 160


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _context(page_text: str, entry: QuantityRef) -> str:
    lo = max(0, entry.start - _CONTEXT_CHARS)
    hi = min(len(page_text), entry.end + _CONTEXT_CHARS)
    return page_text[lo:hi]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _is_trivial(entry: QuantityRef) -> bool:
    if entry.unit:
        return False
    raw = str(entry.value).replace(",", "")
    if not re.fullmatch(r"-?\d+", raw):
        return False
    number = int(raw)
    return abs(number) < 100 or 1900 <= number <= 2099


def features(entity: str, field: str, entry: QuantityRef, page_title: str, page_text: str,
             embed: Optional[Callable[[str], Sequence[float]]], *, index: int = 0,
             n_entries: int = 1) -> List[float]:
    field_tokens = _tokens(field)
    label_tokens = _tokens(entry.label)
    context = _context(page_text, entry)
    context_tokens = _tokens(context)
    entity_tokens = {t for t in _tokens(entity) if len(t) > 3}
    overlap = (len(field_tokens & label_tokens) / len(field_tokens)) if field_tokens else 0.0
    label_cos = context_cos = 0.0
    if embed is not None:
        try:
            f_vec = embed(field)
            label_cos = _cos(f_vec, embed(entry.label)) if entry.label else 0.0
            context_cos = _cos(embed(f"{field} {entity}"), embed(context))
        except Exception:  # noqa: BLE001 -- an encoder failure degrades to lexical-only
            label_cos = context_cos = 0.0
    return [
        float(overlap),
        float(bool(field_tokens & context_tokens)),
        float(bool(entity_tokens & context_tokens)),
        float(_tokens(page_title) == _tokens(entity) and bool(entity_tokens)),
        float(entry.source == "infobox"),
        float(bool(entry.unit)),
        float(_is_trivial(entry)),
        float(index / max(1, n_entries)),
        float(label_cos),
        float(context_cos),
    ]


class OperandRanker:
    """A fitted logistic regression over :data:`FEATURE_NAMES`."""

    def __init__(self, coef: Sequence[float], intercept: float, embed_model: Optional[str],
                 fit_report: Optional[Dict[str, Any]] = None) -> None:
        if len(coef) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} coefficients, got {len(coef)}")
        self.coef = [float(c) for c in coef]
        self.intercept = float(intercept)
        self.embed_model = embed_model
        self.fit_report = dict(fit_report or {})
        self._embed: Optional[Callable[[str], Sequence[float]]] = None
        self._embed_tried = False

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "OperandRanker":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"operand attribution model not found: {path}")
        data = json.loads(path.read_text())
        if list(data.get("feature_names") or []) != FEATURE_NAMES:
            raise ValueError("model feature names do not match this module's FEATURE_NAMES")
        return cls(data["coef"], data["intercept"], data.get("embed_model"), data.get("fit_report"))

    def _embedder(self) -> Optional[Callable[[str], Sequence[float]]]:
        if self._embed_tried:
            return self._embed
        self._embed_tried = True
        if not self.embed_model:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embed_model)
            cache: Dict[str, Sequence[float]] = {}

            def embed(text: str) -> Sequence[float]:
                if text not in cache:
                    cache[text] = model.encode(text, normalize_embeddings=True).tolist()
                return cache[text]

            self._embed = embed
        except Exception:  # noqa: BLE001 -- offline or missing encoder: lexical-only
            self._embed = None
        return self._embed

    def score(self, entity: str, field: str, entries: Sequence[QuantityRef], page_title: str,
              page_text: str) -> List[float]:
        embed = self._embedder()
        out: List[float] = []
        for i, entry in enumerate(entries):
            vec = features(entity, field, entry, page_title, page_text, embed,
                           index=i, n_entries=len(entries))
            logit = self.intercept + sum(c * v for c, v in zip(self.coef, vec))
            out.append(1.0 / (1.0 + math.exp(-logit)))
        return out

    def rank(self, entity: str, field: str, entries: Sequence[QuantityRef], page_title: str,
             page_text: str) -> List[Tuple[float, int]]:
        scores = self.score(entity, field, entries, page_title, page_text)
        return sorted(((s, i) for i, s in enumerate(scores)), key=lambda t: (-t[0], t[1]))
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/operand_attribution_test.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Write the fit script**

Create `scripts/fit_operand_attribution.py`:

```python
#!/usr/bin/env python3
"""Fit the operand-attribution ranker on weak-supervision rows and evaluate on the task dev set.

Never reads ``eval_holdout.jsonl``. Writes ``agent/app/operand_attribution_model.json`` and
prints precision@1 per (test_id, field) on the dev set, split by infobox vs prose positives.

Usage::
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/fit_operand_attribution.py \
        --data agent/idea_test_results/operand_attribution [--no-embed]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "services")); sys.path.insert(0, str(ROOT / "agent"))

from agent.app import operand_attribution as oa  # noqa: E402
from agent.app.quantity_index import QuantityRef  # noqa: E402


def _entry(row: Dict[str, Any]) -> QuantityRef:
    return QuantityRef(label=row["label"], value=row["value"], unit=row["unit"], start=0, end=0,
                       source=row["source"])


def _vectors(rows: List[Dict[str, Any]], embed) -> np.ndarray:
    # `context` is stored pre-cut, so pass it as the page text with start=end=0: the ±160 window
    # then covers the stored context exactly.
    groups: Dict[tuple, int] = defaultdict(int)
    for r in rows:
        groups[(r["page_url"], r["field"])] += 1
    return np.array([
        oa.features(r["entity"], r["field"], _entry(r), r["page_title"], r["context"], embed,
                    index=r["entry_index"], n_entries=groups[(r["page_url"], r["field"])])
        for r in rows
    ], dtype=float)


def precision_at_1(rows: List[Dict[str, Any]], scores: np.ndarray) -> Dict[str, Any]:
    by_group: Dict[tuple, List[tuple]] = defaultdict(list)
    for r, s in zip(rows, scores):
        by_group[(r.get("test_id"), r["page_url"], r["field"])].append((s, r))
    hits = {"infobox": [0, 0], "prose": [0, 0]}
    for _, items in by_group.items():
        positives = [r for _, r in items if r["y"] == 1]
        if not positives:
            continue
        kind = positives[0]["source"]
        top = max(items, key=lambda t: t[0])[1]
        hits[kind][1] += 1
        hits[kind][0] += int(top["y"] == 1)
    return {k: {"hits": v[0], "groups": v[1], "p_at_1": (v[0] / v[1]) if v[1] else None}
            for k, v in hits.items()}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--out", default=str(oa.DEFAULT_MODEL_PATH))
    args = parser.parse_args(argv)
    data = Path(args.data)
    train = [json.loads(l) for l in open(data / "train.jsonl")]
    dev = [json.loads(l) for l in open(data / "eval_dev.jsonl")]

    embed = None
    embed_model = None
    if not args.no_embed:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(oa.EMBED_MODEL_ID)
        cache: Dict[str, Any] = {}

        def embed(text: str):
            if text not in cache:
                cache[text] = model.encode(text, normalize_embeddings=True).tolist()
            return cache[text]
        embed_model = oa.EMBED_MODEL_ID

    x = _vectors(train, embed)
    y = np.array([r["y"] for r in train])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)
    dev_scores = clf.decision_function(_vectors(dev, embed))
    report = {"n_train": int(len(train)), "n_train_pos": int(y.sum()),
              "dev_precision_at_1": precision_at_1(dev, dev_scores),
              "embed": embed_model}
    artifact = {"feature_names": oa.FEATURE_NAMES, "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]), "embed_model": embed_model,
                "fit_report": report}
    Path(args.out).write_text(json.dumps(artifact, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Fit once lexical-only, once with the encoder, and keep the better dev result**

Run:
```bash
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/fit_operand_attribution.py --data agent/idea_test_results/operand_attribution --no-embed --out /tmp/oa_lexical.json
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/fit_operand_attribution.py --data agent/idea_test_results/operand_attribution --out agent/app/operand_attribution_model.json
```
Expected: both print a `dev_precision_at_1` block with `infobox` and `prose` entries. The first encoder run downloads `all-MiniLM-L6-v2` (network needed once). If the encoder run's infobox p@1 is not higher than lexical-only, copy `/tmp/oa_lexical.json` over the artifact and note it in the commit. Spec gate: infobox p@1 ≥ 0.9 on dev before Task 4 proceeds; prose is reported, not gated.

- [ ] **Step 7: Commit**

```bash
git add agent/app/operand_attribution.py agent/app/operand_attribution_model.json scripts/fit_operand_attribution.py agent/tests/operand_attribution_test.py
git commit -m "add the operand attribution ranker with a json weight artifact fitted on infobox weak supervision"
```

---

### Task 4: `host_derive` on the toolkit

**Files:**
- Modify: `agent/app/ledger_tools.py` (new constant after `SHAPE_DERIVE_TAG`; new method after `shape_derive_check`)
- Test: `agent/tests/ledger_tools_test.py`

**Interfaces:**
- Consumes: `answer_numbers.mandate_demanded_operation(mandate) -> {"operation","absolute","reason"}`; `idea_policies.candidate_coverage.extract_named_candidates(mandate) -> List[str]`; `idea_policies.entity_names.named_entities(*texts) -> List[str]`; `OperandRanker`; `EvidenceGraph.add_arith(op, ids, minted_by=...)`; `self._entries` (run-global `(page_id, QuantityRef)` list built by `register_page`) and `self._graph.page(page_id)`.
- Produces: `LedgerToolkit.host_derive(mandate: str, *, ranker=None) -> Dict[str, Any]` with keys `{"demanded_operation", "entities", "operands": [{"entity","page_id","value","unit","score","node_id"}], "value", "unit", "derived_node_id", "reason"}`; `reason` is one of `"computed"`, `"no_unambiguous_shape"`, `"no_index_entries"`, `"fewer_than_two_entities"`, `"operand_not_found:<entity>"`, `"refused:<CODE>"`, `"no_ranker"`. `HOST_DERIVE_TAG = "host_derive"`.

- [ ] **Step 1: Write the failing tests**

Append to `agent/tests/ledger_tools_test.py`:

```python
from agent.app.operand_attribution import FEATURE_NAMES, OperandRanker


def _lexical_ranker() -> OperandRanker:
    coef = [0.0] * len(FEATURE_NAMES)
    coef[FEATURE_NAMES.index("label_token_overlap")] = 4.0
    coef[FEATURE_NAMES.index("entity_is_page_title")] = 2.0
    coef[FEATURE_NAMES.index("is_infobox")] = 1.0
    coef[FEATURE_NAMES.index("is_trivial_bare_int")] = -3.0
    return OperandRanker(coef, -2.0, embed_model=None)


TWO_ENTITY_MANDATE = (
    "What is the absolute difference in height between these two chimneys?\n"
    "1. GRES-2 Power Station — the Ekibastuz chimney\n"
    "2. Inco Superstack — the Sudbury chimney\n")
GRES_PAGE = "GRES-2 Power Station\nHeight\n419.7 m\nCompleted\n1987\nUnits\n8\n"
STACK_PAGE = "Inco Superstack\nHeight\n381 m\nBase width\n35 m\nCompleted\n1972\n"


def test_host_derive_computes_the_demanded_operation_from_pages_only():
    kit = LedgerToolkit()
    kit.register_page("https://en.wikipedia.org/wiki/GRES-2_Power_Station", GRES_PAGE)
    kit.register_page("https://en.wikipedia.org/wiki/Inco_Superstack", STACK_PAGE)

    result = kit.host_derive(TWO_ENTITY_MANDATE, ranker=_lexical_ranker())

    assert result["reason"] == "computed"
    assert result["demanded_operation"] == "difference"
    assert abs(float(result["value"]) - 38.7) < 1e-6
    assert {o["entity"] for o in result["operands"]} == {"GRES-2 Power Station", "Inco Superstack"}
    node = next(n for n in kit.artifact()["nodes"] if n["id"] == result["derived_node_id"])
    assert node["minted_by"] == "host_derive"
    assert node["derivation_valid"] is True


def test_host_derive_takes_one_operand_per_named_entity_never_two_from_one_page():
    """The mint02 failure: both operands from a single entity's page (height minus base width)."""
    kit = LedgerToolkit()
    kit.register_page("https://en.wikipedia.org/wiki/Inco_Superstack", STACK_PAGE)

    result = kit.host_derive(TWO_ENTITY_MANDATE, ranker=_lexical_ranker())

    assert result["reason"] == "operand_not_found:GRES-2 Power Station"
    assert result["derived_node_id"] is None


def test_host_derive_is_silent_without_an_unambiguous_shape():
    kit = LedgerToolkit()
    kit.register_page("https://en.wikipedia.org/wiki/Inco_Superstack", STACK_PAGE)
    result = kit.host_derive("Tell me about the Inco Superstack.", ranker=_lexical_ranker())
    assert result["reason"] == "no_unambiguous_shape"
    assert kit.artifact()["nodes"] == []


def test_host_derive_refuses_mismatched_units_and_records_the_refusal():
    kit = LedgerToolkit()
    kit.register_page("https://en.wikipedia.org/wiki/GRES-2_Power_Station",
                      "GRES-2 Power Station\nHeight\n1,377 ft\n")
    kit.register_page("https://en.wikipedia.org/wiki/Inco_Superstack", STACK_PAGE)
    result = kit.host_derive(TWO_ENTITY_MANDATE, ranker=_lexical_ranker())
    assert result["reason"] == "refused:UNIT_MISMATCH"
    assert kit.artifact()["derivation_refusals"][0]["code"] == "UNIT_MISMATCH"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_tools_test.py -k host_derive -v`
Expected: 4 FAIL with `AttributeError: 'LedgerToolkit' object has no attribute 'host_derive'`.

- [ ] **Step 3: Implement `host_derive`**

In `agent/app/ledger_tools.py`, after `SHAPE_DERIVE_TAG = "shape_derive"` add:

```python
#: `minted_by` tag for nodes minted by :meth:`LedgerToolkit.host_derive`: operands chosen from
#: the mandate's entity roster and the run's own pages by the operand-attribution ranker, the
#: result recomputed in Python, all BEFORE the deliverable exists. Not circular with the answer
#: (unlike `answer_audit`/`shape_derive`, which mint from the deliverable), so a certify signal
#: may consume it; still excluded from clauses 1-5 so the pre-registered chain is unchanged.
HOST_DERIVE_TAG = "host_derive"
```

Add the imports near the top of the file (after the existing `from agent.app.answer_numbers import ...` line):

```python
from agent.app.idea_policies.candidate_coverage import extract_named_candidates
from agent.app.idea_policies.entity_names import named_entities
```

Add the method to `LedgerToolkit`, after `shape_derive_check`:

```python
    def host_derive(self, mandate: Any, *, ranker: Any = None) -> Dict[str, Any]:
        """Compute the operation ``mandate`` demands, host-side, from the pages this run read.

        One operand per named entity: the ranker scores every quantity-index entry on every page
        for ``(entity, field phrase)`` and the top entry wins, so both operands can never come
        from one entity's page (the wrong-entity vindication mint02 found in every True-but-wrong
        cell). Nothing here reads the deliverable, so the result is an INDEPENDENT prediction of
        the answer, minted under :data:`HOST_DERIVE_TAG`.

        :param mandate: the task mandate.
        :param ranker: an :class:`~agent.app.operand_attribution.OperandRanker`; ``None`` loads
            the shipped artifact, and a missing artifact yields ``reason="no_ranker"``.
        :returns: ``{"demanded_operation", "entities", "operands", "value", "unit",
            "derived_node_id", "reason"}``.
        :raises: nothing.
        """
        result: Dict[str, Any] = {
            "demanded_operation": None, "entities": [], "operands": [], "value": None,
            "unit": "", "derived_node_id": None, "reason": "",
        }
        try:
            shape = mandate_demanded_operation(mandate)
            operation = shape.get("operation")
            result["demanded_operation"] = operation
            if operation is None:
                result["reason"] = "no_unambiguous_shape"
                return result
            if not self._entries:
                result["reason"] = "no_index_entries"
                return result
            if ranker is None:
                try:
                    from agent.app.operand_attribution import OperandRanker
                    ranker = OperandRanker.load()
                except FileNotFoundError:
                    result["reason"] = "no_ranker"
                    return result

            entities = extract_named_candidates(str(mandate or ""))
            if len(entities) < 2:
                entities = named_entities(str(mandate or ""))[:2]
            result["entities"] = list(entities)
            if len(entities) < 2:
                result["reason"] = "fewer_than_two_entities"
                return result
            field = _field_phrase(str(mandate or ""))

            pages_by_id: Dict[str, Dict[str, Any]] = {}
            entries_by_page: Dict[str, List[Any]] = {}
            for page_id, entry in self._entries:
                entries_by_page.setdefault(page_id, []).append(entry)
                if page_id not in pages_by_id:
                    pages_by_id[page_id] = self._graph.page(page_id) or {}

            input_ids: List[str] = []
            for entity in entities[:2]:
                best: Optional[Tuple[float, str, Any]] = None
                for page_id, entries in entries_by_page.items():
                    page = pages_by_id[page_id]
                    page_text = str(page.get("text") or "")
                    title = _title_from_url(str(page.get("url") or ""))
                    for score, index in ranker.rank(entity, field, entries, title, page_text)[:1]:
                        if best is None or score > best[0]:
                            best = (score, page_id, entries[index])
                if best is None or best[0] < _HOST_DERIVE_MIN_SCORE:
                    result["reason"] = f"operand_not_found:{entity}"
                    return result
                score, page_id, entry = best
                node = self._mint_source_from_entry(page_id, entry, minted_by=HOST_DERIVE_TAG)
                if node is None:
                    result["reason"] = f"operand_not_found:{entity}"
                    return result
                input_ids.append(node.id)
                result["operands"].append({"entity": entity, "page_id": page_id, "value": entry.value,
                                           "unit": entry.unit, "score": round(score, 4),
                                           "node_id": node.id})
            try:
                derived = self._graph.add_arith(operation, input_ids, minted_by=HOST_DERIVE_TAG)
            except DerivationError as exc:
                self._graph.record_refusal(operation, input_ids, exc)
                result["reason"] = f"refused:{exc.code}"
                return result
            value = derived.value
            if operation == "difference" and shape.get("absolute"):
                try:
                    value = str(abs(float(str(value).replace(",", ""))))
                except ValueError:
                    pass
            result["value"] = value
            result["unit"] = derived.unit or ""
            result["derived_node_id"] = derived.id
            result["reason"] = "computed"
            return result
        except Exception as exc:  # noqa: BLE001 -- host-side audit must never break a run
            result["reason"] = f"error:{type(exc).__name__}"
            return result
```

Add these module-level helpers just above `class LedgerToolkit`:

```python
#: Below this ranker probability an operand is reported as not found rather than guessed.
_HOST_DERIVE_MIN_SCORE = 0.2

_FIELD_STOP = frozenset("what is the absolute difference between these two in of and a an "
                        "ratio sum total how many much which by to for".split())


def _field_phrase(mandate: str) -> str:
    """The content words of the mandate's first sentence, the closest thing to a field name a
    free-form question offers ("height", "length", "average speed")."""
    first = re.split(r"[?\n]", mandate, maxsplit=1)[0]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", first.lower()) if w not in _FIELD_STOP]
    return " ".join(words[:6])


def _title_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1].split("#")[0]
    return re.sub(r"_+", " ", slug).strip()
```

Change `_mint_source_from_entry` to accept a tag:

```python
    def _mint_source_from_entry(self, page_id: str, entry: Any, *,
                                minted_by: str = ANSWER_AUDIT_TAG) -> Optional[Any]:
        page = self._graph.page(page_id)
        page_text = str((page or {}).get("text") or "")
        quote = _quote_for_span(page_text, entry.start, entry.end)
        return self._graph.add_source(page_id, entry.value, quote=quote, unit=entry.unit or None,
                                      minted_by=minted_by)
```

(Existing callers pass no tag and keep `ANSWER_AUDIT_TAG`.)

- [ ] **Step 4: Run the ledger tests**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_tools_test.py -v`
Expected: all PASS. If `test_host_derive_computes_...` fails on the entity list, print `extract_named_candidates(TWO_ENTITY_MANDATE)` and adjust the mandate fixture's enumeration format to what `enumerated_items` parses (numbered lines with an em-dash description are its documented shape).

- [ ] **Step 5: Commit**

```bash
git add agent/app/ledger_tools.py agent/tests/ledger_tools_test.py
git commit -m "add host derive that computes the demanded operation from ranked per entity operands before the answer exists"
```

---

### Task 5: Wire `host_derive` into both hosts

**Files:**
- Modify: `agent/app/testing/execution_sequential.py:96-100` (add `_ledger_host_derive_enabled`), `:749-752` (kit construction condition), `:781-800` (call before `artifact()`)
- Modify: `agent/app/langgraph_solver.py:1545` (flag), `:1736-1741` (kit condition), `:1918-1933` (call)
- Test: `agent/tests/execution_sequential_ledger_test.py` (create) — check for an existing sequential ledger test file first with `ls agent/tests | grep -i "sequential.*ledger\|ledger.*sequential"` and append there if one exists.

**Interfaces:**
- Consumes: `LedgerToolkit.host_derive(mandate)` from Task 4.
- Produces: `output["host_derive"]` in the stored cell when `LEDGER_HOST_MODULES` contains `host_derive`; nodes it mints land in `output["evidence_graph"]`.

- [ ] **Step 1: Write the failing test**

```python
import os

from agent.app.testing import execution_sequential as es


def test_host_derive_token_is_read_from_the_module_list(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive,host_derive")
    assert es._ledger_host_derive_enabled() is True
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive,answer_audit")
    assert es._ledger_host_derive_enabled() is False


def test_host_derive_output_is_written_before_the_artifact(monkeypatch):
    """A node minted by host_derive must be inside the stored evidence graph."""
    calls = []

    class Kit:
        def host_derive(self, mandate):
            calls.append("host_derive")
            return {"reason": "no_index_entries", "derived_node_id": None}

        def artifact(self):
            calls.append("artifact")
            return {"nodes": [], "pages": [], "rejections": [], "derivation_refusals": [],
                    "counts": {}}

    output = es._finish_ledger_output("answer", "mandate", Kit(), answer_audit_enabled=False,
                                      shape_derive_enabled=False, host_derive_enabled=True)
    assert calls == ["host_derive", "artifact"]
    assert output["host_derive"]["reason"] == "no_index_entries"
    assert "evidence_graph" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/execution_sequential_ledger_test.py -v`
Expected: FAIL, `AttributeError: module has no attribute '_ledger_host_derive_enabled'`.

- [ ] **Step 3: Implement in `execution_sequential.py`**

After `_ledger_shape_derive_enabled` (around line 100):

```python
def _ledger_host_derive_enabled() -> bool:
    """True when ``host_derive`` is present in ``LEDGER_HOST_MODULES``."""
    modules = {m.strip().lower() for m in os.environ.get("LEDGER_HOST_MODULES", "").split(",")}
    return "host_derive" in modules
```

Extract the finish-time block (lines ~781-800) into a helper placed above `run_sequential_execution`, and call it from the original site:

```python
def _finish_ledger_output(deliverable: str, mandate: str, ledger_kit: Any, *,
                          answer_audit_enabled: bool, shape_derive_enabled: bool,
                          host_derive_enabled: bool) -> Dict[str, Any]:
    """The finish-time, host-side ledger audits, all mechanical, all called BEFORE
    ``artifact()`` so every node they mint lands in the stored evidence graph."""
    output: Dict[str, Any] = {}
    if answer_audit_enabled:
        output["answer_audit"] = ledger_kit.audit_answer(deliverable, mandate)
    if shape_derive_enabled:
        output["shape_derive"] = ledger_kit.shape_derive_check(deliverable, mandate)
    if host_derive_enabled:
        # Independent of the deliverable by construction (mandate + pages only), so a certify
        # signal may consume it; still tagged and excluded from clauses 1-5.
        output["host_derive"] = ledger_kit.host_derive(mandate)
    output["evidence_graph"] = ledger_kit.artifact()
    return output
```

At the original site replace the `if ledger_kit is not None:` block body with:

```python
    if ledger_kit is not None:
        output.update(_finish_ledger_output(
            output["final_deliverable"], mandate, ledger_kit,
            answer_audit_enabled=answer_audit_enabled,
            shape_derive_enabled=shape_derive_enabled,
            host_derive_enabled=host_derive_enabled))
```

and where `answer_audit_enabled`/`shape_derive_enabled` are read (~line 749) add `host_derive_enabled = _ledger_host_derive_enabled()` and include it in the kit-construction condition:

```python
    ledger_kit = (LedgerToolkit()
                  if (derive_enabled or answer_audit_enabled or shape_derive_enabled
                      or host_derive_enabled) else None)
```

- [ ] **Step 4: Implement in `langgraph_solver.py`**

At line ~1545 add `self._ledger_host_derive_enabled = "host_derive" in self._ledger_host_modules`. In the kit-construction condition (~1740) add `or self._ledger_host_derive_enabled`. In the exit block (~1925), after the `shape_derive` branch:

```python
            if self._ledger_host_derive_enabled:
                result_out["host_derive"] = ledger_kit.host_derive(mandate)
```

(before `result_out["evidence_graph"] = ledger_kit.artifact()`).

- [ ] **Step 5: Run the tests plus the existing host test files**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/execution_sequential_ledger_test.py agent/tests/ledger_tools_test.py agent/tests/langgraph_solver_test.py -v` (if `langgraph_solver_test.py` does not exist, run `ls agent/tests | grep -i langgraph` and use those files).
Expected: all PASS.

- [ ] **Step 6: Live smoke on one cell, $0**

```bash
scripts/run_campaign.sh hostderive_smoke <<'ENVEOF'
SEARCH_PROVIDER=corpus
LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/numeric22
LEDGER_MAX_LIVE_FALLBACKS=0
LEDGER_CONTEXT_FIT=1
LEDGER_ZERO_VISIT_GATE=1
LEDGER_HOST_MODULES=derive,answer_audit,shape_derive,host_derive
LLM_SEED=22222
LLM_PROVIDER=ollama
MODEL_API_URL=http://127.0.0.1:11435/v1
OPENAI_API_KEY=dummy
IDEA_TEST_VALIDATION_MODEL=qwen2.5:7b
IDEA_TEST_IDS=210
IDEA_TEST_EXECUTION_VARIANTS=sequential_react,langgraph_react
IDEA_TEST_RUNS=1
IDEA_TEST_CONCURRENCY=1
IDEA_TEST_JSON_TELEMETRY=1
IDEA_TEST_KEEP_TRACES=1
IDEA_TEST_REPORT_VERBOSITY=3
IDEA_TEST_MODELS=qwen2.5:7b
IDEA_TEST_RUN_ID=hostderive_smoke
ENVEOF
```
Then verify with:
```bash
python3 - <<'EOF'
import json,glob
for f in glob.glob('agent/idea_test_results/hostderive_smoke_*_r1.json'):
    o=json.load(open(f))['execution']['output']; print(f.split('/')[-1][:60], o.get('host_derive'))
EOF
```
Expected: two cells, each with a `host_derive` dict; at least one `reason == "computed"` with value ≈ 38.7. Move the smoke cells to `agent/idea_test_results/smoke_archive/` afterwards so they never join an analysis corpus.

- [ ] **Step 7: Commit**

```bash
git add agent/app/testing/execution_sequential.py agent/app/langgraph_solver.py agent/tests/execution_sequential_ledger_test.py
git commit -m "bind host derive to both ledger hosts behind the host_derive module token"
```

---

### Task 6: The `host_agrees` certify signal in risk coverage

**Files:**
- Modify: `scripts/ledger_risk_coverage.py` (`classify_cell` ~518-612; `build_report` ~630; `print_report` ~923)
- Test: `agent/tests/ledger_risk_coverage_test.py` (append)

**Interfaces:**
- Consumes: `output["host_derive"]` from Task 5; existing `extract_numbers`, `value_backed`, `REL_TOL`.
- Produces: pure function `host_agrees(host_derive: Any, final_numbers: Sequence[float], rel_tol=REL_TOL) -> Dict[str, Any]` returning `{"available": bool, "reason": str, "agrees": Optional[bool]}`; `classify_cell` row gains `host_derive` (that dict) and `host_certified = host_agrees.agrees is True`; report gains a `host_derive` section with coverage/risk pooled, per model, dev/holdout, and the standard inversion lists. Clauses 1–5 untouched; `HOST_DERIVE_TAG` nodes excluded from `derived_nodes` and `backing_values` like the other two tags.

- [ ] **Step 1: Write the failing tests**

```python
from scripts.ledger_risk_coverage import host_agrees  # adjust to the import style the file already uses


def test_host_agrees_is_unavailable_when_the_host_did_not_compute():
    out = host_agrees({"reason": "no_index_entries", "value": None}, [38.7])
    assert out == {"available": False, "reason": "no_index_entries", "agrees": None}


def test_host_agrees_true_when_an_answer_number_matches_within_tolerance():
    out = host_agrees({"reason": "computed", "value": "38.7"}, [38.75, 1972.0])
    assert out["available"] and out["agrees"] is True


def test_host_agrees_false_when_no_answer_number_matches():
    out = host_agrees({"reason": "computed", "value": "38.7"}, [800.7])
    assert out["available"] and out["agrees"] is False


def test_host_agrees_false_on_a_number_free_answer():
    """Unlike clause 5, an answer with no number cannot agree with a computed value."""
    out = host_agrees({"reason": "computed", "value": "38.7"}, [])
    assert out["agrees"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_risk_coverage_test.py -k host_agrees -v`
Expected: FAIL with `ImportError: cannot import name 'host_agrees'`.

- [ ] **Step 3: Implement**

In `scripts/ledger_risk_coverage.py`, after `certify_clauses_minus`:

```python
HOST_DERIVE_TAG = "host_derive"


def host_agrees(host_derive: Any, final_numbers: Sequence[float],
                rel_tol: float = REL_TOL) -> Dict[str, Any]:
    """Whether the deliverable states the value the host computed independently.

    ``available`` is False whenever the host did not compute (no shape, no pages, no ranker,
    refused). A number-free deliverable does NOT agree: the host has a value and the answer
    states none -- the opposite of clause 5's vacuous truth, on purpose.
    """
    if not isinstance(host_derive, dict) or host_derive.get("reason") != "computed":
        reason = (host_derive or {}).get("reason", "absent") if isinstance(host_derive, dict) else "absent"
        return {"available": False, "reason": str(reason), "agrees": None}
    try:
        target = float(str(host_derive.get("value")).replace(",", ""))
    except (TypeError, ValueError):
        return {"available": False, "reason": "unparseable_value", "agrees": None}
    agrees = bool(final_numbers) and any(value_backed(n, [target], rel_tol) for n in final_numbers)
    return {"available": True, "reason": "computed", "agrees": agrees}
```

In `classify_cell`: extend the exclusion at the `minted_by` check to `("answer_audit", "shape_derive", HOST_DERIVE_TAG)`; after `final_numbers` is computed add:

```python
    host = host_agrees(output.get("host_derive"), final_numbers)
```
and put `"host_derive": host, "host_certified": host["agrees"] is True` in the returned row.

In `build_report`, add a `host_derive` section computed the same way the `shape_derive` section is (pooled n_available, coverage = host_certified / n, risk = wrong-rate among host_certified, per model, dev/holdout split via `split_dev_holdout`, plus `certified_but_wrong` and `rejected_but_right` cell lists). Copy the shape_derive section's structure exactly; only the predicate changes. In `print_report`, print it after the shape_derive block.

- [ ] **Step 4: Run tests and the tool on mint02**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_risk_coverage_test.py -v`
Expected: PASS.
Run: `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/ledger_risk_coverage.py --prefix mint02_q7b`
Expected: the new section prints with `n_available=0` (mint02 predates host_derive), every other number identical to `agent/idea_test_results/mint02_report_v3.json`. Diff the JSON output against that file to confirm nothing else moved.

- [ ] **Step 5: Commit**

```bash
git add scripts/ledger_risk_coverage.py agent/tests/ledger_risk_coverage_test.py
git commit -m "add the host agrees certify signal to risk coverage without touching the frozen five clauses"
```

---

### Task 7: Preregister mint03 and write the launch recipe

**Files:**
- Create: `docs/handoffs/MINT03_PREREG_2026-09-XX.md` (date at write time)
- Create via `scripts/prereg.py write`: `agent/idea_test_results/prereg/mint03_{q05,q15,l3b,q7b}.json`

**Interfaces:**
- Consumes: `scripts/prereg.py write --spec <json>` with required fields `run_id, hypothesis, tasks, arms, reps, primary_endpoint, abort_conditions` (`KNOWN_ABORT_CONDITIONS = min_completion_rate, max_infra_failed_rate, max_live_fallbacks`).
- Produces: four prereg manifests, a handoff doc, and the four `run_campaign.sh` env blocks. **Launching is a user decision** (GPU hours, ~6.7 h projected for 192 cells at ladder03's rate); this task stops at "ready to launch".

- [ ] **Step 1: Write the prereg spec JSON (one per model)**

`/tmp/mint03_q7b.json`:
```json
{
  "run_id": "mint03_q7b",
  "hypothesis": "host_derive (mandate + pages only, operand-attribution ranked, one operand per entity) yields a certify signal whose coverage no longer depends on the model calling derive: host_certified coverage >= 25% at risk <= 5% on the dev split, within model, vs the mint02 certify chain's 10.4% at 6.7% pooled.",
  "tasks": ["210","211","212","213","214","215","216","217","218","219","220","221"],
  "arms": ["sequential_react", "langgraph_react"],
  "reps": 2,
  "primary_endpoint": "host_derive section of scripts/ledger_risk_coverage.py --prefix mint03_q7b: pooled dev-split coverage and risk of host_certified. PROMOTE if coverage >= 0.25 AND risk <= 0.05 within model on dev; else honest negative. Secondary (exploratory, not decision-bearing): host_derive availability rate by reason; agreement between host value and OP_A/OP_B-derived truth from the task module (host_value_correct), reported per model. No threshold re-sweeping; holdout 213/217/221 reported once, separately.",
  "abort_conditions": {"min_completion_rate": 1.0, "max_infra_failed_rate": 0.2, "max_live_fallbacks": 0}
}
```
Repeat for `mint03_q05` (qwen2.5:0.5b), `mint03_q15` (qwen2.5:1.5b), `mint03_l3b` (llama3.2:3b), changing only `run_id`.

- [ ] **Step 2: Write the manifests**

```bash
for m in q7b q05 q15 l3b; do PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py write --spec /tmp/mint03_$m.json; done
```
Expected: four files under `agent/idea_test_results/prereg/`. Commit them (the prereg dir is tracked; confirm with `git ls-files agent/idea_test_results/prereg | head`).

- [ ] **Step 3: Write the handoff doc with the launch blocks**

`docs/handoffs/MINT03_PREREG_2026-09-XX.md` must contain: the hypothesis verbatim, the code SHA frozen, the operand-ranker artifact's `fit_report` numbers, the four env blocks (copy `agent/idea_test_results/_campaigns/mint02_q7b.env`, changing `LEDGER_HOST_MODULES=derive,answer_audit,shape_derive,host_derive`, the model, and `IDEA_TEST_RUN_ID`), the launch order (one at a time, lockfile-serialized), and the analysis command. State explicitly: "Launch requires a go from the user; projected wall time ~7 h on the local GPU; $0."

- [ ] **Step 4: Commit**

```bash
git add agent/idea_test_results/prereg/mint03_*.json docs/handoffs/MINT03_PREREG_2026-09-XX.md
git commit -m "preregister the mint03 host derive campaign and record the launch recipe"
```

---

### Task 8: Learned trust channel

**Files:**
- Create: `scripts/ledger_trust_channel.py`
- Test: `agent/tests/ledger_trust_channel_test.py`

**Interfaces:**
- Consumes: `classify_cell` rows (`scripts/ledger_risk_coverage.py:518`) via `discover_cell_files(results_dir, prefix)` + `load_cell`; `sklearn.ensemble.HistGradientBoostingClassifier`; `sklearn.model_selection.GroupKFold`.
- Produces: `cell_features(row) -> Dict[str, float]` (mechanical only: `n_final_numbers, has_evidence_graph, n_derived_nodes, clauses_passed, quote_null_count, quote_checked_count, aa_backed, aa_derived, aa_unbacked, aa_trivial, aa_max_ambiguity, sd_available, sd_n_entries, sd_n_candidates, hd_available, hd_agrees, visits, searches, distinct_hosts, prompt_tokens`), `fit_and_curve(rows) -> Dict` with out-of-fold probabilities, per-model risk-coverage curves at thresholds, and the operating point nearest the certify chain's risk; CLI prints a table comparing chain coverage vs channel coverage at equal risk, per model.

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util, pathlib
_SPEC = importlib.util.spec_from_file_location(
    "ledger_trust_channel", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ledger_trust_channel.py")
ltc = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(ltc)


def _row(test_id, model, wrong, n_derived):
    return {"test_id": test_id, "model": model, "wrong": wrong, "n_final_numbers": 1,
            "has_evidence_graph": True, "n_derived_nodes": n_derived, "clauses_passed": 3,
            "quote_null_count": 0, "quote_checked_count": 2,
            "answer_audit": {"backed": 1, "derived": n_derived, "unbacked": 0, "trivial": 0, "max_ambiguity": 1},
            "shape_derive": {"available": False, "n_entries": 0, "n_candidates": 0},
            "host_derive": {"available": False, "agrees": None},
            "observability": {"visit": {"count": 2}, "search": {"count": 1}, "llm": {"prompt": {"tokens": 1000}}},
            "hosts": 1}


def test_features_never_include_model_identity():
    feats = ltc.cell_features(_row("210", "qwen2.5:7b", False, 1))
    assert "model" not in feats and not any("qwen" in k for k in feats)
    assert set(feats) == set(ltc.FEATURE_NAMES)


def test_folds_are_grouped_by_task_id():
    rows = [_row(t, "m", i % 2 == 0, i % 3) for i, t in enumerate(["210", "211", "212", "214", "215"] * 4)]
    for train_idx, test_idx in ltc.task_folds(rows, n_splits=5):
        train_tasks = {rows[i]["test_id"] for i in train_idx}
        test_tasks = {rows[i]["test_id"] for i in test_idx}
        assert not (train_tasks & test_tasks)


def test_curve_is_monotone_in_threshold_on_coverage():
    rows = [_row(str(210 + i % 9), "m", i % 3 == 0, i % 2) for i in range(60)]
    probs = [(i % 10) / 10 for i in range(60)]
    curve = ltc.risk_coverage_curve(rows, probs)
    coverages = [pt["coverage"] for pt in curve]
    assert coverages == sorted(coverages, reverse=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_trust_channel_test.py -v`
Expected: FAIL, file not found.

- [ ] **Step 3: Implement**

Create `scripts/ledger_trust_channel.py`:

```python
#!/usr/bin/env python3
"""A learned trust channel over MECHANICAL evidence features, giving the full risk-coverage curve.

The certify chain is one operating point. This fits a gradient-boosted classifier on the
per-cell rows `scripts/ledger_risk_coverage.py:classify_cell` already produces, with folds
grouped by task id (reps and models of one task never straddle a fold) and NO model-identity
feature, then reports per-model out-of-fold curves and the coverage at the chain's own risk.

Usage::
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/ledger_trust_channel.py \
        --prefix mint02 [--prefix mint01 ...] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "services")); sys.path.insert(0, str(ROOT / "agent"))

FEATURE_NAMES: List[str] = [
    "n_final_numbers", "has_evidence_graph", "n_derived_nodes", "clauses_passed",
    "quote_null_count", "quote_checked_count", "aa_backed", "aa_derived", "aa_unbacked",
    "aa_trivial", "aa_max_ambiguity", "sd_available", "sd_n_entries", "sd_n_candidates",
    "hd_available", "hd_agrees", "visits", "searches", "distinct_hosts", "prompt_tokens",
]
THRESHOLDS = [i / 100 for i in range(0, 101, 2)]


def cell_features(row: Dict[str, Any]) -> Dict[str, float]:
    aa = row.get("answer_audit") or {}
    sd = row.get("shape_derive") or {}
    hd = row.get("host_derive") or {}
    obs = row.get("observability") or {}
    return {
        "n_final_numbers": float(row.get("n_final_numbers") or 0),
        "has_evidence_graph": float(bool(row.get("has_evidence_graph"))),
        "n_derived_nodes": float(row.get("n_derived_nodes") or 0),
        "clauses_passed": float(row.get("clauses_passed") or 0),
        "quote_null_count": float(row.get("quote_null_count") or 0),
        "quote_checked_count": float(row.get("quote_checked_count") or 0),
        "aa_backed": float(aa.get("backed") or 0), "aa_derived": float(aa.get("derived") or 0),
        "aa_unbacked": float(aa.get("unbacked") or 0), "aa_trivial": float(aa.get("trivial") or 0),
        "aa_max_ambiguity": float(aa.get("max_ambiguity") or 0),
        "sd_available": float(bool(sd.get("available"))),
        "sd_n_entries": float(sd.get("n_entries") or 0),
        "sd_n_candidates": float(sd.get("n_candidates") or 0),
        "hd_available": float(bool(hd.get("available"))),
        "hd_agrees": float(hd.get("agrees") is True),
        "visits": float(((obs.get("visit") or {}).get("count")) or 0),
        "searches": float(((obs.get("search") or {}).get("count")) or 0),
        "distinct_hosts": float(row.get("hosts") or 0),
        "prompt_tokens": float((((obs.get("llm") or {}).get("prompt") or {}).get("tokens")) or 0),
    }


def task_folds(rows: Sequence[Dict[str, Any]], n_splits: int = 5) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    groups = [str(r["test_id"]) for r in rows]
    n_splits = min(n_splits, len(set(groups)))
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(rows)), groups=groups)


def risk_coverage_curve(rows: Sequence[Dict[str, Any]], probs: Sequence[float]) -> List[Dict[str, Any]]:
    curve = []
    n = len(rows)
    for t in THRESHOLDS:
        accepted = [r for r, p in zip(rows, probs) if p >= t]
        wrong = sum(1 for r in accepted if r["wrong"])
        curve.append({"threshold": t, "coverage": (len(accepted) / n) if n else 0.0,
                      "risk": (wrong / len(accepted)) if accepted else None, "n": len(accepted)})
    return curve


def fit_and_curve(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = np.array([[cell_features(r)[k] for k in FEATURE_NAMES] for r in rows])
    y = np.array([0 if r["wrong"] else 1 for r in rows])
    oof = np.zeros(len(rows))
    for train_idx, test_idx in task_folds(rows):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
        clf.fit(x[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    by_model: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_model[str(r.get("model"))].append(i)
    return {
        "n": len(rows), "pooled": risk_coverage_curve(rows, oof.tolist()),
        "per_model": {m: risk_coverage_curve([rows[i] for i in idx], [oof[i] for i in idx])
                      for m, idx in by_model.items()},
        "oof": oof.tolist(),
    }


def coverage_at_risk(curve: Sequence[Dict[str, Any]], max_risk: float) -> Optional[Dict[str, Any]]:
    ok = [pt for pt in curve if pt["risk"] is not None and pt["risk"] <= max_risk]
    return max(ok, key=lambda pt: pt["coverage"]) if ok else None


def _load_rows(prefixes: Sequence[str]) -> List[Dict[str, Any]]:
    from scripts.ledger_risk_coverage import classify_cell, discover_cell_files, load_cell, _default_results_dir
    rows: List[Dict[str, Any]] = []
    for prefix in prefixes:
        for path in discover_cell_files(_default_results_dir(), prefix):
            raw = load_cell(path)
            if raw is None:
                continue
            row = classify_cell(path, raw)
            if row.get("infra_failed") or row.get("split") == "holdout":
                continue
            row["observability"] = (raw.get("execution") or {}).get("observability") or {}
            pages = ((raw.get("execution") or {}).get("output") or {}).get("pages") or []
            row["hosts"] = len({p.get("url", "").split("/")[2] for p in pages if "://" in p.get("url", "")})
            rows.append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--chain-risk", type=float, default=0.067)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = _load_rows(args.prefix)
    result = fit_and_curve(rows)
    summary = {"n": result["n"], "chain_risk": args.chain_risk,
               "pooled_at_chain_risk": coverage_at_risk(result["pooled"], args.chain_risk),
               "per_model_at_chain_risk": {m: coverage_at_risk(c, args.chain_risk)
                                           for m, c in result["per_model"].items()}}
    if args.json:
        print(json.dumps({"summary": summary, "curves": result}, indent=1))
    else:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, then the tool on mint01+mint02**

Run: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/ledger_trust_channel_test.py -v` → PASS.
Run: `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/ledger_trust_channel.py --prefix mint01 --prefix mint02`
Expected: prints pooled and per-model coverage at risk ≤ 0.067. Record the numbers in a short section appended to `docs/STATE_OF_EVIDENCE_2026-09-07.md` under a new heading "Trust channel (development evidence, not confirmatory)". The comparison that matters is per model vs the chain's per-model coverage from `mint02_report_v3.json`. This is development evidence; confirmation waits for mint03.

- [ ] **Step 5: Commit**

```bash
git add scripts/ledger_trust_channel.py agent/tests/ledger_trust_channel_test.py docs/STATE_OF_EVIDENCE_2026-09-07.md
git commit -m "add a task grouped learned trust channel over mechanical ledger features for the full risk coverage curve"
```

---

### Task 9: Fix the two `ledger_api.py` component defects

**Files:**
- Modify: `agent/app/ledger_api.py` (`run(...)` ~434, `SourceServingHttp` ~226)
- Test: `agent/tests/ledger_api_test.py` (append)

**Interfaces:**
- Consumes: `evidence_loop.run_evidence_loop` page-chars setting (`IDEA_TEST_EVIDENCE_LOOP_PAGE_CHARS`, default 6000, `execution_evidence_loop.py:1805`); `clean_operation` (grep `def clean_operation` under `agent/app/` for its location).
- Produces: `run(question, sources, ..., page_chars: Optional[int] = None)`: when `sources` are supplied, `page_chars` defaults to the longest supplied text so nothing is truncated; supplied text is registered verbatim (no `clean_operation` pass) so `recheck_claim` offsets refer to the bytes the caller handed in.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio

from agent.app import ledger_api


def test_a_supplied_source_longer_than_the_default_window_is_not_truncated(monkeypatch):
    long_text = ("filler line\n" * 700) + "The chimney is 419.7 metres tall.\n"   # > 6000 chars
    assert len(long_text) > 6000
    captured = {}

    async def fake_loop(io, mandate, *args, **kwargs):
        captured["page_chars"] = kwargs.get("page_chars")
        return {"ledger": None, "graph": None, "deliverable": "", "pages": []}

    monkeypatch.setattr(ledger_api.evidence_loop, "run_evidence_loop", fake_loop)
    asyncio.run(ledger_api.run("How tall is the chimney?",
                               sources=[{"url": "https://example.com/a", "text": long_text}]))
    assert captured["page_chars"] >= len(long_text)


def test_supplied_source_text_is_served_verbatim():
    http = ledger_api.SourceServingHttp(ledger_api.ConnectorConfig(), {"https://example.com/a": "  raw <b>text</b>  "})
    result = asyncio.run(http.request("GET", "https://example.com/a"))
    assert result.data == "  raw <b>text</b>  "
```
(Adjust `ConnectorConfig()` construction and `result.data` attribute to what `RequestResult` exposes; read `services/shared/request_result.py` first.)

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement** by threading a `page_chars` kwarg from `run()` into `run_evidence_loop` (defaulting to `max(len(s.text) for s in sources)` when sources are given) and by making `SourceServingHttp.request` return the stored text without the cleaning pass. **Step 4: run** `agent/tests/ledger_api_test.py` → PASS. **Step 5: commit** `fix supplied source truncation and verbatim serving in the ledger api facade`.

---

## Self-review against the spec

- L1 → Tasks 4, 5, 6, 7. L2 → Tasks 2, 3. L3 → Task 8. L5 → Tasks 1, 9 (second-hostname policy and `unified_verdict` on the mint hosts are NOT in this plan; they are listed as follow-ups in Task 7's handoff doc so they are not lost).
- Placeholders: Task 9 steps 2–5 are compressed but name the exact change; Task 6 step 3's report section says "copy the shape_derive section's structure", which is a real instruction because that section exists at `build_report`.
- Type consistency: `host_derive()` returns `reason` strings that `host_agrees()` checks (`"computed"`); `HOST_DERIVE_TAG` is `"host_derive"` in both `ledger_tools.py` and `ledger_risk_coverage.py`; `OperandRanker.rank` returns `(score, index)` tuples consumed in Task 4 as `for score, index in ranker.rank(...)[:1]`.
