"""Seeded generator of derivation tasks from the frozen ``numeric22`` corpus.

Why this exists (see ``docs/LEDGER_FINAL01_TUNING_RESULT.md``): the sealed-holdout suite has 22
paired tasks, and reportable-comparison confidence needs roughly 60. Hand-authoring 40 more is an
authoring-budget problem; this script turns it into a compute-budget problem by MECHANICALLY
composing a question from two quantities that are demonstrably present in real corpus pages,
under one arithmetic operation (difference / sum / ratio / quotient). Ground truth is known by
construction (it is computed from the same numbers the question names), so leakage cannot come
from an author's memory of a fact -- it can only come from the corpus itself, which is why every
emitted task is checked against the WHOLE corpus (see :func:`_is_leaked`), not just its own two
source pages.

Pipeline, in order:

1. :func:`load_operands` -- run ``agent.app.quantity_index.build_index`` (the frozen, reused
   extractor; this module does not re-derive a number grammar) over every document's page text,
   keep only INFOBOX quantities (a field LABEL is required for a question to read as sensible --
   see the "plausibility" note below), and parse each one with
   ``agent.app.testing.evidence_graph.parse_quantity`` to get a magnitude and a ``(currency,
   unit)`` dimension. Quantities with no unit and no currency (a bare scaled number, e.g. "121
   crore" with nothing after it) are dropped -- there is no canonical unit to pair them on.
2. :func:`group_by_dimension` -- bucket operands by their EXACT dimension (unit string, normalized
   only for case/dash, never converted -- ``"m"`` and ``"km"`` are different buckets forever; see
   ``docs/LEDGER_PLAN_2026-09-01.md`` section 7, "no unit conversion, ever"). Pairing only within
   one bucket is what makes a cross-unit pair structurally impossible to emit.
3. :func:`candidate_pairs` -- within each bucket, every CROSS-DOCUMENT pair is a candidate; same-
   document pairs are dropped before anything else runs, which is what makes the Wikipedia
   dual-unit idiom (``"1,642 m (5,387 ft)"``, one fact restated in two units on the SAME page/row)
   structurally unreachable as a pair here. Each bucket's candidate list is then shuffled with a
   bucket-specific seeded RNG (derived from the master seed via a SHA-256 digest, so ordering
   never depends on Python's dict/set hashing) so which pairs :func:`generate` reaches first is
   seed-controlled.
4. Filters applied to each candidate, in order, each with its own discard counter (see
   :func:`generate`'s returned ``stats``):
     * ``label_family`` -- PLAUSIBILITY. See :func:`_plausible_pair`'s docstring for the judgment
       call and why an earlier, looser version of it was rejected: the two labels must be
       byte-identical after normalization (the SAME fact type on two different entities -- "Max.
       depth" only pairs with another entity's own "Max. depth"), matching the shape every
       authored task in ``test_210``..``test_221`` already uses.
     * ``degenerate`` -- equal magnitudes are rejected outright (a-a, a/a=1 for ANY chosen
       operation), and a ratio/quotient that would round to 1.0 within 5% is rejected too (values
       technically unequal but not meaningfully different).
     * ``retrieval`` -- :func:`_is_retrievable` runs a per-operand query (``"<title> <label>"``)
       through the SAME ``BM25Index`` that backs ``ConnectorSearchCorpus`` (see
       ``agent/app/connector_search_corpus.py``; token-overlap scoring, not exact-key lookup) and
       requires the operand's own source document in the top-``k``. A task whose evidence cannot
       be found this way is measuring retrieval failure, not derivation.
     * ``leaked`` -- :func:`_is_leaked` scans EVERY document's raw text (not just the two source
       pages) for the derived value as a written token; a verbatim hit anywhere discards the task.
5. :func:`_build_task` renders the accepted pair into one task record (question text, both
   operands with full provenance, the operation, the exact derived value, and the search terms
   used for the retrievability check, so a reader can rerun that check by hand).

Determinism: the whole pipeline is seeded exclusively through :func:`_seed_for` (a SHA-256 digest
of the master seed plus a bucket key, never Python's built-in hash), iterates buckets in a key-
sorted order, and serializes tasks as line-delimited JSON with ``sort_keys=True`` and floats
rounded to 6 places. Same seed -> byte-identical output file; this is pinned by
``agent/tests/generate_derivation_tasks_test.py``.

Regenerate with::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/generate_derivation_tasks.py \\
        --seed 20260901 --n 40 \\
        --corpus-dir agent/idea_test_results/corpus/numeric22 \\
        --out-dir agent/idea_test_results/generated

What this generator CANNOT produce (see the handoff report for the full structural comparison
against the authored 22): a chain that requires reading a THIRD fact to know which two operands to
compare; an adversarial unit-mismatch trap (every pair is same-unit by construction); a plausible-
but-unsupported decoy value; a task where the operand is buried in prose rather than sitting in an
infobox row (only infobox quantities are used, for label plausibility -- see step 1 above); or any
control over how many distractor numbers share the source page (whatever the real page happens to
contain).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "services"), str(_REPO_ROOT / "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.app.quantity_index import build_index  # noqa: E402
from agent.app.testing.evidence_graph import parse_quantity, normalize_for_match  # noqa: E402
from agent.app.connector_search_corpus import BM25Index, CorpusDocument, title_from_url  # noqa: E402

DEFAULT_CORPUS_DIR = str(_REPO_ROOT / "agent/idea_test_results/corpus/numeric22")
DEFAULT_OUT_DIR = str(_REPO_ROOT / "agent/idea_test_results/generated")
DEFAULT_SEED = 20260901
DEFAULT_N = 40
DEFAULT_TOPK = 5

OPERATIONS: Tuple[str, ...] = ("difference", "sum", "ratio", "quotient")

#: Ratio/quotient values within this fraction of 1.0 are treated as "not meaningfully different".
_NEAR_UNITY_BAND = 0.05
#: Acceptance tolerance recorded on every emitted task (matches the authored tier-5 tasks' band).
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class Operand:
    """One infobox quantity, addressable for pairing."""
    doc_index: int
    url: str
    title: str
    label: str
    value_text: str
    unit_text: str
    magnitude: float
    dimension: Tuple[str, str]
    source: str


def _seed_for(master_seed: int, key: str) -> int:
    """A deterministic sub-seed derived from ``master_seed`` and ``key`` via SHA-256.

    Never uses Python's built-in ``hash()`` -- that is salted per-process for strings, so a
    pipeline keyed on it would not reproduce across runs (or even within one run across
    invocations with ``PYTHONHASHSEED`` unset).
    """
    digest = hashlib.sha256(f"{master_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _plausible_pair(a: Operand, b: Operand) -> bool:
    """PLAUSIBILITY gate (see module docstring, step 4): the two labels must be the SAME FACT
    TYPE, byte-identical after normalization (``"Max. depth"`` only pairs with another entity's
    own ``"Max. depth"``).

    This is the judgment call the task brief asks to be explicit about. An earlier version of
    this filter grouped labels into coarse semantic families (any "height"/"elevation"/"depth"/
    "length" label counted as one "size" family) so that, for instance, a lake's "Max. depth"
    could pair with a different lake's "Average depth". Inspecting the output that heuristic
    actually produced surfaced a worse failure than the one it was built to avoid: it also let a
    bridge's "Height" pair with an unrelated mountain range's "Elevation" -- same unit, same
    coarse family, but not a fact a human would ever ask to compare. Every authored task in
    ``agent/app/idea_tests/test_210..221`` compares the identical field on two comparable
    entities (chimney height vs chimney height, dam capacity vs dam capacity, lake depth vs lake
    depth) -- never two different fields, however related they sound. Requiring an exact label
    match reproduces that shape mechanically instead of trying to approximate it with synonym
    families, at the cost of rejecting some pairs a human might judge sensible (see the report's
    structural comparison for the resulting yield trade-off).
    """
    return normalize_for_match(a.label) == normalize_for_match(b.label)


def load_corpus_documents(corpus_dir: str) -> List[Dict[str, str]]:
    """Read ``documents.jsonl`` from ``corpus_dir`` as plain dicts (url/title/description/text).

    Malformed lines are skipped, never fatal -- a partially-corrupt corpus degrades to fewer
    documents rather than aborting generation (mirrors ``connector_search_corpus.load_documents``).
    """
    path = Path(corpus_dir) / "documents.jsonl"
    out: List[Dict[str, str]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def load_operands(documents: List[Dict[str, str]]) -> List[Operand]:
    """Every INFOBOX quantity across ``documents``, in stable (doc order, extraction order).

    Only ``source == "infobox"`` entries are kept: an infobox row carries a field LABEL, and a
    label is what lets the question read as sensible (a labelless prose number like "419.7
    metres" has no ready-made noun phrase to build a question around). Entries whose parsed
    quantity has neither a unit nor a currency (a bare scaled number) are dropped too -- there is
    no canonical unit to pair them on.
    """
    operands: List[Operand] = []
    for doc_index, doc in enumerate(documents):
        text = str(doc.get("text") or "")
        if not text:
            continue
        url = str(doc.get("url") or "")
        raw_title = str(doc.get("title") or "").strip()
        if raw_title.endswith(" - Wikipedia"):
            raw_title = raw_title[: -len(" - Wikipedia")].strip()
        # 191/313 corpus documents carry no title field at all -- fall back to a readable name
        # derived from the URL's last path segment (the same helper the corpus search backend
        # itself uses), never the raw URL, which would otherwise leak into every question and
        # search-term string built from this operand.
        title = raw_title or title_from_url(url) or url
        for entry in build_index(text):
            if entry.source != "infobox" or not entry.label:
                continue
            parsed = parse_quantity(f"{entry.value} {entry.unit}")
            if not parsed.ok:
                continue
            unit, currency = parsed.dimension[1], parsed.dimension[0]
            if not unit and not currency:
                continue
            operands.append(Operand(
                doc_index=doc_index, url=url, title=title, label=entry.label,
                value_text=entry.value, unit_text=entry.unit, magnitude=parsed.magnitude,
                dimension=parsed.dimension, source=entry.source,
            ))
    return operands


def group_by_dimension(operands: List[Operand]) -> Dict[Tuple[str, str], List[int]]:
    """``operands`` indices bucketed by exact dimension, insertion order = ``operands`` order."""
    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, op in enumerate(operands):
        groups.setdefault(op.dimension, []).append(idx)
    return groups


def candidate_pairs(operands: List[Operand], seed: int) -> List[Tuple[int, int]]:
    """Deterministic candidate index pairs, restricted to different source documents.

    Within each dimension bucket, EVERY cross-document pair is a candidate (``itertools.
    combinations`` over the bucket's indices, same-document pairs dropped before shuffling -- the
    filter that also makes a same-page dual-unit restatement structurally unreachable here, since
    such a restatement's two halves never appear as a pair). The full candidate list for one
    bucket is then shuffled with a SHA-256-derived sub-seed (see :func:`_seed_for`) unique to that
    bucket, so which pairs :func:`generate` reaches first is seed-controlled without discarding
    any pair up front. Buckets are visited in sorted-key order so the result never depends on
    dict/set iteration order.
    """
    groups = group_by_dimension(operands)
    pairs: List[Tuple[int, int]] = []
    for dimension in sorted(groups.keys()):
        indices = list(groups[dimension])
        if len(indices) < 2:
            continue
        bucket_pairs = [(i, j) for i, j in itertools.combinations(indices, 2)
                         if operands[i].doc_index != operands[j].doc_index]
        rng = random.Random(_seed_for(seed, f"dim:{dimension[0]}|{dimension[1]}"))
        rng.shuffle(bucket_pairs)
        pairs.extend(bucket_pairs)
    return pairs


def compute_derived(a_mag: float, b_mag: float, operation: str) -> Optional[float]:
    if operation == "difference":
        return abs(a_mag - b_mag)
    if operation == "sum":
        return a_mag + b_mag
    if operation == "ratio":
        lo, hi = sorted((abs(a_mag), abs(b_mag)))
        if lo == 0:
            return None
        return hi / lo
    if operation == "quotient":
        if b_mag == 0:
            return None
        return a_mag / b_mag
    return None


def _is_degenerate(a: Operand, b: Operand, operation: str, derived: Optional[float]) -> Optional[str]:
    if a.magnitude == b.magnitude:
        return "equal_values"
    if derived is None:
        return "undefined"
    if operation in ("ratio", "quotient") and abs(derived - 1.0) < _NEAR_UNITY_BAND:
        return "ratio_near_unity"
    return None


def _renderings(value: float) -> List[str]:
    """Plausible verbatim renderings of ``value``, used to scan raw corpus text for a leak."""
    out = {f"{value:.0f}", f"{value:.1f}", f"{value:.2f}"}
    if value == int(value):
        out.add(str(int(value)))
    return sorted(out)


def _is_leaked(derived: float, documents: List[Dict[str, str]]) -> bool:
    """True when ``derived`` appears verbatim (as a written number token) on ANY corpus page."""
    renderings = _renderings(derived)
    for doc in documents:
        text = str(doc.get("text") or "")
        if not text:
            continue
        for rendering in renderings:
            if re.search(rf"(?<!\d){re.escape(rendering)}(?!\d)", text):
                return True
    return False


def _query_for(operand: Operand) -> str:
    return f"{operand.title} {operand.label}".strip()


def _is_retrievable(operand: Operand, index: BM25Index, k: int) -> bool:
    hits = index.search(_query_for(operand), k)
    return any(hit.url == operand.url for hit in hits)


def _build_task(task_id: str, a: Operand, b: Operand, operation: str, derived: float,
                 seed: int) -> Dict[str, Any]:
    unit = a.unit_text
    op_phrase = {
        "difference": "the ABSOLUTE DIFFERENCE between",
        "sum": "the SUM of",
        "ratio": "the RATIO (larger divided by smaller) between",
        "quotient": "the QUOTIENT (the first divided by the second) of",
    }[operation]
    question = (
        f"Read {a.label!r} for {a.title} (a value in {a.unit_text}) and "
        f"{b.label!r} for {b.title} (a value in {b.unit_text}). "
        f"Then compute {op_phrase} the two values."
    )
    return {
        "task_id": task_id,
        "seed": seed,
        "operation": operation,
        "unit": unit,
        "expected_value": round(float(derived), 6),
        "tolerance": DEFAULT_TOLERANCE,
        "question": question,
        "operand_a": {
            "url": a.url, "title": a.title, "label": a.label,
            "value_text": a.value_text, "unit_text": a.unit_text,
            "magnitude": round(a.magnitude, 6), "search_terms": _query_for(a),
        },
        "operand_b": {
            "url": b.url, "title": b.title, "label": b.label,
            "value_text": b.value_text, "unit_text": b.unit_text,
            "magnitude": round(b.magnitude, 6), "search_terms": _query_for(b),
        },
    }


def generate(seed: int, corpus_dir: str, n_target: int, k: int = DEFAULT_TOPK) -> Dict[str, Any]:
    """Run the full pipeline and return ``{"tasks": [...], "stats": {...}}``.

    Stops once ``n_target`` tasks are accepted, or once every candidate pair has been tried --
    whichever comes first. Deterministic: same ``(seed, corpus_dir, n_target, k)`` always yields
    the same tasks in the same order.
    """
    documents = load_corpus_documents(corpus_dir)
    operands = load_operands(documents)
    index = BM25Index([CorpusDocument(url=d.get("url", ""), title=d.get("title", ""),
                                       description=d.get("description", ""),
                                       text=d.get("text", "")) for d in documents])
    pairs = candidate_pairs(operands, seed)

    stats = {
        "documents": len(documents),
        "operands_extracted": len(operands),
        "candidate_pairs": len(pairs),
        "discard_not_plausible": 0,
        "discard_degenerate": 0,
        "discard_not_retrievable": 0,
        "discard_leaked": 0,
        "accepted": 0,
    }
    tasks: List[Dict[str, Any]] = []
    op_cycle_rng = random.Random(_seed_for(seed, "operation_cycle"))
    for pair_index, (i, j) in enumerate(pairs):
        if len(tasks) >= n_target:
            break
        a, b = operands[i], operands[j]
        if not _plausible_pair(a, b):
            stats["discard_not_plausible"] += 1
            continue
        operation = op_cycle_rng.choice(OPERATIONS)
        derived = compute_derived(a.magnitude, b.magnitude, operation)
        reason = _is_degenerate(a, b, operation, derived)
        if reason is not None:
            stats["discard_degenerate"] += 1
            continue
        if not (_is_retrievable(a, index, k) and _is_retrievable(b, index, k)):
            stats["discard_not_retrievable"] += 1
            continue
        if _is_leaked(derived, documents):
            stats["discard_leaked"] += 1
            continue
        task_id = f"gen{seed}_{len(tasks):04d}"
        tasks.append(_build_task(task_id, a, b, operation, derived, seed))
        stats["accepted"] += 1
    return {"tasks": tasks, "stats": stats}


def write_outputs(result: Dict[str, Any], out_dir: str, seed: int) -> Tuple[Path, Path]:
    """Write the tasks as line-delimited JSON and a companion report; return both paths.

    Serialization is fixed (``sort_keys=True``, ``ensure_ascii=True``, no extra whitespace) so
    that regenerating with the same inputs produces a byte-identical file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks_path = out / f"derivation_tasks_seed{seed}.jsonl"
    report_path = out / f"derivation_tasks_seed{seed}_report.json"
    with tasks_path.open("w", encoding="utf-8") as handle:
        for task in result["tasks"]:
            handle.write(json.dumps(task, sort_keys=True, ensure_ascii=True) + "\n")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(result["stats"], handle, sort_keys=True, ensure_ascii=True, indent=2)
        handle.write("\n")
    return tasks_path, report_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args(argv)

    result = generate(args.seed, args.corpus_dir, args.n, args.k)
    tasks_path, report_path = write_outputs(result, args.out_dir, args.seed)
    print(f"wrote {len(result['tasks'])} tasks -> {tasks_path}")
    print(f"stats -> {report_path}: {json.dumps(result['stats'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
