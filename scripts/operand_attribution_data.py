#!/usr/bin/env python3
"""Build the leak-free EVAL half of the operand-attribution dataset, and score rankers on it.

This is the diagnostic half of Phase 2b: it answers "on the real pages the agent actually fetched,
does :mod:`agent.app.operand_attribution` pick the operand the task asks for?" — and nothing else.
There is deliberately **no training set here**. The superseded plan (2026-09-07) fitted a logistic
ranker on weak supervision drawn from the same stored pages; the adversarial review found that set
leaked eval pages into training, had zero prose positives by construction, and rested on a page
count that dedup'd away. So this script builds the eval side only, and keeps
:func:`assert_no_overlap` on the shelf for the day training is revisited.

Three choices worth stating, because each one was a defect in the superseded design:

* **Positives are labelled by OFFSET IDENTITY, never by relative tolerance.** The operand value
  from the task module is located as WRITTEN in the page text (with its thousands separators, in
  any of the spellings :func:`value_variants` enumerates), and every index entry whose span
  intersects that occurrence is a positive. A tolerance-based label would mark any nearby number
  "correct" and quietly grade the ranker against itself.
* **A slot's pages are found by the module's own ``slug_rx``**, not by string-equal URLs: the
  stored URLs are percent-encoded and the tasks carry non-ASCII titles (``Puskás_Aréna``).
* **A slot with no located positive is reported, not dropped silently.** Per task the report says
  how many pages were found and how many slots got a positive; a task where nothing could be
  labelled is a finding about the corpus, and hiding it would inflate every p@1 below.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/operand_attribution_data.py \\
        --out-dir agent/idea_test_results/operand_attribution

$0, offline, read-only: it touches nothing but result JSON already on disk.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "agent", ROOT / "services"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from agent.app import operand_attribution as oa  # noqa: E402
from agent.app.quantity_index import QuantityRef, build_index  # noqa: E402

#: The derived-arithmetic cluster. 210-217 are two-operand tasks (``OP_A``/``OP_B``); 218-221 are
#: five-entity argmax tasks (``ENTITIES``).
EVAL_TASK_IDS: Tuple[str, ...] = tuple(str(n) for n in range(210, 222))

#: Held out so a later fitted ranker has a set it was never tuned against. One task from each
#: shape family: a difference (213), a ratio (217), an argmax (221).
HOLDOUT_IDS = frozenset({"213", "217", "221"})

#: Runs whose cells carry ``execution.output.pages[]`` for this cluster.
DEFAULT_PREFIXES: Tuple[str, ...] = ("mint01", "mint02", "ladder03")

DEFAULT_RESULTS_DIR = "agent/idea_test_results"

#: For the argmax tasks the field phrase is shared across all five entity slots and is not stored
#: on the module, so it is transcribed here from each module's own ``get_task_statement()`` prose
#: (the wording a host would parse out of the mandate), paired with the ``ENTITIES`` key holding
#: the ground-truth value. Field names genuinely vary per task — 218 measures a river, 220 a
#: bridge — so there is no mechanical rule to derive these from.
ARGMAX_METRICS: Dict[str, List[Tuple[str, str]]] = {
    "218": [("length_km", "the river's LENGTH (in kilometres)"),
            ("basin_km2", "the river's DRAINAGE BASIN SIZE / basin area "
                          "(in square kilometres, km^2)")],
    "219": [("height_m", "the waterfall's total HEIGHT (in metres)"),
            ("width_m", "the waterfall's WIDTH (in metres)")],
    "220": [("total_m", "the bridge's TOTAL LENGTH (in metres)"),
            ("span_m", "the bridge's LONGEST (main) SPAN (in metres)")],
    "221": [("height_m", "the building's architectural HEIGHT (in metres)"),
            ("floors", "the building's FLOOR COUNT (number of floors)")],
}


@dataclass(frozen=True)
class EvalSlot:
    """One thing a mandate asks to be read off one entity's page.

    ``entity`` and ``field_phrase`` are exactly the two attributes
    :func:`agent.app.operand_attribution.features` duck-types on, so an :class:`EvalSlot` is
    interchangeable with ``mandate_slots.Slot`` at the ranker's boundary.
    """

    entity: str
    field_phrase: str
    target_value: float
    slug_rx: str
    url: str = ""


# ------------------------------------------------------------------ task modules

def load_task_module(test_id: str):
    """Import ``agent.app.idea_tests.test_<test_id>_*`` as a real package module.

    Real-package import (not ``importlib.util.spec_from_file_location``) on purpose: the module
    constants ``OP_A`` / ``ENTITIES`` must keep object identity with what the runner sees, which a
    detached spec load would break. Same pattern as ``scripts/rescore_results.py``.
    """
    matches = sorted(glob.glob(str(ROOT / "agent" / "app" / "idea_tests" / f"test_{test_id}_*.py")))
    if not matches:
        raise FileNotFoundError(f"no task module for test id {test_id}")
    return importlib.import_module("agent.app.idea_tests." + Path(matches[0]).stem)


def slots_for_module(test_id: str, module: Any) -> List[EvalSlot]:
    """The read-this-number slots a task declares, in mandate order.

    Branches on the module's own shape (``OP_A``/``OP_B`` vs ``ENTITIES``) rather than on the id,
    so a stub module is exercised by the same code path as a real one.
    """
    if hasattr(module, "OP_A") and hasattr(module, "OP_B"):
        return [
            EvalSlot(entity=str(op["label"]), field_phrase=str(op["fact"]),
                     target_value=float(op["value"]), slug_rx=str(op.get("slug_rx") or ""),
                     url=str(op.get("url") or ""))
            for op in (module.OP_A, module.OP_B)
        ]
    entities = getattr(module, "ENTITIES", None)
    if not entities:
        return []
    metrics = ARGMAX_METRICS.get(test_id)
    if not metrics:
        raise KeyError(f"test {test_id} has ENTITIES but no ARGMAX_METRICS entry")
    slots: List[EvalSlot] = []
    for entity in entities:
        for key, phrase in metrics:
            if key not in entity:
                raise KeyError(f"test {test_id} entity {entity.get('key')!r} has no {key!r}")
            slots.append(EvalSlot(entity=str(entity["name"]), field_phrase=phrase,
                                  target_value=float(entity[key]),
                                  slug_rx=str(entity.get("slug_rx") or "")))
    return slots


# ------------------------------------------------------------------ stored pages

def iter_result_files(results_dir: str, prefixes: Sequence[str], test_id: str) -> Iterator[Path]:
    """Cell files for one task under any of ``prefixes``, newest-name-last, streaming.

    Skips the derived siblings (``*_summary.json``, ``*_report_v*.json``) and the ``.jsonl``
    transcripts, the way every other stored-cell reader in this repo does.
    """
    for prefix in prefixes:
        pattern = os.path.join(results_dir, f"{prefix}*_{test_id}_*.json")
        for name in sorted(glob.glob(pattern)):
            if name.endswith("_summary.json") or "_report_v" in os.path.basename(name):
                continue
            yield Path(name)


def stored_pages(results_dir: str, prefixes: Sequence[str], test_id: str) -> List[Dict[str, str]]:
    """Every distinct page stored for ``test_id``, dedup'd by ``content_hash``.

    :returns: ``[{"url", "content_hash", "text"}, ...]`` in first-seen order. A cell that fails to
        parse is skipped with a note on stderr rather than aborting the build.
    """
    seen: Dict[str, Dict[str, str]] = {}
    for path in iter_result_files(results_dir, prefixes, test_id):
        try:
            cell = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 -- a corrupt cell must not sink the build
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        if str((cell.get("test_metadata") or {}).get("test_id") or "") != test_id:
            continue
        for page in (cell.get("execution") or {}).get("output", {}).get("pages") or []:
            content_hash = str(page.get("content_hash") or "")
            text = str(page.get("text") or "")
            if not content_hash or not text or content_hash in seen:
                continue
            seen[content_hash] = {"url": str(page.get("url") or ""),
                                  "content_hash": content_hash, "text": text}
    return list(seen.values())


def pages_for_slot(slot: EvalSlot, pages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """The subset of ``pages`` whose URL matches the slot's module-declared ``slug_rx``."""
    if not slot.slug_rx:
        return []
    pattern = re.compile(slot.slug_rx, re.IGNORECASE)
    return [page for page in pages if pattern.search(unquote(page["url"]))]


# ------------------------------------------------------------------ offset-identity labelling

def value_variants(value: float) -> List[str]:
    """Every spelling of ``value`` a Wikipedia page plausibly writes, longest first.

    ``4350.0`` -> ``["4,350", "4 350", "4350"]``; ``419.7`` -> ``["419.7"]``. Longest first so a
    comma-grouped occurrence is found before its bare-digit prefix. Deliberately NOT exhaustive:
    a page that writes ``533 million`` for ``533000000`` yields no match, and the row is then
    reported as "no positive located" instead of being labelled by a tolerance.
    """
    variants: List[str] = []
    if float(value).is_integer():
        digits = str(abs(int(value)))
        sign = "-" if value < 0 else ""
        grouped = "{:,}".format(abs(int(value)))
        variants = [sign + grouped, sign + grouped.replace(",", " "), sign + digits]
    else:
        text = repr(float(value))
        whole, _, frac = text.partition(".")
        try:
            grouped = "{:,}".format(int(whole))
        except ValueError:
            grouped = whole
        variants = [f"{grouped}.{frac}", f"{grouped.replace(',', ' ')}.{frac}", text]
    # ``dict.fromkeys`` for order-preserving dedup and a STABLE sort, so equal-length spellings
    # keep the order above rather than a set's hash order: the labels must not move between runs.
    return sorted(dict.fromkeys(variants), key=lambda variant: -len(variant))


def find_value_spans(text: str, value: float) -> List[Tuple[int, int]]:
    """Character spans in ``text`` where ``value`` is written, as written.

    Guarded on both sides so ``381`` does not match inside ``3812`` or ``1,381``.
    """
    spans: List[Tuple[int, int]] = []
    for variant in value_variants(value):
        pattern = re.compile(r"(?<![0-9.,])" + re.escape(variant) + r"(?![0-9])")
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in spans:
                spans.append(span)
    return spans


def positive_indices(entries: Sequence[QuantityRef], text: str, value: float) -> List[int]:
    """Indices of the entries whose span covers an occurrence of ``value`` as written.

    "Covers" is span intersection: ``build_index`` records ``start``/``end`` around the value text
    itself, so an entry overlapping the located occurrence IS that occurrence.
    """
    spans = find_value_spans(text, value)
    if not spans:
        return []
    hits: List[int] = []
    for index, entry in enumerate(entries):
        if any(entry.start < end and start < entry.end for start, end in spans):
            hits.append(index)
    return hits


def _entry_dict(entry: QuantityRef) -> Dict[str, Any]:
    return {"label": entry.label, "value": entry.value, "unit": entry.unit,
            "start": entry.start, "end": entry.end, "source": entry.source}


def entry_from_dict(data: Dict[str, Any]) -> QuantityRef:
    """Rebuild a :class:`QuantityRef` from a stored row's entry dict."""
    return QuantityRef(label=str(data.get("label") or ""), value=str(data.get("value") or ""),
                       unit=str(data.get("unit") or ""), start=int(data.get("start") or 0),
                       end=int(data.get("end") or 0), source=str(data.get("source") or ""))


# ------------------------------------------------------------------ the eval set

def task_eval_set(ids: Sequence[str] = EVAL_TASK_IDS,
                  results_prefixes: Sequence[str] = DEFAULT_PREFIXES, *,
                  results_dir: str = DEFAULT_RESULTS_DIR,
                  module_loader: Callable[[str], Any] = load_task_module,
                  index_limit: int = 40) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the eval rows and a per-task coverage report.

    One row per ``(slot, page)`` pair: the slot, the page identity and text, the full quantity
    index of that page, and the indices of the entries that ARE the asked-for operand.

    :returns: ``(rows, report)``. ``report`` carries, per task, how many distinct pages were found
        and how many slots got at least one positive — the numbers that say whether a p@1 below is
        measuring anything.
    """
    rows: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    for test_id in ids:
        module = module_loader(test_id)
        slots = slots_for_module(test_id, module)
        pages = stored_pages(results_dir, results_prefixes, test_id)
        slots_with_page = 0
        slots_with_positive = 0
        for slot in slots:
            matched = pages_for_slot(slot, pages)
            if matched:
                slots_with_page += 1
            found_positive = False
            for page in matched:
                entries = build_index(page["text"], limit=index_limit)
                if not entries:
                    continue
                hits = positive_indices(entries, page["text"], slot.target_value)
                found_positive = found_positive or bool(hits)
                rows.append({
                    "test_id": test_id,
                    "entity": slot.entity,
                    "field_phrase": slot.field_phrase,
                    "target_value": slot.target_value,
                    "page_url": page["url"],
                    "content_hash": page["content_hash"],
                    "page_text": page["text"],
                    "entries": [_entry_dict(entry) for entry in entries],
                    "positive_indices": hits,
                    "positive_sources": sorted({entries[i].source for i in hits}),
                })
            if found_positive:
                slots_with_positive += 1
        report.append({
            "test_id": test_id,
            "split": "holdout" if test_id in HOLDOUT_IDS else "dev",
            "slots": len(slots),
            "pages_found": len(pages),
            "slots_with_page": slots_with_page,
            "slots_with_positive": slots_with_positive,
            "rows": sum(1 for row in rows if row["test_id"] == test_id),
        })
    return rows, report


def split_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]],
                                                        List[Dict[str, Any]]]:
    """``(dev, holdout)`` split by :data:`HOLDOUT_IDS`."""
    dev = [row for row in rows if row["test_id"] not in HOLDOUT_IDS]
    holdout = [row for row in rows if row["test_id"] in HOLDOUT_IDS]
    return dev, holdout


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    """Write ``rows`` one JSON object per line; returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file written by :func:`write_jsonl`."""
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assert_no_overlap(train_urls: Iterable[str], eval_paths: Sequence[Any]) -> None:
    """Raise if any training-side page URL also appears in an eval file.

    Kept for the day training is revisited. The superseded plan's ranker was fitted on rows drawn
    from the same stored pages it was then evaluated on; this is the check that would have caught
    it, and it is cheap enough to run before every fit.

    :raises ValueError: naming the offending URLs.
    """
    wanted = {str(url) for url in train_urls if url}
    leaked: Dict[str, List[str]] = {}
    for path in eval_paths:
        for row in read_jsonl(Path(path)):
            url = str(row.get("page_url") or "")
            if url in wanted:
                leaked.setdefault(url, []).append(str(path))
    if leaked:
        detail = "; ".join(f"{url} in {sorted(set(paths))}"
                           for url, paths in sorted(leaked.items()))
        raise ValueError(f"training pages leak into the eval set: {detail}")


# ------------------------------------------------------------------ scoring

def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` by the Lentz continued fraction.

    Only used when scipy is absent; scipy's implementation is preferred where available.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_reg(b, a, 1.0 - x)
    tiny = 1e-30
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def _beta_quantile(p: float, a: float, b: float) -> float:
    """``x`` such that ``I_x(a, b) == p``, by bisection. Fallback for a scipy-less venv."""
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _betainc_reg(a, b, mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def clopper_pearson(hits: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact-binomial (Clopper-Pearson) confidence interval for ``hits/n``.

    Exact rather than normal-approximate because the per-source n here is in the tens and often
    at the 0 or 1 boundary, where a Wald interval is simply wrong.

    :returns: ``(lower, upper)``; ``(0.0, 0.0)`` when ``n == 0``.
    """
    if n <= 0:
        return (0.0, 0.0)
    try:
        from scipy.stats import beta as _beta  # noqa: PLC0415 -- optional dependency
        lower = 0.0 if hits == 0 else float(_beta.ppf(alpha / 2.0, hits, n - hits + 1))
        upper = 1.0 if hits == n else float(_beta.ppf(1.0 - alpha / 2.0, hits + 1, n - hits))
    except Exception:  # noqa: BLE001 -- no scipy in this venv: use the local quantile
        lower = 0.0 if hits == 0 else _beta_quantile(alpha / 2.0, hits, n - hits + 1)
        upper = 1.0 if hits == n else _beta_quantile(1.0 - alpha / 2.0, hits + 1, n - hits)
    return (lower, upper)


def precision_at_1(ranker: Any, rows: Sequence[Dict[str, Any]],
                   alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Top-1 accuracy of ``ranker`` over ``rows``, split by infobox versus prose positives.

    A row with no located positive is not scoreable and is counted under ``"unlabelled"`` rather
    than silently dropped. A row whose positives span both sources is filed under ``"infobox"``
    (the infobox row is the one a host would want) and noted in ``both_sources``.

    :returns: ``{"infobox"|"prose": {"hits", "n", "p_at_1", "ci_low", "ci_high"}, "unlabelled": n}``
    """
    buckets = {"infobox": [0, 0], "prose": [0, 0]}
    unlabelled = 0
    both = 0
    for row in rows:
        hits = list(row.get("positive_indices") or [])
        if not hits:
            unlabelled += 1
            continue
        entries = [entry_from_dict(entry) for entry in row["entries"]]
        sources = {entries[i].source for i in hits}
        both += int(len(sources) > 1)
        kind = "infobox" if "infobox" in sources else "prose"
        slot = EvalSlot(entity=str(row.get("entity") or ""),
                        field_phrase=str(row.get("field_phrase") or ""),
                        target_value=float(row.get("target_value") or 0.0), slug_rx="")
        ranked = ranker.rank(slot, entries, page_url=str(row.get("page_url") or ""),
                             page_text=str(row.get("page_text") or ""))
        # Identity, not equality: ``rank`` returns the very objects it was passed, and two
        # distinct entries can compare equal after a page repeats a value.
        index_by_identity = {id(entry): index for index, entry in enumerate(entries)}
        top_index = index_by_identity.get(id(ranked[0][1])) if ranked else None
        buckets[kind][1] += 1
        buckets[kind][0] += int(top_index is not None and top_index in hits)
    out: Dict[str, Dict[str, Any]] = {}
    for kind, (hit_count, n) in buckets.items():
        low, high = clopper_pearson(hit_count, n, alpha)
        out[kind] = {"hits": hit_count, "n": n,
                     "p_at_1": (hit_count / n) if n else None,
                     "ci_low": low, "ci_high": high}
    out["unlabelled"] = {"n": unlabelled, "both_sources": both}
    return out


def format_precision_table(named_scores: Sequence[Tuple[str, Dict[str, Dict[str, Any]]]]) -> str:
    """The p@1 table as it goes into a report: one line per ranker, both sources side by side."""
    lines = [f"{'ranker':<16} {'infobox p@1':>12} {'95% CI':>16} {'n':>5} "
             f"{'prose p@1':>10} {'95% CI':>16} {'n':>5}"]
    for name, scores in named_scores:
        cells = [f"{name:<16}"]
        for kind in ("infobox", "prose"):
            block = scores[kind]
            value = "-" if block["p_at_1"] is None else f"{block['p_at_1']:.3f}"
            width = 12 if kind == "infobox" else 10
            cells.append(f"{value:>{width}}")
            cells.append(f"[{block['ci_low']:.3f}, {block['ci_high']:.3f}]".rjust(16))
            cells.append(f"{block['n']:>5}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


# ------------------------------------------------------------------ cli

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES),
                        help="comma-separated run-id prefixes to mine for stored pages")
    parser.add_argument("--out-dir", default="agent/idea_test_results/operand_attribution")
    parser.add_argument("--tests", default=",".join(EVAL_TASK_IDS),
                        help="comma-separated test ids to build")
    args = parser.parse_args(argv)

    ids = [t.strip() for t in args.tests.split(",") if t.strip()]
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    rows, report = task_eval_set(ids, prefixes, results_dir=args.results_dir)
    dev, holdout = split_rows(rows)
    out_dir = Path(args.out_dir)
    write_jsonl(dev, out_dir / "eval_dev.jsonl")
    write_jsonl(holdout, out_dir / "eval_holdout.jsonl")

    print(f"{'test':>5} {'split':>8} {'slots':>6} {'pages':>6} {'w/page':>7} "
          f"{'w/pos':>6} {'rows':>6}")
    for entry in report:
        print(f"{entry['test_id']:>5} {entry['split']:>8} {entry['slots']:>6} "
              f"{entry['pages_found']:>6} {entry['slots_with_page']:>7} "
              f"{entry['slots_with_positive']:>6} {entry['rows']:>6}")
    print(f"\nwrote {len(dev)} dev rows and {len(holdout)} holdout rows to {out_dir}")

    print("\nprecision@1 on dev (diagnostic, not a gate)")
    print(format_precision_table([
        (oa.default_ranker().name, precision_at_1(oa.default_ranker(), dev)),
        (oa.document_order_ranker().name, precision_at_1(oa.document_order_ranker(), dev)),
    ]))
    unlabelled = precision_at_1(oa.default_ranker(), dev)["unlabelled"]
    print(f"rows with no located positive: {unlabelled['n']}  "
          f"rows whose positives span both sources: {unlabelled['both_sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
