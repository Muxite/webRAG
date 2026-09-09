"""A run, told as an ordered list of beats you can draw one frame at a time.

What this is
------------
A **projection** of one stored per-cell result JSON into :class:`Beat` records -- a storyboard.
Each beat answers the same four questions in the same four fields, so a renderer can lay out one
frame grammar and reuse it for every step of every run:

  * ``title``   -- what is being done          ("LOCATE")
  * ``subject`` -- the text being handled      ("508.2")
  * ``origin``  -- where it came from          ("p3 - Taipei_101 - chars 1074-1081 - infobox")
  * ``target``  -- where it goes               ("E1")

plus ``ledger_after``, the accumulating state, which is what turns a pile of frames into a story:
the viewer watches the evidence rail fill.

Why a projection, not a second writer
-------------------------------------
This is the same call :mod:`agent.app.ledger_trace` made, for the same reason. Everything here is
derived from what a run ALREADY records -- ``telemetry_raw.timings`` (via ``ledger_trace``),
``output.evidence_graph``, ``output.host_derive``, ``output.answer_audit`` and
``output.confidence`` -- so it runs against every cell already on disk, including cells written
before this module existed, and it can never disagree with the run because it has no independent
source of truth.

The control plane comes entirely from :func:`agent.app.ledger_trace.project_cell`, so the
double-instrument merge (one LLM call recorded twice), the credential scrubbing and the closed
status vocabulary are inherited rather than re-derived.

Narrative order
---------------
Beats are ordered the way the run actually earned them, not the way the JSON happens to be laid
out:

1. ``question``  -- the mandate.
2. per search: ``search``.
3. per page fetched: ``visit``, then ``locate`` + ``mint`` for every span located on THAT page.
   A span can only be found on a page that was fetched, so this nesting is a fact about the run,
   not a presentational choice.
4. pages registered without a model visit (host prefetch) come after, marked as such.
5. ``rank``    -- per-slot operand attribution, with the score and the floor it had to clear.
6. ``derive`` / ``refuse`` -- the evidence plane. These have no interval; they cost no time.
7. ``audit``   -- every number in the final answer, backed / derived / unbacked.
8. ``verdict`` -- ANSWER / PARTIAL / ABSTAIN, with the arithmetic that produced it.

Honesty rules, inherited from the data and enforced here
--------------------------------------------------------
* **Tri-state is preserved.** ``quote_verified is None`` ("never checked") stays ``None`` and is
  never collapsed to ``False`` ("checked, and absent"). Same for ``derivation_valid`` and
  ``operand_supported``. A renderer that wants a boolean has to decide for itself, in the open.
* **Absent is never zero.** A beat with no interval carries no duration key at all.
* **Refusals and conflicts are beats.** The Ledger declining to assert something is the behaviour
  the whole subsystem exists to produce. It is the one thing a demo must not quietly drop.
* **Nothing is invented.** Every beat carries ``trace_ref`` (a real ``TraceNode`` id) or an
  evidence node id, so any frame can be walked back to the run that produced it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.app import ledger_trace
from agent.app.ledger_trace import LedgerTrace, TraceNode

#: Bumped whenever :meth:`Beat.as_dict` or :meth:`Storyboard.as_dict` changes shape.
SCHEMA_VERSION = 1

# --- beat kinds. Closed vocabulary: a renderer switches on this, so an unlisted kind would be
# --- drawn by no branch and would vanish from the story. ----------------------------------
KIND_QUESTION = "question"
KIND_SEARCH = "search"
KIND_VISIT = "visit"
KIND_LOCATE = "locate"
KIND_MINT = "mint"
KIND_RANK = "rank"
KIND_DERIVE = "derive"
KIND_REFUSE = "refuse"
KIND_AUDIT = "audit"
KIND_VERDICT = "verdict"

BEAT_KINDS = (
    KIND_QUESTION, KIND_SEARCH, KIND_VISIT, KIND_LOCATE, KIND_MINT,
    KIND_RANK, KIND_DERIVE, KIND_REFUSE, KIND_AUDIT, KIND_VERDICT,
)

#: Statuses are ``ledger_trace``'s, not a second vocabulary.
STATUS_OK = ledger_trace.STATUS_OK
STATUS_ERROR = ledger_trace.STATUS_ERROR
STATUS_EMPTY = ledger_trace.STATUS_EMPTY
STATUS_REFUSED = ledger_trace.STATUS_REFUSED
STATUS_INVALID = ledger_trace.STATUS_INVALID
STATUS_UNKNOWN = ledger_trace.STATUS_UNKNOWN

#: Characters of page text kept either side of a located span. Stored page text is a FLATTENED
#: infobox -- one token per line for long stretches -- so a window measured in characters can be
#: many lines tall. 160 keeps the span and its label on screen together once the renderer
#: collapses the line breaks. The raw text is handed over unmodified; collapsing is the
#: renderer's job, so the offsets reported here stay literally true of the stored page.
WINDOW_CHARS = 160

#: ``host_derive``'s confidence floor, mirrored here for the ``rank`` beat's annotation. Read from
#: ``ledger_tools`` so the two cannot drift.
try:  # pragma: no cover - exercised implicitly; the fallback exists for a trimmed import graph
    from agent.app.ledger_tools import _HOST_DERIVE_MIN_SCORE as HOST_DERIVE_MIN_SCORE
except Exception:  # pragma: no cover
    HOST_DERIVE_MIN_SCORE = 0.93


def _text(value: Any, cap: int = 400) -> str:
    """A bounded, JSON-safe string. ``None`` becomes ``""`` -- absent text is not the word None."""
    if value is None:
        return ""
    out = value if isinstance(value, str) else str(value)
    return out[:cap]


def _short_url(url: Any) -> str:
    """A URL reduced to the part a reader identifies it by: host tail plus last path segment."""
    raw = _text(url, 300)
    if not raw:
        return ""
    body = raw.split("://", 1)[-1]
    host, _, path = body.partition("/")
    tail = [seg for seg in path.split("/") if seg]
    return f"{host}/{tail[-1]}" if tail else host


@dataclass(frozen=True)
class Handle:
    """One fact on the evidence rail, under the ``E1``/``D1`` naming the Ledger itself uses.

    :param ref: ``E1``, ``E2``, ... for located spans; ``D1``, ``D2``, ... for derived values.
        Assigned in admission order, exactly as :meth:`Ledger.handle_for` does, so a handle in a
        frame means the same thing as a handle in a prompt.
    :param verified: tri-state. ``True`` located on the page, ``False`` checked and absent,
        ``None`` never checked. Never collapsed.
    """

    ref: str
    node_id: str
    kind: str
    value: str
    unit: str = ""
    origin: str = ""
    verified: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ref": self.ref, "node_id": self.node_id, "kind": self.kind, "value": self.value}
        if self.unit:
            out["unit"] = self.unit
        if self.origin:
            out["origin"] = self.origin
        if self.verified is not None:
            out["verified"] = self.verified
        return out


@dataclass(frozen=True)
class LedgerSnapshot:
    """The accumulating state after a beat -- what the deck's right-hand rail draws.

    Snapshots are cumulative and immutable: beat *n* holds everything known after *n* steps, so a
    renderer never replays the story itself to work out what to show.
    """

    handles: Tuple[Handle, ...] = ()
    pages: Tuple[str, ...] = ()
    refusals: int = 0

    @property
    def counts(self) -> Dict[str, int]:
        """Rail totals. ``verified`` counts only ``True`` -- an unchecked quote is not a verified one."""
        sources = [h for h in self.handles if h.kind == "source"]
        return {
            "pages": len(self.pages),
            "sources": len(sources),
            "derived": len([h for h in self.handles if h.kind == "derived"]),
            "verified": len([h for h in sources if h.verified is True]),
            "unverified": len([h for h in sources if h.verified is False]),
            "unchecked": len([h for h in sources if h.verified is None]),
            "refusals": self.refusals,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "handles": [h.as_dict() for h in self.handles],
            "pages": list(self.pages),
            "counts": self.counts,
        }


@dataclass(frozen=True)
class Beat:
    """One frame of the story.

    :param subject: the text being handled -- the value, the query, the answer number.
    :param origin: where that text came from, in reader-facing terms.
    :param target: where it goes -- the handle it becomes, or the slot it fills.
    :param status: ``ledger_trace``'s vocabulary, so ``refused`` here means what it means there.
    :param detail: bounded, kind-specific scalars. Never page text (except the ``locate`` window,
        which IS the point of that beat), never a prompt, never a credential.
    :param trace_ref: the ``TraceNode`` id this beat came from, or ``""`` for beats that live on
        the evidence plane and have no step in the timing record.
    """

    index: int
    kind: str
    title: str
    subject: str = ""
    origin: str = ""
    target: str = ""
    status: str = STATUS_OK
    detail: Dict[str, Any] = field(default_factory=dict)
    ledger_after: LedgerSnapshot = field(default_factory=LedgerSnapshot)
    trace_ref: str = ""
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Seconds, or ``None`` when this beat has no interval. Derivations cost no time."""
        if self.t_start is None or self.t_end is None:
            return None
        return max(0.0, self.t_end - self.t_start)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable, with every absent field OMITTED rather than nulled."""
        out: Dict[str, Any] = {"index": self.index, "kind": self.kind, "title": self.title, "status": self.status}
        for key in ("subject", "origin", "target", "trace_ref"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.t_start is not None:
            out["t_start"] = round(self.t_start, 4)
        if self.t_end is not None:
            out["t_end"] = round(self.t_end, 4)
        if self.detail:
            out["detail"] = self.detail
        out["ledger_after"] = self.ledger_after.as_dict()
        return out


@dataclass(frozen=True)
class Storyboard:
    """One run's beats, plus the identity a frame's header needs."""

    beats: Tuple[Beat, ...]
    mandate: str = ""
    task_id: str = ""
    model: str = ""
    variant: str = ""
    verdict: str = ""
    verdict_basis: str = ""
    score: Optional[float] = None
    source_file: str = ""

    def __len__(self) -> int:
        return len(self.beats)

    def of_kind(self, kind: str) -> List[Beat]:
        """Every beat of one kind, in story order."""
        return [b for b in self.beats if b.kind == kind]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for beat in self.beats:
            out[beat.kind] = out.get(beat.kind, 0) + 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mandate": self.mandate,
            "task_id": self.task_id,
            "model": self.model,
            "variant": self.variant,
            "verdict": self.verdict,
            "counts": self.counts(),
            "beats": [b.as_dict() for b in self.beats],
        }
        if self.verdict_basis:
            out["verdict_basis"] = self.verdict_basis
        if self.score is not None:
            out["score"] = self.score
        if self.source_file:
            out["source_file"] = self.source_file
        return out


class _Builder:
    """Accumulates beats and the rail state. Private: :func:`build` is the surface."""

    def __init__(self) -> None:
        self.beats: List[Beat] = []
        self.handles: List[Handle] = []
        self.pages: List[str] = []
        self.refusals = 0
        self._by_node: Dict[str, Handle] = {}

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(handles=tuple(self.handles), pages=tuple(self.pages), refusals=self.refusals)

    def add_handle(self, node_id: str, kind: str, value: str, unit: str, origin: str,
                   verified: Optional[bool]) -> Handle:
        prefix = "E" if kind == "source" else "D"
        seq = len([h for h in self.handles if h.kind == kind]) + 1
        handle = Handle(ref=f"{prefix}{seq}", node_id=node_id, kind=kind, value=value,
                        unit=unit, origin=origin, verified=verified)
        self.handles.append(handle)
        self._by_node[node_id] = handle
        return handle

    def ref_for(self, node_id: str) -> str:
        """The handle name for an evidence node id, or the truncated id when it never got one."""
        handle = self._by_node.get(node_id)
        return handle.ref if handle else (node_id[:8] if node_id else "")

    def emit(self, kind: str, title: str, **kwargs: Any) -> Beat:
        beat = Beat(index=len(self.beats), kind=kind, title=title,
                    ledger_after=self.snapshot(), **kwargs)
        self.beats.append(beat)
        return beat


def _page_lookup(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(p.get("page_id")): p for p in (graph.get("pages") or []) if isinstance(p, dict)}


def _window(text: str, start: int, end: int) -> Dict[str, Any]:
    """The span plus the text either side of it, with the offsets that locate it.

    Returned as three separate strings rather than one marked-up blob so a renderer chooses its
    own emphasis (a highlight, a colour, a box) without parsing anything back out.
    """
    if not text or start < 0 or end <= start or end > len(text):
        return {}
    lead = text[max(0, start - WINDOW_CHARS):start]
    tail = text[end:end + WINDOW_CHARS]
    return {
        "before": lead,
        "span": text[start:end],
        "after": tail,
        "start": start,
        "end": end,
        "truncated_left": start - WINDOW_CHARS > 0,
        "truncated_right": end + WINDOW_CHARS < len(text),
    }


def _source_origin(node: Dict[str, Any], page: Optional[Dict[str, Any]]) -> str:
    """A source span's provenance line: page id, page identity, and the exact offsets."""
    bits: List[str] = []
    page_id = _text(node.get("page_id"), 16)
    if page_id:
        bits.append(page_id)
    url = node.get("source_url") or (page or {}).get("url")
    if url:
        bits.append(_short_url(url))
    start, end = node.get("start"), node.get("end")
    if isinstance(start, int) and isinstance(end, int) and start >= 0:
        bits.append(f"chars {start}-{end}")
    return " · ".join(bits)


def _mint_status(node: Dict[str, Any]) -> str:
    """A minted span's status, holding the tri-state open.

    ``verified is False`` is an ERROR (the value was looked for on the page and was not there).
    ``verified is None`` is UNKNOWN (it was never checked). Folding the second into the first
    would report a check that never ran as a check that failed.
    """
    verified = node.get("verified")
    if verified is True:
        return STATUS_OK
    if verified is False:
        return STATUS_ERROR
    return STATUS_UNKNOWN


def _visit_beats(builder: _Builder, node: TraceNode, pages: Dict[str, Dict[str, Any]],
                 sources_by_page: Dict[str, List[Dict[str, Any]]]) -> None:
    """One ``visit`` beat, then every span located on the page it stored."""
    page_id = node.page_id
    page = pages.get(page_id) if page_id else None
    url = node.detail.get("url") or (page or {}).get("url")
    detail: Dict[str, Any] = {}
    for key in ("http_status", "chars"):
        if node.detail.get(key) is not None:
            detail[key] = node.detail[key]
    if page:
        detail["chars"] = page.get("stored_chars", page.get("chars"))
        if page.get("truncated"):
            detail["truncated"] = True
    if page_id and page_id not in builder.pages:
        builder.pages.append(page_id)
    builder.emit(
        KIND_VISIT, "VISIT",
        subject=_short_url(url), origin=_text(url, 300), target=page_id or "",
        status=node.status, detail=detail, trace_ref=node.id,
        t_start=node.t_start, t_end=node.t_end,
    )
    if page_id:
        _spans_on_page(builder, page_id, page, sources_by_page.pop(page_id, []))


def _spans_on_page(builder: _Builder, page_id: str, page: Optional[Dict[str, Any]],
                   sources: Sequence[Dict[str, Any]]) -> None:
    """``locate`` then ``mint``, for every span read off one page, in document order."""
    text = _text((page or {}).get("text"), 10 ** 7)
    for node in sorted(sources, key=lambda n: (n.get("start") if isinstance(n.get("start"), int) else 10 ** 9)):
        node_id = _text(node.get("id"), 64)
        value = _text(node.get("value"), 200)
        unit = _text(node.get("unit"), 32)
        origin = _source_origin(node, page)
        window = _window(text, node.get("start", -1), node.get("end", -1))
        if window:
            builder.emit(
                KIND_LOCATE, "LOCATE",
                subject=value, origin=origin, target="verbatim span",
                status=STATUS_OK if node.get("quote_verified") is not False else STATUS_ERROR,
                detail={"window": window, "occurrences": node.get("occurrences", 0),
                        "unit_bearing": bool(node.get("unit_bearing")),
                        "quote_verified": node.get("quote_verified")},
            )
        handle = builder.add_handle(node_id, "source", value, unit, origin, node.get("verified"))
        detail = {"verified": node.get("verified"), "quote_verified": node.get("quote_verified"),
                  "occurrences": node.get("occurrences", 0), "page_id": page_id}
        if node.get("quote_fail_reason"):
            detail["quote_fail_reason"] = _text(node["quote_fail_reason"], 64)
        if node.get("minted_by"):
            detail["minted_by"] = _text(node["minted_by"], 32)
        builder.emit(
            KIND_MINT, "MINT",
            subject=f"{value} {unit}".strip(), origin=origin, target=handle.ref,
            status=_mint_status(node), detail=detail,
        )


def _rank_beats(builder: _Builder, host_derive: Dict[str, Any]) -> None:
    """One beat per operand slot: what filled it, how well it scored, and against what floor."""
    slots = host_derive.get("slots") or []
    floor = host_derive.get("min_score")
    floor = floor if isinstance(floor, (int, float)) else HOST_DERIVE_MIN_SCORE
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        entry = slot.get("entry") if isinstance(slot.get("entry"), dict) else None
        reason = _text(slot.get("reason"), 48)
        score = slot.get("score")
        subject = f"{entry.get('value')} {entry.get('unit', '')}".strip() if entry else "no operand"
        if reason and reason != "selected":
            # The reason is the whole point of a losing slot: "below_min_score" and
            # "no_candidate_page" are different failures with different fixes.
            subject = f"{subject} ({reason})"
        origin_bits = [_text(slot.get("page_id"), 16) or "no page"]
        if slot.get("url"):
            origin_bits.append(_short_url(slot["url"]))
        if entry and entry.get("label"):
            origin_bits.append(f"label: {_text(entry['label'], 60)}")
        if entry and entry.get("source"):
            origin_bits.append(_text(entry["source"], 16))
        detail: Dict[str, Any] = {
            "entity": _text(slot.get("entity"), 120),
            "field_phrase": _text(slot.get("field_phrase"), 160),
            "reason": reason,
            "min_score": round(float(floor), 4),
        }
        if isinstance(score, (int, float)):
            detail["score"] = round(float(score), 4)
        if entry and isinstance(entry.get("start"), int):
            detail["start"], detail["end"] = entry.get("start"), entry.get("end")
        builder.emit(
            KIND_RANK, "RANK",
            subject=subject, origin=" · ".join(origin_bits),
            target=f"operand[{slot.get('index')}]",
            status=STATUS_OK if reason == "selected" else STATUS_REFUSED,
            detail=detail,
        )


def _derive_beats(builder: _Builder, trace: LedgerTrace, derived: Dict[str, Dict[str, Any]]) -> None:
    """The evidence plane: every derivation built, and every one the Ledger refused to build."""
    for node in trace.query(kind=ledger_trace.KIND_DERIVE):
        operation = _text(node.detail.get("operation"), 32)
        inputs = [builder.ref_for(nid) for nid in node.evidence_in]
        if node.status == STATUS_REFUSED:
            builder.refusals += 1
            builder.emit(
                KIND_REFUSE, "REFUSE",
                subject=_text(node.detail.get("code"), 64),
                origin=f"{operation}({', '.join(inputs)})" if inputs else operation,
                target="not asserted",
                status=STATUS_REFUSED,
                detail={"code": _text(node.detail.get("code"), 64),
                        "message": _text(node.detail.get("message"), 300),
                        "operation": operation},
                trace_ref=node.id,
            )
            continue
        out_id = node.evidence_out[0] if node.evidence_out else ""
        data = derived.get(out_id, {})
        value = _text(data.get("value"), 120)
        unit = _text(data.get("unit"), 32)
        origin = f"{operation}({', '.join(inputs)})" if inputs else operation
        handle = builder.add_handle(out_id, "derived", value, unit, origin, data.get("operand_supported"))
        detail = {"operation": operation, "inputs": inputs,
                  "derivation_valid": data.get("derivation_valid"),
                  "operand_supported": data.get("operand_supported")}
        if node.detail.get("message"):
            detail["message"] = _text(node.detail["message"], 200)
        builder.emit(
            KIND_DERIVE, "DERIVE",
            subject=f"{value} {unit}".strip(), origin=origin, target=handle.ref,
            status=node.status, detail=detail, trace_ref=node.id,
        )


def _audit_beat(builder: _Builder, audit: Dict[str, Any]) -> None:
    """Every number in the final answer, with whether it earned its support -- and, when it did
    not, WHICH of the three independent support conditions failed.

    ``answer_supported`` is a conjunction: every non-trivial number backed or derived, no unit
    inconsistency, and ambiguity within tolerance. A frame that reported only the first condition
    would show "28 of 28 backed" beside "not supported" and read as a bug in the system rather
    than the finding it is -- on the mint04 cell the blocker is AMBIGUITY (a figure that matches
    several different evidence values backs none of them uniquely).
    """
    numbers = [n for n in (audit.get("numbers") or []) if isinstance(n, dict)]
    if not numbers:
        return
    rows = []
    for entry in numbers:
        rows.append({
            "text": _text(entry.get("text"), 40),
            "unit": _text(entry.get("unit"), 24),
            "status": _text(entry.get("status"), 24),
            "trivial": bool(entry.get("trivial")),
            "ambiguity": int(entry.get("ambiguity") or 0),
            "unit_consistent": entry.get("unit_consistent"),
            "ref": builder.ref_for(_text(entry.get("node_id"), 64)),
            "page_id": _text(entry.get("page_id"), 16),
        })
    non_trivial = [r for r in rows if not r["trivial"]]
    unbacked = [r for r in non_trivial if r["status"] not in ("backed", "derived")]
    inconsistent = [r for r in non_trivial if r["unit_consistent"] is False]
    ambiguous = [r for r in non_trivial if r["ambiguity"] > 1]
    supported = bool(audit.get("answer_supported"))
    blockers = {"unbacked": len(unbacked), "unit inconsistent": len(inconsistent),
                "ambiguously backed": len(ambiguous)}
    named = [f"{count} {label}" for label, count in blockers.items() if count]
    builder.emit(
        KIND_AUDIT, "AUDIT",
        subject=f"{len(non_trivial) - len(unbacked)} of {len(non_trivial)} non-trivial numbers backed",
        origin="final answer",
        target="answer_supported" if supported else ("blocked: " + ", ".join(named) if named else "not supported"),
        status=STATUS_OK if supported else STATUS_ERROR,
        detail={"numbers": rows, "numbers_total": audit.get("numbers_total"),
                "answer_supported": supported, "blockers": blockers},
    )


def build(cell: Any, source_file: str = "") -> Storyboard:
    """Project one stored cell into a storyboard.

    :param cell: the parsed contents of an ``agent/idea_test_results/*.json`` file.
    :param source_file: the path it was read from, carried through for the frame header.
    :returns: the storyboard. A cell with no evidence graph still yields a valid, short story
        (question, whatever ran, verdict) rather than raising -- a run that found nothing is a
        result, and refusing to draw it would hide exactly the case worth seeing.
    """
    cell = cell if isinstance(cell, dict) else {}
    execution = cell.get("execution") if isinstance(cell.get("execution"), dict) else {}
    output = execution.get("output") if isinstance(execution.get("output"), dict) else {}
    telemetry = execution.get("telemetry_raw") if isinstance(execution.get("telemetry_raw"), dict) else {}
    graph = output.get("evidence_graph") if isinstance(output.get("evidence_graph"), dict) else {}
    validation = cell.get("validation") if isinstance(cell.get("validation"), dict) else {}
    metadata = cell.get("test_metadata") if isinstance(cell.get("test_metadata"), dict) else {}

    trace = ledger_trace.project_cell(cell)
    pages = _page_lookup(graph)
    all_nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    sources_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for node in all_nodes:
        if str(node.get("kind")) == "source":
            sources_by_page.setdefault(_text(node.get("page_id"), 16), []).append(node)
    derived = {_text(n.get("id"), 64): n for n in all_nodes if str(n.get("kind")) == "derived"}

    builder = _Builder()
    mandate = _text(telemetry.get("mandate"), 2000)
    builder.emit(
        KIND_QUESTION, "QUESTION",
        subject=mandate, origin=_text(metadata.get("test_id") or metadata.get("id"), 32),
        target="open ledger",
        detail={"model": _text(cell.get("model"), 64), "variant": _text(cell.get("execution_variant"), 64)},
    )

    # The timed plane, in the order it happened. Nodes with no interval sort last, stably.
    timed = [n for n in trace.nodes if n.kind in (ledger_trace.KIND_SEARCH, ledger_trace.KIND_VISIT)]
    timed.sort(key=lambda n: (n.t_start if n.t_start is not None else float("inf"), n.id))
    for node in timed:
        if node.kind == ledger_trace.KIND_SEARCH:
            builder.emit(
                KIND_SEARCH, "SEARCH",
                subject=_text(node.detail.get("query"), 200),
                origin="search backend",
                target=f"{node.detail.get('result_count', 0)} results",
                status=node.status,
                detail={k: v for k, v in node.detail.items() if k in ("query", "result_count")},
                trace_ref=node.id, t_start=node.t_start, t_end=node.t_end,
            )
        else:
            _visit_beats(builder, node, pages, sources_by_page)

    # Pages the model never visited: host prefetch put them there. Reported, never hidden -- a
    # span found on a prefetched page has different provenance from one the model went and got.
    for page_id in sorted(sources_by_page.keys(), key=lambda p: (len(p), p)):
        page = pages.get(page_id)
        if not page_id:
            continue
        if page_id not in builder.pages:
            builder.pages.append(page_id)
        builder.emit(
            KIND_VISIT, "PREFETCH",
            subject=_short_url((page or {}).get("url")),
            origin=_text((page or {}).get("source") or "host_prefetch", 40),
            target=page_id,
            detail={"chars": (page or {}).get("stored_chars"), "prefetched": True},
        )
        _spans_on_page(builder, page_id, page, sources_by_page[page_id])

    host_derive = output.get("host_derive") if isinstance(output.get("host_derive"), dict) else {}
    if host_derive:
        _rank_beats(builder, host_derive)

    _derive_beats(builder, trace, derived)

    audit = output.get("answer_audit") if isinstance(output.get("answer_audit"), dict) else {}
    if audit:
        _audit_beat(builder, audit)

    verdict = _text(output.get("confidence"), 32)
    basis = output.get("confidence_basis") if isinstance(output.get("confidence_basis"), dict) else {}
    basis_text = _text(basis.get("detail"), 400)
    if verdict:
        builder.emit(
            KIND_VERDICT, "VERDICT",
            subject=verdict, origin=_text(basis.get("source"), 64) or "computed in code",
            # NOT the deliverable: a raw markdown answer truncated to fit a header line reads as
            # a rendering failure, and the AUDIT beat has already shown the answer's numbers with
            # their backing. The closing frame's job is the verdict and how it was reached.
            target="run complete",
            status=STATUS_OK if verdict == "ANSWER" else (STATUS_EMPTY if verdict == "ABSTAIN" else STATUS_UNKNOWN),
            # The answer itself, bounded. A closing frame that never restates what the run
            # concluded leaves a viewer unable to say what happened -- which is the question the
            # opening frame asked. It goes in `detail` so the renderer can give it a real panel
            # rather than squeezing raw markdown into a one-line header.
            detail={"basis": basis_text, "counts": builder.snapshot().counts,
                    "answer": _text(output.get("final_deliverable"), 1800)},
        )

    score = validation.get("overall_score")
    return Storyboard(
        beats=tuple(builder.beats),
        mandate=mandate,
        task_id=_text(metadata.get("test_id") or metadata.get("id"), 32),
        model=_text(cell.get("model"), 64),
        variant=_text(cell.get("execution_variant"), 64),
        verdict=verdict,
        verdict_basis=basis_text,
        score=float(score) if isinstance(score, (int, float)) else None,
        source_file=source_file,
    )


def build_file(path: str) -> Storyboard:
    """Project the stored cell at ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        return build(json.load(handle), source_file=path)
