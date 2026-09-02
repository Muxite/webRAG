"""One normalized, typed, drawable record of what the Ledger did on a run.

What this is
------------
A flat list of :class:`TraceNode` -- stable id, typed ``kind``, typed ``status``, a parent link,
a time interval, a small typed ``detail`` dict, and cross-references into the artifacts that
already hold the content (``page_id`` into the page store, ``evidence_in`` / ``evidence_out``
into :mod:`agent.app.testing.evidence_graph`). Nothing else. Four properties are load-bearing:

1. **Queryable.** "Every derive that refused, with its operands" is
   ``trace.query(kind="derive", status="refused")`` and then ``node.evidence_in``; "every visit
   that returned non-200" is ``[n for n in trace.query(kind="visit") if n.status == "error"]``.
   Neither re-parses a free-text string, because neither fact is stored as free text.
2. **Drawable.** Every node has an id, a parent that chains to the single ``run`` root, and
   ``[t_start, t_end]`` when one is known -- enough for a timeline or a DAG with no further
   inference. :func:`to_mermaid` is the proof.
3. **Cross-referenced.** A ``derive`` node names the evidence-node ids it consumed and produced;
   a ``visit`` node names the ``page_id`` it stored. That is what turns a pile of timings into a
   traversal from "the agent fetched this URL" to "therefore this number is asserted".
4. **Smaller than what it subsumes.** It stores ids and small scalars, never content. Evidence
   VALUES, page TEXT and prompt/completion bodies stay in their one canonical home and are
   joined on demand (:func:`evidence_for`).

Why a projection, not a second writer
-------------------------------------
The trace is built from what a run ALREADY records -- ``telemetry_raw.timings`` plus
``output.evidence_graph`` -- so it can be validated against the ~1000 stored cells in
``agent/idea_test_results`` with no new run, and so a stored cell from before this module
existed is still readable. It also has to be a projection: half the trace's value is the
cross-reference into the evidence graph, and :class:`~agent.app.telemetry.TelemetrySession`
never sees the evidence graph -- the graph is assembled downstream, in the execution variant.
A live writer could emit the timing half and would then have to be joined to the evidence half
anyway, i.e. it would be this function with worse coverage.
:meth:`agent.app.telemetry.TelemetrySession.ledger_trace` exposes the timing half live for the
in-process case; the full trace comes from :func:`project_cell`.

Sourced from ``timings``, deliberately
--------------------------------------
The rolled-up ``execution.observability`` counters are known-wrong in ways that would silently
corrupt a trace (``search.count`` counts search RESULT ROWS, not searches: 12 against 2 real
calls; ``llm.calls`` counts connector_io events, in AND out, so it doubles). The raw ``timings``
entries are correct and are the only input used here.

The doubled ``llm_call``
------------------------
Two instrumentation layers wrap one model call -- an inner one carrying
``completion_chars``/``attempts`` and an outer one carrying only the model -- so ``timings``
holds two entries per call whose intervals nest and whose durations agree to well under a
millisecond (565 pairs, 0 unpaired, across 40 stored cells). That is one call recorded twice,
not two calls, so :func:`project` merges a nesting same-kind pair into ONE node with the union
of the two payloads and the outer interval. This module does not "fix" the underlying counter;
it just declines to encode the same fact twice.

Absent is never zero
--------------------
An unknown interval is ``None`` and is OMITTED from :meth:`TraceNode.as_dict`, never written as
``0.0``; a visit with no reported status code has no ``http_status`` key, rather than a ``0``
that would read as a failure. Absent and zero are different claims about the run.

No secrets
----------
Payload keys whose NAME is credential-shaped are dropped before they can reach a node, and a
URL's query string is stripped of credential-shaped parameters (:data:`_SECRET_KEY_RE`). Values
are never inspected for secret SHAPE -- that is a guess; a key name is a fact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import re

from agent.app.evidence_store import canonicalize_url

#: Bumped whenever :meth:`TraceNode.as_dict` changes shape, so a stored trace can be read back.
SCHEMA_VERSION = 1

KIND_RUN = "run"
KIND_LLM = "llm"
KIND_SEARCH = "search"
KIND_VISIT = "visit"
KIND_HTTP = "http"
KIND_CHROMA = "chroma"
KIND_DERIVE = "derive"
KIND_DECISION = "decision"
KIND_OTHER = "other"

#: ``timings`` names -> trace kind. An unlisted name becomes :data:`KIND_OTHER` and keeps its
#: original name in ``detail["name"]`` -- an instrument this module has not met yet is still
#: worth drawing, and silently dropping it would make the trace lie about what ran.
_TIMING_KINDS = {
    "llm_call": KIND_LLM,
    "search": KIND_SEARCH,
    # The backend half of the same search -- ``search_query`` nests inside ``search`` with the
    # same query and an identical interval, so mapping it to the same kind lets the
    # double-instrument merge collapse the pair instead of double-counting the call.
    "search_query": KIND_SEARCH,
    "visit": KIND_VISIT,
    "http_request": KIND_HTTP,
    "browser_fetch": KIND_HTTP,
    "chroma_store": KIND_CHROMA,
    "chroma_retrieve": KIND_CHROMA,
    "chroma_query": KIND_CHROMA,
    "chroma_add": KIND_CHROMA,
    "chroma_get_or_create": KIND_CHROMA,
    "chroma_list_collections": KIND_CHROMA,
    "embedding": KIND_CHROMA,
}

STATUS_OK = "ok"
STATUS_ERROR = "error"
#: A call that succeeded and returned nothing. Distinct from ``error`` (it worked) and from
#: ``ok`` (it bought nothing), which is exactly the distinction a yield question needs.
STATUS_EMPTY = "empty"
#: A derivation the graph declined to build at all (unit mismatch, missing operand, ...).
STATUS_REFUSED = "refused"
#: A derivation that was built and whose postcondition does NOT hold.
STATUS_INVALID = "invalid"
#: A derivation whose postcondition was never assessed. Never checked is not the same claim as
#: passed, and not the same claim as failed -- it gets its own status rather than folding in.
STATUS_UNKNOWN = "unknown"

_SECRET_KEY_RE = re.compile(
    r"(?:^|[_\-.])?(api[_\-]?key|apikey|key|token|secret|password|passwd|pwd|auth|authorization"
    r"|credential|bearer|cookie|session[_\-]?id|access[_\-]?token|refresh[_\-]?token|signature|sig)"
    r"(?:$|[_\-.])?",
    re.IGNORECASE,
)

#: Two intervals this close, one inside the other, are one operation seen by two instruments.
_MERGE_TOLERANCE_SECONDS = 0.01

_MESSAGE_CAP = 160


def _is_secret_key(name: Any) -> bool:
    """True when a payload key's NAME is credential-shaped."""
    return bool(_SECRET_KEY_RE.search(str(name or "")))


def _scrub_url(url: Any) -> str:
    """A URL with credential-shaped query parameters removed, other parameters intact.

    A search or fetch URL is one of the few payload fields that can carry a key inside an
    otherwise innocuous string, so the query string is filtered by parameter NAME. An
    unparseable URL degrades to the raw text with any query string dropped entirely, which is
    the conservative direction.
    """
    raw = str(url or "")
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw.split("?", 1)[0]
    if not parts.query:
        return raw
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_secret_key(k)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))


def _clean(value: Any) -> Any:
    """A payload value reduced to something small and JSON-safe."""
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float) or value is None:
        return value
    if isinstance(value, str):
        return value[:_MESSAGE_CAP]
    return str(value)[:_MESSAGE_CAP]


@dataclass
class TraceNode:
    """One typed step of a run.

    Every field earns its place against one of the four properties in the module docstring:

    :param id: stable within a trace and reproducible across two projections of the same run --
        an id you cannot re-derive cannot be referenced from a rendering or a follow-up query.
    :param kind: the closed vocabulary above. Typed so a filter is a comparison, not a regex.
    :param status: the closed outcome vocabulary. Typed for the same reason, and separate from
        ``kind`` because "which visits failed" and "what ran" are different questions.
    :param parent: the containing node's id, or ``None`` for the single ``run`` root. This is
        the edge that lets a renderer lay the trace out with no interval arithmetic of its own.
    :param t_start: session-relative seconds, or ``None`` when the step has no interval (a
        derivation refusal happens on the evidence plane, not the clock). Never ``0.0`` as a
        stand-in for unknown.
    :param t_end: as ``t_start``.
    :param detail: a small dict of typed, kind-specific scalars (``query``/``result_count``,
        ``url``/``http_status``, ``operation``/``code``). Bounded and credential-filtered; never
        a body, never a prompt, never page text.
    :param evidence_in: evidence-graph node ids this step consumed.
    :param evidence_out: evidence-graph node ids this step produced.
    :param page_id: the page-store id this step stored or read. Ids only -- the content lives
        in the page store, and duplicating it here is exactly what this module refuses to do.
    """

    id: str
    kind: str
    status: str = STATUS_OK
    parent: Optional[str] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    evidence_in: Tuple[str, ...] = ()
    evidence_out: Tuple[str, ...] = ()
    page_id: str = ""

    @property
    def duration(self) -> Optional[float]:
        """Seconds this step took, or ``None`` when it has no interval."""
        if self.t_start is None or self.t_end is None:
            return None
        return max(0.0, self.t_end - self.t_start)

    def as_dict(self) -> Dict[str, Any]:
        """This node as a JSON-serializable dict, with every absent field OMITTED.

        Omission rather than a null/zero placeholder is both the size win and the honesty win:
        a reader can tell "no interval" from "a zero-length interval" because only one of them
        has the key.
        """
        out: Dict[str, Any] = {"id": self.id, "kind": self.kind, "status": self.status}
        if self.parent is not None:
            out["parent"] = self.parent
        if self.t_start is not None:
            out["t_start"] = round(self.t_start, 4)
        if self.t_end is not None:
            out["t_end"] = round(self.t_end, 4)
        if self.detail:
            out["detail"] = dict(self.detail)
        if self.evidence_in:
            out["evidence_in"] = list(self.evidence_in)
        if self.evidence_out:
            out["evidence_out"] = list(self.evidence_out)
        if self.page_id:
            out["page_id"] = self.page_id
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceNode":
        """Rebuild a node from :meth:`as_dict` output. Missing keys mean absent, not zero."""
        data = data or {}
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", KIND_OTHER)),
            status=str(data.get("status", STATUS_OK)),
            parent=data.get("parent"),
            t_start=data.get("t_start"),
            t_end=data.get("t_end"),
            detail=dict(data.get("detail") or {}),
            evidence_in=tuple(data.get("evidence_in") or ()),
            evidence_out=tuple(data.get("evidence_out") or ()),
            page_id=str(data.get("page_id", "")),
        )


class LedgerTrace:
    """A run's nodes plus the query surface that makes them worth storing this way."""

    def __init__(self, nodes: Optional[Iterable[TraceNode]] = None) -> None:
        self.nodes: List[TraceNode] = list(nodes or [])

    # ------------------------------------------------------------------ query

    def query(
        self,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        page_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> List[TraceNode]:
        """Nodes matching every supplied filter, in trace order.

        :param kind: exact ``kind`` match.
        :param status: exact ``status`` match.
        :param page_id: exact ``page_id`` match.
        :param evidence_id: nodes that consumed OR produced this evidence-graph node -- the
            reverse edge, i.e. "what did the Ledger do with this fact?".
        :returns: the matching nodes.
        """
        out = []
        for node in self.nodes:
            if kind is not None and node.kind != kind:
                continue
            if status is not None and node.status != status:
                continue
            if page_id is not None and node.page_id != page_id:
                continue
            if evidence_id is not None and evidence_id not in node.evidence_in and evidence_id not in node.evidence_out:
                continue
            out.append(node)
        return out

    def get(self, node_id: str) -> Optional[TraceNode]:
        """The node with this id, or ``None``."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def children_of(self, node_id: str) -> List[TraceNode]:
        """Direct children of a node, in trace order."""
        return [n for n in self.nodes if n.parent == node_id]

    def root(self) -> Optional[TraceNode]:
        """The single parentless ``run`` node."""
        for node in self.nodes:
            if node.parent is None:
                return node
        return None

    def counts(self) -> Dict[str, Dict[str, int]]:
        """``{"kind": {...}, "status": {...}}`` -- the one rollup worth precomputing."""
        kinds: Dict[str, int] = {}
        statuses: Dict[str, int] = {}
        for node in self.nodes:
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
            statuses[node.status] = statuses.get(node.status, 0) + 1
        return {"kind": kinds, "status": statuses}

    # ------------------------------------------------------------- serialize

    def as_dict(self) -> Dict[str, Any]:
        """The whole trace as a JSON-serializable artifact.

        A ``detail`` value shared by EVERY node of a kind (in practice the model name, repeated
        once per LLM call) is hoisted into ``defaults`` and omitted from the nodes -- one
        encoding of one fact. :meth:`from_dict` puts it back.
        """
        payload = [node.as_dict() for node in self.nodes]
        defaults: Dict[str, Any] = {}
        models = {n.detail.get("model") for n in self.nodes if n.kind == KIND_LLM}
        models.discard(None)
        if len(models) == 1:
            defaults["model"] = models.pop()
            for row, node in zip(payload, self.nodes):
                if node.kind == KIND_LLM and "detail" in row:
                    row["detail"].pop("model", None)
                    if not row["detail"]:
                        row.pop("detail")
        out: Dict[str, Any] = {"schema": SCHEMA_VERSION, "counts": self.counts(), "nodes": payload}
        if defaults:
            out["defaults"] = defaults
        return out

    @classmethod
    def from_dict(cls, artifact: Dict[str, Any]) -> "LedgerTrace":
        """Rebuild a trace from :meth:`as_dict` output, re-expanding ``defaults``."""
        artifact = artifact or {}
        defaults = artifact.get("defaults") or {}
        nodes = [TraceNode.from_dict(row) for row in (artifact.get("nodes") or []) if isinstance(row, dict)]
        if "model" in defaults:
            for node in nodes:
                if node.kind == KIND_LLM:
                    node.detail.setdefault("model", defaults["model"])
        return cls(nodes)


# ------------------------------------------------------------------ projection


def _timing_detail(kind: str, name: str, payload: Dict[str, Any], error: Any) -> Tuple[Dict[str, Any], str]:
    """The small typed ``detail`` for one timing, plus its status.

    Named payload fields are lifted into stable, kind-independent key names (``status`` becomes
    ``http_status``, ``search_provenance`` becomes ``provenance``) so a query does not have to
    know which instrument wrote the row.
    """
    payload = {k: v for k, v in (payload or {}).items() if not _is_secret_key(k)}
    detail: Dict[str, Any] = {}
    status = STATUS_OK

    if kind == KIND_LLM:
        for key in ("model", "completion_chars", "attempts"):
            if payload.get(key) is not None:
                detail[key] = _clean(payload[key])
    elif kind == KIND_SEARCH:
        if payload.get("query") is not None:
            detail["query"] = _clean(payload["query"])
        # ``search`` reports ``result_count``, its ``search_query`` backend half reports
        # ``results``. One fact, so it gets one key name here.
        count = payload.get("result_count", payload.get("results"))
        if count is not None:
            detail["result_count"] = int(count)
            if detail["result_count"] == 0:
                status = STATUS_EMPTY
        if payload.get("search_provenance"):
            detail["provenance"] = _clean(payload["search_provenance"])
    elif kind in (KIND_VISIT, KIND_HTTP):
        if payload.get("url") is not None:
            detail["url"] = _scrub_url(payload["url"])
        if payload.get("method"):
            detail["method"] = _clean(payload["method"])
        if payload.get("status") is not None:
            try:
                detail["http_status"] = int(payload["status"])
            except (TypeError, ValueError):
                detail["http_status"] = _clean(payload["status"])
            if isinstance(detail["http_status"], int) and not 200 <= detail["http_status"] < 300:
                status = STATUS_ERROR
        if payload.get("used_browser"):
            detail["used_browser"] = True
    else:
        detail["name"] = name
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                detail[key] = _clean(value)

    if error:
        detail["error"] = _clean(error)
    return detail, status


def _nodes_from_timings(timings: Any) -> List[TraceNode]:
    """One node per timing, ordered by start time, doubled-instrument pairs already merged."""
    rows = []
    for entry in timings or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        kind = _TIMING_KINDS.get(name, KIND_OTHER)
        t_start = entry.get("t_start")
        t_end = entry.get("t_end")
        t_start = float(t_start) if isinstance(t_start, (int, float)) else None
        t_end = float(t_end) if isinstance(t_end, (int, float)) else None
        detail, status = _timing_detail(kind, name, entry.get("payload") or {}, entry.get("error"))
        if entry.get("success") is False:
            status = STATUS_ERROR
        rows.append({
            "kind": kind, "name": name, "status": status, "detail": detail,
            "t_start": t_start, "t_end": t_end,
        })

    rows.sort(key=lambda r: (
        r["t_start"] if r["t_start"] is not None else float("inf"),
        -(r["t_end"] if r["t_end"] is not None else 0.0),
    ))
    rows = _merge_double_instrumented(rows)

    nodes = []
    for index, row in enumerate(rows):
        nodes.append(TraceNode(
            id=f"t{index:04d}",
            kind=row["kind"],
            status=row["status"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            detail=row["detail"],
        ))
    return nodes


def _details_agree(outer: Dict[str, Any], inner: Dict[str, Any]) -> bool:
    """True when two payloads can only be two views of ONE operation.

    Nesting plus a matching duration is not on its own proof: two genuinely PARALLEL calls
    launched together and finishing together would nest and agree on duration too, and merging
    those would erase a branch of the fan-out -- the exact thing the interval stamps were added
    to make provable. So the payloads have to be compatible as well: every key they share must
    carry the same value, and one keyset must contain the other. The real double-instrument pair
    passes trivially (the outer records ``{model}``, the inner ``{model, completion_chars,
    attempts}``); two parallel model calls do not, because both carry their own
    ``completion_chars``.
    """
    shared = set(outer) & set(inner)
    if any(outer[key] != inner[key] for key in shared):
        return False
    return set(outer) <= set(inner) or set(inner) <= set(outer)


def _merge_double_instrumented(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a same-kind pair whose intervals nest and agree, into one node.

    Two instruments wrapping one call is not two calls. The OUTER interval is kept (it is the
    one that bounds the real elapsed time) and the two payloads are unioned, with the inner
    row's values winning because the inner instrument is the one closest to the call and so
    carries the specifics (``completion_chars``, ``attempts``).
    """
    merged: List[Dict[str, Any]] = []
    consumed = set()
    for i, outer in enumerate(rows):
        if i in consumed:
            continue
        if outer["t_start"] is not None and outer["t_end"] is not None:
            for j in range(i + 1, len(rows)):
                if j in consumed:
                    continue
                inner = rows[j]
                if inner["kind"] != outer["kind"] or inner["t_start"] is None or inner["t_end"] is None:
                    continue
                if inner["t_start"] < outer["t_start"] or inner["t_end"] > outer["t_end"] + 1e-9:
                    continue
                span_out = outer["t_end"] - outer["t_start"]
                span_in = inner["t_end"] - inner["t_start"]
                if abs(span_out - span_in) > _MERGE_TOLERANCE_SECONDS:
                    continue
                if not _details_agree(outer["detail"], inner["detail"]):
                    continue
                detail = dict(outer["detail"])
                detail.update(inner["detail"])
                outer = dict(outer, detail=detail)
                if inner["status"] == STATUS_ERROR:
                    outer["status"] = STATUS_ERROR
                consumed.add(j)
                break
        merged.append(outer)
    return merged


def _assign_parents(nodes: List[TraceNode], root: TraceNode) -> None:
    """Parent each node to the INNERMOST node whose interval strictly contains it, else the root.

    Containment is the structure the run already has -- an ``http_request`` really does happen
    inside the ``visit`` that issued it -- so reading it off the intervals gives a correct tree
    without any instrument having to be taught to pass a parent id down. A node with no
    interval hangs off the root.
    """
    timed = [n for n in nodes if n.t_start is not None and n.t_end is not None]
    for node in nodes:
        best: Optional[TraceNode] = None
        if node.t_start is not None and node.t_end is not None:
            for candidate in timed:
                if candidate is node:
                    continue
                if candidate.t_start > node.t_start or candidate.t_end < node.t_end:
                    continue
                if candidate.t_start == node.t_start and candidate.t_end == node.t_end:
                    continue  # a tie is not containment; both hang off the root
                if best is None or (candidate.t_start >= best.t_start and candidate.t_end <= best.t_end):
                    best = candidate
        node.parent = best.id if best is not None else root.id
    root.parent = None


def _page_index(evidence_graph: Any) -> Dict[str, str]:
    """Canonical URL -> ``page_id``, with a query-stripped fallback key.

    ``canonicalize_url`` deliberately KEEPS the query string, which is right for page identity
    and wrong for matching a visit whose URL picked up a tracking parameter on the way. Both
    keys are indexed and the exact one is tried first, so the looser match is only ever a
    fallback.
    """
    index: Dict[str, str] = {}
    for page in ((evidence_graph or {}).get("pages") or []):
        if not isinstance(page, dict):
            continue
        page_id = str(page.get("page_id") or "")
        url = canonicalize_url(page.get("url"))
        if not page_id or not url:
            continue
        index.setdefault(url, page_id)
        index.setdefault(url.split("?", 1)[0], page_id)
    return index


def _resolve_page(index: Dict[str, str], url: Any) -> str:
    """The ``page_id`` a visited URL landed in, or ``""`` when nothing stored it."""
    canonical = canonicalize_url(url)
    if not canonical:
        return ""
    return index.get(canonical) or index.get(canonical.split("?", 1)[0], "")


def _derive_nodes(evidence_graph: Any, start_index: int, parent_id: str) -> List[TraceNode]:
    """One node per DERIVED evidence node and one per derivation refusal.

    A refusal is a first-class step of the run: it is the Ledger declining to assert something,
    which is the behaviour the whole subsystem exists to produce, and it is invisible in
    ``timings`` because it costs no time. It has no interval, so it has none -- not a zero one.
    """
    graph = evidence_graph or {}
    nodes: List[TraceNode] = []
    index = start_index

    for data in (graph.get("nodes") or []):
        if not isinstance(data, dict) or str(data.get("kind")) != "derived":
            continue
        valid = data.get("derivation_valid")
        status = STATUS_OK if valid is True else (STATUS_INVALID if valid is False else STATUS_UNKNOWN)
        detail: Dict[str, Any] = {"operation": _clean(data.get("operation") or "")}
        if data.get("derivation_detail"):
            detail["message"] = _clean(data["derivation_detail"])
        nodes.append(TraceNode(
            id=f"d{index:04d}",
            kind=KIND_DERIVE,
            status=status,
            parent=parent_id,
            detail=detail,
            evidence_in=tuple(str(x) for x in (data.get("input_ids") or ())),
            evidence_out=(str(data.get("id")),) if data.get("id") else (),
        ))
        index += 1

    for row in (graph.get("derivation_refusals") or []):
        if not isinstance(row, dict):
            continue
        detail = {"operation": _clean(row.get("operation") or ""), "code": _clean(row.get("code") or "")}
        if row.get("message"):
            detail["message"] = _clean(row["message"])
        nodes.append(TraceNode(
            id=f"d{index:04d}",
            kind=KIND_DERIVE,
            status=STATUS_REFUSED,
            parent=parent_id,
            detail=detail,
            evidence_in=tuple(str(x) for x in (row.get("input_ids") or ())),
        ))
        index += 1
    return nodes


def _decision_nodes(decisions: Any, start_index: int, parent_id: str) -> List[TraceNode]:
    """One node per recorded engine decision, when the run recorded any."""
    nodes = []
    for offset, row in enumerate(decisions or []):
        if not isinstance(row, dict):
            continue
        detail: Dict[str, Any] = {"stage": _clean(row.get("stage") or "")}
        if row.get("chosen") is not None:
            detail["chosen"] = _clean(row["chosen"])
        if row.get("score") is not None:
            detail["score"] = row["score"]
        if row.get("grounded") is not None:
            detail["grounded"] = bool(row["grounded"])
        nodes.append(TraceNode(
            id=f"c{start_index + offset:04d}",
            kind=KIND_DECISION,
            status=STATUS_OK,
            parent=parent_id,
            detail=detail,
        ))
    return nodes


def project(telemetry_raw: Any, evidence_graph: Any) -> LedgerTrace:
    """Build a trace from a run's raw telemetry and its evidence graph.

    :param telemetry_raw: ``TelemetrySession.summary()`` output (slimmed or not). Anything that
        is not a dict is treated as an empty run rather than raising -- a stored cell from a
        verbosity level that dropped the block should still project to a valid, if bare, trace.
    :param evidence_graph: ``EvidenceGraph.to_dict()`` output, or ``None``.
    :returns: the trace, always with exactly one ``run`` root.
    """
    telemetry_raw = telemetry_raw if isinstance(telemetry_raw, dict) else {}
    nodes = _nodes_from_timings(telemetry_raw.get("timings"))

    ends = [n.t_end for n in nodes if n.t_end is not None]
    duration = telemetry_raw.get("duration")
    if not isinstance(duration, (int, float)):
        duration = max(ends) if ends else 0.0
    root = TraceNode(
        id="run0",
        kind=KIND_RUN,
        status=STATUS_OK,
        t_start=0.0,
        t_end=float(max(duration, max(ends) if ends else 0.0)),
    )

    page_index = _page_index(evidence_graph)
    for node in nodes:
        if node.kind in (KIND_VISIT, KIND_HTTP):
            node.page_id = _resolve_page(page_index, node.detail.get("url"))

    _assign_parents(nodes, root)
    all_nodes = [root] + nodes
    all_nodes.extend(_derive_nodes(evidence_graph, len(all_nodes), root.id))
    all_nodes.extend(_decision_nodes(telemetry_raw.get("decisions"), len(all_nodes), root.id))
    return LedgerTrace(all_nodes)


def project_cell(cell: Any) -> LedgerTrace:
    """Build a trace from one stored per-cell result JSON.

    :param cell: the parsed contents of an ``agent/idea_test_results/*.json`` file.
    :returns: the trace. A cell with no telemetry block still yields a valid bare trace.
    """
    cell = cell if isinstance(cell, dict) else {}
    execution = cell.get("execution") if isinstance(cell.get("execution"), dict) else {}
    output = execution.get("output") if isinstance(execution.get("output"), dict) else {}
    return project(execution.get("telemetry_raw"), output.get("evidence_graph"))


# ---------------------------------------------------------------- cross-reference


def evidence_for(node: TraceNode, evidence_graph: Any) -> Dict[str, List[Any]]:
    """Resolve a trace node's evidence references against the graph that holds the content.

    This is the join the trace is designed around: the trace keeps ids, the graph keeps values,
    and neither stores the other's half. An id that names no node comes back under
    ``missing_inputs`` rather than being dropped -- that a derivation named an operand which
    does not exist is precisely the ``MISSING_OPERAND`` refusal a reader is chasing.

    :param node: a trace node.
    :param evidence_graph: ``EvidenceGraph.to_dict()`` output, or ``None``.
    :returns: ``{"inputs", "outputs", "missing_inputs", "missing_outputs", "pages"}``.
    """
    by_id = {}
    for data in ((evidence_graph or {}).get("nodes") or []):
        if isinstance(data, dict) and data.get("id"):
            by_id[str(data["id"])] = data
    pages = {}
    for page in ((evidence_graph or {}).get("pages") or []):
        if isinstance(page, dict) and page.get("page_id"):
            pages[str(page["page_id"])] = page

    out: Dict[str, List[Any]] = {"inputs": [], "outputs": [], "missing_inputs": [], "missing_outputs": [], "pages": []}
    for key, missing_key, ids in (
        ("inputs", "missing_inputs", node.evidence_in),
        ("outputs", "missing_outputs", node.evidence_out),
    ):
        for eid in ids:
            if eid in by_id:
                out[key].append(by_id[eid])
            else:
                out[missing_key].append(eid)
    page_ids = {node.page_id} if node.page_id else set()
    page_ids.update(str(n.get("page_id") or "") for n in out["inputs"] + out["outputs"])
    out["pages"] = [pages[pid] for pid in sorted(p for p in page_ids if p in pages)]
    return out


# ---------------------------------------------------------------------- rendering

_SHAPES = {
    KIND_RUN: ("([", "])"),
    KIND_DERIVE: ("{{", "}}"),
    KIND_LLM: ("[", "]"),
    KIND_SEARCH: ("[/", "/]"),
    KIND_VISIT: ("[(", ")]"),
    KIND_HTTP: ("[", "]"),
}

_STATUS_CLASS = {
    STATUS_ERROR: "bad",
    STATUS_REFUSED: "bad",
    STATUS_INVALID: "bad",
    STATUS_EMPTY: "warn",
    STATUS_UNKNOWN: "warn",
}


def _mermaid_label(node: TraceNode) -> str:
    """A short, mermaid-safe one-liner for a node box, COMPOSED from the typed fields.

    Nodes carry no display string of their own: a stored ``label`` would be a second encoding
    of ``kind`` (``"llm_call"`` next to ``kind: "llm"``) or of a detail field (``"UNIT_MISMATCH"``
    next to ``code: "UNIT_MISMATCH"``). The renderer composes one instead.
    """
    bits = [node.detail.get("name") or node.kind]
    detail = node.detail
    for key in ("stage", "operation", "code", "query", "url", "http_status", "result_count", "completion_chars"):
        if detail.get(key) is not None:
            bits.append(f"{key}={detail[key]}")
    if node.page_id:
        bits.append(node.page_id)
    if node.duration is not None and node.duration >= 0.001:
        bits.append(f"{node.duration:.2f}s")
    text = " ".join(str(b) for b in bits)[:90]
    return re.sub(r'["\[\]{}()<>|#;]', " ", text)


def to_mermaid(trace: LedgerTrace, evidence_graph: Any = None, max_nodes: int = 200) -> str:
    """Render a trace as a mermaid ``graph TD``, ready to paste into a mermaid block.

    Parent links become solid edges; evidence references become dotted edges to evidence nodes
    pulled LIVE from ``evidence_graph`` -- the values are drawn from their one home rather than
    copied into the trace, which is the whole point of storing ids.

    :param trace: the trace.
    :param evidence_graph: optional graph, for the evidence layer.
    :param max_nodes: cap on trace nodes drawn, so a long run still renders.
    :returns: mermaid source.
    """
    lines = ["graph TD"]
    nodes = trace.nodes[:max_nodes]
    drawn = {n.id for n in nodes}
    by_eid = {}
    for data in ((evidence_graph or {}).get("nodes") or []):
        if isinstance(data, dict) and data.get("id"):
            by_eid[str(data["id"])] = data

    classes: Dict[str, List[str]] = {}
    for node in nodes:
        open_b, close_b = _SHAPES.get(node.kind, ("[", "]"))
        lines.append(f'    {node.id}{open_b}"{_mermaid_label(node)}"{close_b}')
        css = _STATUS_CLASS.get(node.status)
        if css:
            classes.setdefault(css, []).append(node.id)
    for node in nodes:
        if node.parent and node.parent in drawn:
            lines.append(f"    {node.parent} --> {node.id}")

    evidence_seen = set()
    for node in nodes:
        for eid in tuple(node.evidence_in) + tuple(node.evidence_out):
            safe = "e_" + re.sub(r"\W", "_", eid)[:24]
            if safe not in evidence_seen:
                evidence_seen.add(safe)
                data = by_eid.get(eid)
                if data is None:
                    label = f"{eid[:12]} (missing)"
                    classes.setdefault("bad", []).append(safe)
                else:
                    value = re.sub(r'["\[\]{}()<>|#;]', " ", str(data.get("value", ""))[:40])
                    label = f"{eid[:8]} {value}"
                lines.append(f'    {safe}(["{label}"])')
            if eid in node.evidence_in:
                lines.append(f"    {safe} -.-> {node.id}")
            else:
                lines.append(f"    {node.id} -.-> {safe}")

    lines.append("    classDef bad stroke:#c0392b,stroke-width:2px;")
    lines.append("    classDef warn stroke:#d68910,stroke-width:2px;")
    for css, ids in classes.items():
        lines.append(f"    class {','.join(sorted(set(ids)))} {css};")
    return "\n".join(lines)


def to_json(trace: LedgerTrace, indent: int = 2) -> str:
    """The trace as JSON text."""
    return json.dumps(trace.as_dict(), indent=indent, ensure_ascii=True)
