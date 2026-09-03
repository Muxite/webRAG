"""
Testing utilities.
"""

import json
from collections import Counter
from typing import Dict, Any, List

from agent.app.agent_io import is_malformed_url_text
from agent.app.idea_test_utils import count_words, count_chars
from agent.app.model_costs import estimate_cost, format_cost

_CHARS_PER_TOKEN = 4.0

_INFRA_STATUS_CODES = frozenset({402, 422, 429, 500, 502, 503, 504})
_INFRA_TIMING_NAMES = frozenset({"http_request", "search_query", "visit"})

# Bug A: infra.failed severity threshold. Occasional transient fetch hiccups are normal
# operating condition on a web-research run, not a corrupt cell — a bare "any single failed
# op" OR (the old behavior) flagged cells with e.g. 14/16 http_request successes that went on
# to produce a perfectly valid score. `failed` now fires only when a per-op infra-classified
# failure rate is a MATERIAL FRACTION of that op's attempts (> 50%), or the op produced zero
# successes outright (a total outage, which is the rate==1.0 case of the same rule for any
# sample size >= 1). failure_count/ops still report every classified failure regardless of
# this gate — nothing is hidden, only the boolean stops being a silent exclusion trigger.
#
# Threshold basis (empirical, not intuition): replayed this rule against the 80 cells of
# agent/idea_test_results/paid_wide_sweep_20260823_rep1_*_r1.json, of which 11 were flagged
# infra_failed=true under the old any-op-OR rule and ALL 80 produced valid scores. The worst
# observed per-op rate among those 11 was exactly 8/16 (0.5) on http_request; every other op
# in every flagged cell was well below that. A strictly-greater-than-0.5 threshold clears all
# 11 while still catching a majority-failed op. See scratch replay script referenced in the
# handoff for this change.
_INFRA_FAILURE_RATE_THRESHOLD = 0.5


def _is_infra_timing(timing: Dict[str, Any]) -> bool:
    """Classify a failed telemetry timing as infra failure or task/model failure.

    A malformed/invented URL, or a coerced-and-still-broken tool argument, fails BEFORE any
    HTTP status exists -- landing in the (formerly unconditional) "status is None -> assume
    infra" branch below and hiding the model's own mistake from its score: the more a model
    mangles a URL, the more of its own failures used to vanish from analysis. Two layers fix
    that, cheapest/most-confident first:

    1. ``payload["failure_class"]`` -- set by ``AgentIO.search``/``.visit`` (agent_io.py) going
       forward, once per call, from a signal richer than what any stored cell has ever
       persisted (the coercion outcome, and for a network-level failure, the underlying
       exception text). ``"model"``/``"infra"`` here are authoritative; ``"unknown"`` means
       even that richer signal couldn't tell, and falls through exactly like no signal at all.
    2. For a ``visit`` timing recorded before that field existed (no ``failure_class`` key at
       all), the raw ``url`` text is still enough to catch the unambiguous case: empty, or
       still shaped like a Python list/dict repr (``is_malformed_url_text``, shared with
       agent_io.py so both places agree on what "unambiguous" means). That is what makes the
       per-model infra-failed table re-measurable against every cell already on disk, not just
       cells recorded after this fix ships.

    Deliberately does NOT try to reclassify the ambiguous case (a syntactically valid URL that
    still failed with no status) for old data: a DNS failure on an invented host and a genuine
    transient network failure produce the identical persisted shape (url, status=None, no
    exception text), and guessing either way would either hide a real model failure or
    wrongly un-quarantine a real infra failure. That case keeps the pre-existing (and, for
    genuine infra, correct) any-status-None-on-visit/search-is-infra heuristic.
    """
    if timing.get("success"):
        return False
    payload = timing.get("payload") or {}
    if payload.get("infra_failed") is True:
        return True
    name = timing.get("name")
    status = payload.get("status")

    failure_class = payload.get("failure_class")
    if failure_class == "model":
        return False
    if failure_class == "infra":
        return True

    if name == "visit" and status is None:
        url = payload.get("url")
        if isinstance(url, str) and is_malformed_url_text(url):
            return False

    if status is None:
        return name in _INFRA_TIMING_NAMES
    try:
        return int(status) in _INFRA_STATUS_CODES
    except (TypeError, ValueError):
        return False


def _summarize_infra(timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up infra-classified failures for the result JSON.

    `failed` is gated by severity, not by the mere presence of any single failed op — see
    _INFRA_FAILURE_RATE_THRESHOLD above for the rationale and empirical basis. `failure_count`
    and `ops` are unaffected: they report every infra-classified failure regardless of rate.
    `rates` additionally exposes the observed per-op failure rate so a consumer can apply its
    own cutoff instead of trusting this one.
    """
    per_op: Dict[str, Dict[str, int]] = {}
    for t in timings:
        name = t.get("name", "unknown")
        entry = per_op.setdefault(name, {"total": 0, "success": 0, "infra_fail": 0})
        entry["total"] += 1
        if t.get("success"):
            entry["success"] += 1
        if _is_infra_timing(t):
            entry["infra_fail"] += 1

    failure_count = sum(e["infra_fail"] for e in per_op.values())
    ops = sorted(name for name, e in per_op.items() if e["infra_fail"] > 0)

    rates = {}
    failed = False
    for name in ops:
        e = per_op[name]
        rate = e["infra_fail"] / e["total"] if e["total"] else 0.0
        rates[name] = round(rate, 4)
        if rate > _INFRA_FAILURE_RATE_THRESHOLD or e["success"] == 0:
            failed = True

    return {
        "failed": failed,
        "failure_count": failure_count,
        "ops": ops,
        "rates": rates,
    }


def build_validation_evidence(telemetry_raw: Dict[str, Any], max_docs: int = 40,
                              max_chars_per_doc: int = 50000) -> Dict[str, Any]:
    """Project telemetry.documents_seen into evidence validators consume.

    Only documents_seen is arm-agnostic (recorded by every variant).
    """
    visited: List[Dict[str, str]] = []
    search_urls: List[str] = []
    for entry in (telemetry_raw or {}).get("documents_seen") or []:
        if not isinstance(entry, dict):
            continue
        doc = entry.get("document") or {}
        if not isinstance(doc, dict):
            continue
        source = entry.get("source")
        url = str(doc.get("url") or "").strip()
        if source == "visit":
            if len(visited) >= max_docs:
                continue
            visited.append({"url": url, "content": str(doc.get("content") or "")[:max_chars_per_doc]})
        elif source == "search" and url:
            search_urls.append(url)
    return {"visited": visited, "search_urls": search_urls}


def summarize_observability(result: Dict[str, Any], telemetry, model_name: str = "") -> Dict[str, Any]:
    """
    Summarize observability metrics from telemetry.
    :param result: Test result payload.
    :param telemetry: Telemetry session.
    :param model_name: Execution model name, used to price token usage in USD.
    :return: Observability summary.
    """
    output = result.get("output", {})
    final_text = ""
    if isinstance(output, dict):
        final_deliverable = output.get("final_deliverable", "")
        if isinstance(final_deliverable, dict):
            final_text = json.dumps(final_deliverable, ensure_ascii=True)
        elif isinstance(final_deliverable, str):
            final_text = final_deliverable
        elif isinstance(final_deliverable, list):
            final_text = json.dumps(final_deliverable, ensure_ascii=True)
        else:
            final_text = str(final_deliverable)
    
    final_chars = count_chars(final_text)
    final_words = count_words(final_text)
    
    llm_prompt_chars = 0
    llm_prompt_words = 0
    llm_completion_chars = 0
    llm_completion_words = 0
    llm_prompt_tokens = 0
    llm_completion_tokens = 0

    # Bug: this used to increment ``llm_calls`` once for EVERY ``connector_io`` event tagged
    # ``connector == "ConnectorLLM"`` -- but a real call always logs a MATCHED PAIR of those, one
    # "in" (before the call) and one "out" (after it): see ``ConnectorLLM.query_llm``'s own
    # ``_record_io`` calls (both its success and its caught-failure branch each log exactly one
    # "in" and one "out"), and ``langgraph_solver._record_io_parity``'s synthesized "in"/"out"
    # pair per assistant turn (covers BOTH LangGraph transports -- native tool-calling and the
    # emulated prompted-JSON loop -- neither of which touches ``ConnectorLLM`` directly). So the
    # old count was always exactly 2x the real number of calls. Verified live: `phi3_both_210`
    # (agent/idea_test_results/phi3_both_210_phi3:mini_langgraph_react_*.json) has 35 real model
    # turns and reported ``llm.calls: 70``.
    #
    # Counting only the "out" direction (every call logs exactly one, success or failure alike)
    # gives the true call count directly -- confirmed against ``telemetry.llm_usage`` (also one
    # entry per real call) on every transport this repo has: the native AgentIO/ConnectorLLM
    # engine, LangGraph's native tool-calling arm, and the emulated arm.
    #
    # (``telemetry.timings`` -- the ground truth for every OTHER count in this function -- was
    # considered here too, but it has two gaps this field can't afford: LangGraph's native
    # tool-calling transport records no per-call timing at all (only the synthesized
    # ``connector_io`` parity events exist for it, which would read as a hard 0, not "unknown"),
    # and a SEPARATE bug -- fixed at its source in ``agent_io.py``'s ``AgentIO.query_llm``, which
    # used to record its own redundant ``"llm_call"`` timing on top of ``ConnectorLLM``'s own --
    # left already-stored cells with a doubled ``"llm_call"`` timing count that this function
    # cannot retroactively de-duplicate. The ``connector_io`` "out" events are not affected by
    # either gap: ``ConnectorLLM`` logs them itself, once per call, independent of ``AgentIO``.)
    llm_calls = 0
    for entry in telemetry.events:
        if entry.get("event") != "connector_io":
            continue
        payload = entry.get("payload") or {}
        if payload.get("connector") != "ConnectorLLM":
            continue
        io_payload = payload.get("payload") or {}
        if payload.get("direction") == "out":
            llm_calls += 1
        # char/word totals are NOT part of this bug -- "in" carries prompt_chars/words, "out"
        # carries completion_chars/words, so summing both directions' payloads was already
        # additive, never doubled.
        llm_prompt_chars += int(io_payload.get("prompt_chars", 0))
        llm_prompt_words += int(io_payload.get("prompt_words", 0))
        llm_completion_chars += int(io_payload.get("completion_chars", 0))
        llm_completion_words += int(io_payload.get("completion_words", 0))

    for usage in telemetry.llm_usage:
        usage_payload = usage.get("usage") or {}
        llm_prompt_tokens += int(usage_payload.get("prompt_tokens", 0))
        llm_completion_tokens += int(usage_payload.get("completion_tokens", 0))
    
    chroma_store_chars = 0
    chroma_store_words = 0
    chroma_store_count = 0
    for entry in telemetry.chroma_stored:
        docs = entry.get("documents") or []
        chroma_store_count += len(docs)
        for doc in docs:
            chroma_store_chars += count_chars(doc)
            chroma_store_words += count_words(doc)
    
    chroma_retrieve_chars = 0
    chroma_retrieve_words = 0
    chroma_retrieve_count = 0
    for entry in telemetry.chroma_retrieved:
        docs = entry.get("documents") or []
        chroma_retrieve_count += len(docs)
        for doc in docs:
            chroma_retrieve_chars += count_chars(doc)
            chroma_retrieve_words += count_words(doc)
    
    # Bug: this was published as ``search["count"]`` -- read by several scripts (see the
    # observability-counters report) as though it were the number of SEARCH CALLS made. It is
    # not: it is the number of RESULT DOCUMENTS seen across every search call (one
    # ``record_document_seen`` per result item -- see ``AgentIO.search``'s loop over
    # ``results``), so it scales with each call's result count, not the call count. Verified
    # live against ``agent/idea_test_results/ledgerfinal01_*`` cells: every one of them has
    # this value at EXACTLY 6x the real number of ``search`` timing entries (6 results/call).
    # Kept, honestly named, as ``search["documents_seen"]`` below -- it is a real and useful
    # number, just not a call count.
    search_documents_seen_count = 0
    search_documents_seen_chars = 0
    search_documents_seen_words = 0
    visit_count = 0
    visit_chars = 0
    visit_words = 0

    for entry in telemetry.documents_seen:
        source = entry.get("source")
        document = entry.get("document") or {}
        if source == "search":
            search_documents_seen_count += 1
            text = " ".join(
                str(value) for value in [document.get("title"), document.get("url"), document.get("description")] if value
            )
            search_documents_seen_chars += count_chars(text)
            search_documents_seen_words += count_words(text)
        elif source == "visit":
            # Unlike search, one visit call records exactly one document (the page itself --
            # see ``AgentIO.visit``'s single ``record_document_seen`` call), so this count is
            # already 1:1 with real visit calls and needs no analogous fix.
            visit_count += 1
            content = document.get("content") or ""
            visit_chars += count_chars(content)
            visit_words += count_words(content)

    # The TRUE call count: one ``telemetry.record_timing(name="search", ...)`` per real search
    # operation (``AgentIO.search``), success or failure alike -- see ``timings_summary`` below,
    # built from the same ground truth. Computed directly here (not read off
    # ``timings_summary``) so this block does not depend on that dict's construction order.
    # Absent is legitimately 0 here, not unknown: ``telemetry.timings`` is the complete log of
    # every operation a run performed, so no "search" entries in it means no search call
    # happened, not that the count couldn't be computed.
    search_call_count = sum(1 for entry in telemetry.timings if entry.get("name") == "search")

    # Per-search provenance ("corpus"/"live"/"none"), when a backend records it -- currently
    # only ConnectorSearchCorpus (agent_io.search folds ``connector_search.provenance`` into
    # each "search" timing's payload as ``search_provenance``). Every other backend, and every
    # cell stored before this field existed, carries no such payload key at all: that is
    # "unknown", not zero, so the block below is only added when at least one timing actually
    # carries the key.
    search_live_fallbacks = 0
    search_corpus_hits = 0
    search_empty_results = 0
    search_provenance_seen = False
    for entry in telemetry.timings:
        if entry.get("name") != "search":
            continue
        provenance = (entry.get("payload") or {}).get("search_provenance")
        if provenance is None:
            continue
        search_provenance_seen = True
        if provenance == "corpus":
            search_corpus_hits += 1
        elif provenance == "live":
            search_live_fallbacks += 1
        elif provenance == "none":
            search_empty_results += 1

    fixture_hits = 0
    fixture_misses = 0
    for entry in telemetry.events:
        if entry.get("event") != "connector_io":
            continue
        io_payload = (entry.get("payload") or {}).get("payload") or {}
        fixture_flag = io_payload.get("fixture")
        if fixture_flag == "hit":
            fixture_hits += 1
        elif fixture_flag == "miss":
            fixture_misses += 1

    decisions = list(getattr(telemetry, "decisions", []) or [])
    grounded_flag = None
    for d in decisions:
        if isinstance(d, dict) and "grounded" in d:
            grounded_flag = d["grounded"]  # last grounded-bearing decision wins
    missing_reqs = []
    replans = 0
    if isinstance(result, dict):
        if "grounded" in result:
            grounded_flag = result.get("grounded")
        missing_reqs = result.get("missing_requirements", []) or []
        replans = int(result.get("grounding_replans", 0) or 0)
    stage_counts = Counter(d.get("stage") for d in decisions if isinstance(d, dict))

    step_confidence = None
    if isinstance(output, dict):
        raw_confidences = output.get("step_confidences")
        if isinstance(raw_confidences, list) and raw_confidences:
            seq = []
            for entry in raw_confidences:
                if not isinstance(entry, dict):
                    continue
                val = entry.get("confidence")
                try:
                    seq.append(float(val))
                except (TypeError, ValueError):
                    continue
            if seq:
                step_confidence = {
                    "count": len(seq),
                    "mean": round(sum(seq) / len(seq), 4),
                    "sequence": seq,
                    "trace": raw_confidences[:200],
                }

    timings_summary = {}
    timings_per_call = []
    for timing in telemetry.timings:
        name = timing.get("name", "unknown")
        duration = timing.get("duration", 0.0)
        success = timing.get("success", False)
        if name not in timings_summary:
            timings_summary[name] = {
                "count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "min_duration": float("inf"),
                "max_duration": 0.0,
                "success_count": 0,
                "error_count": 0,
            }
        entry = timings_summary[name]
        entry["count"] += 1
        entry["total_duration"] += duration
        entry["avg_duration"] = round(entry["total_duration"] / entry["count"], 4)
        entry["min_duration"] = min(entry["min_duration"], duration)
        entry["max_duration"] = max(entry["max_duration"], duration)
        if success:
            entry["success_count"] += 1
        else:
            entry["error_count"] += 1
        # ``t_start``/``t_end`` are session-relative seconds (see TelemetrySession.record_timing).
        # They are what makes "did these calls overlap?" answerable from the persisted result
        # JSON alone; absent on entries recorded before the field existed, so this stays a
        # conditional widening rather than a schema break for old artifacts.
        call_entry = {
            "name": name,
            "duration": round(duration, 4),
            "success": success,
        }
        for bound in ("t_start", "t_end"):
            if bound in timing:
                call_entry[bound] = round(float(timing[bound]), 4)
        timings_per_call.append(call_entry)
    for entry in timings_summary.values():
        if entry["min_duration"] == float("inf"):
            entry["min_duration"] = 0.0
        entry["total_duration"] = round(entry["total_duration"], 4)
        entry["min_duration"] = round(entry["min_duration"], 4)
        entry["max_duration"] = round(entry["max_duration"], 4)

    cost_estimated = False
    cost_prompt_tokens = llm_prompt_tokens
    cost_completion_tokens = llm_completion_tokens
    if llm_prompt_tokens == 0 and llm_completion_tokens == 0 and (llm_prompt_chars or llm_completion_chars):
        cost_estimated = True
        cost_prompt_tokens = int(llm_prompt_chars / _CHARS_PER_TOKEN)
        cost_completion_tokens = int(llm_completion_chars / _CHARS_PER_TOKEN)
    cost_usd = (
        estimate_cost(model_name, cost_prompt_tokens, cost_completion_tokens)
        if model_name
        else None
    )

    return {
        "final_output": {
            "chars": final_chars,
            "words": final_words,
            "kilobytes": round(final_chars / 1024, 2),
        },
        "llm": {
            "calls": llm_calls,
            "prompt": {
                "chars": llm_prompt_chars,
                "words": llm_prompt_words,
                "kilobytes": round(llm_prompt_chars / 1024, 2),
                "tokens": llm_prompt_tokens,
            },
            "completion": {
                "chars": llm_completion_chars,
                "words": llm_completion_words,
                "kilobytes": round(llm_completion_chars / 1024, 2),
                "tokens": llm_completion_tokens,
            },
            "total_tokens": llm_prompt_tokens + llm_completion_tokens,
        },
        "cost": {
            "model": model_name,
            "usd": cost_usd,
            "usd_str": format_cost(cost_usd),
            "estimated": cost_estimated,
            "prompt_tokens": cost_prompt_tokens,
            "completion_tokens": cost_completion_tokens,
        },
        "chroma": {
            "store": {
                "count": chroma_store_count,
                "chars": chroma_store_chars,
                "words": chroma_store_words,
                "kilobytes": round(chroma_store_chars / 1024, 2),
            },
            "retrieve": {
                "count": chroma_retrieve_count,
                "chars": chroma_retrieve_chars,
                "words": chroma_retrieve_words,
                "kilobytes": round(chroma_retrieve_chars / 1024, 2),
            },
        },
        "search": {
            # The real call count -- see ``search_call_count``'s definition above. This is the
            # field every existing reader (``scripts/bench_common.py``, ``analyze_dag_mechanism.py``,
            # ``replay_chain_failures.py``, ``capspec_report.py``, ``agent/app/testing/rubric.py``)
            # already reads as a call count; it was simply wrong before this fix, not renamed.
            "count": search_call_count,
            # Renamed from the old (wrong) ``count`` -- the number of RESULT DOCUMENTS seen
            # across every search call, not a call count. See the comment above
            # ``search_documents_seen_count`` for how this was verified.
            "documents_seen": {
                "count": search_documents_seen_count,
                "chars": search_documents_seen_chars,
                "words": search_documents_seen_words,
                "kilobytes": round(search_documents_seen_chars / 1024, 2),
            },
            **({
                "live_fallbacks": search_live_fallbacks,
                "corpus_hits": search_corpus_hits,
                "empty_results": search_empty_results,
            } if search_provenance_seen else {}),
        },
        "visit": {
            "count": visit_count,
            "chars": visit_chars,
            "words": visit_words,
            "kilobytes": round(visit_chars / 1024, 2),
        },
        "fixtures": {
            "hits": fixture_hits,
            "misses": fixture_misses,
            "miss_rate": round(fixture_misses / (fixture_hits + fixture_misses), 3)
            if (fixture_hits + fixture_misses) else 0.0,
        },
        "grounding": {
            "grounded": grounded_flag,
            "missing": missing_reqs,
            "replans": replans,
        },
        "decisions": {
            "count": len(decisions),
            "by_stage": dict(stage_counts),
            "trace": decisions[:200],
        },
        "timings": timings_summary,
        "timings_per_call": timings_per_call,
        "infra": _summarize_infra(telemetry.timings),
        "step_confidence": step_confidence,
        "events_count": len(telemetry.events),
    }


# Size drivers, and the reason ``telemetry_raw`` used to be dropped wholesale: page bodies and
# embedding vectors. None of them is needed to reconstruct what a run DID.
_TELEMETRY_BULK_FIELDS = ("documents_seen", "chroma_stored", "chroma_retrieved")

# Raw prompt/completion text, present on connector_io events only when LLM I/O capture is on.
# It belongs in the JSONL trace, not multiplied into every result JSON -- keeping it here would
# reintroduce exactly the bloat the old wholesale pop was defending against.
_TELEMETRY_RAW_TEXT_FIELDS = ("prompt_text", "completion_text", "messages")


def _strip_raw_text(event):
    """Drop captured prompt/completion text from one telemetry event."""
    if not isinstance(event, dict):
        return event
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return event
    inner = payload.get("payload")
    if not isinstance(inner, dict):
        return event
    if not any(field in inner for field in _TELEMETRY_RAW_TEXT_FIELDS):
        return event
    trimmed = {k: v for k, v in inner.items() if k not in _TELEMETRY_RAW_TEXT_FIELDS}
    return {**event, "payload": {**payload, "payload": trimmed}}


def slim_telemetry_raw(telemetry_raw):
    """Strip the bulk from a telemetry summary while keeping its forensic content.

    ``idea_test_runner`` popped the whole ``telemetry_raw`` block below verbosity 3, and every
    execution variant separately unlinked its JSONL trace on success -- so a default run kept
    no per-step record at all, in either place. That is what left a 96-cell baseline with zero
    recoverable traces.

    The size concern behind the pop was legitimate; it just applied to the wrong scope. Page
    bodies and embeddings live in :data:`_TELEMETRY_BULK_FIELDS`, while ``timings`` (carrying
    the call intervals that make concurrency provable), ``decisions`` and ``events`` are small
    and are exactly what offline analysis reads.

    :param telemetry_raw: The session summary, or ``None``.
    :returns: A slimmed copy, or the input unchanged when it is not a dict.
    """
    if not isinstance(telemetry_raw, dict):
        return telemetry_raw
    slim = {k: v for k, v in telemetry_raw.items() if k not in _TELEMETRY_BULK_FIELDS}
    events = slim.get("events")
    if isinstance(events, list):
        slim["events"] = [_strip_raw_text(event) for event in events]
    return slim
