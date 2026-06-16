"""
Testing utilities.
"""

import json
from collections import Counter
from typing import Dict, Any

from agent.app.idea_test_utils import count_words, count_chars
from agent.app.model_costs import estimate_cost, format_cost

# Rough chars-per-token used only when the provider omits a usage block so we can
# still place a (flagged) cost estimate on the chart instead of $0.
_CHARS_PER_TOKEN = 4.0


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
    llm_calls = 0
    
    for entry in telemetry.events:
        if entry.get("event") != "connector_io":
            continue
        payload = entry.get("payload") or {}
        if payload.get("connector") != "ConnectorLLM":
            continue
        io_payload = payload.get("payload") or {}
        llm_prompt_chars += int(io_payload.get("prompt_chars", 0))
        llm_prompt_words += int(io_payload.get("prompt_words", 0))
        llm_completion_chars += int(io_payload.get("completion_chars", 0))
        llm_completion_words += int(io_payload.get("completion_words", 0))
        llm_calls += 1
    
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
    
    search_count = 0
    search_chars = 0
    search_words = 0
    visit_count = 0
    visit_chars = 0
    visit_words = 0
    
    for entry in telemetry.documents_seen:
        source = entry.get("source")
        document = entry.get("document") or {}
        if source == "search":
            search_count += 1
            text = " ".join(
                str(value) for value in [document.get("title"), document.get("url"), document.get("description")] if value
            )
            search_chars += count_chars(text)
            search_words += count_words(text)
        elif source == "visit":
            visit_count += 1
            content = document.get("content") or ""
            visit_chars += count_chars(content)
            visit_words += count_words(content)
    
    # Fixture hit/miss counts: a non-zero miss rate means a model saw evidence the
    # prewarm did not cover, which is the asymmetry strict replay is meant to remove.
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

    # Decision trace + grounding verdict (thought-process observability).
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
        timings_per_call.append({
            "name": name,
            "duration": round(duration, 4),
            "success": success,
        })
    for entry in timings_summary.values():
        if entry["min_duration"] == float("inf"):
            entry["min_duration"] = 0.0
        entry["total_duration"] = round(entry["total_duration"], 4)
        entry["min_duration"] = round(entry["min_duration"], 4)
        entry["max_duration"] = round(entry["max_duration"], 4)

    # USD cost: price the reported token usage when available, otherwise fall back
    # to a chars/token approximation flagged as estimated so headline numbers aren't
    # silently understated when a provider (e.g. some OpenRouter slugs) omits usage.
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
            "count": search_count,
            "chars": search_chars,
            "words": search_words,
            "kilobytes": round(search_chars / 1024, 2),
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
        "events_count": len(telemetry.events),
    }
