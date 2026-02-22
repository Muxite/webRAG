"""
Testing utilities.
"""

import json
from typing import Dict, Any

from agent.app.idea_test_utils import count_words, count_chars


def summarize_observability(result: Dict[str, Any], telemetry) -> Dict[str, Any]:
    """
    Summarize observability metrics from telemetry.
    :param result: Test result payload.
    :param telemetry: Telemetry session.
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
    
    timings_summary = {}
    for timing in telemetry.timings:
        name = timing.get("name", "unknown")
        if name not in timings_summary:
            timings_summary[name] = {
                "count": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "error_count": 0,
            }
        timings_summary[name]["count"] += 1
        timings_summary[name]["total_duration"] += timing.get("duration", 0.0)
        if timing.get("success"):
            timings_summary[name]["success_count"] += 1
        else:
            timings_summary[name]["error_count"] += 1
    
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
        "timings": timings_summary,
        "events_count": len(telemetry.events),
    }
