"""Web-read tool (read-only, v1). SEARCH registers URL entities; READ fetches+extracts
one by its id. Decoupled from the network via injected callables so the logic is
unit-testable with fakes; runtime factories wire self-hosted SearXNG + httpx fetch +
HTML→text extraction (optionally IdeaEngine's observation.clean).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .base import ToolContext, ToolResult

SearchFn = Callable[[str, int], List[Dict[str, Any]]]   # (query, k) -> [{title,url,snippet}]
FetchFn = Callable[[str], str]                          # url -> html/text
ExtractFn = Callable[[str], str]                        # html -> clean text


class WebReadTool:
    name = "web"

    def __init__(self, search_fn: SearchFn, fetch_fn: FetchFn,
                 extract_fn: Optional[ExtractFn] = None, max_chars: int = 6000) -> None:
        self.search_fn = search_fn
        self.fetch_fn = fetch_fn
        self.extract_fn = extract_fn or _regex_extract
        self.max_chars = max_chars

    def execute(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        op = request.get("op")
        if op == "search":
            query = str(request.get("query", "")).strip()
            k = int(request.get("k", 5) or 5)
            try:
                results = self.search_fn(query, k) or []
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, f"search failed: {exc}", error="search_error")
            new_entities = [("url", (r.get("title") or r.get("url") or "?")[:60], r.get("url"))
                            for r in results if r.get("url")]
            preview = "; ".join((r.get("title") or r.get("url") or "?")[:40] for r in results[:5])
            return ToolResult(True, f"search({query!r}): {len(results)} results — {preview or 'none'}",
                              data={"results": results}, new_entities=new_entities)
        if op == "read":
            url = request.get("url")
            if not url:
                return ToolResult(False, "read: no url", error="missing")
            try:
                html = self.fetch_fn(url) or ""
                text = (self.extract_fn(html) or "")[: self.max_chars]
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, f"read failed: {exc}", error="fetch_error")
            head = text[:200].replace("\n", " ")
            return ToolResult(True, f"read {str(url)[:50]}: {len(text)} chars — “{head[:120]}”",
                              data={"url": url, "text": text})
        return ToolResult(False, f"unknown web op {op!r}", error="unknown_op")


# --- runtime adapters (fully local: SearXNG + httpx + HTML→text) ----------------
def _regex_extract(html: str) -> str:
    txt = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html or "")
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def searxng_search_fn(base_url: str) -> SearchFn:
    """Self-hosted SearXNG JSON search — the fully-local search backend."""
    def _search(query: str, k: int = 5) -> List[Dict[str, Any]]:
        import httpx
        r = httpx.get(f"{base_url.rstrip('/')}/search",
                      params={"q": query, "format": "json"}, timeout=30)
        r.raise_for_status()
        rows = (r.json().get("results") or [])[:k]
        return [{"title": d.get("title"), "url": d.get("url"), "snippet": d.get("content")}
                for d in rows]
    return _search


def httpx_fetch_fn() -> FetchFn:
    def _fetch(url: str) -> str:
        import httpx
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 (local-agent)"},
                      timeout=30, follow_redirects=True)
        return r.text
    return _fetch


def observation_extract_fn() -> ExtractFn:
    """Reuse IdeaEngine's Wikipedia-tuned BeautifulSoup cleaner if importable; else regex-strip.
    Puts services/ on the path so the reuse actually happens (the regex fallback leaks raw
    Wikipedia template/JSON markup, which reads as garbage to a weak model)."""
    try:
        from agent.app.observation import clean_operation
    except Exception:
        import sys
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[3]      # webRAG/
        for p in (root / "services", root / "services" / "agent"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            from agent.app.observation import clean_operation
        except Exception:
            return _regex_extract

    def _extract(html: str) -> str:
        try:
            out = clean_operation(html)
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, dict):
                out = out.get("text") or out.get("content") or ""
            return str(out)
        except Exception:
            return _regex_extract(html)
    return _extract
