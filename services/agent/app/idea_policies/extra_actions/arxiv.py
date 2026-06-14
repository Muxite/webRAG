"""arxiv_search action — query arXiv's free Atom API for papers.

No API key required. Endpoint:
  https://export.arxiv.org/api/query?search_query=...&max_results=...
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, ok

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def _strip_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


class ArxivSearchAction(LeafAction):
    """Search arXiv papers by query string.

    Reads from node details:
      - `query` (str): arXiv search expression (required). Plain terms work;
        `cat:cs.AI`, `au:hinton`, etc. supported per arXiv docs.
      - `max_results` (int): default 5, capped at 20.

    Returns `{query, count, papers: [{title, authors, summary, url, published}]}`.
    """

    name = "arxiv_search"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        query = details.get("query")
        if not isinstance(query, str) or not query.strip():
            return fail(self.name, "missing 'query' detail (str)")
        max_results = max(1, min(int(details.get("max_results") or 5), 20))
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={quote_plus(query)}&max_results={max_results}"
            "&sortBy=relevance&sortOrder=descending"
        )
        try:
            body = await io.fetch_url(url)
        except Exception as exc:  # noqa: BLE001
            return fail(self.name, f"fetch failed: {exc}", retryable=True)
        if not body:
            return fail(self.name, "empty response", retryable=True)
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            return fail(self.name, f"XML parse failed: {exc}")
        papers: List[Dict[str, Any]] = []
        for entry in root.findall("a:entry", _ATOM_NS):
            authors = [
                _strip_ws(name_el.text)
                for name_el in entry.findall("a:author/a:name", _ATOM_NS)
            ]
            link = ""
            for link_el in entry.findall("a:link", _ATOM_NS):
                if link_el.get("rel") in (None, "alternate"):
                    link = link_el.get("href") or ""
                    break
            papers.append({
                "title": _strip_ws(entry.findtext("a:title", default="", namespaces=_ATOM_NS)),
                "summary": _strip_ws(entry.findtext("a:summary", default="", namespaces=_ATOM_NS)),
                "authors": authors,
                "url": link,
                "published": _strip_ws(entry.findtext("a:published", default="", namespaces=_ATOM_NS)),
            })
        return ok(self.name, query=query, count=len(papers), papers=papers)
