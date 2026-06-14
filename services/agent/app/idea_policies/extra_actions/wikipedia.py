"""wikipedia_summary action — fetch an article extract via Wikipedia's REST API.

No API key required. Endpoint:
  https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}
"""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import quote

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, fetch_json, ok


class WikipediaSummaryAction(LeafAction):
    """Look up a Wikipedia article summary by title.

    Reads from node details:
      - `title` (str): article title (required).
      - `lang` (str): language code, default "en".

    Returns `{title, extract, url, thumbnail, lang}`.
    """

    name = "wikipedia_summary"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        title = details.get("title")
        if not isinstance(title, str) or not title.strip():
            return fail(self.name, "missing 'title' detail (str)")
        lang = (details.get("lang") or "en").strip()
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}"
        resp = await fetch_json(io, url)
        if not resp.get("_ok"):
            return fail(self.name, resp.get("error", "fetch failed"), retryable=True)
        data = resp["data"]
        return ok(
            self.name,
            title=data.get("title"),
            extract=data.get("extract") or "",
            url=(data.get("content_urls") or {}).get("desktop", {}).get("page"),
            thumbnail=(data.get("thumbnail") or {}).get("source"),
            lang=lang,
        )
