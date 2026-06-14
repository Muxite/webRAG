"""url_metadata action — extract OpenGraph / Twitter Card / title from a URL.

Lightweight alternative to a full `visit`: just fetch the head/top of the
HTML and pull out the metadata tags. Useful for link previews and quick
relevance checks before deciding whether to visit a page properly.

Pulls in `bs4.BeautifulSoup` which is already a transitive dep of the
agent app (see `actions.py`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, ok


def _meta(soup, *, name: Optional[str] = None, prop: Optional[str] = None) -> Optional[str]:
    selector = {"name": name} if name else {"property": prop}
    tag = soup.find("meta", selector)
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None


class UrlMetadataAction(LeafAction):
    """Fetch a URL and extract <title>, OG, and Twitter Card metadata.

    Reads from node details:
      - `url` (str): the URL (required).

    Returns `{url, title, description, image, site_name, og: {...},
    twitter: {...}}`.
    """

    name = "url_metadata"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        url = details.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return fail(self.name, "missing or invalid 'url' detail")
        try:
            body = await io.fetch_url(url)
        except Exception as exc:  # noqa: BLE001
            return fail(self.name, f"fetch failed: {exc}", retryable=True)
        if not body:
            return fail(self.name, "empty response body", retryable=True)

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(body[:200_000], "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        og = {
            "title": _meta(soup, prop="og:title"),
            "description": _meta(soup, prop="og:description"),
            "image": _meta(soup, prop="og:image"),
            "site_name": _meta(soup, prop="og:site_name"),
            "type": _meta(soup, prop="og:type"),
            "url": _meta(soup, prop="og:url"),
        }
        twitter = {
            "card": _meta(soup, name="twitter:card"),
            "title": _meta(soup, name="twitter:title"),
            "description": _meta(soup, name="twitter:description"),
            "image": _meta(soup, name="twitter:image"),
        }
        description = (
            og.get("description")
            or twitter.get("description")
            or _meta(soup, name="description")
            or ""
        )
        return ok(
            self.name,
            url=url,
            title=og.get("title") or title,
            description=description,
            image=og.get("image") or twitter.get("image"),
            site_name=og.get("site_name"),
            og={k: v for k, v in og.items() if v},
            twitter={k: v for k, v in twitter.items() if v},
        )
