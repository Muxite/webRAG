"""hacker_news_top action — fetch the current top stories from HN.

No API key required. Endpoints:
  https://hacker-news.firebaseio.com/v0/topstories.json
  https://hacker-news.firebaseio.com/v0/item/{id}.json

Story metadata is fetched concurrently for the top N IDs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, fetch_json, ok


class HackerNewsTopAction(LeafAction):
    """Return the current top N Hacker News stories.

    Reads from node details:
      - `count` (int): how many top stories, default 10, max 30.

    Returns `{count, stories: [{id, title, url, score, by, descendants,
    hn_url}]}`.
    """

    name = "hacker_news_top"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        count = max(1, min(int(details.get("count") or 10), 30))

        top_resp = await fetch_json(io, "https://hacker-news.firebaseio.com/v0/topstories.json")
        if not top_resp.get("_ok"):
            return fail(self.name, top_resp.get("error", "top stories fetch failed"), retryable=True)
        ids = top_resp["data"][:count] if isinstance(top_resp["data"], list) else []
        if not ids:
            return fail(self.name, "no story IDs returned")

        async def _fetch_story(sid: int) -> Dict[str, Any] | None:
            item = await fetch_json(io, f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
            if not item.get("_ok"):
                return None
            d = item["data"]
            if not isinstance(d, dict):
                return None
            return {
                "id": d.get("id"),
                "title": d.get("title") or "",
                "url": d.get("url") or "",
                "score": int(d.get("score") or 0),
                "by": d.get("by"),
                "descendants": int(d.get("descendants") or 0),
                "hn_url": f"https://news.ycombinator.com/item?id={d.get('id')}",
            }

        stories_raw = await asyncio.gather(*(_fetch_story(sid) for sid in ids))
        stories: List[Dict[str, Any]] = [s for s in stories_raw if s]
        return ok(self.name, count=len(stories), stories=stories)
