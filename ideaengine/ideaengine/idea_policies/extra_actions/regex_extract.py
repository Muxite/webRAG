"""regex_extract action — pull regex matches out of arbitrary text."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, ok


class RegexExtractAction(LeafAction):
    """Extract regex matches from text. Useful chained after a visit/think.

    Reads from node details:
      - `pattern`: Python regex string (required).
      - `text`: text to search (required).
      - `flags`: optional list of flag names ("i", "m", "s", "x").
      - `max_matches`: optional cap on returned matches (default 50).

    Returns: `{matches: [str|tuple], count: int}`.
    """

    name = "regex_extract"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        pattern = details.get("pattern")
        text = details.get("text")
        if not isinstance(pattern, str) or not pattern:
            return fail(self.name, "missing 'pattern' detail (str)")
        if not isinstance(text, str):
            return fail(self.name, "missing 'text' detail (str)")
        max_matches = int(details.get("max_matches") or 50)
        flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
        flag_value = 0
        for flag in details.get("flags") or []:
            flag_value |= flag_map.get(str(flag).lower(), 0)
        try:
            compiled = re.compile(pattern, flag_value)
        except re.error as exc:
            return fail(self.name, f"invalid regex: {exc}", error_type="InvalidPattern")
        matches = compiled.findall(text)[:max_matches]
        return ok(self.name, matches=matches, count=len(matches), pattern=pattern)
