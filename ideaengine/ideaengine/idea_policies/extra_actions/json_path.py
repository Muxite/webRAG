"""json_path action — dotted-path lookup against a JSON value.

Implements a tiny subset of JSONPath: dotted segments + `[index]` accessors.
`a.b.c[0].d`. No filters, no wildcards. Good enough for navigating API
responses without dragging in a jsonpath dependency.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, ok


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def _tokenize_path(path: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    for match in _PATH_TOKEN.finditer(path):
        key, idx = match.group(1), match.group(2)
        if key is not None:
            tokens.append(("key", key))
        elif idx is not None:
            tokens.append(("idx", idx))
    return tokens


def resolve_json_path(data: Any, path: str) -> Tuple[bool, Any]:
    """Walk `data` per `path`; return `(found, value)`."""
    if not path or path == "$":
        return True, data
    cursor = data
    for kind, token in _tokenize_path(path):
        if kind == "key":
            if isinstance(cursor, dict) and token in cursor:
                cursor = cursor[token]
            else:
                return False, None
        else:  # idx
            try:
                i = int(token)
            except ValueError:
                return False, None
            if isinstance(cursor, list) and -len(cursor) <= i < len(cursor):
                cursor = cursor[i]
            else:
                return False, None
    return True, cursor


class JsonPathAction(LeafAction):
    """Look up one or more dotted paths inside a JSON document.

    Reads from node details:
      - `json` (dict/list/str): the document. If str, parsed as JSON first.
      - `path` (str): single path expression.
      - `paths` (list[str]): or many paths at once; returns a dict of results.

    Returns:
      - Single path: `{path, value, found}`
      - Many paths:  `{values: {path: value}, missing: [paths_not_found]}`
    """

    name = "json_path"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        raw = details.get("json")
        if raw is None:
            return fail(self.name, "missing 'json' detail")
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                return fail(self.name, f"json parse failed: {exc}", error_type="InvalidJson")
        else:
            data = raw

        path = details.get("path")
        paths = details.get("paths")
        if path:
            found, value = resolve_json_path(data, path)
            return ok(self.name, path=path, value=value, found=found)
        if paths and isinstance(paths, list):
            values: Dict[str, Any] = {}
            missing: List[str] = []
            for p in paths:
                found, value = resolve_json_path(data, str(p))
                if found:
                    values[str(p)] = value
                else:
                    missing.append(str(p))
            return ok(self.name, values=values, missing=missing)
        return fail(self.name, "provide 'path' (str) or 'paths' (list)")
