"""pypi_package_info action — fetch package metadata from PyPI's JSON API.

No API key required. Endpoint:
  https://pypi.org/pypi/{package}/json
"""

from __future__ import annotations

from typing import Any, Dict

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, fetch_json, ok


class PypiPackageInfoAction(LeafAction):
    """Read PyPI metadata for a package.

    Reads from node details:
      - `package` (str): the package name (required).

    Returns `{name, version, summary, author, home_page, license,
    requires_python, releases_count, project_urls}`.
    """

    name = "pypi_package_info"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        package = details.get("package")
        if not isinstance(package, str) or not package.strip():
            return fail(self.name, "missing 'package' detail (str)")

        resp = await fetch_json(io, f"https://pypi.org/pypi/{package.strip()}/json")
        if not resp.get("_ok"):
            return fail(self.name, resp.get("error", "fetch failed"), retryable=True)
        data = resp["data"]
        if not isinstance(data, dict):
            return fail(self.name, "unexpected response shape")
        info = data.get("info") or {}
        releases = data.get("releases") or {}
        return ok(
            self.name,
            name=info.get("name") or package,
            version=info.get("version"),
            summary=info.get("summary") or "",
            description_excerpt=(info.get("description") or "")[:1000],
            author=info.get("author"),
            home_page=info.get("home_page"),
            license=info.get("license"),
            requires_python=info.get("requires_python"),
            releases_count=len(releases),
            project_urls=info.get("project_urls") or {},
        )
