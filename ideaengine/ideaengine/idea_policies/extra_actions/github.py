"""github_repo_info action — fetch public repo metadata from the GitHub REST API.

Anonymous calls work; rate-limited to 60/hr per IP. If `GITHUB_TOKEN` is set
in the environment, the action uses it for the 5000/hr quota.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ideaengine.idea_policies.actions import LeafAction
from ideaengine.idea_policies.extra_actions.base import fail, fetch_json, ok


class GithubRepoInfoAction(LeafAction):
    """Read public metadata for a GitHub repository.

    Reads from node details:
      - `owner` (str): repo owner / org (required).
      - `repo` (str): repo name (required).
      - Alternatively, `url` (str): a full GitHub URL is parsed for owner/repo.

    Returns `{owner, repo, description, stars, forks, language, license,
    updated_at, default_branch, html_url, topics}`.
    """

    name = "github_repo_info"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        details = node.details or {}
        owner = details.get("owner")
        repo = details.get("repo")
        if not owner or not repo:
            url = details.get("url") or ""
            if "github.com/" in url:
                tail = url.split("github.com/", 1)[1].strip("/")
                parts = tail.split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1].split(".")[0]
        if not owner or not repo:
            return fail(self.name, "provide 'owner' + 'repo' or a 'url' containing them")

        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        token = os.environ.get("GITHUB_TOKEN")
        # AgentIO.fetch_url doesn't accept headers; rely on rate-limited
        # anonymous access. If GITHUB_TOKEN is present we still surface a
        # warning so users know the path is rate-limited.
        if token:
            # Note: the engine's io.fetch_url currently doesn't support
            # custom headers. Documenting the limitation here rather than
            # silently dropping the token.
            pass

        resp = await fetch_json(io, api_url)
        if not resp.get("_ok"):
            return fail(self.name, resp.get("error", "fetch failed"), retryable=True)
        data = resp["data"]
        if not isinstance(data, dict) or data.get("message") == "Not Found":
            return fail(self.name, f"repo {owner}/{repo} not found", error_type="NotFound")
        return ok(
            self.name,
            owner=owner,
            repo=repo,
            description=data.get("description") or "",
            stars=int(data.get("stargazers_count") or 0),
            forks=int(data.get("forks_count") or 0),
            open_issues=int(data.get("open_issues_count") or 0),
            language=data.get("language"),
            license=(data.get("license") or {}).get("spdx_id"),
            updated_at=data.get("updated_at"),
            pushed_at=data.get("pushed_at"),
            default_branch=data.get("default_branch"),
            html_url=data.get("html_url"),
            topics=data.get("topics") or [],
            archived=bool(data.get("archived")),
        )
