"""Visit semantic-dedup helpers extracted from :class:`IdeaDagEngine`.

Behavior-preserving helpers that fold URL-less planner visit candidates into a
URL-bearing sibling when the sibling's path slug matches the URL-less node's
text. Weak planners often emit a visit like "Visit Axolotl Wikipedia page"
without a URL alongside a hook-injected sibling carrying the explicit URL;
without this pass the engine would dispatch both and the URL-less one fails.
State is passed in explicitly so the engine stays the orchestrator.
"""

from __future__ import annotations

import logging
from typing import List
from urllib.parse import unquote, urlparse

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.idea_policies import DetailKey
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.base import IdeaActionType


def url_slug_tokens(url: str) -> List[str]:
    """Pull the trailing path slug from a URL and return its lowercased tokens."""
    if not isinstance(url, str) or not url:
        return []
    try:
        parsed = urlparse(url)
    except ValueError:
        return []
    path = (parsed.path or "").rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    if not slug:
        return []
    slug = unquote(slug).replace("_", " ").replace("-", " ").lower()
    # Strip wrapping parens like "Pando_(tree)" → "pando tree"
    slug = slug.replace("(", " ").replace(")", " ")
    tokens = [t for t in slug.split() if len(t) >= 3]
    return tokens


def semantic_dedup_visits(
    graph: IdeaDag,
    parent_id: str,
    step_index: int,
    require_hook_source: bool,
    logger: logging.Logger,
) -> None:
    """Fold URL-less planner visit candidates into URL-bearing siblings.

    Weak planners (notably Gemini 2.5 Flash) often emit visit candidates
    with a title like "Visit Axolotl Wikipedia page" but no `url` or
    `link_idea` detail. The mandate-injection hook then adds a sibling
    with the explicit URL. The engine sees them as distinct, dispatches
    both, and the URL-less one fails — or worse, in sequential mode the
    evaluator may pick the URL-less candidate, prune the correct sibling,
    and execute the wrong thing.

    This pre-execution pass detects each URL-less visit, scans the
    URL-bearing siblings for one whose path slug appears in the URL-less
    node's title/goal text, and marks the URL-less node SKIPPED with a
    diagnostic reason. The URL-bearing sibling runs unopposed.

    No-op when no visit candidates lack URLs, or when no slug matches.
    """
    parent = graph.get_node(parent_id)
    if not parent or not parent.children:
        return

    url_bearing: List[tuple] = []  # (node_id, url, slug_tokens)
    url_less: List[tuple] = []     # (node_id, node, search_text)

    for child_id in parent.children:
        child = graph.get_node(child_id)
        if not child or child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.SKIPPED, IdeaNodeStatus.FAILED):
            continue
        action = NodeDetailsExtractor.get_action(child.details)
        if action != IdeaActionType.VISIT.value:
            continue
        url = (
            child.details.get(DetailKey.URL.value)
            or child.details.get("optional_url")
            or child.details.get("url")
        )
        link_idea = child.details.get("link_idea")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            tokens = url_slug_tokens(url)
            if tokens:
                url_bearing.append((child_id, url, tokens))
        elif not url and not link_idea:
            title = (child.title or "").lower()
            goal = (child.details.get("goal") or "").lower()
            parent_goal = (child.details.get("parent_goal") or "").lower()
            search_text = " ".join([title, goal, parent_goal])
            url_less.append((child_id, child, search_text))

    if not url_bearing or not url_less:
        return

    # Hook-only gate: only fold against URL-bearing siblings that came
    # from MandateUrlInjectionHook (i.e. the mandate text literally
    # contained the URL). This protects chain-of-links tests where
    # planner-generated URL-less candidates were intended for sequential
    # link discovery rather than as duplicates of the literal URL.
    def _is_hook_injected(source_id: str) -> bool:
        source = graph.get_node(source_id)
        if not source:
            return False
        justification = (source.details.get(DetailKey.JUSTIFICATION.value) or "")
        return justification.startswith("Mandate requires visiting")

    folded = 0
    gate_blocked = 0
    for nid, node, search_text in url_less:
        for source_id, source_url, tokens in url_bearing:
            # Match: every slug token must appear as a substring in the
            # URL-less node's text. Slugs like "Axolotl" → ["axolotl"]
            # match "Visit Axolotl Wikipedia page". Multi-word slugs like
            # "Pando tree" require both "pando" and "tree" present.
            if not all(tok in search_text for tok in tokens):
                continue
            if require_hook_source and not _is_hook_injected(source_id):
                gate_blocked += 1
                continue
            node.status = IdeaNodeStatus.SKIPPED
            node.details[DetailKey.ACTION_ERROR.value] = (
                f"Semantic dedup: URL-less visit folded into sibling {source_id} "
                f"(URL: {source_url})"
            )
            node.details["__semantic_dedup_source"] = source_id
            folded += 1
            break

    if gate_blocked:
        logger.info(
            f"[STEP {step_index}] SEMANTIC_DEDUP: gate blocked {gate_blocked} fold(s) "
            f"(URL-bearing sibling was not from MandateUrlInjectionHook)"
        )

    if folded:
        logger.info(
            f"[STEP {step_index}] SEMANTIC_DEDUP: folded {folded} URL-less visit "
            f"candidates into URL-bearing siblings (under parent {parent_id[:8]})"
        )
