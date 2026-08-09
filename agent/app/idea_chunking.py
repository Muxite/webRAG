"""Document chunking helpers extracted from :class:`IdeaDagEngine`.

Self-contained, behavior-preserving functions that decide whether a visited
document is large enough to split, create per-chunk SEARCH sub-problems under
the originating visit node, slice the raw text into overlapping windows, and
detect ordering dependencies between chunk siblings. All state is passed in
explicitly so the engine stays the sole stateful orchestrator.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.idea_policies import DetailKey


def should_chunk_document(
    graph: IdeaDag,
    node: IdeaNode,
    chunk_threshold: int,
    logger: logging.Logger,
) -> bool:
    from agent.app.idea_policies.action_constants import (
        NodeDetailsExtractor,
        ActionResultKey,
        ActionResultExtractor,
    )
    from agent.app.idea_policies.base import IdeaActionType

    action = NodeDetailsExtractor.get_action(node.details)
    if action != IdeaActionType.VISIT.value:
        return False

    result = node.details.get(DetailKey.ACTION_RESULT.value)
    if not isinstance(result, dict):
        return False

    if not ActionResultExtractor.is_success(result):
        return False

    content_total = result.get(ActionResultKey.CONTENT_TOTAL_CHARS.value) or 0

    if content_total > chunk_threshold:
        logger.info(f"[CHUNKING] Document size {content_total} chars exceeds threshold {chunk_threshold}")
        return True

    return False


async def create_chunk_subproblems(
    graph: IdeaDag,
    visit_node: IdeaNode,
    chunk_size: int,
    chunk_overlap: int,
    logger: logging.Logger,
) -> Optional[List[str]]:
    from agent.app.idea_policies.action_constants import ActionResultKey
    from agent.app.idea_policies.base import IdeaActionType

    result = visit_node.details.get(DetailKey.ACTION_RESULT.value)
    if not isinstance(result, dict):
        return None

    content_full = result.get(ActionResultKey.CONTENT_FULL.value) or ""
    if not content_full or not isinstance(content_full, str):
        return None

    chunks = chunk_text(content_full, chunk_size, chunk_overlap)
    if len(chunks) <= 1:
        return None

    original_goal = visit_node.details.get(DetailKey.GOAL.value) or visit_node.details.get(DetailKey.INTENT.value) or visit_node.title
    url = result.get(ActionResultKey.URL.value) or ""

    chunk_nodes = []
    for i, chunk in enumerate(chunks):
        chunk_title = f"Search chunk {i+1}/{len(chunks)} of {visit_node.title[:40]}..."
        chunk_details = {
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: original_goal,
            DetailKey.CHUNK_INDEX.value: i,
            DetailKey.TOTAL_CHUNKS.value: len(chunks),
            DetailKey.CHUNK_CONTENT.value: chunk,
            DetailKey.ORIGINAL_GOAL.value: original_goal,
            DetailKey.GOAL.value: f"Find information in chunk {i+1} relevant to: {original_goal}",
            DetailKey.REQUIRES_DATA.value: {
                "type": "chunk_from_visit",
                "source_node_id": visit_node.node_id
            },
            DetailKey.JUSTIFICATION.value: f"Document too large ({len(content_full)} chars) - searching chunk {i+1}/{len(chunks)} for: {original_goal}",
        }

        chunk_node = graph.add_child(
            parent_id=visit_node.node_id,
            title=chunk_title,
            details=chunk_details,
            status=IdeaNodeStatus.PENDING,
        )
        chunk_nodes.append(chunk_node.node_id)

    logger.info(f"[CHUNKING] Created {len(chunk_nodes)} chunk sub-problems for document from {url[:60]}...")
    return chunk_nodes


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end - chunk_overlap)

    return chunks


def detect_chunk_dependencies(graph: IdeaDag, candidate_ids: List[str]) -> bool:
    chunk_nodes = {}
    for node_id in candidate_ids:
        node = graph.get_node(node_id)
        if not node:
            continue

        chunk_index = node.details.get(DetailKey.CHUNK_INDEX.value)
        total_chunks = node.details.get(DetailKey.TOTAL_CHUNKS.value)
        source_node_id = None
        requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
        if isinstance(requires_data, dict):
            source_node_id = requires_data.get("source_node_id")

        if chunk_index is not None and total_chunks is not None and source_node_id:
            if source_node_id not in chunk_nodes:
                chunk_nodes[source_node_id] = []
            chunk_nodes[source_node_id].append((node_id, chunk_index))

    for source_id, chunks in chunk_nodes.items():
        if len(chunks) > 1:
            chunks_sorted = sorted(chunks, key=lambda x: x[1] or 0)
            for i, (node_id, idx) in enumerate(chunks_sorted):
                if i > 0 and idx is not None and chunks_sorted[i-1][1] is not None:
                    if idx <= chunks_sorted[i-1][1]:
                        return True

    return False


def is_chunk_node(graph: IdeaDag, node_id: str) -> bool:
    node = graph.get_node(node_id)
    if not node:
        return False

    return node.details.get(DetailKey.CHUNK_CONTENT.value) is not None
