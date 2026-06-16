"""Pure node-state classification and execution-readiness scans from
:class:`IdeaDagEngine`.

Behavior-preserving helpers that answer "is this node ready to act?",
"which nodes are still pending execution?", "what is the globally best
executable node?", and "give me a JSON-safe copy of an action result". None
of these touch engine I/O, memory, telemetry or the checkpointer — they are
pure functions of the graph plus the current step index and score thresholds,
so they live outside the stateful orchestrator. The engine keeps thin
delegating wrappers (mirroring the ``idea_sequencing`` / ``idea_chunking``
extractions) so call sites and external behavior are unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies import DetailKey
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.idea_policies.action_constants import NodeDetailsExtractor


def is_action_ready(node: IdeaNode, step_index: int) -> bool:
    """Whether ``node``'s action may run at ``step_index``.

    A node is not ready if it has already terminally failed/skipped, or if it
    is still inside a retry cooldown window.
    """
    if node.status in (IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
        return False
    cooldown = node.details.get(DetailKey.ACTION_COOLDOWN_UNTIL.value)
    if isinstance(cooldown, int) and step_index < cooldown:
        return False
    return True


def get_pending_executable_nodes(graph: IdeaDag) -> List[IdeaNode]:
    """Action-bearing (non-merge) nodes that have neither a result nor a
    terminal status — i.e. work the engine still owes before it can finalize.
    """
    pending: List[IdeaNode] = []
    for node in graph.iter_depth_first():
        action = NodeDetailsExtractor.get_action(node.details)
        if action and not NodeDetailsExtractor.is_merge_action(node.details):
            has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
            if not has_result and node.status not in (
                IdeaNodeStatus.DONE,
                IdeaNodeStatus.FAILED,
                IdeaNodeStatus.SKIPPED,
            ):
                pending.append(node)
    return pending


def select_best_global(
    graph: IdeaDag,
    min_score: float,
    allow_unscored: bool,
    step_index: int,
) -> Tuple[Optional[Any], Optional[str]]:
    """Highest-scored executable node anywhere in the graph plus its parent id.

    Skips the root, already-executed DONE nodes, nodes not yet ready
    (cooldown), and nodes below ``min_score`` (or unscored when not allowed).
    """
    best = None
    for node in graph.iter_depth_first():
        if node.parent_id is None:
            continue
        if (
            node.details.get(DetailKey.ACTION_RESULT.value) is not None
            and node.status == IdeaNodeStatus.DONE
        ):
            continue
        if not is_action_ready(node, step_index):
            continue
        if node.score is None and not allow_unscored:
            continue
        if node.score is not None and node.score < min_score:
            continue
        if best is None or (node.score or 0.0) > (best.score or 0.0):
            best = node
    if not best:
        return None, None
    parent_id = best.parent_id
    if parent_id is None and best.parent_ids:
        parent_id = best.parent_ids[0]
    return best, parent_id


def sanitize_action_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce an action result into a JSON-/checkpoint-safe dict.

    Scalars pass through; nested dicts recurse; sequences are flattened to a
    list (dict items recurse, others are stringified); anything else is
    stringified.
    """
    if not isinstance(result, dict):
        return {"error": f"Invalid result type: {type(result)}"}

    sanitized: Dict[str, Any] = {}
    for key, value in result.items():
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_action_result(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [
                sanitize_action_result(item) if isinstance(item, dict) else str(item)
                for item in value
            ]
        else:
            sanitized[key] = str(value)
    return sanitized
