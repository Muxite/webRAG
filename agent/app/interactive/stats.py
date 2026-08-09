"""Stats tracker for agent-debug."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.idea_policies.action_constants import NodeDetailsExtractor


class StatsTracker:

    def __init__(self, graph: IdeaDag):
        self._graph = graph
        self._start = time.monotonic()
        self._steps = 0
        self._depth = 0

    def tick(self, depth: int = 0) -> None:
        self._steps += 1
        self._depth = depth

    def set_depth(self, depth: int) -> None:
        self._depth = depth

    def snapshot(self) -> Dict[str, Any]:
        status_counts: Counter = Counter()
        action_counts: Counter = Counter()
        for node in self._graph.iter_depth_first():
            status_counts[node.status] += 1
            act = NodeDetailsExtractor.get_action(node.details)
            if act:
                action_counts[str(act)] += 1

        return {
            "steps": self._steps,
            "total": self._graph.node_count(),
            "done": status_counts.get(IdeaNodeStatus.DONE, 0),
            "active": status_counts.get(IdeaNodeStatus.ACTIVE, 0),
            "pending": status_counts.get(IdeaNodeStatus.PENDING, 0),
            "failed": status_counts.get(IdeaNodeStatus.FAILED, 0),
            "blocked": status_counts.get(IdeaNodeStatus.BLOCKED, 0),
            "skipped": status_counts.get(IdeaNodeStatus.SKIPPED, 0),
            "actions": dict(action_counts),
            "depth": self._depth,
            "elapsed": time.monotonic() - self._start,
        }
