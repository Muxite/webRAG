from __future__ import annotations

from typing import Any, Dict, Optional

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies.base import SelectionPolicy


class BestScoreSelectionPolicy(SelectionPolicy):
    """
    Select child node with the highest score.
    :param settings: Settings dictionary.
    :returns: BestScoreSelectionPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings=settings)

    def select(self, graph: IdeaDag, parent_id: str) -> Optional[IdeaNode]:
        """
        Select the best child node based on score.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node identifier.
        :returns: Selected IdeaNode or None.
        """
        require_score = bool(self.settings.get("require_score", True))
        return graph.select_best_child(parent_id, require_score=require_score)
