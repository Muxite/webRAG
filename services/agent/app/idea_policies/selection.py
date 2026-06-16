from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.idea_policies.base import SelectionPolicy
from agent.app.idea_policies.config import IdeaConfig


class BestScoreSelectionPolicy(SelectionPolicy):
    """
    Select child node with the highest score.
    :param settings: Settings dictionary.
    :returns: BestScoreSelectionPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings=settings)
        self._cfg = IdeaConfig.from_settings(self.settings)

    def select(self, graph: IdeaDag, parent_id: str) -> Optional[IdeaNode]:
        """
        Select the best child node based on score.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node identifier.
        :returns: Selected IdeaNode or None.
        """
        require_score = self._cfg.policy.require_score
        return graph.select_best_child(parent_id, require_score=require_score)
