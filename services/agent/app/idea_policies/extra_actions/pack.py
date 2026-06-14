"""Bundle of curated extra actions.

`ExtraActionPack` is a tiny container — a list of action classes plus a
`build_instances(settings)` helper. The full `ActionPack` abstraction with
prompt fragments, dataflow rules, and post-expansion hooks is part of the
Phase 1 extraction; this in-tree shim gives callers something concrete to
import today.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.arxiv import ArxivSearchAction
from agent.app.idea_policies.extra_actions.datetime_now import DatetimeNowAction
from agent.app.idea_policies.extra_actions.github import GithubRepoInfoAction
from agent.app.idea_policies.extra_actions.hacker_news import HackerNewsTopAction
from agent.app.idea_policies.extra_actions.json_path import JsonPathAction
from agent.app.idea_policies.extra_actions.pypi import PypiPackageInfoAction
from agent.app.idea_policies.extra_actions.regex_extract import RegexExtractAction
from agent.app.idea_policies.extra_actions.unit_convert import UnitConvertAction
from agent.app.idea_policies.extra_actions.url_metadata import UrlMetadataAction
from agent.app.idea_policies.extra_actions.weather import OpenMeteoWeatherAction
from agent.app.idea_policies.extra_actions.wikipedia import WikipediaSummaryAction


class ExtraActionPack:
    """A curated set of small, useful plugin actions.

    Today the engine's `LeafActionRegistry` is keyed by the `IdeaActionType`
    enum, so these can't yet be auto-selected by the planner. Phase 1 of
    the extraction generalizes the registry to string keys; once that lands,
    `ExtraActionPack.install(registry)` will wire each action in by name.

    Until then, callers instantiate actions directly:

        from agent.app.idea_policies.extra_actions import WikipediaSummaryAction
        action = WikipediaSummaryAction()
        result = await action.execute(graph, node_id, io)
    """

    name = "extras"

    ACTION_CLASSES: List[Type[LeafAction]] = [
        WikipediaSummaryAction,
        ArxivSearchAction,
        GithubRepoInfoAction,
        HackerNewsTopAction,
        PypiPackageInfoAction,
        OpenMeteoWeatherAction,
        UrlMetadataAction,
        RegexExtractAction,
        JsonPathAction,
        UnitConvertAction,
        DatetimeNowAction,
    ]

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self.settings = dict(settings or {})

    def build_instances(self) -> Dict[str, LeafAction]:
        """Instantiate each action class with the pack's shared settings."""
        return {cls.name: cls(settings=self.settings) for cls in self.ACTION_CLASSES}

    def names(self) -> List[str]:
        return [cls.name for cls in self.ACTION_CLASSES]
