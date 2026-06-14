"""Extra `LeafAction` plugins beyond the built-in web research pack.

Each action is small, single-purpose, opt-in. Import the specific actions you
want, or use `ExtraActionPack` for the curated bundle. None of these are
auto-registered with the engine — the user composes them explicitly:

    from ideaengine.idea_policies.extra_actions import ExtraActionPack
    pack = ExtraActionPack()
    # ... wire into your engine config when Phase 1's ActionRegistry lands ...

Until the string-keyed action registry ships (Phase 1), these are usable
standalone: instantiate the action class and call `.execute(graph, node_id, io)`
from your own glue code or tests.
"""

from ideaengine.idea_policies.extra_actions.pack import ExtraActionPack
from ideaengine.idea_policies.extra_actions.wikipedia import WikipediaSummaryAction
from ideaengine.idea_policies.extra_actions.arxiv import ArxivSearchAction
from ideaengine.idea_policies.extra_actions.github import GithubRepoInfoAction
from ideaengine.idea_policies.extra_actions.hacker_news import HackerNewsTopAction
from ideaengine.idea_policies.extra_actions.pypi import PypiPackageInfoAction
from ideaengine.idea_policies.extra_actions.weather import OpenMeteoWeatherAction
from ideaengine.idea_policies.extra_actions.url_metadata import UrlMetadataAction
from ideaengine.idea_policies.extra_actions.regex_extract import RegexExtractAction
from ideaengine.idea_policies.extra_actions.json_path import JsonPathAction
from ideaengine.idea_policies.extra_actions.unit_convert import UnitConvertAction
from ideaengine.idea_policies.extra_actions.datetime_now import DatetimeNowAction

__all__ = [
    "ExtraActionPack",
    "WikipediaSummaryAction",
    "ArxivSearchAction",
    "GithubRepoInfoAction",
    "HackerNewsTopAction",
    "PypiPackageInfoAction",
    "OpenMeteoWeatherAction",
    "UrlMetadataAction",
    "RegexExtractAction",
    "JsonPathAction",
    "UnitConvertAction",
    "DatetimeNowAction",
]
