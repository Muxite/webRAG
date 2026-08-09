"""Extra `LeafAction` plugins beyond the built-in web research pack.

Each action is small, single-purpose, opt-in. Import the specific actions you
want, or use `ExtraActionPack` for the curated bundle. None of these are
auto-registered with the engine — the user composes them explicitly:

    from agent.app.idea_policies.extra_actions import ExtraActionPack
    pack = ExtraActionPack()
    # ... wire into your engine config when Phase 1's ActionRegistry lands ...

Until the string-keyed action registry ships (Phase 1), these are usable
standalone: instantiate the action class and call `.execute(graph, node_id, io)`
from your own glue code or tests.
"""

from agent.app.idea_policies.extra_actions.pack import ExtraActionPack
from agent.app.idea_policies.extra_actions.calculator_tools import (
    CalculatorToolPack,
    RunPythonAction,
)
from agent.app.idea_policies.extra_actions.sandbox_tools import (
    CountLinesAction,
    DiskUsageAction,
    FindFilesAction,
    HeadFileAction,
    ListDirAction,
    ReadFileAction,
    SandboxToolPack,
    WordCountAction,
    WriteFileAction,
)
from agent.app.idea_policies.extra_actions.wikipedia import WikipediaSummaryAction
from agent.app.idea_policies.extra_actions.arxiv import ArxivSearchAction
from agent.app.idea_policies.extra_actions.github import GithubRepoInfoAction
from agent.app.idea_policies.extra_actions.hacker_news import HackerNewsTopAction
from agent.app.idea_policies.extra_actions.pypi import PypiPackageInfoAction
from agent.app.idea_policies.extra_actions.weather import OpenMeteoWeatherAction
from agent.app.idea_policies.extra_actions.url_metadata import UrlMetadataAction
from agent.app.idea_policies.extra_actions.regex_extract import RegexExtractAction
from agent.app.idea_policies.extra_actions.json_path import JsonPathAction
from agent.app.idea_policies.extra_actions.unit_convert import UnitConvertAction
from agent.app.idea_policies.extra_actions.datetime_now import DatetimeNowAction

__all__ = [
    "ExtraActionPack",
    # Workdir file + read-only shell tools. A higher-stakes capability class than the rest of
    # this package: opt-in only (install the pack AND allow the names AND wire a sandbox onto
    # the run's AgentIO), never part of ExtraActionPack. See `sandbox_tools.py`.
    "SandboxToolPack",
    # A one-off Python calculator, opt-in on the same three-way gate but packaged separately so a
    # caller can grant "compute" without granting file/shell access. See `calculator_tools.py`.
    "CalculatorToolPack",
    "RunPythonAction",
    "ReadFileAction",
    "WriteFileAction",
    "ListDirAction",
    "CountLinesAction",
    "WordCountAction",
    "HeadFileAction",
    "DiskUsageAction",
    "FindFilesAction",
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
