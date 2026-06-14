# Idea Engine — Plugin Pack

Small, useful `LeafAction` plugins beyond the built-in web research set
(`search`, `visit`, `think`, `save`, `merge`). Each one is opt-in, single
purpose, and ~50–150 lines. None of them ship as engine defaults — you
import what you want.

All eleven actions live under
`services/agent/app/idea_policies/extra_actions/`. The four pure-logic ones
have unit tests in `services/agent/tests/extra_actions_test.py` (20/20
passing). The seven network-backed ones run against free public APIs; no
keys required.

The string-keyed `ActionRegistry` and prompt-fragment composition arrive in
Phase 1 of the extraction. Until then, these plugins are **directly
callable** — instantiate the class, call `.execute(graph, node_id, io)`
from your own glue code or tests. The contracts are stable; the registry
plumbing is what changes.

---

## The eleven plugins

| Action | What it does | Inputs | Network |
|---|---|---|---|
| `wikipedia_summary` | Article extract via Wikipedia REST | `title`, `lang?` | yes |
| `arxiv_search` | Paper search via arXiv Atom API | `query`, `max_results?` | yes |
| `github_repo_info` | Public repo metadata | `owner+repo` or `url` | yes |
| `hacker_news_top` | Top N current HN stories | `count?` | yes |
| `pypi_package_info` | PyPI package metadata | `package` | yes |
| `open_meteo_weather` | Current conditions at lat/lon | `lat`, `lon`, `units?` | yes |
| `url_metadata` | OG / Twitter Card / `<title>` | `url` | yes |
| `regex_extract` | Pull regex matches from text | `pattern`, `text`, `flags?` | no |
| `json_path` | Dotted-path lookup on JSON | `json`, `path` / `paths` | no |
| `unit_convert` | Length/mass/temp/time/data conversion | `value`, `from_unit`, `to_unit` | no |
| `datetime_now` | Current time with offset + shift | `tz_offset_hours?`, `add_*?` | no |

---

## Usage today (pre-Phase 1)

The engine's `LeafActionRegistry` is still enum-keyed
(`IdeaActionType.SEARCH`, etc.) so the planner can't yet auto-pick these
by name. Two ways to use them in the meantime:

### A. Call directly from your own glue code

```python
from agent.app.idea_policies.extra_actions import (
    WikipediaSummaryAction,
    ExtraActionPack,
)

action = WikipediaSummaryAction()
node = graph.get_node(some_node_id)
node.details.update({"title": "Cryptography", "lang": "en"})
result = await action.execute(graph, some_node_id, agent_io)
print(result["extract"])
```

### B. Embed inside a custom built-in action

If you want the planner to *trigger* a plugin without waiting for Phase 1,
extend one of the built-in actions to dispatch to a plugin based on a
detail flag:

```python
class HybridSearchAction(SearchLeafAction):
    """If details.search_mode == 'arxiv', use arxiv_search instead of web."""

    async def execute(self, graph, node_id, io):
        node = graph.get_node(node_id)
        if (node.details or {}).get("search_mode") == "arxiv":
            return await ArxivSearchAction(self.settings).execute(graph, node_id, io)
        return await super().execute(graph, node_id, io)
```

Then register `HybridSearchAction` in place of `SearchLeafAction`:

```python
registry = LeafActionRegistry(settings)
registry._registry[IdeaActionType.SEARCH] = HybridSearchAction  # private but works
engine = IdeaDagEngine(io=io, actions=registry, ...)
```

This is a stop-gap. Phase 1 ships a proper string registry that makes the
indirection cleaner.

---

## Usage after Phase 1 (preview)

```python
from ideaengine import IdeaEngine, ActionRegistry, WebActionPack
from ideaengine.extras import ExtraActionPack

registry = ActionRegistry()
registry.install(WebActionPack())
registry.install(ExtraActionPack())   # adds the eleven above

engine = IdeaEngine(io=io, actions=registry)
result = await engine.run("Summarize the Wikipedia page on Tor and find related arXiv papers")
# Planner can pick wikipedia_summary + arxiv_search nodes from the registry.
```

---

## Action result shape

Every plugin returns the same envelope as the built-in actions:

```python
{
  "action": "wikipedia_summary",
  "success": True,
  # ...action-specific fields...
}
# Failure:
{
  "action": "wikipedia_summary",
  "success": False,
  "error": "...",
  "error_type": "ToolError",
  "retryable": False,
}
```

Network actions set `retryable=True` on transient HTTP failures so the
engine's existing retry / cooldown machinery kicks in.

---

## Writing your own plugin

Subclass `LeafAction`. Override `execute(graph, node_id, io)`. Read inputs
from `graph.get_node(node_id).details`. Return a dict via the `ok()` /
`fail()` helpers in `extra_actions/base.py`.

Optional hooks:

- `name: ClassVar[str]` — string used in `node.details[ACTION]` and (post
  Phase 1) in `allowed_actions`.
- `post_execute_provides(node, result) -> Optional[str]` — if your action
  produces a `DataContract` on success, return the contract name so
  downstream nodes can declare `REQUIRES_DATA` against it. Register the
  contract in `idea_policies/data_contracts.py`.

Minimal template:

```python
# my_plugin.py
from typing import Any, Dict
from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.extra_actions.base import fail, ok

class MyPluginAction(LeafAction):
    name = "my_plugin"

    async def execute(self, graph, node_id: str, io: Any) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return fail(self.name, f"node {node_id} not found")
        # ...do your work...
        return ok(self.name, payload="result here")
```

---

## What I'd add next (cheap wins)

If the eleven above land well, the next batch of small plugins worth
shipping:

- **`rss_feed`** — read an RSS/Atom feed → list of items. ~80 lines.
- **`whois_lookup`** — domain ownership via the python-whois library.
- **`dns_lookup`** — A/AAAA/MX records via `dnspython`. ~60 lines.
- **`stack_overflow_search`** — Stack Exchange API, free tier. ~100 lines.
- **`opensearch_query`** — generic search across any OpenSearch endpoint.
- **`html_table_extract`** — pull tables out of HTML into rows.
- **`base64_encode_decode`** — for inspecting tokens, certs, etc.
- **`url_shortener_expand`** — follow short URLs to their destination.
- **`country_info`** — restcountries.com (free) → currency, languages, etc.
- **`stock_quote`** — Yahoo Finance unofficial endpoint or Alpha Vantage
  (with key support).
- **`cron_expression_explain`** — describe a cron string in English.
- **`uuid_generate`** — UUID v4/v7 generation. Tiny, useful.

All zero-key, well under 200 lines each, perfect candidates for the
plugin marketplace described in `IDEA_ENGINE_FEATURES.md` §L4.

---

## File layout

```
services/agent/app/idea_policies/extra_actions/
  __init__.py            # re-exports
  base.py                # ok() / fail() / fetch_json() helpers
  pack.py                # ExtraActionPack container
  arxiv.py               # ArxivSearchAction
  datetime_now.py        # DatetimeNowAction
  github.py              # GithubRepoInfoAction
  hacker_news.py         # HackerNewsTopAction
  json_path.py           # JsonPathAction + resolve_json_path helper
  pypi.py                # PypiPackageInfoAction
  regex_extract.py       # RegexExtractAction
  unit_convert.py        # UnitConvertAction
  url_metadata.py        # UrlMetadataAction
  weather.py             # OpenMeteoWeatherAction
  wikipedia.py           # WikipediaSummaryAction

services/agent/tests/
  extra_actions_test.py  # 20 unit tests, no network
```
