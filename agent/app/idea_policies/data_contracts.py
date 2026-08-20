"""
Data contracts for cross-node dependencies in the Idea DAG.

A `DataContract` describes a piece of evidence that one node produces and
another node consumes. The engine uses contracts to decide when a node whose
`REQUIRES_DATA` detail points at a peer is actually unblocked.

Each contract owns:
- `name`: the string written into `REQUIRES_DATA.type` and `PROVIDES_DATA.type`
- `is_ready`: a predicate over `(source_action_result, source_node)` returning
  True when the downstream consumer can safely run.
- `value_for`: an optional `(source_action_result, source_node, slot) -> value`
  reader used by the resolved-value channel when a consumer declares
  `requires_data.slot` (`IdeaDagEngine._resolve_slot`). Fallback only: the
  source node's `waypoint` is tried first. `chunk_from_visit` has none - a
  page's body text carries no value a slot can be filled from deterministically.
- `default_for_action`: action name whose successful completion implicitly tags
  the node with this contract (e.g. SEARCH success → `urls_from_search`).
  Used by `_handle_action_result`'s auto-tagging path.

The four built-in contracts (`urls_from_search`, `urls_from_visit`,
`url_from_think`, `chunk_from_visit`) replicate the behavior previously
hard-coded inside `IdeaDagEngine._has_required_data`. Custom action packs
register their own contracts via `ContractRegistry.register`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaNode


ReadyPredicate = Callable[[Dict[str, Any], "IdeaNode"], bool]
#: ``(source_action_result, source_node, slot) -> value | None``. See `DataContract.value_for`.
ValueGetter = Callable[[Dict[str, Any], "IdeaNode", str], Optional[str]]


@dataclass(frozen=True)
class DataContract:
    """A typed evidence channel between producing and consuming nodes."""

    name: str
    is_ready: ReadyPredicate
    default_for_action: Optional[str] = None
    consumer_actions: tuple[str, ...] = field(default_factory=tuple)
    #: The resolved-value channel's FALLBACK reader: the best value this contract's structured
    #: output offers for a consumer's declared `requires_data.slot`. Only reads what the tool
    #: itself returned (a result URL, an outgoing link) -- never free text, which the waypoint
    #: audit showed cannot be extracted deterministically. `IdeaDagEngine._resolve_slot` tries
    #: the source's `waypoint` first and falls back here; returning None is always safe (the
    #: consumer then runs exactly as it does today). None means "this contract carries nothing
    #: a slot can be filled from".
    value_for: Optional[ValueGetter] = None


class ContractRegistry:
    """Lookup table for `DataContract`s keyed by name and producing action."""

    def __init__(self) -> None:
        self._contracts: Dict[str, DataContract] = {}
        self._default_by_action: Dict[str, str] = {}

    def register(self, contract: DataContract) -> None:
        self._contracts[contract.name] = contract
        if contract.default_for_action:
            self._default_by_action[contract.default_for_action] = contract.name

    def get(self, name: Optional[str]) -> Optional[DataContract]:
        if not name:
            return None
        return self._contracts.get(name)

    def contract_for_action(self, action_name: Optional[str]) -> Optional[DataContract]:
        """Default contract a successful execution of `action_name` provides."""
        if not action_name:
            return None
        contract_name = self._default_by_action.get(action_name)
        return self._contracts.get(contract_name) if contract_name else None

    def names(self) -> Iterable[str]:
        return self._contracts.keys()


def _urls_from_search_is_ready(result: Dict[str, Any], _source_node: "IdeaNode") -> bool:
    from agent.app.idea_policies.action_constants import ActionResultKey

    results = result.get(ActionResultKey.RESULTS.value) or []
    return bool(results)


def _urls_from_visit_is_ready(result: Dict[str, Any], _source_node: "IdeaNode") -> bool:
    links = result.get("links") or result.get("links_full") or []
    return bool(links)


def _url_from_think_is_ready(result: Dict[str, Any], source_node: "IdeaNode") -> bool:
    from agent.app.idea_policies.action_constants import ActionResultKey, NodeDetailsExtractor

    url = result.get(ActionResultKey.URL.value) or result.get("extracted_url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return True
    url_from_details = NodeDetailsExtractor.get_url(source_node.details)
    return isinstance(url_from_details, str) and url_from_details.startswith(("http://", "https://"))


def _chunk_from_visit_is_ready(result: Dict[str, Any], _source_node: "IdeaNode") -> bool:
    from agent.app.idea_policies.action_constants import ActionResultKey

    return bool(result.get(ActionResultKey.CONTENT_FULL.value))


# The three URL-bearing contracts' value getters. Each answers ONE question -- "what URL did
# this source actually produce?" -- and every one of them declines any slot but `url`: no
# contract's structured output carries a query string or a link idea, and inventing one from
# free text is exactly the extraction that the waypoint audit measured as unreliable.


def _http(value: Any) -> Optional[str]:
    """`value` as a usable http(s) URL, or None. The single admissibility rule for all getters."""
    if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
        return value.strip()
    return None


def _urls_from_search_value(result: Dict[str, Any], _source_node: "IdeaNode", slot: str) -> Optional[str]:
    """The search's first result URL: the same pool ``VisitLeafAction`` scavenges, addressed."""
    if slot != "url":
        return None
    from agent.app.idea_policies.action_constants import ActionResultKey

    results = result.get(ActionResultKey.RESULTS.value) or result.get("results") or []
    if not isinstance(results, list):
        return None
    for item in results:
        if isinstance(item, dict):
            url = _http(item.get("url") or item.get("link") or item.get("href")
                        or item.get("source") or item.get("page_url"))
        else:
            url = _http(item)
        if url:
            return url
    return None


def _urls_from_visit_value(result: Dict[str, Any], _source_node: "IdeaNode", slot: str) -> Optional[str]:
    """The visited page's first outgoing link. Deliberately unranked: ranking a link set by
    node goal text is the design that died three times over (see the resolved-value channel
    design doc); this is a last-resort fallback behind the waypoint, not a selector."""
    if slot != "url":
        return None
    links = result.get("links") or result.get("links_full") or []
    if not isinstance(links, list):
        return None
    for item in links:
        if isinstance(item, dict):
            url = _http(item.get("url") or item.get("href") or item.get("link"))
        else:
            url = _http(item)
        if url:
            return url
    return None


def _url_from_think_value(result: Dict[str, Any], source_node: "IdeaNode", slot: str) -> Optional[str]:
    """The URL the think node recorded -- on its result, else on its own details."""
    if slot != "url":
        return None
    from agent.app.idea_policies.action_constants import ActionResultKey, NodeDetailsExtractor

    return (
        _http(result.get(ActionResultKey.URL.value))
        or _http(result.get("extracted_url"))
        or _http(NodeDetailsExtractor.get_url(source_node.details))
    )


URLS_FROM_SEARCH = DataContract(
    name="urls_from_search",
    is_ready=_urls_from_search_is_ready,
    default_for_action="search",
    consumer_actions=("visit", "think"),
    value_for=_urls_from_search_value,
)

URLS_FROM_VISIT = DataContract(
    name="urls_from_visit",
    is_ready=_urls_from_visit_is_ready,
    default_for_action="visit",
    consumer_actions=("visit", "think"),
    value_for=_urls_from_visit_value,
)

URL_FROM_THINK = DataContract(
    name="url_from_think",
    is_ready=_url_from_think_is_ready,
    default_for_action=None,
    consumer_actions=("visit",),
    value_for=_url_from_think_value,
)

CHUNK_FROM_VISIT = DataContract(
    name="chunk_from_visit",
    is_ready=_chunk_from_visit_is_ready,
    default_for_action=None,
    consumer_actions=("search", "think"),
)


def default_contract_registry() -> ContractRegistry:
    registry = ContractRegistry()
    registry.register(URLS_FROM_SEARCH)
    registry.register(URLS_FROM_VISIT)
    registry.register(URL_FROM_THINK)
    registry.register(CHUNK_FROM_VISIT)
    return registry
