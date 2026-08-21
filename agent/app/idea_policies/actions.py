from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, List, Set, Tuple
import uuid
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.observation import clean_operation
from agent.app.llm_backends import json_instruction_from_response_format, accepts_reasoning_effort
from agent.app.model_tiers import tier_token_multiplier, is_reasoning_model
from agent.app.idea_dag_schemas import MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST
from agent.app.idea_policies.base import IdeaActionType, DetailKey, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.action_constants import (
    ActionResultKey,
    PromptKey,
    ContextKey,
    ActionResultBuilder,
    PromptBuilder,
    ContextBuilder,
    is_transient_tool_error,
)

#: Where a parent records the pages its visit children have already claimed, so a sibling
#: batch does not fetch one page four times. Dunder-prefixed like ``__semantic_dedup_source``:
#: engine bookkeeping written onto a node, never a planner-authored detail.
VISIT_URL_CLAIMS_KEY = "__visit_url_claims"


class LeafAction(ABC):
    #: One-line, model-facing summary of what this action does. Left ``None`` on purpose:
    #: :meth:`menu_description` then harvests the FIRST line of the class docstring, so an
    #: action documents itself once (in its docstring) rather than in two places that drift.
    #: Set it explicitly only when the docstring's opening line is a poor prompt line.
    description: ClassVar[Optional[str]] = None
    #: One-line argument shape for the expansion prompt, in the same notation the hardcoded
    #: ACTIONS block uses for ``visit`` (e.g. ``details={title, lang?}`` — ``?`` = optional).
    #: ``None`` means "no args worth stating"; the menu line then carries the description only.
    args_hint: ClassVar[Optional[str]] = None

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})
        self._cfg = IdeaConfig.from_settings(self.settings)
        self._logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def menu_description(cls) -> str:
        """The one-line description used by the expansion prompt's extra-actions menu.

        Prefers the explicit :attr:`description` ClassVar, else the first line of the class
        docstring, else an empty string. Docstring harvesting keeps the prose in ONE place.
        """
        explicit = getattr(cls, "description", None)
        if isinstance(explicit, str) and explicit.strip():
            return " ".join(explicit.split())
        doc = (cls.__doc__ or "").strip()
        if not doc:
            return ""
        return " ".join(doc.splitlines()[0].split())

    @classmethod
    def menu_line(cls) -> str:
        """This action's full ``- name: details={...}. Description`` prompt line.

        Mirrors the shape of the hardcoded ACTIONS entries so an extra action reads like a
        first-class one to the model rather than a bolted-on afterthought.
        """
        name = str(getattr(cls, "name", "") or "")
        parts = [part for part in (str(cls.args_hint or "").strip(), cls.menu_description()) if part]
        body = ". ".join(parts)
        return f"- {name}: {body}" if body else f"- {name}"

    def _log_structured(self, level: str, message: str, **kwargs) -> None:
        """
        Log structured message for AWS CloudWatch.
        :param level: Log level (info, warning, error, debug)
        :param message: Main log message
        :param kwargs: Additional structured fields
        """
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        
        if kwargs:
            fields = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            log_method(f"{message} | {fields}")
        else:
            log_method(message)

    @abstractmethod
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        raise NotImplementedError()

    def post_execute_provides(self, node: "IdeaNode", result: Dict[str, Any]) -> Optional[str]:
        """
        Name of the `DataContract` this action satisfies on successful execution.

        Returning `None` (the default) means the action does not auto-tag the
        node's `PROVIDES_DATA` detail. Subclasses override to declare what
        downstream nodes can rely on this action having produced.
        """
        return None

    def _max_observation_chars(self) -> int:
        return self._cfg.action.max_observation_chars

    def _effective_intent(self, node: "IdeaNode") -> Optional[str]:
        """Resolve the extraction intent for a leaf, folding in an optional ``expect``
        contract (``expansion_expect_contract_enabled``).

        When a leaf carries a measurable-output contract (``DetailKey.EXPECT``), append it
        to the free-text intent as an explicit "Report exactly: ..." target so the executor
        grounds on the exact declared value + source URL. Returns the plain intent UNCHANGED
        when no ``expect`` is present, so the default path is byte-identical.
        """
        intent = node.details.get(DetailKey.INTENT.value)
        expect = node.details.get(DetailKey.EXPECT.value)
        if not isinstance(expect, str) or not expect.strip():
            return intent
        contract = f"Report exactly: {expect.strip()}"
        if isinstance(intent, str) and intent.strip():
            return f"{intent.strip()}\n{contract}"
        return contract

    def _effective_model(self, io: AgentIO, model_name: Optional[str]) -> Optional[str]:
        """The model a micro-prompt actually runs on: the explicit override, else the
        connector's current execution model (so price/reasoning tiering keys on the real
        executor even when the call passes ``model_name=None``)."""
        if model_name:
            return model_name
        try:
            return io.connector_llm.get_model()
        except Exception:  # noqa: BLE001
            return None

    def _micro_prompt_reasoning_effort(
        self, model_name: Optional[str], default: Optional[str] = None
    ) -> Optional[str]:
        """A3b: a reasoning model's PERCEPTION/selection micro-prompt should spend minimal
        hidden reasoning so it can't starve its own completion budget (the compiled-path
        content=None bug). Returns ``"minimal"`` when the discipline flag is on and the model
        is a reasoning model; otherwise the caller's ``default`` (byte-identical when off).

        The hint needs BOTH: the model must actually be a reasoning model (else minimal effort is
        meaningless) AND the wire must accept the ``reasoning_effort`` param (else it's stripped)."""
        if (
            self._cfg.action.native_reasoning_effort_discipline_enabled
            and is_reasoning_model(model_name)
            and accepts_reasoning_effort(model_name)
        ):
            return "minimal"
        return default

    def _executor_max_tokens(
        self, model_name: Optional[str], base: Optional[int], *, price_tier: bool = True
    ) -> Optional[int]:
        """Resolve a native micro-prompt token budget under the opt-in tiering flags.

        A5 (``price_tier_param_tiering_enabled``): scale ``base`` by the executor's price-tier
        multiplier (cheap ``1.0`` -> unchanged; mid/premium get headroom). A3b
        (``native_reasoning_effort_discipline_enabled``): floor a reasoning model's budget to
        ``native_reasoning_min_tokens_floor`` so a tight micro-prompt budget can't starve it.
        Returns ``base`` unchanged when both flags are off (byte-identical). ``price_tier=False``
        skips the A5 multiplier (used where only the anti-starvation floor is wanted).

        The floor keys on ``is_reasoning_model`` (the "does it starve?" predicate) — NOT on
        wire-acceptance — so o-series models (which the wire allowlist omits) still get headroom,
        and a non-reasoning model like gpt-4.1 is not needlessly floored."""
        if base is None:
            return base
        tokens = int(base)
        if price_tier and self._cfg.action.price_tier_param_tiering_enabled:
            tokens = int(round(tokens * tier_token_multiplier(model_name)))
        if (
            self._cfg.action.native_reasoning_effort_discipline_enabled
            and is_reasoning_model(model_name)
        ):
            tokens = max(tokens, self._cfg.action.native_reasoning_min_tokens_floor)
        return tokens

    def _timeout_seconds(self, key: str) -> Optional[float]:
        # Accepts a full settings key (e.g. "search_timeout_seconds"); resolves
        # it against TimeoutConfig, falling back to the generic action timeout.
        name = key[:-len("_timeout_seconds")] if key.endswith("_timeout_seconds") else key
        value = getattr(self._cfg.timeouts, name, None)
        if value is None:
            value = self._cfg.timeouts.action
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_retryable(self, error: Exception) -> bool:
        # Shared with the sequential arm's tool retry so both arms agree on what "transient"
        # means — see action_constants.is_transient_tool_error.
        return is_transient_tool_error(error)

    def _limit_text(self, text: str) -> Dict[str, Any]:
        max_chars = self._max_observation_chars()
        raw = text or ""
        if len(raw) <= max_chars:
            return {"content": raw, "is_truncated": False, "total_chars": len(raw)}
        return {
            "content": raw[:max_chars],
            "is_truncated": True,
            "total_chars": len(raw),
        }

    @staticmethod
    def _copy_details_safely(details: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(details, dict):
            return {}
        
        result = {}
        for key, value in details.items():
            if value is None:
                result[str(key)] = None
            elif isinstance(value, (str, int, float, bool)):
                result[str(key)] = value
            elif isinstance(value, dict):
                result[str(key)] = LeafAction._copy_details_safely(value)
            elif isinstance(value, (list, tuple)):
                result[str(key)] = [
                    LeafAction._copy_details_safely(item) if isinstance(item, dict) else str(item)
                    for item in value
                ]
            else:
                result[str(key)] = str(value)
        return result

    def _failure(
        self,
        action: IdeaActionType,
        node_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import traceback
        
        error_str = str(error)
        error_type = type(error).__name__
        
        root_cause = error_str
        if hasattr(error, '__cause__') and error.__cause__:
            root_cause = f"{error_str} (caused by: {str(error.__cause__)})"
        
        http_status = None
        status_match = re.search(r"status[=:]?\s*([0-9]{3})", error_str, re.IGNORECASE)
        if status_match:
            http_status = int(status_match.group(1))
        
        tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        tb_summary = tb_str.split('\n')[-3:-1] if len(tb_str.split('\n')) > 3 else []
        
        return ActionResultBuilder.failure(
            action=action.value,
            error=error_str,
            error_type=error_type,
            retryable=self._is_retryable(error),
            node_id=node_id,
            context=context or {},
            root_cause=root_cause,
            http_status=http_status,
            traceback_summary=tb_summary,
            timestamp=None,
        )


class SearchLeafAction(LeafAction):
    name = "search"

    def post_execute_provides(self, node, result: Dict[str, Any]) -> Optional[str]:
        return "urls_from_search"

    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        query = None
        count = None
        intent = None
        vector_context = []
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            query = NodeDetailsExtractor.get_query(node.details, fallback_title=node.title)
            intent = self._effective_intent(node)
            count = int(node.details.get(DetailKey.COUNT.value, self._cfg.action.default_search_count))
            
            chunk_content = node.details.get(DetailKey.CHUNK_CONTENT.value)
            if chunk_content:
                self._logger.info(f"[SEARCH] Chunk-based search: searching within chunk {node.details.get(DetailKey.CHUNK_INDEX.value, '?')}/{node.details.get(DetailKey.TOTAL_CHUNKS.value, '?')}")
                results = self._search_in_chunk(chunk_content, query, count)
            else:
                timeout_seconds = self._timeout_seconds("search_timeout_seconds")
                self._logger.debug(f"[SEARCH] query='{query}', intent='{intent}', count={count}")
                results = await io.search(str(query), count=count, timeout_seconds=timeout_seconds)
                self._logger.info(f"[SEARCH] {len(results) if results else 0} results for '{query[:50]}...'")
            
            return ActionResultBuilder.success(
                action=IdeaActionType.SEARCH.value,
                node_id=node_id,
                query=query,
                intent=intent,
                vector_context=[str(doc) for doc in vector_context] if vector_context else [],
                count=count,
                results=results or [],
            )
        except Exception as exc:
            if not query and node:
                query = (
                    node.details.get(DetailKey.QUERY.value)
                    or node.details.get(DetailKey.PROMPT.value)
                    or node.title
                )
            if count is None and node:
                try:
                    count = int(node.details.get(DetailKey.COUNT.value, self._cfg.action.default_search_count))
                except (ValueError, TypeError):
                    count = self._cfg.action.default_search_count
            failure = self._failure(
                action=IdeaActionType.SEARCH,
                node_id=node_id,
                error=exc,
                context={DetailKey.QUERY.value: query},
            )
            if query is not None:
                failure[ActionResultKey.QUERY.value] = query
            if count is not None:
                failure[ActionResultKey.COUNT.value] = count
            return failure
    
    def _search_in_chunk(self, chunk_content: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        import re
        
        query_terms = re.findall(r'\b\w+\b', query.lower())
        if not query_terms:
            return []
        
        chunk_lower = chunk_content.lower()
        matches = []
        
        for term in query_terms:
            if term in chunk_lower:
                start_idx = chunk_lower.find(term)
                if start_idx >= 0:
                    snippet_start = max(0, start_idx - 100)
                    snippet_end = min(len(chunk_content), start_idx + len(term) + 100)
                    snippet = chunk_content[snippet_start:snippet_end].strip()
                    
                    matches.append({
                        "title": f"Match for '{term}' in chunk",
                        "snippet": snippet,
                        "url": f"chunk://{start_idx}",
                        "relevance": 1.0,
                    })
        
        if not matches:
            snippet_start = 0
            snippet_end = min(500, len(chunk_content))
            matches.append({
                "title": "Chunk content",
                "snippet": chunk_content[snippet_start:snippet_end],
                "url": "chunk://0",
                "relevance": 0.5,
            })
        
        return matches[:max_results]


class VisitLeafAction(LeafAction):
    name = "visit"

    def post_execute_provides(self, node, result: Dict[str, Any]) -> Optional[str]:
        from agent.app.idea_policies.action_constants import ActionResultKey

        content = (
            result.get(ActionResultKey.CONTENT.value)
            or result.get(ActionResultKey.CONTENT_FULL.value)
            or result.get("content")
            or ""
        )
        if not content or not content.strip():
            return None
        return "urls_from_visit"

    def _is_valid_url(self, candidate: str) -> bool:
        if not candidate or not isinstance(candidate, str):
            return False
        candidate = candidate.strip()
        return candidate.startswith(("http://", "https://"))
    
    def _clean_and_fix_link(self, href: str, base_url: str) -> Optional[str]:
        if not href or not isinstance(href, str):
            return None
        
        href = href.strip()
        
        if not href or href == "#" or href.startswith("#"):
            return None
        
        if href.startswith(("javascript:", "mailto:", "tel:", "data:", "file:", "ftp:")):
            return None
        
        if href.startswith("//"):
            href = "https:" + href
        
        try:
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)
            
            if not parsed.scheme or parsed.scheme not in ("http", "https"):
                return None
            
            if not parsed.netloc:
                parsed_base = urlparse(base_url)
                if parsed_base.netloc:
                    parsed = parsed._replace(netloc=parsed_base.netloc)
                    parsed = parsed._replace(scheme=parsed_base.scheme)
                else:
                    return None
            
            cleaned_path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
            
            cleaned_query = parsed.query
            if cleaned_query:
                query_params = parse_qs(cleaned_query, keep_blank_values=False)
                cleaned_query = urlencode(query_params, doseq=True)
            
            cleaned_url = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                cleaned_path,
                parsed.params,
                cleaned_query,
                ""
            ))
            
            if cleaned_url == base_url:
                return None
            
            return cleaned_url
        except Exception:
            return None
    
    # Wikipedia non-content namespaces / chrome that are never valid navigation targets.
    _WIKI_CHROME = re.compile(
        r"/wiki/(main_page|special:|wikipedia:|help:|portal:|template:|category:|file:|"
        r"talk:|user:|draft:|module:|mediawiki:|book:)",
        re.IGNORECASE,
    )

    def _is_wiki_chrome(self, url: str) -> bool:
        return bool(self._WIKI_CHROME.search(url or ""))

    def _filter_and_prioritize_links(self, links: List[str], base_url: str) -> List[str]:
        seen: Set[str] = set()
        cleaned_links: List[str] = []

        for link in links:
            cleaned = self._clean_and_fix_link(link, base_url)
            if cleaned and cleaned not in seen and not self._is_wiki_chrome(cleaned):
                seen.add(cleaned)
                cleaned_links.append(cleaned)

        return cleaned_links
    
    def _attach_links_to_content(self, content: str, links: List[str], max_links: int = 20) -> str:
        if not links:
            return content
        
        links_to_attach = links[:max_links]
        links_section = "\n\n--- Links found on this page ---\n"
        for i, link in enumerate(links_to_attach, 1):
            links_section += f"{i}. {link}\n"
        
        if len(links) > max_links:
            links_section += f"\n... and {len(links) - max_links} more links (see 'links' field in action result)\n"
        
        return content + links_section

    def _extract_urls_from_parent_search_results(self, graph: IdeaDag, node: IdeaNode, max_depth: int = 3) -> List[str]:
        """
        Extract URLs from parent and sibling search results.
        :param graph: The idea DAG
        :param node: Current node
        :param max_depth: Maximum depth to search
        :returns: List of valid URLs
        """
        from agent.app.idea_policies.action_constants import ActionResultKey, NodeDetailsExtractor
        from agent.app.idea_dag import IdeaNodeStatus
        
        required_data = node.details.get(DetailKey.REQUIRES_DATA.value)
        if required_data and isinstance(required_data, dict):
            source_node_id = required_data.get("source_node_id")
            data_type = required_data.get("type", "")
            if source_node_id and data_type == "urls_from_search":
                source_node = graph.get_node(source_node_id)
                if source_node:
                    self._logger.info(
                        f"[VISIT] Checking REQUIRES_DATA source node {source_node_id[:16]}... "
                        f"(status: {source_node.status.value})"
                    )
                    if source_node.status == IdeaNodeStatus.DONE:
                        result = source_node.details.get(DetailKey.ACTION_RESULT.value)
                        if result and isinstance(result, dict):
                            action_type = result.get(ActionResultKey.ACTION.value) or result.get("action")
                            if action_type == IdeaActionType.SEARCH.value:
                                search_results = result.get(ActionResultKey.RESULTS.value) or result.get("results", [])
                                # Pool EVERY result URL of the named source (mis-indentation
                                # here used to return inside the loop, i.e. the first hit only —
                                # which handed the caller a single take-it-or-leave-it URL, so a
                                # dead first result failed the whole visit and the `link_idea`
                                # best-match pick below never had anything to choose between).
                                if isinstance(search_results, list) and search_results:
                                    urls_from_source = []
                                    for item in search_results:
                                        if isinstance(item, dict):
                                            candidate_url = (
                                                item.get("url") or item.get("link") or item.get("href")
                                                or item.get("source") or item.get("page_url")
                                            )
                                            if candidate_url:
                                                candidate_url = str(candidate_url).strip()
                                                if self._is_valid_url(candidate_url) and candidate_url not in urls_from_source:
                                                    urls_from_source.append(candidate_url)
                                    if urls_from_source:
                                        self._log_structured(
                                            "info",
                                            "[VISIT] Extracted URLs from REQUIRES_DATA source node",
                                            source_node_id=source_node_id[:16],
                                            urls_found=len(urls_from_source),
                                            action="visit",
                                            operation="extract_urls_from_required_source"
                                        )
                                        return urls_from_source
                                    else:
                                        self._logger.warning(
                                            f"[VISIT] REQUIRES_DATA source node {source_node_id[:16]}... "
                                            f"has {len(search_results)} results but no valid URLs extracted"
                                        )
                                else:
                                    self._logger.warning(
                                        f"[VISIT] REQUIRES_DATA source node {source_node_id[:16]}... "
                                        f"has no results list (type: {type(search_results).__name__})"
                                    )
                            else:
                                self._logger.warning(
                                    f"[VISIT] REQUIRES_DATA source node {source_node_id[:16]}... "
                                    f"is not a search node (action: {action_type})"
                                )
                        else:
                            self._logger.warning(
                                f"[VISIT] REQUIRES_DATA source node {source_node_id[:16]}... "
                                f"has no action_result"
                            )
                    else:
                        self._logger.warning(
                            f"[VISIT] REQUIRES_DATA source node {source_node_id[:16]}... "
                            f"not yet completed (status: {source_node.status.value})"
                        )
        
        visited = set()
        queue = [(node, 0, "self")]
        all_urls = []
        search_nodes_checked = []
        
        while queue:
            current, depth, relation = queue.pop(0)
            if depth >= max_depth or current.node_id in visited:
                continue
            visited.add(current.node_id)
            
            current_action = NodeDetailsExtractor.get_action(current.details)
            current_status = current.status
            
            if current_action == IdeaActionType.SEARCH.value:
                search_nodes_checked.append({
                    "node_id": current.node_id,
                    "status": current_status.value,
                    "relation": relation,
                    "depth": depth
                })
                
                if current_status == IdeaNodeStatus.DONE:
                    result = current.details.get(DetailKey.ACTION_RESULT.value)
                    if result and isinstance(result, dict):
                        action_type = result.get(ActionResultKey.ACTION.value) or result.get("action")
                        if action_type == IdeaActionType.SEARCH.value:
                            search_results = result.get(ActionResultKey.RESULTS.value) or result.get("results", [])
                            if isinstance(search_results, list):
                                urls_found_in_node = 0
                                for item in search_results:
                                    if isinstance(item, dict):
                                        candidate_url = (
                                            item.get("url") or item.get("link") or item.get("href")
                                            or item.get("source") or item.get("page_url")
                                        )
                                        if candidate_url:
                                            candidate_url = str(candidate_url).strip()
                                            if self._is_valid_url(candidate_url) and candidate_url not in all_urls:
                                                all_urls.append(candidate_url)
                                                urls_found_in_node += 1
                                if urls_found_in_node > 0:
                                    self._log_structured(
                                        "info",
                                        "[VISIT] Extracted URLs from search node",
                                        urls_found=urls_found_in_node,
                                        relation=relation,
                                        search_node_id=current.node_id[:16],
                                        search_node_status=current_status.value,
                                        depth=depth,
                                        action="visit",
                                        operation="extract_urls_from_search"
                                    )
                            else:
                                if search_results and isinstance(search_results, list) and len(search_results) > 0:
                                    first = search_results[0]
                                    sample_keys = list(first.keys())[:8] if isinstance(first, dict) else []
                                    self._logger.debug(
                                        f"[VISIT] Search node {current.node_id[:8]}... results format: "
                                        f"first_item_keys={sample_keys}"
                                    )
                                else:
                                    self._logger.debug(
                                        f"[VISIT] Search node {current.node_id[:8]}... has no results list "
                                        f"(type: {type(search_results).__name__})"
                                    )
                        else:
                            self._logger.debug(
                                f"[VISIT] Node {current.node_id[:8]}... marked as search but action_type={action_type}"
                            )
                    else:
                        self._logger.debug(
                            f"[VISIT] Search node {current.node_id[:8]}... (status: {current_status.value}) "
                            f"has no action_result"
                        )
                else:
                    self._logger.debug(
                        f"[VISIT] Search node {current.node_id[:8]}... not yet completed "
                        f"(status: {current_status.value})"
                    )
            
            if current.parent_id:
                parent = graph.get_node(current.parent_id)
                if parent:
                    queue.append((parent, depth + 1, "parent"))
            
            if current.parent_id and relation == "self":
                parent = graph.get_node(current.parent_id)
                if parent:
                    for sibling_id in parent.children:
                        if sibling_id != current.node_id:
                            sibling = graph.get_node(sibling_id)
                            if sibling:
                                queue.append((sibling, depth, "sibling"))
        
        if search_nodes_checked:
            search_node_summary = ",".join([f"{n['relation']}:{n['status']}" for n in search_nodes_checked])
            self._log_structured(
                "info",
                "[VISIT] Checked search nodes for URL extraction",
                search_nodes_checked=len(search_nodes_checked),
                search_node_summary=search_node_summary,
                action="visit",
                operation="extract_urls_from_search"
            )
        
        if all_urls:
            self._log_structured(
                "info",
                "[VISIT] Successfully extracted URLs from search results",
                total_urls=len(all_urls),
                action="visit",
                operation="extract_urls_from_search"
            )
        else:
            search_node_details = ",".join([f"{n['node_id'][:8]}...({n['status']})" for n in search_nodes_checked])
            self._log_structured(
                "warning",
                "[VISIT] No URLs extracted from search results",
                search_nodes_checked=len(search_nodes_checked),
                search_node_details=search_node_details,
                action="visit",
                operation="extract_urls_from_search",
                issue="no_urls_found"
            )
        
        return all_urls
    
    #: Path/host markers of site CHROME (account, edit, donation, portal plumbing) that survives
    #: link extraction. Harmless in a link index, but a chrome link can carry the leaf's own words
    #: in a `returnto=`/campaign parameter, so it out-ranks real content in a lexical pool.
    _CHROME_URL_MARKERS: ClassVar[Tuple[str, ...]] = (
        "/w/index.php", "/wiki/special:", "/wiki/help:", "/wiki/portal:",
        "/wiki/wikipedia:", "/wiki/talk:", "action=edit", "returnto=",
    )
    _CHROME_HOST_MARKERS: ClassVar[Tuple[str, ...]] = ("donate.", "login.", "auth.")

    @classmethod
    def _is_page_chrome_url(cls, url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        host = (parsed.netloc or "").lower()
        target = f"{(parsed.path or '').lower()}?{(parsed.query or '').lower()}"
        return (
            any(host.startswith(marker) for marker in cls._CHROME_HOST_MARKERS)
            or any(marker in target for marker in cls._CHROME_URL_MARKERS)
        )

    def _drop_chrome_urls(self, urls: List[str], where: str) -> List[str]:
        """``urls`` minus site chrome, when ``action.visit_chrome_link_filter`` is on.

        The chrome test is cheap and the same one the dead-URL harvest already applies; what this
        adds is applying it to the OTHER pools a URL-less visit resolves from, where a donation
        appeal or a create-account form is otherwise a selectable "page" (3.0% of executed sibling
        visits in the recorded corpus landed on one). Everything dropped is a page that can never
        answer a research leaf, so an empty remainder is left to the caller's own no-URL handling
        rather than silently restored.
        """
        if not urls or not self._cfg.action.visit_chrome_link_filter:
            return urls
        kept = [u for u in urls if not self._is_page_chrome_url(u)]
        if len(kept) != len(urls):
            self._logger.info(
                f"[VISIT] Chrome filter: dropped {len(urls) - len(kept)} site-chrome URL(s) "
                f"from {where} ({len(kept)} left)"
            )
        return kept

    def _harvest_relative_page_links(
        self,
        graph: IdeaDag,
        node: IdeaNode,
        link_idea: str,
        limit: int,
        max_depth: int = 3,
    ) -> List[str]:
        """Links an ANCESTOR's visited page offered, ranked by overlap with ``link_idea``.

        The recovery pool for a declared URL that turned out not to exist. The Chroma link index
        cannot serve that case: it is only written for a visit that was FOLLOWING links
        (``link_count > 1`` or a ``link_idea``), and a leaf that declares its own URL is neither
        -- so the previous hop's link menu lives only in its own stored action result. On a chain
        the next hop's real page is routinely IN that menu (the Brooklyn Bridge page links the
        Roebling bridge that task 135's planner kept inventing a title for).

        Zero-overlap links are dropped rather than ranked last, which is also what keeps page
        chrome (donation/portal/login links, which name nothing the leaf asked for) out of the
        pool the selection step then sees.
        """
        from agent.app.idea_visit_dedup import url_slug_tokens

        wanted = set(self._URL_WORD_RE.findall((link_idea or "").lower()))
        if not wanted or limit <= 0:
            return []

        seen_nodes: Set[str] = set()
        queue: List[Tuple[IdeaNode, int]] = [(node, 0)]
        scored: List[Tuple[int, int, str]] = []
        seen_urls: Set[str] = set()

        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth or current.node_id in seen_nodes:
                continue
            seen_nodes.add(current.node_id)

            result = current.details.get(DetailKey.ACTION_RESULT.value)
            if isinstance(result, dict) and result.get("action") == IdeaActionType.VISIT.value:
                links = result.get("links_full") or result.get("links") or []
                contexts = result.get("link_contexts") or {}
                if isinstance(links, list):
                    for url in links:
                        if not isinstance(url, str) or url in seen_urls:
                            continue
                        if not self._is_valid_url(url) or self._is_page_chrome_url(url):
                            continue
                        seen_urls.add(url)
                        anchor = contexts.get(url, "") if isinstance(contexts, dict) else ""
                        tokens = set(url_slug_tokens(url)) | set(
                            self._URL_WORD_RE.findall(str(anchor).lower())
                        )
                        overlap = len(
                            {t for t in tokens if t in wanted} - self._URL_NOISE_TOKENS
                        )
                        if overlap:
                            scored.append((overlap, depth, url))

            if current.parent_id:
                parent = graph.get_node(current.parent_id)
                if parent:
                    queue.append((parent, depth + 1))

        if not scored:
            return []
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        harvested = [url for _, _, url in scored[:limit]]
        self._logger.info(
            f"[VISIT] Harvested {len(harvested)} link(s) from an ancestor's page for "
            f"'{link_idea[:60]}' (top: {harvested[0][:80]})"
        )
        return harvested

    def _extract_url_from_parents(self, graph: IdeaDag, node: IdeaNode, max_depth: int = 3) -> Optional[str]:
        visited = set()
        queue = [(node, 0)]
        
        node_title_lower = node.title.lower() if node.title else ""
        node_intent = node.details.get(DetailKey.INTENT.value, "")
        node_intent_lower = node_intent.lower() if isinstance(node_intent, str) else ""
        
        all_candidates = []
        
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth or current.node_id in visited:
                continue
            visited.add(current.node_id)
            
            result = current.details.get(DetailKey.ACTION_RESULT.value)
            if result and isinstance(result, dict):
                action_type = result.get("action")

                if action_type == IdeaActionType.SEARCH.value:
                    search_results = result.get("results", [])
                    if isinstance(search_results, list):
                        for item in search_results[:10]:
                            if isinstance(item, dict):
                                candidate_url = item.get("url") or item.get("link") or item.get("href")
                                if candidate_url:
                                    candidate_url = str(candidate_url).strip()
                                    if self._is_valid_url(candidate_url):
                                        all_candidates.append((candidate_url, depth, "search"))
                
                if action_type == IdeaActionType.VISIT.value:
                    visit_links = result.get("links", []) or result.get("links_full", [])
                    link_contexts = result.get("link_contexts", {})
                    if isinstance(visit_links, list):
                        for candidate_url in visit_links:
                            if candidate_url and isinstance(candidate_url, str) and self._is_valid_url(candidate_url):
                                context = link_contexts.get(candidate_url, "") if isinstance(link_contexts, dict) else ""
                                all_candidates.append((candidate_url, depth, "visit", context))
            
            if current.parent_id:
                parent = graph.get_node(current.parent_id)
                if parent:
                    queue.append((parent, depth + 1))
        
        if self._cfg.action.visit_chrome_link_filter:
            kept_urls = set(self._drop_chrome_urls(
                [candidate[0] for candidate in all_candidates], "ancestor links"
            ))
            all_candidates = [c for c in all_candidates if c[0] in kept_urls]

        if not all_candidates:
            return None

        scored = []
        for candidate in all_candidates:
            url = candidate[0]
            url_lower = url.lower()
            score = 0
            
            score += (max_depth - candidate[1]) * 10
            
            if "wikipedia" in node_title_lower or "wikipedia" in node_intent_lower:
                if "wikipedia.org" in url_lower:
                    score += 50
            if "guido" in node_title_lower or "guido" in node_intent_lower:
                if "guido" in url_lower or "van_rossum" in url_lower:
                    score += 30
            
            if len(candidate) > 2 and candidate[2] == "visit":
                score += 5
            
            if len(candidate) > 3:
                context = candidate[3].lower() if candidate[3] else ""
                if context and any(word in context for word in node_title_lower.split()[:5]):
                    score += 20
            
            scored.append((score, url))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored:
            return scored[0][1]
        
        return None
    
    def _extract_url_from_think_node(self, graph: IdeaDag, node: IdeaNode) -> Optional[str]:
        from agent.app.idea_policies.base import DetailKey
        from agent.app.idea_policies.action_constants import ActionResultKey
        
        requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
        if not isinstance(requires_data, dict):
            return None
        
        source_node_id = requires_data.get("source_node_id")
        if not source_node_id:
            return None
        
        source_node = graph.get_node(source_node_id)
        if not source_node:
            return None
        
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        source_action = NodeDetailsExtractor.get_action(source_node.details)
        if source_action != IdeaActionType.THINK.value:
            return None
        
        source_result = source_node.details.get(DetailKey.ACTION_RESULT.value)
        if isinstance(source_result, dict):
            extracted_url = source_result.get(ActionResultKey.URL.value) or source_result.get("extracted_url")
            if extracted_url and isinstance(extracted_url, str) and extracted_url.startswith(("http://", "https://")):
                return extracted_url
        
        url_from_details = NodeDetailsExtractor.get_url(source_node.details)
        if url_from_details and isinstance(url_from_details, str) and url_from_details.startswith(("http://", "https://")):
            return url_from_details
        
        return None
    
    def _extract_url_from_sibling_results(self, graph: IdeaDag, node: IdeaNode) -> Optional[str]:
        import re

        parent = None
        pids = node.parent_ids if node.parent_ids else ([node.parent_id] if node.parent_id else [])
        for pid in pids:
            if not pid:
                continue
            parent = graph.get_node(pid)
            if parent:
                break
        if not parent:
            return None
        
        node_title_lower = (node.title or "").lower()
        node_intent = (node.details.get(DetailKey.INTENT.value) or "").lower()
        
        sibling_links: List[str] = []
        for sibling_id in parent.children:
            if sibling_id == node.node_id:
                continue
            sibling = graph.get_node(sibling_id)
            if not sibling or sibling.status.value != "done":
                continue
            result = sibling.details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(result, dict):
                continue
            if not result.get("success") and result.get(ActionResultKey.ACTION.value) != IdeaActionType.SEARCH.value:
                continue

            action_type = result.get(ActionResultKey.ACTION.value) or result.get("action")
            if action_type == IdeaActionType.SEARCH.value:
                search_results = result.get(ActionResultKey.RESULTS.value) or result.get("results", [])
                if isinstance(search_results, list):
                    for item in search_results:
                        if isinstance(item, dict):
                            u = item.get("url") or item.get("link") or item.get("href") or item.get("source")
                            if u and isinstance(u, str) and self._is_valid_url(u.strip()):
                                sibling_links.append(u.strip())

            links = result.get("links", []) or result.get("links_full", [])
            if isinstance(links, list):
                sibling_links.extend(links)

            content = result.get("content_with_links", "") or result.get("content", "")
            if isinstance(content, str):
                found = re.findall(r'https?://[^\s\]\)\"\'<>]+', content)
                sibling_links.extend(found)
        
        if not sibling_links:
            return None
        
        seen = set()
        unique_links = []
        for link in sibling_links:
            if link not in seen and self._is_valid_url(link):
                seen.add(link)
                unique_links.append(link)
        
        unique_links = self._drop_chrome_urls(unique_links, "sibling results")
        if not unique_links:
            return None

        search_terms = node_title_lower + " " + node_intent
        best_link = None
        best_score = 0
        
        for link in unique_links:
            link_lower = link.lower()
            score = 0
            for word in search_terms.split():
                if len(word) > 3 and word in link_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_link = link
        
        if best_link:
            self._logger.info(f"[VISIT] Found URL from sibling results: {best_link[:80]}")
            return best_link
        
        self._logger.info(f"[VISIT] Using first sibling link as fallback: {unique_links[0][:80]}")
        return unique_links[0]

    @staticmethod
    def _normalize_visit_url(url: Optional[str]) -> str:
        """Compare key for "these two leaves would fetch the same page".

        Scheme/host case and a trailing slash are not a different page; the query string is.
        The fragment is dropped because the fetch never sends it.
        """
        if not url or not isinstance(url, str):
            return ""
        try:
            parsed = urlparse(url.strip())
        except ValueError:
            return ""
        if not parsed.scheme or not parsed.netloc:
            return ""
        return urlunparse((
            parsed.scheme.lower(), parsed.netloc.lower(), (parsed.path or "").rstrip("/"),
            "", parsed.query, "",
        ))

    def _first_parent(self, graph: IdeaDag, node: IdeaNode) -> Optional[IdeaNode]:
        pids = node.parent_ids if node.parent_ids else ([node.parent_id] if node.parent_id else [])
        for pid in pids:
            if not pid:
                continue
            parent = graph.get_node(pid)
            if parent:
                return parent
        return None

    def _sibling_claimed_urls(self, graph: IdeaDag, node: IdeaNode) -> Set[str]:
        """Pages a visit sibling under the same parent has already fetched or claimed.

        Two sources, because a sibling batch runs concurrently: what a finished sibling
        recorded (``visit_url`` / its result's ``url``), and what an in-flight sibling wrote
        into the parent's claim map before awaiting its own fetch.
        """
        parent = self._first_parent(graph, node)
        if not parent:
            return set()
        out: Set[str] = set()
        claims = parent.details.get(VISIT_URL_CLAIMS_KEY)
        if isinstance(claims, dict):
            out.update(key for key, owner in claims.items() if key and owner != node.node_id)
        for sibling_id in parent.children:
            if sibling_id == node.node_id:
                continue
            sibling = graph.get_node(sibling_id)
            if not sibling:
                continue
            result = sibling.details.get(DetailKey.ACTION_RESULT.value)
            urls = [sibling.details.get("visit_url")]
            if isinstance(result, dict) and result.get("action") == IdeaActionType.VISIT.value:
                urls.append(result.get("url"))
            out.update(key for key in (self._normalize_visit_url(u) for u in urls) if key)
        return out

    def _claim_visit_urls(self, graph: IdeaDag, node: IdeaNode, urls: List[str]) -> None:
        """Record, on the parent, that this leaf is about to fetch these pages."""
        parent = self._first_parent(graph, node)
        if not parent:
            return
        claims = parent.details.get(VISIT_URL_CLAIMS_KEY)
        if not isinstance(claims, dict):
            claims = {}
            parent.details[VISIT_URL_CLAIMS_KEY] = claims
        for url in urls:
            key = self._normalize_visit_url(url)
            if key:
                claims.setdefault(key, node.node_id)

    def _links_worth_indexing(
        self,
        links: List[str],
        link_contexts: Dict[str, str],
        link_idea: str,
        limit: int,
    ) -> List[str]:
        """The ``limit`` links of this page worth paying the embedding cost for.

        Every stored link is embedded CLIENT-side by ChromaDB's default function before the
        add is sent, so the store's price is linear in link count (~19ms each) and is paid on
        the event loop. A link-dense page (a Wikipedia article yields ~990 links after chrome
        filtering) therefore costs ~19s -- more than the whole visit budget. Cutting the list
        by document order alone would bias the index to the lead section, so when the visit
        names what it is looking for the survivors are ranked by lexical overlap of their
        anchor+path with that idea (document order breaks ties), which keeps a target link
        that sits deep in the body.
        """
        if limit <= 0 or len(links) <= limit:
            return links
        haystack = set(self._URL_WORD_RE.findall((link_idea or "").lower()))
        if not haystack:
            return links[:limit]
        scored: List[Tuple[int, int, str]] = []
        for index, url in enumerate(links):
            text = f"{link_contexts.get(url) or ''} {urlparse(url).path or ''}".lower()
            tokens = set(self._URL_WORD_RE.findall(text)) - self._URL_NOISE_TOKENS
            scored.append((len(tokens & haystack), -index, url))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        kept = {url for _, _, url in scored[:limit]}
        return [url for url in links if url in kept]

    async def _store_links_in_chroma(
        self,
        base_url: str,
        links: List[str],
        link_contexts: Dict[str, str],
        io: AgentIO,
        link_idea: str = "",
    ) -> bool:
        if not links or not getattr(io, "connector_chroma", None):
            return False

        limit = self._cfg.action.visit_link_store_max
        if limit > 0 and len(links) > limit:
            self._logger.info(
                f"[VISIT] Page has {len(links)} links; indexing the {limit} most relevant "
                f"(embedding every one costs ~{len(links) * 0.02:.0f}s of the visit budget)"
            )
            links = self._links_worth_indexing(links, link_contexts, link_idea, limit)

        try:
            if not base_url or not isinstance(base_url, str):
                self._logger.warning(f"[VISIT] Invalid base_url for link storage: {base_url}")
                return False
            
            import hashlib
            url_hash = hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]
            # Scoped to this run's collection namespace (``io.collection_name``, e.g.
            # ``idea_test_{test_id}_{run_stamp}`` or ``agent_visit_{correlation_id}``) so a
            # page's outbound-link index cannot bleed into an unrelated later run/task whose
            # link_idea happens to embed close to it -- the read side below only ever matches
            # collections in this same namespace.
            run_scope = getattr(io, "collection_name", None) or "agent_memory"
            collection_name = f"links_{run_scope}_{url_hash}"
            docs: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []
            
            for idx, url in enumerate(links):
                anchor = (link_contexts.get(url) or "").strip()
                parsed = urlparse(url)
                path = parsed.path or "/"
                
                doc = f"{anchor} | {path}" if anchor else path
                docs.append(doc)
                metadatas.append({
                    "url": url,
                    "anchor": anchor,
                    "path": path,
                    "host": parsed.netloc.lower(),
                    "source_url": base_url,
                })
                url_id_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
                ids.append(f"link_{idx}_{url_id_hash}")
            
            if docs:
                # Batched (add_to_chroma_parallel falls back to a single add when the set
                # fits one batch): the client-side embedding inside each add blocks the
                # event loop for its whole batch, so one op per ~50 docs keeps the stall
                # near a second and lets `chroma_op_timeout` actually fire between batches.
                store = getattr(
                    io.connector_chroma, "add_to_chroma_parallel", io.connector_chroma.add_to_chroma
                )
                success = await store(
                    collection=collection_name,
                    ids=ids,
                    metadatas=metadatas,
                    documents=docs,
                )
                if success:
                    self._logger.debug(f"[VISIT] Stored {len(links)} links in Chroma collection '{collection_name}'")
                return bool(success)
        except Exception as exc:
            self._logger.warning(f"[VISIT] Failed to store links in Chroma: {exc}")
        return False
    
    async def _query_links_from_chroma(
        self,
        link_idea: str,
        io: AgentIO,
        top_k: int = 10,
    ) -> List[str]:
        if not link_idea or not getattr(io, "connector_chroma", None):
            return []
        
        try:
            if not hasattr(io.connector_chroma, "list_collections"):
                self._logger.debug(f"[VISIT] ConnectorChroma does not support list_collections")
                return []
            
            all_collections = await io.connector_chroma.list_collections()
            if not all_collections:
                self._logger.debug(f"[VISIT] No collections found in ChromaDB")
                return []
            
            # Scoped to this run's collection namespace -- see the matching comment in
            # _store_links_in_chroma. Without this prefix a query would match ANY link
            # index ever written by ANY run/task, including unrelated ones from before or
            # after this run.
            run_scope = getattr(io, "collection_name", None) or "agent_memory"
            run_prefix = f"links_{run_scope}_"
            link_collections = [c for c in all_collections if c and c.startswith(run_prefix)]

            if not link_collections:
                self._logger.debug(f"[VISIT] No link collections found for query: {link_idea}")
                return []
            
            all_results: List[Tuple[float, str]] = []
            
            for collection_name in link_collections:
                try:
                    query_result = await io.connector_chroma.query_chroma(
                        collection=collection_name,
                        query_texts=[link_idea],
                        n_results=min(top_k, 20),
                    )
                    if query_result and "metadatas" in query_result and "distances" in query_result:
                        meta_lists = query_result.get("metadatas") or []
                        dist_lists = query_result.get("distances") or []
                        if meta_lists and dist_lists:
                            for meta, dist in zip(meta_lists[0], dist_lists[0]):
                                if isinstance(meta, dict):
                                    url_value = meta.get("url")
                                    if url_value and isinstance(url_value, str):
                                        distance = float(dist) if isinstance(dist, (int, float)) else 1.0
                                        all_results.append((distance, url_value))
                except Exception as coll_exc:
                    self._logger.debug(f"[VISIT] Failed to query collection {collection_name}: {coll_exc}")
                    continue
            
            all_results.sort(key=lambda x: x[0])
            unique_urls: List[str] = []
            seen = set()
            for _, url in all_results:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
                    if len(unique_urls) >= top_k:
                        break
            
            self._logger.debug(f"[VISIT] Found {len(unique_urls)} links matching '{link_idea}' from {len(link_collections)} collections")
            return unique_urls
        except Exception as exc:
            self._logger.warning(f"[VISIT] Failed to query links from Chroma: {exc}")
            return []
    
    #: Host/path noise that says nothing about WHICH page a URL is.
    _URL_NOISE_TOKENS = frozenset({
        "www", "com", "org", "net", "edu", "gov", "int", "html", "htm", "php", "asp",
        "wiki", "page", "pages", "index", "article", "articles",
    })
    _URL_WORD_RE = re.compile(r"[a-z0-9]+")

    def _pick_link_by_name(self, link_idea: str, candidate_urls: List[str]) -> Optional[str]:
        """The ONE candidate URL whose page the ``link_idea`` literally names — or None.

        A deterministic pre-step for :meth:`_select_links_with_llm`, for the common case where
        the visit is not "follow the link that means X" but "open THIS entity's page" (every
        plan-library leaf, and planner leaves like "Visit the Axolotl Wikipedia page"). There
        the model call is pure overhead — and worse on a small/local executor, where a fan-out
        of visits fires one concurrent selection prompt each: measured on a 7B model, four of
        five hit the 20s action watchdog and the one that answered picked the wrong page
        (``/Muztagh_Tower`` for a "Muztagh Ata" leaf).

        Deliberately narrow — it answers ONLY about candidates whose whole slug is named in the
        ``link_idea``; a descriptive idea ("the rocket that launched the mission") names no
        slug at all, so that case still reaches the model, which is what it is for. Among
        named candidates it prefers, in order: the most specific match (so "Muztagh Ata"
        outranks a bare "Muztagh" hub page), then a host the idea also names (so "the Wikipedia
        page" beats a content mirror carrying the identical slug), then search rank.
        """
        from agent.app.idea_visit_dedup import url_slug_tokens

        haystack = set(self._URL_WORD_RE.findall((link_idea or "").lower()))
        if not haystack:
            return None

        named: List[Tuple[int, float, int, str]] = []
        for index, url in enumerate(candidate_urls):
            slug = [t for t in url_slug_tokens(url) if t not in self._URL_NOISE_TOKENS]
            if not slug:
                continue
            matched = [t for t in slug if t in haystack]
            if len(matched) < len(slug):
                continue  # part of this page's identity is NOT what the idea asked for
            host = [
                t for t in self._URL_WORD_RE.findall((urlparse(url).netloc or "").lower())
                if len(t) >= 3 and t not in self._URL_NOISE_TOKENS
            ]
            host_hit = 1.0 if any(t in haystack for t in host) else 0.0
            named.append((len(matched), host_hit, index, url))

        if not named:
            return None
        best = max(named, key=lambda entry: (entry[0], entry[1], -entry[2]))
        self._logger.info(
            f"[VISIT] Link idea names one page outright; picked {best[3][:80]} "
            f"deterministically (no selection prompt)"
        )
        return best[3]

    async def _select_links_with_llm(
        self,
        link_idea: str,
        candidate_urls: List[str],
        link_count: int,
        io: AgentIO,
    ) -> List[str]:
        if not candidate_urls or link_count <= 0:
            return []
        
        if len(candidate_urls) <= link_count:
            return candidate_urls[:link_count]
        
        try:
            # No explicit selection model -> use the connector's current (execution) model
            # via model_name=None, so the agent can reason about which discovered link
            # matches a descriptive link_idea (e.g. "rocket that launched the mission" ->
            # Saturn V). Cost attributes to the execution model, like other leaf LLM calls.
            model_name = self._cfg.action.visit_link_selection_model or self._cfg.evaluation.model or None

            candidates_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(candidate_urls)])
            system_content = f"Select the top {link_count} URLs that best match the user's request. Return JSON with a 'selected' array of URLs in order of preference."
            user_content = f"User wants: {link_idea}\n\nCandidate URLs:\n{candidates_text}\n\nReturn JSON: {{\"selected\": [\"url1\", \"url2\", ...]}}"

            messages = PromptBuilder.build_messages(system_content=system_content, user_content=user_content)
            # A3b/A5 discipline for this perception/selection micro-prompt (opt-in; no-op default):
            # minimal reasoning-effort + a token floor for reasoning models, and price-tier budget
            # scaling — so a small 500-token budget can't starve a reasoning executor.
            _effective_model = self._effective_model(io, model_name)
            payload = io.build_llm_payload(
                messages=messages,
                json_mode=True,
                model_name=model_name,
                temperature=0.2,
                max_tokens=self._executor_max_tokens(_effective_model, 500),
                reasoning_effort=self._micro_prompt_reasoning_effort(_effective_model),
            )
            
            timeout_seconds = self._timeout_seconds("llm_timeout_seconds")
            response = await io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=timeout_seconds,
            )
            
            if response:
                try:
                    data = json.loads(response)
                    selected = data.get("selected", [])
                    if isinstance(selected, list):
                        valid_selected = [url for url in selected if url in candidate_urls]
                        if valid_selected:
                            self._logger.debug(f"[VISIT] LLM selected {len(valid_selected)} links from {len(candidate_urls)} candidates")
                            return valid_selected[:link_count]
                        # Nothing the model picked was in the (possibly poisoned/off-topic)
                        # candidate pool. If it still answered with well-formed absolute
                        # URL(s), prefer those over silently substituting the first
                        # candidate -- the model's own answer is more likely right than an
                        # arbitrary index-0 pick, especially when the pool itself is
                        # off-topic (see the run-scoping fix above). Fall back to
                        # candidate_urls only when the answer itself is unusable.
                        off_list_urls = [
                            url for url in selected
                            if isinstance(url, str) and self._looks_like_url(url)
                        ]
                        if off_list_urls:
                            self._logger.warning(
                                f"[VISIT] LLM selected {len(off_list_urls)} URL(s) not in the "
                                f"{len(candidate_urls)}-candidate pool; using the model's answer "
                                f"instead of falling back to candidate_urls[0] (selected="
                                f"{off_list_urls[:link_count]}, pool sample={candidate_urls[:3]})"
                            )
                            return off_list_urls[:link_count]
                        self._logger.warning(
                            f"[VISIT] LLM link selection returned no usable URL (selected="
                            f"{selected!r}); falling back to the first {link_count} candidate(s)"
                        )
                    else:
                        self._logger.warning(
                            f"[VISIT] LLM link selection response had a non-list 'selected' field "
                            f"(selected={selected!r}); falling back to the first {link_count} candidate(s)"
                        )
                except json.JSONDecodeError:
                    self._logger.warning(f"[VISIT] Failed to parse LLM link selection response: {response[:200]}")
        except Exception as exc:
            self._logger.warning(f"[VISIT] LLM link selection failed: {exc}")

        return candidate_urls[:link_count]

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        """Basic sanity check: absolute http(s) URL with a real host, not garbage.

        Used only to decide whether an off-candidate-list model answer is trustworthy
        enough to use in place of the ``candidate_urls[:link_count]`` fallback -- not a
        full RFC validator.
        """
        value = value.strip()
        if not value or not (value.startswith("http://") or value.startswith("https://")):
            return False
        try:
            parsed = urlparse(value)
        except ValueError:
            return False
        return bool(parsed.netloc) and "." in parsed.netloc
    
    def _parse_visit_html(self, raw_html: str, url: str) -> Dict[str, Any]:
        """
        CPU-bound HTML parsing for a visited page. Pure/synchronous so it can be
        offloaded to a thread executor, keeping the event loop responsive while
        concurrent page fetches and LLM calls proceed.

        :param raw_html: Raw page HTML.
        :param url: Source URL (for link resolution).
        :returns: Dict of parsed/derived fields consumed by _visit_single_page.
        """
        cleaned = clean_operation(raw_html) or ""
        soup = BeautifulSoup(raw_html, "html.parser")
        raw_links = []
        link_contexts = {}

        for tag in soup.find_all("a", href=True):
            href = tag.get("href")
            if href:
                raw_links.append(href)
                link_text = tag.get_text(strip=True)
                if link_text:
                    link_contexts[href] = link_text[:200]

        page_title = ""
        title_tag = soup.find("title")
        if title_tag:
            page_title = title_tag.get_text(strip=True)

        h1_text = ""
        h1_tag = soup.find("h1")
        if h1_tag:
            h1_text = h1_tag.get_text(separator=" ", strip=True)

        cleaned_links = self._filter_and_prioritize_links(raw_links, url)
        cleaned_link_contexts = {}
        for raw_link in raw_links:
            fixed_link = self._clean_and_fix_link(raw_link, url)
            if fixed_link and raw_link in link_contexts:
                cleaned_link_contexts[fixed_link] = link_contexts[raw_link]

        content_payload = self._limit_text(cleaned)
        content_text = content_payload.get("content") or cleaned or ""
        if not content_text or len(content_text.strip()) == 0:
            content_text = soup.get_text(separator="\n", strip=True)
            if content_text:
                content_payload = self._limit_text(content_text)
                content_text = content_payload.get("content") or content_text

        max_links_for_llm = self._cfg.action.max_links_per_visit
        links_for_llm = cleaned_links[:max_links_for_llm]
        content_with_links = self._attach_links_to_content(content_text, links_for_llm, max_links=max_links_for_llm)

        final_content = content_payload.get("content") or content_text or ""
        content_total_chars = content_payload.get("total_chars", len(final_content))

        return {
            "cleaned": cleaned,
            "cleaned_links": cleaned_links,
            "cleaned_link_contexts": cleaned_link_contexts,
            "page_title": page_title,
            "h1_text": h1_text,
            "content_text": content_text,
            "content_payload": content_payload,
            "links_for_llm": links_for_llm,
            "content_with_links": content_with_links,
            "final_content": final_content,
            "content_total_chars": content_total_chars,
        }

    async def _visit_single_page(
        self,
        url: str,
        graph: IdeaDag,
        node: IdeaNode,
        io: AgentIO,
        intent: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, str]]:
        from agent.app.idea_policies.action_constants import ErrorType
        
        blocked_reason = graph.is_site_blocked(str(url))
        if blocked_reason:
            self._logger.warning(f"[VISIT] Site blocked: {url} - {blocked_reason}")
            return (
                ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error=f"Site blocked: {blocked_reason}",
                    error_type=ErrorType.BLOCKED_SITE.value,
                    retryable=False,
                    url=url,
                    intent=intent,
                ),
                [],
                {},
            )
        
        timeout_seconds = self._timeout_seconds("fetch_timeout_seconds")
        
        use_browser = bool(getattr(io, "connector_browser", None))
        if use_browser:
            self._logger.info(
                f"[VISIT] Using browser connector for {url[:80]}... "
                f"(will wait for page load, mimic human behavior)"
            )
        else:
            self._logger.info(
                f"[VISIT] Using HTTP connector for {url[:80]}... "
                f"(browser not available, using direct HTTP)"
            )
        
        raw_html = await io.fetch_url(str(url), timeout_seconds=timeout_seconds)
        if not raw_html:
            return (
                ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error="Failed to fetch URL - no content returned",
                    error_type=ErrorType.NETWORK_ERROR.value,
                    retryable=True,
                    url=url,
                ),
                [],
                {},
            )
        
        # HTML parsing (clean_operation + BeautifulSoup) is CPU-bound and pure
        # Python; run it in a thread executor so it doesn't freeze the event
        # loop while sibling page fetches / LLM calls proceed concurrently.
        parsed = await asyncio.get_running_loop().run_in_executor(
            None, self._parse_visit_html, raw_html, str(url)
        )
        cleaned = parsed["cleaned"]
        cleaned_links = parsed["cleaned_links"]
        cleaned_link_contexts = parsed["cleaned_link_contexts"]
        page_title = parsed["page_title"]
        h1_text = parsed["h1_text"]
        content_text = parsed["content_text"]
        content_payload = parsed["content_payload"]
        links_for_llm = parsed["links_for_llm"]
        content_with_links = parsed["content_with_links"]
        final_content = parsed["final_content"]
        content_total_chars = parsed["content_total_chars"]

        # Storing every page's links in Chroma (embedding 1000+ link contexts) is the
        # dominant per-visit cost and is only useful when this visit will FOLLOW links
        # (link_count>1 or a link_idea). For single-URL fact reads it is pure waste, so
        # skip it — this keeps the graph cheap/fast on small tasks. Navigation visits
        # (046/047) still populate Chroma.
        _lc = node.details.get("link_count")
        try:
            _lc = int(_lc) if _lc is not None else 1
        except (TypeError, ValueError):
            _lc = 1
        _following_links = _lc > 1 or bool(node.details.get("link_idea"))
        if _following_links:
            _idea = str(
                node.details.get("link_idea") or node.details.get("link_concept") or intent or node.title or ""
            )
            await self._store_links_in_chroma(
                str(url), cleaned_links, cleaned_link_contexts, io, link_idea=_idea
            )
        else:
            self._logger.debug(f"[VISIT] Single-URL read; skipping Chroma link store for {str(url)[:60]}")

        if not content_text or len(content_text.strip()) == 0:
            self._logger.warning(
                f"[VISIT] No extractable content from {url[:80]}. "
                f"HTML length: {len(raw_html)}, cleaned length: {len(cleaned)}"
            )
        
        if not final_content or len(final_content.strip()) == 0:
            self._logger.error(
                f"[VISIT] Failed to extract any content from {url[:80]}. "
                f"Marking visit as failed."
            )
            return (
                ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error=f"Visit succeeded but no content could be extracted from page. HTML length: {len(raw_html)} chars",
                    error_type=ErrorType.PARSE_ERROR.value,
                    retryable=True,
                    url=url,
                    intent=intent,
                ),
                cleaned_links,
                cleaned_link_contexts,
            )
        
        self._logger.info(
            f"[VISIT] Successfully retrieved content from {url[:80]}: "
            f"{content_total_chars} chars, {len(cleaned_links)} links, "
            f"title: {page_title[:60] if page_title else 'N/A'}"
        )
        
        result = ActionResultBuilder.success(
            action=IdeaActionType.VISIT.value,
            url=url,
            intent=intent,
            content=final_content,
            content_is_truncated=content_payload.get("is_truncated", False),
            content_total_chars=content_total_chars,
            content_full=cleaned,
            content_with_links=content_with_links,
            links=links_for_llm,
            links_full=cleaned_links,
            links_count=len(cleaned_links),
            link_contexts=cleaned_link_contexts,
            page_title=page_title if page_title else None,
            h1_text=h1_text if h1_text else None,
            source_url=url,
        )
        
        return result, cleaned_links, cleaned_link_contexts
    
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        intent = None
        vector_context = []
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            from agent.app.idea_policies.action_constants import ErrorType

            intent = self._effective_intent(node)

            link_count = node.details.get("link_count")
            if link_count is None:
                link_count = 1
            else:
                try:
                    link_count = int(link_count)
                except (ValueError, TypeError):
                    link_count = 1
            
            link_idea = node.details.get("link_idea") or node.details.get("link_concept") or ""
            original_optional_url = node.details.get("optional_url") or NodeDetailsExtractor.get_url(node.details)
            optional_url = original_optional_url
            
            if optional_url:
                optional_url_str = str(optional_url).strip()
                if optional_url_str.lower() in ("none", "null", ""):
                    optional_url = None
                elif not self._is_valid_url(optional_url_str):
                    optional_url = None
            
            if optional_url and not self._is_valid_url(optional_url):
                if optional_url.startswith("<") or optional_url.startswith("{") or "chosen" in optional_url.lower():
                    self._logger.warning(f"[VISIT] Clearing placeholder URL: {optional_url[:80]}")
                    optional_url = None
            
            # Does this node NAME the search whose URLs it is meant to open? If so that source
            # is authoritative and the opportunistic scavenging below must not pre-empt it:
            # `_extract_url_from_parents` / `_extract_url_from_sibling_results` return the first
            # URL any relative produced, which under a fan-out of per-entity page reads is
            # routinely a SIBLING ENTITY's search hit — visited immediately (link_count == 1
            # short-circuits) and never reconciled against this leaf's own `link_idea`.
            has_named_url_source = False
            required_data = node.details.get(DetailKey.REQUIRES_DATA.value)
            if required_data and isinstance(required_data, dict):
                source_node_id = required_data.get("source_node_id")
                data_type = required_data.get("type", "")
                if source_node_id and data_type == "urls_from_search":
                    source_node = graph.get_node(source_node_id)
                    if source_node:
                        from agent.app.idea_dag import IdeaNodeStatus
                        if source_node.status != IdeaNodeStatus.DONE:
                            self._log_structured(
                                "warning",
                                "[VISIT] Node requires URLs from search node but source not ready",
                                node_id=node_id[:16],
                                source_node_id=source_node_id[:16],
                                source_status=source_node.status.value,
                                action="visit",
                                issue="dependency_not_ready"
                            )
                        else:
                            has_named_url_source = True
                            self._logger.info(
                                f"[VISIT] Node {node_id[:8]}... depends on search node {source_node_id[:8]}... "
                                f"(status: {source_node.status.value})"
                            )

            # A URL this leaf did NOT ask for is one the cascade below handed it, and the
            # cascade is sibling-blind: `_extract_url_from_parents`, `_extract_url_from_
            # sibling_results` and the Chroma link query each rank the SAME pool the same way
            # for every sibling that arrives without a URL, so a fan-out of per-entity page
            # reads collapses onto one page (52 of 101 duplicate sibling-visit groups in the
            # recorded corpus resolved this way; ASSUMPTION_AUDIT.md T1-4).
            sibling_dedup = self._cfg.action.visit_sibling_url_dedup and not has_named_url_source
            url_was_declared = bool(optional_url and self._is_valid_url(str(optional_url)))
            claimed_urls: Set[str] = (
                self._sibling_claimed_urls(graph, node) if sibling_dedup else set()
            )
            dropped_duplicate: Optional[str] = None

            if not has_named_url_source and (not optional_url or not self._is_valid_url(optional_url)):
                think_url = self._extract_url_from_think_node(graph, node)
                if think_url:
                    optional_url = think_url
                    self._logger.info(f"[VISIT] Extracted URL from think node: {think_url[:80]}")
                else:
                    extracted_url = self._extract_url_from_parents(graph, node)
                    if extracted_url:
                        optional_url = extracted_url
                        self._logger.info(f"[VISIT] Extracted URL from parent nodes: {extracted_url[:80]}")
                    else:
                        sibling_url = self._extract_url_from_sibling_results(graph, node)
                        if sibling_url:
                            optional_url = sibling_url
                            self._logger.info(f"[VISIT] Extracted URL from sibling results: {sibling_url[:80]}")
                        else:
                            self._logger.debug(f"[VISIT] No URL found in node details, think node, parents, or siblings")

            # Only a fallback-resolved URL is dropped. An explicitly declared one is this
            # leaf's own instruction, so a duplicate there is the planner's call to make.
            if (
                sibling_dedup
                and not url_was_declared
                and optional_url
                and self._normalize_visit_url(str(optional_url)) in claimed_urls
            ):
                dropped_duplicate = str(optional_url)
                optional_url = None
                self._logger.info(
                    f"[VISIT] Sibling URL dedup: dropped fallback URL already claimed by a "
                    f"sibling: {dropped_duplicate[:80]}"
                )

            if not link_idea and not (optional_url and self._is_valid_url(optional_url)):
                link_idea = intent or node.title or ""
                if link_idea:
                    link_idea = link_idea[:200]
                    self._logger.info(f"[VISIT] Auto-generated link_idea from context: '{link_idea[:60]}...'")
            
            max_sites = self._cfg.action.visit_max_sites_per_action
            link_count = min(link_count, max_sites)
            
            urls_to_visit: List[str] = []
            optional_success = False
            
            # A declared URL whose fetch RAISES (permanent HTTP status: `io.fetch_url` raises on
            # 404/403) used to abort the action here, skipping the recovery cascade below that a
            # RETURNED failure already falls through to. Held instead and re-raised only if the
            # cascade resolves nothing new, so the failure surface is unchanged when it can't help.
            declared_url_error: Optional[BaseException] = None

            if optional_url and self._is_valid_url(optional_url):
                # Claimed BEFORE the fetch is awaited: a concurrent sibling that resolves
                # while this one is in flight has to see the page as taken.
                if sibling_dedup:
                    self._claim_visit_urls(graph, node, [optional_url])
                if self._cfg.action.visit_dead_url_fallback_enabled:
                    try:
                        result, _, _ = await self._visit_single_page(optional_url, graph, node, io, intent)
                    except Exception as fetch_exc:
                        declared_url_error = fetch_exc
                        result = None
                        self._logger.warning(
                            f"[VISIT] Declared URL fetch raised ({str(fetch_exc)[:120]}); "
                            f"falling through to URL recovery"
                        )
                else:
                    result, _, _ = await self._visit_single_page(optional_url, graph, node, io, intent)
                if result and result.get("success"):
                    content_length = result.get("content_total_chars") or len(result.get("content", "") or "")
                    optional_success = True
                    urls_to_visit.append(optional_url)
                    self._logger.info(
                        f"[VISIT] Optional URL visited successfully: {optional_url[:60]}... "
                        f"Content: {content_length} chars, Status: SUCCESS"
                    )
                    if link_count == 1:
                        if io.telemetry:
                            io.telemetry.record_document_seen(
                                source="visit",
                                document={"url": optional_url, "content": result.get("content_with_links", result.get("content", ""))},
                            )
                        return result
                else:
                    error_from_result = result.get("error", "Unknown error") if result else "No result returned"
                    self._logger.warning(
                        f"[VISIT] Optional URL visit failed: {optional_url[:60]}... "
                        f"Error: {error_from_result[:100]}"
                    )
            
            if not optional_success or link_count > 1:
                candidate_urls = []

                # A declared URL suppresses the auto-generated link_idea above (a leaf that names
                # its page has no link to discover). Once that page turns out not to exist, the
                # leaf DOES need one, or the stored-link query below is skipped and the recovery
                # cascade has nothing left to try.
                if declared_url_error is not None and not link_idea:
                    link_idea = (intent or node.title or "")[:200]
                    if link_idea:
                        self._logger.info(
                            f"[VISIT] Dead declared URL: recovering via link_idea "
                            f"'{link_idea[:60]}...'"
                        )

                self._logger.info(
                    f"[VISIT] Attempting to extract URLs from search results. "
                    f"Node: {node_id[:8]}..., link_idea: '{link_idea[:60] if link_idea else 'None'}...', "
                    f"link_count: {link_count}"
                )
                parent_search_urls = self._extract_urls_from_parent_search_results(graph, node)
                if parent_search_urls:
                    self._logger.info(
                        f"[VISIT] Successfully extracted {len(parent_search_urls)} URLs from search results. "
                        f"Sample URLs: {', '.join([url[:50] + '...' if len(url) > 50 else url for url in parent_search_urls[:3]])}"
                    )
                    candidate_urls.extend(parent_search_urls)
                else:
                    sibling_url = self._extract_url_from_sibling_results(graph, node)
                    if sibling_url:
                        candidate_urls.append(sibling_url)
                        self._logger.info(f"[VISIT] Fallback: extracted 1 URL from sibling results")
                    else:
                        self._logger.warning(
                            f"[VISIT] No URLs extracted from search results. "
                            f"This may indicate: (1) search node hasn't completed, "
                            f"(2) search returned no results, or (3) search results format is unexpected."
                        )
                
                if link_idea and len(candidate_urls) < link_count:
                    query_top_k = self._cfg.action.visit_link_query_top_k
                    chroma_urls = await self._query_links_from_chroma(link_idea, io, top_k=query_top_k)
                    if chroma_urls:
                        seen = set(candidate_urls)
                        for url in chroma_urls:
                            if url not in seen:
                                candidate_urls.append(url)
                                seen.add(url)

                if declared_url_error is not None:
                    harvested = self._harvest_relative_page_links(
                        graph, node, link_idea or node.title or "",
                        limit=self._cfg.action.visit_link_query_top_k,
                    )
                    seen = set(candidate_urls)
                    for url in harvested:
                        if url not in seen:
                            candidate_urls.append(url)
                            seen.add(url)

                if declared_url_error is not None and candidate_urls:
                    dead = self._normalize_visit_url(str(optional_url))
                    candidate_urls = [
                        u for u in candidate_urls if self._normalize_visit_url(u) != dead
                    ]

                candidate_urls = self._drop_chrome_urls(candidate_urls, "the resolved pool")

                # Re-read the claims: the extraction above awaited, so a sibling may have
                # taken a page since this leaf started resolving. Dropping the taken ones
                # here is what lets the selection below pick the NEXT candidate rather than
                # re-picking the one every sibling ranks first.
                if sibling_dedup and candidate_urls:
                    claimed_urls = self._sibling_claimed_urls(graph, node)
                    kept = [
                        u for u in candidate_urls
                        if self._normalize_visit_url(u) not in claimed_urls
                    ]
                    if len(kept) != len(candidate_urls):
                        self._logger.info(
                            f"[VISIT] Sibling URL dedup: dropped "
                            f"{len(candidate_urls) - len(kept)} candidate URL(s) already "
                            f"claimed by a sibling ({len(kept)} left)"
                        )
                        candidate_urls = kept

                if candidate_urls:
                    if link_count > len(urls_to_visit):
                        needed = link_count - len(urls_to_visit)
                        if len(candidate_urls) > needed:
                            named = (
                                self._pick_link_by_name(link_idea or node.title, candidate_urls)
                                if needed == 1 else None
                            )
                            if named:
                                urls_to_visit.append(named)
                            else:
                                selected = await self._select_links_with_llm(link_idea or node.title, candidate_urls, needed, io)
                                urls_to_visit.extend(selected)
                        else:
                            urls_to_visit.extend(candidate_urls[:needed])
                else:
                    if declared_url_error is not None:
                        raise declared_url_error
                    if not optional_url or not optional_success:
                        parent_urls_found = len(parent_search_urls) if 'parent_search_urls' in locals() else 0
                        chroma_urls_found = len(chroma_urls) if 'chroma_urls' in locals() else 0
                        error_msg = (
                            f"Visit node missing valid URL or link_idea. "
                            f"Node title: '{node.title}'. "
                            f"Details: optional_url='{original_optional_url}', link_idea='{link_idea}', "
                            f"link_count={link_count}. "
                            f"URL extraction attempts: parent_search_urls={parent_urls_found}, "
                            f"chroma_urls={chroma_urls_found}. "
                            f"Details should contain 'url'/'optional_url' or 'link_idea' for semantic link discovery."
                        )
                        if dropped_duplicate:
                            error_msg += (
                                f" Every resolvable URL was already claimed by a sibling "
                                f"(first dropped: {dropped_duplicate}); this leaf would have "
                                f"re-fetched a page the batch already has."
                            )
                        self._log_structured(
                        "error",
                        "[VISIT] Visit node missing valid URL or link_idea",
                        node_id=node_id[:16],
                        node_title=node.title[:100] if node.title else "None",
                        optional_url=original_optional_url[:100] if original_optional_url else "None",
                        link_idea=link_idea[:100] if link_idea else "None",
                        link_count=link_count,
                        parent_search_urls_found=parent_urls_found,
                        chroma_urls_found=chroma_urls_found,
                        action="visit",
                        error_type=ErrorType.INVALID_URL.value,
                        retryable=False
                    )
                        return ActionResultBuilder.failure(
                            action=IdeaActionType.VISIT.value,
                            error=error_msg,
                            error_type=ErrorType.INVALID_URL.value,
                            retryable=False,
                            url=optional_url or node.title,
                        )
            
            if not urls_to_visit:
                if declared_url_error is not None:
                    raise declared_url_error
                if optional_url and self._is_valid_url(optional_url):
                    urls_to_visit = [optional_url]
                else:
                    parent_urls_found = len(parent_search_urls) if 'parent_search_urls' in locals() else 0
                    chroma_urls_found = len(chroma_urls) if 'chroma_urls' in locals() else 0
                    error_msg = (
                        f"Visit node: no URLs to visit. "
                        f"link_count={link_count}, link_idea='{link_idea}', "
                        f"optional_url='{original_optional_url}'. "
                        f"URL extraction results: parent_search_urls={parent_urls_found}, "
                        f"chroma_urls={chroma_urls_found}. "
                        f"Diagnosis: {'No URLs found in parent search results' if parent_urls_found == 0 else 'URLs found but not selected'}. "
                        f"Action: Ensure visit node has valid 'url'/'optional_url' field or depends on a search node that provides URLs."
                    )
                    self._log_structured(
                        "error",
                        "[VISIT] No URLs to visit after extraction attempts",
                        node_id=node_id[:16],
                        link_count=link_count,
                        link_idea=link_idea[:100] if link_idea else "None",
                        optional_url=original_optional_url[:100] if original_optional_url else "None",
                        parent_search_urls_found=parent_urls_found,
                        chroma_urls_found=chroma_urls_found,
                        diagnosis="no_urls_found" if parent_urls_found == 0 else "urls_not_selected",
                        action="visit",
                        error_type=ErrorType.INVALID_URL.value,
                        retryable=False
                    )
                    return ActionResultBuilder.failure(
                        action=IdeaActionType.VISIT.value,
                        error=error_msg,
                        error_type=ErrorType.INVALID_URL.value,
                        retryable=False,
                    )
            
            visited_results: List[Dict[str, Any]] = []
            all_links: List[str] = []
            all_link_contexts: Dict[str, str] = {}
            combined_content: List[str] = []
            all_page_titles: List[str] = []
            all_h1_texts: List[str] = []
            
            # Dedup URLs preserving input order (first URL stays primary), then
            # fetch pages concurrently. aiohttp's ClientSession is safe for
            # concurrent requests; a semaphore caps in-flight fetches so a burst
            # of CPU-bound HTML parsing can't starve the event loop.
            unique_urls = list(dict.fromkeys(urls_to_visit))
            if sibling_dedup:
                self._claim_visit_urls(graph, node, unique_urls)
            concurrency = max(1, self._cfg.action.visit_page_concurrency)
            semaphore = asyncio.Semaphore(concurrency)

            async def _visit_with_limit(target_url: str):
                async with semaphore:
                    return await self._visit_single_page(target_url, graph, node, io, intent)

            page_outcomes = await asyncio.gather(
                *[_visit_with_limit(u) for u in unique_urls],
                return_exceptions=True,
            )

            # gather preserves input order, so visited_results stays deterministic
            # and primary_result remains the first successful input URL.
            for url_to_visit, outcome in zip(unique_urls, page_outcomes):
                if isinstance(outcome, BaseException):
                    self._logger.warning(f"[VISIT] Error visiting {url_to_visit}: {outcome}")
                    continue

                result, page_links, page_link_contexts = outcome
                if result:
                    if result.get("success"):
                        visited_results.append(result)
                        all_links.extend(page_links)
                        all_link_contexts.update(page_link_contexts)
                        content = result.get("content") or result.get("content_full") or ""
                        if content:
                            combined_content.append(f"=== {url_to_visit} ===\n{content}")
                        if result.get("page_title"):
                            all_page_titles.append(f"{url_to_visit}: {result.get('page_title')}")
                        if result.get("h1_text"):
                            all_h1_texts.append(f"{url_to_visit}: {result.get('h1_text')}")
                    else:
                        self._logger.warning(f"[VISIT] Failed to visit {url_to_visit}: {result.get('error', 'Unknown error')}")
            
            if not visited_results:
                if declared_url_error is not None:
                    raise declared_url_error
                attempted_urls = ", ".join([url[:60] for url in urls_to_visit[:3]])
                error_msg = (
                    f"All URL visits failed. Attempted {len(urls_to_visit)} URL(s): {attempted_urls}"
                    f"{'...' if len(urls_to_visit) > 3 else ''}. "
                    f"Check network connectivity, site availability, or bot blocking."
                )
                self._logger.error(f"[VISIT] {error_msg}")
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error=error_msg,
                    error_type=ErrorType.NETWORK_ERROR.value,
                    retryable=True,
                )
            
            primary_result = visited_results[0]
            combined_content_text = "\n\n".join(combined_content)
            max_links_for_llm = self._cfg.action.max_links_per_visit
            links_for_llm = list(dict.fromkeys(all_links))[:max_links_for_llm]
            
            if h1_text := primary_result.get("h1_text"):
                node.details["h1_text"] = h1_text
            if page_title := primary_result.get("page_title"):
                node.details["page_title"] = page_title
            
            content_with_links = self._attach_links_to_content(combined_content_text, links_for_llm, max_links=max_links_for_llm)
            
            if io.telemetry:
                for result in visited_results:
                    if result.get("url"):
                        io.telemetry.record_document_seen(
                            source="visit",
                            document={"url": result.get("url"), "content": result.get("content_with_links", "")},
                        )
            
            primary_url = urls_to_visit[0] if urls_to_visit else None
            total_content_chars = len(combined_content_text)
            self._logger.info(
                f"[VISIT] Multi-page visit completed successfully: "
                f"{len(visited_results)}/{len(urls_to_visit)} pages visited, "
                f"total content: {total_content_chars} chars, "
                f"primary URL: {primary_url[:80] if primary_url else 'N/A'}"
                        )
            
            return ActionResultBuilder.success(
                action=IdeaActionType.VISIT.value,
                url=urls_to_visit[0] if urls_to_visit else None,
                urls_visited=urls_to_visit,
                intent=intent,
                vector_context=[str(doc) for doc in vector_context] if vector_context else [],
                content=combined_content_text[:self._max_observation_chars()] if len(combined_content_text) > self._max_observation_chars() else combined_content_text,
                content_is_truncated=len(combined_content_text) > self._max_observation_chars(),
                content_total_chars=len(combined_content_text),
                content_full=combined_content_text,
                content_with_links=content_with_links,
                links=links_for_llm,
                links_full=list(dict.fromkeys(all_links)),
                links_count=len(set(all_links)),
                link_contexts=all_link_contexts,
                page_title="; ".join(all_page_titles) if all_page_titles else primary_result.get("page_title"),
                h1_text="; ".join(all_h1_texts) if all_h1_texts else primary_result.get("h1_text"),
                sites_visited=len(visited_results),
            )
        except Exception as exc:
            resolved_url = optional_url if optional_url else None
            if not resolved_url and node:
                resolved_url = (
                    node.details.get(DetailKey.URL.value)
                    or node.details.get(DetailKey.LINK.value)
                    or node.title
                )
            
            error_str = str(exc)
            error_lower = error_str.lower()
            is_bot_block = False
            block_reason = None
            
            status_match = re.search(r"status=([0-9]{3})", error_str)
            if status_match:
                status = int(status_match.group(1))
                if status == 403:
                    is_bot_block = True
                    block_reason = "HTTP 403 Forbidden (bot blocking)"
                elif status == 401:
                    is_bot_block = True
                    block_reason = "HTTP 401 Unauthorized (authentication required)"
            
            bot_indicators = [
                ("forbidden", "HTTP 403 Forbidden"),
                ("access denied", "Access denied"),
                ("cloudflare", "Cloudflare bot protection"),
                ("bot detection", "Bot detection"),
                ("captcha", "CAPTCHA challenge"),
                ("blocked", "Site blocked"),
            ]
            for indicator, reason in bot_indicators:
                if indicator in error_lower:
                    is_bot_block = True
                    block_reason = reason
                    break
            
            if is_bot_block and resolved_url:
                graph.mark_site_blocked(str(resolved_url), block_reason or "Bot blocking detected")
                self._logger.warning(f"[VISIT] Marking site as blocked: {resolved_url} - {block_reason}")
            
            failure = self._failure(
                action=IdeaActionType.VISIT,
                node_id=node_id,
                error=exc,
                context={DetailKey.URL.value: resolved_url},
            )
            if resolved_url is not None:
                failure[ActionResultKey.URL.value] = resolved_url
            if is_bot_block:
                failure[ActionResultKey.RETRYABLE.value] = False
                failure[ActionResultKey.ERROR.value] = f"{block_reason}: {error_str}"
            return failure


class ThinkLeafAction(LeafAction):
    name = "think"

    def _extract_url_from_parent_result(self, graph: IdeaDag, node: IdeaNode) -> Optional[str]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        from agent.app.idea_policies.base import DetailKey
        
        requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
        if not isinstance(requires_data, dict):
            return None
        
        source_node_id = requires_data.get("source_node_id")
        if not source_node_id:
            return None
        
        source_node = graph.get_node(source_node_id)
        if not source_node:
            return None
        
        source_result = source_node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(source_result, dict):
            return None
        
        action_type = source_result.get(ActionResultKey.ACTION.value)
        
        if action_type == IdeaActionType.VISIT.value:
            links_inline = source_result.get("_links_inline")
            if links_inline and isinstance(links_inline, str):
                import re
                link_pattern = r'\[link:\s*(https?://[^\]]+)\]'
                matches = re.findall(link_pattern, links_inline, re.IGNORECASE)
                if matches:
                    return matches[0].strip()
            
            links = source_result.get("links") or source_result.get("links_full") or []
            if isinstance(links, list) and len(links) > 0:
                for link in links:
                    if isinstance(link, str) and link.startswith(("http://", "https://")):
                        return link
        
        elif action_type == IdeaActionType.SEARCH.value:
            results = source_result.get(ActionResultKey.RESULTS.value) or []
            if isinstance(results, list) and len(results) > 0:
                for result in results:
                    if isinstance(result, dict):
                        url = result.get("url") or result.get("link")
                        if url and isinstance(url, str) and url.startswith(("http://", "https://")):
                            return url
        
        return None
    
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            thinking_content = NodeDetailsExtractor.get_query(node.details, fallback_title=node.title) or ""
            
            extracted_url = None
            target_facts = node.details.get("target_facts", [])
            if isinstance(target_facts, list):
                for fact in target_facts:
                    if isinstance(fact, str) and ("chosen_url" in fact.lower() or "url" in fact.lower()):
                        extracted_url = self._extract_url_from_parent_result(graph, node)
                        if extracted_url:
                            node.details[DetailKey.URL.value] = extracted_url
                            node.details[DetailKey.LINK.value] = extracted_url
                            self._logger.info(f"[THINK] Extracted URL from parent: {extracted_url[:60]}...")
                        break
            
            if extracted_url and hasattr(io, 'store_chroma'):
                try:
                    doc_text = f"Selected URL: {extracted_url}\n\n{node.title}"
                    metadata = {
                        "node_id": node.node_id,
                        "action": "think",
                        "title": node.title[:200] if len(node.title) > 200 else node.title,
                        "extracted_url": extracted_url,
                    }
                    timeout_seconds = self._timeout_seconds("chroma_timeout_seconds")
                    await io.store_chroma(
                        documents=[doc_text],
                        metadatas=[metadata],
                        ids=[str(uuid.uuid4())],
                        timeout_seconds=timeout_seconds,
                    )
                    self._logger.debug(f"[THINK] Saved extracted URL to ChromaDB for node {node_id}")
                except Exception as chroma_exc:
                    self._logger.warning(f"[THINK] Failed to save to ChromaDB: {chroma_exc}")
            elif thinking_content and hasattr(io, 'store_chroma'):
                try:
                    doc_text = f"{node.title}\n\n{thinking_content}" if thinking_content != node.title else thinking_content
                    metadata = {
                        "node_id": node.node_id,
                        "action": "think",
                        "title": node.title[:200] if len(node.title) > 200 else node.title,
                    }
                    timeout_seconds = self._timeout_seconds("chroma_timeout_seconds")
                    await io.store_chroma(
                        documents=[doc_text],
                        metadatas=[metadata],
                        ids=[str(uuid.uuid4())],
                        timeout_seconds=timeout_seconds,
                    )
                    self._logger.debug(f"[THINK] Saved thinking content to ChromaDB for node {node_id}")
                except Exception as chroma_exc:
                    self._logger.warning(f"[THINK] Failed to save to ChromaDB: {chroma_exc}")
            
            details_copy = self._copy_details_safely(node.details)
            result = ActionResultBuilder.success(
                action=IdeaActionType.THINK.value,
                node_id=node.node_id,
                title=node.title,
                details=details_copy,
                thinking_content=thinking_content,
            )
            if extracted_url:
                result[ActionResultKey.URL.value] = extracted_url
                result["extracted_url"] = extracted_url
            return result
        except Exception as exc:
            return self._failure(action=IdeaActionType.THINK, node_id=node_id, error=exc)


class SaveLeafAction(LeafAction):
    name = "save"

    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            docs = node.details.get(DetailKey.DOCUMENTS.value)
            if docs is None:
                doc = node.details.get(DetailKey.DOCUMENT.value)
                docs = [doc] if doc else []
            if not docs:
                intent = node.details.get(DetailKey.INTENT.value) or ""
                title = node.title or ""
                fallback_text = f"{title}\n{intent}".strip()
                if fallback_text:
                    docs = [fallback_text]
                    self._logger.info(f"[SAVE] No documents provided, constructed from title/intent ({len(fallback_text)} chars)")
                else:
                    return ActionResultBuilder.success(
                        action=IdeaActionType.SAVE.value,
                        success=True,
                        count=0,
                    )
            metadatas = node.details.get(DetailKey.METADATAS.value) or [{"node_id": node_id, "action": "save", "title": node.title[:200]} for _ in range(len(docs))]
            ids = [str(uuid.uuid4()) for _ in range(len(docs))]
            timeout_seconds = self._timeout_seconds("chroma_timeout_seconds")
            success = await io.store_chroma(documents=docs, metadatas=metadatas, ids=ids, timeout_seconds=timeout_seconds)
            return ActionResultBuilder.success(
                action=IdeaActionType.SAVE.value,
                success=bool(success),
                count=len(docs),
            )
        except Exception as exc:
            return self._failure(action=IdeaActionType.SAVE, node_id=node_id, error=exc)


class PlanLibrarySearchLeafAction(LeafAction):
    """Ask the plan library for a pre-authored strategy for this node — READ-ONLY.

    The model invokes this when it would rather retrieve a proven composition strategy than
    invent one. The action ranks the template corpus, and on a confident hit spends the
    pipeline's one slot-extraction call to bind the winner to this mandate's entities; it then
    REPORTS what it found and stops. It never expands the graph: re-expansion bookkeeping (the
    ``_got_reexpanded`` marker, the lineage ``_got_reexpand_count`` budget, the
    ``max_total_nodes`` ceiling) lives on ``IdeaDagEngine`` and stays single-sourced there —
    ``IdeaDagEngine._maybe_plan_library_reexpand`` turns an ``adopted`` result into children.

    The result carries the bound ``slot_values`` alongside ``adopted_template_id`` precisely so
    that hook can rebuild the expansion deterministically (``candidates_from_template`` is
    pure) without a second Chroma query or a second extraction call — re-running extraction
    would not only cost that call but could produce values that disagree with the verdict
    already recorded here.

    Unreachable unless the engine patches ``plan_library_search`` into ``allowed_actions``
    (only when ``plan_library_enabled`` + ``plan_library_action_enabled``), so the default
    action menu — and every run that never arms the flags — is unchanged.
    """

    name = "plan_library_search"

    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            # Imported here, not at module scope: the retrieval stack reaches into Chroma and
            # the LLM slot-filler, and nothing else in this module needs it — an unarmed run
            # must not pay for (or risk) importing it at all.
            from agent.app.idea_policies import plan_library_search as _plan_search
            from agent.app.idea_policies.plan_library import ORIGIN_ACTION
            from agent.app.plan_library import retrieval as _plan_retrieval

            # A fresh corpus per execution: a LeafAction is constructed per action, so there is
            # no instance to cache on (the engine caches its own on ``_plan_library_corpus``),
            # and an on-demand search is rare — at most a handful of nodes per run.
            library = _plan_retrieval.PlanLibrary()
            resolution = await _plan_search.resolve(
                library,
                graph,
                node,
                io=io,
                call_site=_plan_retrieval.CALL_SITE_ON_DEMAND,
                origin=ORIGIN_ACTION,
                model_name=self._effective_model(io, None) or self._cfg.expansion.model,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=self._timeout_seconds("expansion_timeout_seconds"),
            )
            retrieval = resolution.retrieval
            adopted = resolution.adopted
            self._log_structured(
                "info",
                "[PLAN_LIBRARY_SEARCH] retrieval complete",
                node_id=node_id,
                decision=getattr(retrieval, "decision", None),
                adopted=adopted,
                template_id=resolution.template_id,
            )
            return ActionResultBuilder.success(
                action=IdeaActionType.PLAN_LIBRARY_SEARCH.value,
                node_id=node_id,
                query=getattr(retrieval, "query_text", ""),
                decision=getattr(retrieval, "decision", None),
                reason=getattr(retrieval, "reason", ""),
                matches=[m.as_dict() for m in getattr(retrieval, "candidates", None) or []],
                adopted=adopted,
                adopted_template_id=resolution.template_id,
                # What the template was bound with — the engine's rebuild input, and (once
                # sanitized onto the node) a readable record of the plan that was adopted.
                slot_values=(getattr(resolution.fill, "slot_values", None) if adopted else None),
                leaf_count=(len(resolution.expansion.candidates) if adopted else 0),
                # The uuid4 join key into both retrieval logs.
                retrieval_id=getattr(retrieval, "retrieval_id", None),
            )
        except Exception as exc:
            return self._failure(
                action=IdeaActionType.PLAN_LIBRARY_SEARCH, node_id=node_id, error=exc
            )


class MergeLeafAction(LeafAction):
    name = "merge"

    # Reason-before-answer variant of the shipped ``merge_system_prompt``, selected under
    # ``merge_goal_evaluation_first_enabled``. Only the ``goal_achieved`` /
    # ``goal_evaluation`` pair is swapped; every other byte, including the doubled braces
    # the settings copy carries, is identical.
    #
    # Copied from the SETTINGS value rather than derived from it at runtime. A runtime
    # ``str.replace`` would silently match nothing if the shipped text were ever edited,
    # turning the flag into an invisible no-op; the derivation is asserted by
    # ``reason_first_ordering_test`` instead, so that edit fails loudly.
    #
    # It is a source constant rather than a new settings key on purpose: ``settings.get``
    # prefers the JSON, and this subsystem already carries one fossil default that never
    # runs because of it. A second JSON-shadows-source surface is not worth the symmetry.
    _GOAL_EVAL_FIRST_SYSTEM_PROMPT = (
        "You are the Aggregate operation in a Graph-of-Thought system. Combine child "
        "node results into a coherent summary. Remove redundancy, extract key findings. "
        "Evaluate if the original goal has been achieved. Return JSON: {{summary: string, "
        "key_findings: [string, ...], goal_evaluation: string, goal_achieved: boolean, "
        "missing_requirements: [string, ...]}}."
    )

    # Page payload keys dropped wholesale before serialization. ``content_full`` and
    # ``content_with_links`` are near-verbatim copies of ``content``; the link keys are
    # navigation metadata the synthesis stage never reads.
    _DROP_KEYS = (
        "content_full", "content_with_links", "links_full", "link_contexts",
        "links", "_links_inline",
    )
    # Evidence budget per child, in-family with ``_collect_all_visit_content``'s 3000.
    _CONTENT_CHARS = 2000
    # Catch-all for any other unexpectedly long string field.
    _FIELD_CHARS = 5000
    _MAX_DEPTH = 4

    @classmethod
    def _compact_payload(cls, value: Any, _depth: int = 0) -> Any:
        """Recursively strip/truncate page payload inside a merged child's ``result``.

        ``SimpleMergePolicy.merge`` nests the action result one level down, under
        ``result`` (and ``ACTION_RESULTS`` may nest a list below that), so compaction
        has to descend rather than inspect only the entry's top-level keys.
        """
        if _depth >= cls._MAX_DEPTH:
            return value
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                if k in cls._DROP_KEYS:
                    continue
                if k == "content" and isinstance(v, str):
                    out[k] = v[:cls._CONTENT_CHARS] + "..." if len(v) > cls._CONTENT_CHARS else v
                elif isinstance(v, (dict, list)):
                    out[k] = cls._compact_payload(v, _depth + 1)
                elif isinstance(v, str) and len(v) > cls._FIELD_CHARS:
                    out[k] = v[:cls._FIELD_CHARS] + "..."
                else:
                    out[k] = v
            return out
        if isinstance(value, list):
            return [cls._compact_payload(v, _depth + 1) for v in value]
        if isinstance(value, str) and len(value) > cls._FIELD_CHARS:
            return value[:cls._FIELD_CHARS] + "..."
        return value

    @classmethod
    def _compact_merged_results(cls, merged_results: Any) -> Any:
        """Compact each merged child, preserving its bookkeeping fields verbatim.

        ``node_id`` / ``title`` / ``status`` / ``score`` / ``evaluation`` / ``is_merge`` /
        ``waypoint`` survive unchanged; only the nested ``result`` payload shrinks.
        """
        compacted = []
        for mr in merged_results:
            if not isinstance(mr, dict):
                compacted.append(mr)
                continue
            compacted.append(cls._compact_payload(mr))
        return compacted

    #: Synthesis fields that carry the merge's own assertions, as opposed to its bookkeeping.
    #: ``missing_requirements`` is excluded deliberately: a figure named there is one the
    #: completion says it does NOT have, so checking its provenance would be backwards.
    _CLAIM_FIELDS = ("summary", "key_findings", "goal_evaluation")

    @classmethod
    def _claim_text(cls, synthesized_data: Any) -> str:
        """What this merge ASSERTS, flattened to one string for the numeric provenance check."""
        if isinstance(synthesized_data, str):
            return synthesized_data
        if not isinstance(synthesized_data, dict):
            return ""
        parts: List[str] = []
        for key in cls._CLAIM_FIELDS:
            value = synthesized_data.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value if isinstance(v, (str, int, float)))
        return " ".join(p for p in parts if p)

    @staticmethod
    def _authored_goal(node: Any, ignore_self_label: bool = False) -> str:
        """A node's own GOAL/ORIGINAL_GOAL text, skipping structural merge labels.

        ``ignore_self_label`` is set only for the merge node itself. Every node the engine
        touches gets a GOAL stamped (``idea_engine`` threads one onto each child, defaulting to
        the child's own title), so "GOAL echoes the title" is normal and MEANINGFUL for a
        decompose step -- its title is a real sub-goal description. On a merge node it is
        meaningless: the title is ``Merge: {parent.title}`` (or whatever label the planner gave
        its synthesis step), a structural name for the operation rather than a research
        question.
        """
        details = node.details if isinstance(getattr(node, "details", None), dict) else {}
        title = (getattr(node, "title", "") or "").strip()
        for key in (DetailKey.GOAL.value, DetailKey.ORIGINAL_GOAL.value):
            value = details.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip()
            if ignore_self_label and (
                text.lower() == title.lower() or text.lower().startswith("merge:")
            ):
                continue
            return text
        return ""

    def _resolve_merge_goal(self, graph: IdeaDag, node: Any) -> str:
        """The task goal this merge synthesizes toward -- never the merge node's own label.

        A merge node carries no research question of its own. Two of the three ways one comes
        into existence leave it with nothing usable: a planner-authored ``action: merge`` child
        gets GOAL stamped from its own title ("Synthesize the findings"), and a
        ``create_merge_node`` node minted outside ``_handle_merge_creation`` gets no GOAL at all,
        so the old ``GOAL -> ORIGINAL_GOAL -> node.title`` chain always bottomed out at the
        literal ``Merge: {parent.title}``. The node's non-empty title made the parent fallback
        below unreachable, so the merge PROMPT's ``{original_goal}`` -- and the goal-relevance
        checks -- saw a structural label instead of the research question.

        Resolution walks UP to the nearest ancestor carrying real goal text, so a legitimate
        intermediate sub-goal still wins over the root mandate (a merge under a decompose step
        synthesizes toward THAT step), and bottoms out at the root's goal/mandate.
        """
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor

        own = self._authored_goal(node, ignore_self_label=True)
        if own:
            return own

        root_id = graph.root_id()
        seen = {getattr(node, "node_id", None)}
        current = graph.get_node(node.parent_id) if node.parent_id else None
        while current is not None and current.node_id not in seen and current.node_id != root_id:
            seen.add(current.node_id)
            ancestor_goal = self._authored_goal(current)
            if ancestor_goal:
                return ancestor_goal
            title = (current.title or "").strip()
            if title and not NodeDetailsExtractor.is_merge_action(current.details):
                return title
            current = graph.get_node(current.parent_id) if current.parent_id else None

        root = graph.get_node(root_id)
        if root is None:
            return ""
        # The root's GOAL is preferred over its raw ``mandate`` detail: the engine stamps it
        # from ``root_title``, which is the mandate with the harness's "Task Statement" boiler-
        # plate already stripped off.
        root_goal = self._authored_goal(root)
        if root_goal:
            return root_goal
        mandate = root.details.get("mandate") if isinstance(root.details, dict) else None
        if isinstance(mandate, str) and mandate.strip():
            return mandate.strip()
        return (root.title or "").strip()

    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        import json
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")

            merged_results = node.details.get(DetailKey.MERGED_RESULTS.value) or []
            if not merged_results:
                return ActionResultBuilder.failure(
                    action=IdeaActionType.MERGE.value,
                    error="No merged results to synthesize",
                )
            
            system_template = self.settings.get("merge_system_prompt", "")
            # Opt-in reason-before-answer ordering. Substituted only when a base template
            # already resolved, so the flag cannot resurrect the no-prompts concatenation
            # fallback below by turning an empty template into a non-empty one.
            if system_template and self._cfg.merge.goal_evaluation_first_enabled:
                system_template = self._GOAL_EVAL_FIRST_SYSTEM_PROMPT
            user_template = self.settings.get("merge_user_prompt", "")
            planning_addendum = str(
                self.settings.get(
                    "merge_planning_addendum",
                    "Preserve provenance and separate confirmed facts from open questions.",
                )
            ).strip()
            if planning_addendum:
                system_template = f"{system_template}\n\n{planning_addendum}" if system_template else planning_addendum
            
            if not system_template or not user_template:
                self._logger.warning("No merge prompts found, using simple concatenation")
                synthesized = json.dumps(merged_results, ensure_ascii=True)
                return ActionResultBuilder.success(
                    action=IdeaActionType.MERGE.value,
                    synthesized=synthesized,
                    child_count=len(merged_results),
                )
            
            original_goal = self._resolve_merge_goal(graph, node)
            parent_intent = node.details.get(DetailKey.INTENT.value) or ""
            parent_justification = node.details.get(DetailKey.PARENT_JUSTIFICATION.value) or node.details.get(DetailKey.JUSTIFICATION.value) or ""

            if node.parent_id:
                parent = graph.get_node(node.parent_id)
                if parent:
                    if not parent_intent:
                        parent_intent = parent.details.get(DetailKey.INTENT.value) or ""
                    if not parent_justification:
                        parent_justification = parent.details.get(DetailKey.JUSTIFICATION.value) or parent.details.get(DetailKey.PARENT_JUSTIFICATION.value) or ""
            
            compacted = self._compact_merged_results(merged_results)
            merged_json = json.dumps(compacted, ensure_ascii=True)
            _merge_cap = self._cfg.merge.max_json_chars
            if len(merged_json) > _merge_cap:
                self._logger.info(f"[MERGE] merged_json {len(merged_json)} chars > cap {_merge_cap}; truncating")
                merged_json = merged_json[:_merge_cap] + ' ...[truncated]'
            user_content = user_template.format(
                merged_json=merged_json,
                original_goal=original_goal or "",
                parent_intent=parent_intent,
                parent_justification=parent_justification
            )
            
            # The merge schema declares optional fields (goal_evaluation,
            # missing_requirements) that are not in ``required``. OpenAI/Azure strict
            # structured output requires ``required`` to enumerate every property, so
            # convey the shape as a prompt instruction and use ``json_object`` mode
            # instead (mirrors the expansion stage; provider-agnostic).
            json_schema = self.settings.get("merge_json_schema")
            # The second of merge's two ordering sources: this schema is dumped into the
            # SAME system message as the template above. Reorder both or the model reads
            # two conflicting orders.
            if self._cfg.merge.goal_evaluation_first_enabled:
                json_schema = MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST
            schema_hint = (
                json_instruction_from_response_format({"type": "json_schema", "json_schema": json_schema})
                if json_schema
                else None
            )
            if schema_hint:
                system_template = f"{system_template}\n\n{schema_hint}" if system_template else schema_hint

            messages = PromptBuilder.build_messages(
                system_content=system_template,
                user_content=user_content,
            )

            model_name = self._cfg.merge.model or self._cfg.final.model
            reasoning_effort = self._cfg.generation.reasoning_effort
            text_verbosity = self._cfg.generation.text_verbosity

            # A3b anti-starvation floor for the merge aggregation call (opt-in; no-op default —
            # merge's budget is already large). Keeps the configured reasoning_effort (merge is
            # aggregation/reasoning, not perception) and skips the A5 multiplier to avoid
            # ballooning aggregation cost.
            payload = io.build_llm_payload(
                messages=messages,
                json_mode=True,
                model_name=model_name,
                temperature=self._cfg.merge.temperature,
                max_tokens=self._executor_max_tokens(
                    self._effective_model(io, model_name), self._cfg.merge.max_tokens, price_tier=False
                ),
                json_schema=None,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
            )
            
            timeout_seconds = self._timeout_seconds("llm_timeout_seconds")
            try:
                merge_preview = json.dumps(messages, indent=2, ensure_ascii=True)
            except Exception:
                merge_preview = str(messages)
            if len(merge_preview) > 2000:
                merge_preview = merge_preview[:2000] + "... [truncated]"
            self._logger.debug(f"[MERGE] LLM Input preview: {merge_preview}")
            response = await io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=timeout_seconds,
            )
            response_preview = response[:2000] + "... [truncated]" if isinstance(response, str) and len(response) > 2000 else response
            self._logger.debug(f"[MERGE] LLM Output preview: {response_preview}")
            
            if not response:
                return ActionResultBuilder.failure(
                    action=IdeaActionType.MERGE.value,
                    error="LLM returned empty response",
                )
            
            try:
                synthesized_data = json.loads(response)
            except json.JSONDecodeError:
                synthesized_data = {"summary": response, "goal_achieved": False, "goal_evaluation": "Failed to parse LLM response", "missing_requirements": []}
            
            goal_achieved = synthesized_data.get("goal_achieved", False)
            # "Key absent" and "key present and false" are the same `False` to the caller but
            # different failures: the second is a verdict, the first is the model never
            # answering the question (the 2026-08-21 A/B measured this on ~4% of 160 real
            # completions -- llama3.2:3b echoing the input blob back under the schema's field
            # names, one 14b `goaled_achieved` typo alongside a goal_evaluation that plainly
            # said ACHIEVED). Default is unchanged -- not-achieved is the safe direction, and
            # the unparseable-response fallback above supplies the key itself, so a parse
            # failure keeps its own distinct diagnosis instead of being relabelled here.
            goal_achieved_field_missing = "goal_achieved" not in synthesized_data
            if goal_achieved_field_missing:
                self._logger.warning(
                    f"[MERGE] {node_id}: model completion did not include a usable goal_achieved "
                    "field, defaulting to not-achieved (schema-adherence failure, not a genuine "
                    "negative verdict)"
                )
            goal_evaluation = synthesized_data.get("goal_evaluation", "")
            missing_requirements = synthesized_data.get("missing_requirements", [])

            if not isinstance(goal_achieved, bool):
                goal_achieved = bool(goal_achieved)

            # Internal-consistency guard: one completion cannot both declare the goal met and
            # list what is still missing. Honouring the ``true`` marks this node AND its parent
            # DONE, permanently terminating a branch the model's own other field says is
            # incomplete -- so the contradiction is resolved toward the more specific claim and
            # falls through to the not-achieved branch below.
            if goal_achieved and missing_requirements:
                self._logger.warning(
                    f"[MERGE] {node_id}: goal_achieved=true but missing_requirements={missing_requirements} "
                    "in the same completion -- downgrading to not-achieved (internally contradictory verdict)"
                )
                goal_achieved = False

            # Race value-agreement check: the routes of a race group returned DIFFERENT values
            # for the quantity they were racing for, so whatever this synthesis says about them
            # agreeing is unsupported ("all three routes confirm 575 meters" was a real 2026-08-21
            # completion about a number in no fetched page). Detection is unconditional, acting
            # on it is gated -- see ``MergeConfig.race_value_agreement_enabled``. The conflict is
            # routed through ``missing_requirements`` rather than flipping the boolean directly,
            # so it rides the consistency guard above and shows up in the run's own report of
            # what is still open.
            from agent.app.idea_policies.merge import race_value_conflicts

            conflicts = race_value_conflicts(graph, node)
            if conflicts:
                node.details["race_value_disagreement"] = conflicts
                self._logger.warning(
                    f"[MERGE] {node_id}: race group(s) {conflicts} returned conflicting values "
                    "for the asked-for datum -- the routes do not agree"
                )
                if self._cfg.merge.race_value_agreement_enabled:
                    existing = (
                        list(missing_requirements)
                        if isinstance(missing_requirements, (list, tuple)) else []
                    )
                    missing_requirements = existing + [
                        f"independent routes in race group '{label}' returned conflicting "
                        "values for the asked-for quantity; the conflict is unresolved"
                        for label in conflicts
                    ]
                    if goal_achieved:
                        self._logger.warning(
                            f"[MERGE] {node_id}: downgrading to not-achieved "
                            "(merge_race_value_agreement_enabled)"
                        )
                    goal_achieved = False

            # Evidence-provenance check: an ACHIEVED verdict backed only by unvisited search
            # snippets names the right answer without ever having obtained it. Detection is
            # unconditional (a silent failure mode nobody could measure otherwise); acting on
            # it is gated, because this overrules a self-consistent verdict on external
            # grounds -- see ``MergeConfig.require_visited_evidence_enabled``.
            if goal_achieved:
                from agent.app.idea_policies.merge import (
                    GOAL_EVIDENCE_SNIPPET,
                    goal_evidence_provenance,
                )
                # The same ``original_goal`` the synthesis prompt was built from: this check
                # measures overlap against the goal the model was asked about, and the two
                # diverging is how a merge ends up validated against text it never saw.
                if goal_evidence_provenance(original_goal, merged_results) == GOAL_EVIDENCE_SNIPPET:
                    node.details["goal_achieved_snippet_only"] = True
                    self._logger.warning(
                        f"[MERGE] {node_id}: goal_achieved=true but every goal-relevant item is an "
                        "unvisited search-result snippet -- no fetched content addresses the goal"
                    )
                    if self._cfg.merge.require_visited_evidence_enabled:
                        self._logger.warning(
                            f"[MERGE] {node_id}: downgrading to not-achieved "
                            "(merge_require_visited_evidence_enabled)"
                        )
                        goal_achieved = False

            # Numeric-token provenance (B1): the narrower sibling of the check above. It asks
            # nothing of the wording -- only whether the FIGURES the completion asserts appear
            # in raw text the run actually fetched. Live-observed shape it targets: a merge
            # completion narrating "all three routes confirm 575 meters", a number in no
            # fetched page anywhere, achieved, no URLs. Detection unconditional, downgrade
            # gated -- see ``MergeConfig.require_numeric_provenance_enabled``.
            if goal_achieved:
                from agent.app.idea_policies.grounding import answer_numeric_provenance

                numeric = answer_numeric_provenance(
                    graph, self._claim_text(synthesized_data), merged_results
                )
                if numeric.unsupported:
                    node.details["goal_achieved_numeric_unverified"] = numeric.unverified_values()
                    self._logger.warning(
                        f"[MERGE] {node_id}: goal_achieved=true but none of its measurements "
                        f"{numeric.unverified_values()} appear in any fetched page text -- "
                        "unsourced numeric claim"
                    )
                    if self._cfg.merge.require_numeric_provenance_enabled:
                        self._logger.warning(
                            f"[MERGE] {node_id}: downgrading to not-achieved "
                            "(merge_require_numeric_provenance_enabled)"
                        )
                        goal_achieved = False

            node.details[DetailKey.GOAL_ACHIEVED.value] = goal_achieved
            if goal_achieved_field_missing:
                node.details["goal_achieved_field_missing"] = True
            if goal_evaluation:
                node.details["goal_evaluation"] = goal_evaluation
            if missing_requirements:
                node.details["missing_requirements"] = missing_requirements
            
            if goal_achieved:
                self._logger.info(f"[MERGE] Goal achieved for node {node_id}: {original_goal or 'N/A'}")
                node.status = IdeaNodeStatus.DONE
                
                if node.parent_id:
                    parent = graph.get_node(node.parent_id)
                    if parent:
                        parent.details[DetailKey.GOAL_ACHIEVED.value] = True
                        if parent.status == IdeaNodeStatus.ACTIVE:
                            parent.status = IdeaNodeStatus.DONE
                            self._logger.info(f"[MERGE] Marked parent node {node.parent_id} as DONE due to goal achievement")
            else:
                self._logger.warning(f"[MERGE] Goal NOT achieved for node {node_id}: {original_goal or 'N/A'}. Missing: {missing_requirements}")
                node.details["merge_incomplete"] = True
                node.details["merge_should_skip"] = True
            
            return ActionResultBuilder.success(
                action=IdeaActionType.MERGE.value,
                synthesized=synthesized_data,
                child_count=len(merged_results),
                raw_response=response,
                goal_achieved=goal_achieved,
                goal_evaluation=goal_evaluation,
                missing_requirements=missing_requirements,
            )
        except Exception as exc:
            return self._failure(action=IdeaActionType.MERGE, node_id=node_id, error=exc)


class VerifyLeafAction(LeafAction):
    """Cross-check a claim against gathered evidence.

    A `verify` node takes a target ``claim`` plus the evidence already gathered by
    ancestor/sibling ``visit`` nodes, optionally fetches one authoritative
    ``optional_url``, then emits a structured verdict via a single LLM call. It is a
    graph-only differentiator: `parametric` (no sources) and `naive_rag` (no second
    round) cannot reconcile sources or flag a contradicting authority.
    """

    name = "verify"

    # Reason-before-answer variant of the shipped ``verify_system_prompt``, selected under
    # ``verify_reason_first_enabled``. Only the position of ``reasoning`` moves.
    #
    # Copied from the SETTINGS value, NOT from ``_DEFAULT_SYSTEM_PROMPT`` below: the two
    # differ in typography (settings uses an ASCII hyphen where the constant uses an em
    # dash, and doubles its braces), and settings is what actually ships. Deriving from the
    # constant would change three things at once and stop the A/B isolating field order.
    #
    # promptbench v2 (2026-08-19): pooled A2-A1 = +0.142, CI [+0.053, +0.232],
    # permutation p = 0.0119, positive on 5/5 models. Opt-in, default OFF.
    #
    # Known risk no offline test can catch: ``verify_max_tokens`` defaults to 1024, and
    # reasoning-first spends output budget on prose BEFORE the verdict, so a verbose model
    # can truncate and fall into the UNVERIFIABLE JSONDecodeError branch below. The
    # "<one sentence>" constraint is retained for exactly this reason; any live A/B on this
    # flag must report parse-failure rate, not just accuracy.
    _REASON_FIRST_SYSTEM_PROMPT = (
        "You are the Verify operation in a Graph-of-Thought fact-checking system. "
        "Given a CLAIM and EVIDENCE collected from web pages, decide whether the claim "
        "is supported. Rely ONLY on the provided evidence - never on prior knowledge. "
        "If the evidence contradicts the claim, identify the authoritative source URL "
        "that contradicts it and quote the exact contradicting sentence. Return strict "
        "JSON: {{\"reasoning\": \"<one sentence>\", \"verdict\": "
        "\"TRUE\"|\"PARTIALLY_TRUE\"|\"FALSE\"|\"UNVERIFIABLE\", \"confidence\": 0.0-1.0, "
        "\"supporting_url\": \"<url or empty>\", \"contradicting_url\": \"<url or empty>\", "
        "\"quote\": \"<verbatim sentence from evidence>\"}}. Use UNVERIFIABLE only when "
        "the evidence does not address the claim."
    )

    _DEFAULT_SYSTEM_PROMPT = (
        "You are the Verify operation in a Graph-of-Thought fact-checking system. "
        "Given a CLAIM and EVIDENCE collected from web pages, decide whether the claim "
        "is supported. Rely ONLY on the provided evidence — never on prior knowledge. "
        "If the evidence contradicts the claim, identify the authoritative source URL "
        "that contradicts it and quote the exact contradicting sentence. "
        'Return strict JSON: {"verdict": "TRUE"|"PARTIALLY_TRUE"|"FALSE"|"UNVERIFIABLE", '
        '"confidence": 0.0-1.0, "supporting_url": "<url or empty>", '
        '"contradicting_url": "<url or empty>", "quote": "<verbatim sentence from '
        'evidence>", "reasoning": "<one sentence>"}. '
        "Use UNVERIFIABLE only when the evidence does not address the claim."
    )

    def _collect_evidence(
        self, graph: IdeaDag, node: IdeaNode, max_chars: int = 30000, max_depth: int = 4
    ) -> List[Dict[str, str]]:
        """Gather visit content + search snippets from ancestors and siblings."""
        visited_nodes: Set[str] = set()
        queue: List[Tuple[IdeaNode, int]] = [(node, 0)]
        evidence: List[Dict[str, str]] = []
        budget = max_chars

        while queue and budget > 0:
            current, depth = queue.pop(0)
            if current.node_id in visited_nodes or depth > max_depth:
                continue
            visited_nodes.add(current.node_id)

            result = current.details.get(DetailKey.ACTION_RESULT.value)
            if isinstance(result, dict) and result.get(ActionResultKey.SUCCESS.value):
                action_type = result.get(ActionResultKey.ACTION.value)
                if action_type == IdeaActionType.VISIT.value:
                    url = result.get(ActionResultKey.URL.value) or ""
                    content = (
                        result.get(ActionResultKey.CONTENT.value)
                        or result.get(ActionResultKey.CONTENT_FULL.value)
                        or ""
                    )
                    if content:
                        snippet = content[:budget]
                        budget -= len(snippet)
                        evidence.append({"url": str(url), "content": snippet})
                elif action_type == IdeaActionType.SEARCH.value:
                    for item in (result.get(ActionResultKey.RESULTS.value) or [])[:10]:
                        if isinstance(item, dict):
                            line = f"{item.get('title', '')} — {item.get('snippet', '')} ({item.get('url', '')})"
                            evidence.append({"url": str(item.get("url", "")), "content": line[:500]})

            # Walk to parent and to siblings (one hop).
            pids = current.parent_ids if current.parent_ids else ([current.parent_id] if current.parent_id else [])
            for pid in pids:
                if pid:
                    parent = graph.get_node(pid)
                    if parent:
                        queue.append((parent, depth + 1))
                        for sib_id in parent.children:
                            if sib_id != current.node_id:
                                sib = graph.get_node(sib_id)
                                if sib:
                                    queue.append((sib, depth + 1))
        return evidence

    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")

            from agent.app.idea_policies.action_constants import NodeDetailsExtractor

            claim = (
                node.details.get(DetailKey.CLAIM.value)
                or node.details.get(DetailKey.INTENT.value)
                or NodeDetailsExtractor.get_query(node.details, fallback_title=node.title)
                or ""
            )
            claim = str(claim).strip()
            if not claim:
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VERIFY.value,
                    error="Verify node missing a 'claim' to check.",
                    node_id=node_id,
                )

            evidence = self._collect_evidence(graph, node)

            # Optionally fetch one authoritative page named on the node.
            optional_url = NodeDetailsExtractor.get_url(node.details) or node.details.get("optional_url")
            if optional_url and isinstance(optional_url, str) and optional_url.startswith(("http://", "https://")):
                visitor = VisitLeafAction(settings=self.settings)
                fetch_result, _, _ = await visitor._visit_single_page(optional_url, graph, node, io, intent=claim)
                if fetch_result and fetch_result.get(ActionResultKey.SUCCESS.value):
                    content = fetch_result.get(ActionResultKey.CONTENT.value) or ""
                    if content:
                        evidence.insert(0, {"url": str(optional_url), "content": content[:30000]})

            if not evidence:
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VERIFY.value,
                    error="Verify node found no gathered evidence to cross-check. Visit sources first.",
                    error_type="ValidationError",
                    node_id=node_id,
                )

            evidence_text = "\n\n".join(
                f"[SOURCE {i + 1}] {e['url']}\n{e['content']}" for i, e in enumerate(evidence)
            )
            system_content = self.settings.get("verify_system_prompt") or self._DEFAULT_SYSTEM_PROMPT
            # Opt-in reason-before-answer ordering.
            if self._cfg.verify.reason_first_enabled:
                system_content = self._REASON_FIRST_SYSTEM_PROMPT
            user_content = f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence_text}"
            messages = PromptBuilder.build_messages(system_content=system_content, user_content=user_content)

            model_name = self._cfg.verify.model  # None -> connector's current execution model
            # A3b anti-starvation floor for the verify micro-prompt (opt-in; no-op default): the
            # 1024-token verify budget can starve a reasoning executor, so floor it when the flag
            # is on. No effort override (verify is a reasoning check, not perception).
            payload = io.build_llm_payload(
                messages=messages,
                json_mode=True,
                model_name=model_name,
                temperature=self._cfg.verify.temperature,
                max_tokens=self._executor_max_tokens(
                    self._effective_model(io, model_name), self._cfg.verify.max_tokens, price_tier=False
                ),
            )
            timeout_seconds = self._timeout_seconds("llm_timeout_seconds")
            response = await io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=timeout_seconds,
            )
            if not response:
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VERIFY.value,
                    error="Verify LLM returned empty response",
                    node_id=node_id,
                )

            try:
                verdict_data = json.loads(response)
            except json.JSONDecodeError:
                verdict_data = {
                    "verdict": "UNVERIFIABLE",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse verify response",
                }

            return ActionResultBuilder.success(
                action=IdeaActionType.VERIFY.value,
                node_id=node_id,
                claim=claim,
                verdict=verdict_data.get("verdict", "UNVERIFIABLE"),
                confidence=verdict_data.get("confidence", 0.0),
                supporting_url=verdict_data.get("supporting_url", ""),
                contradicting_url=verdict_data.get("contradicting_url", ""),
                quote=verdict_data.get("quote", ""),
                reasoning=verdict_data.get("reasoning", ""),
                evidence_sources=[e["url"] for e in evidence],
                raw_response=response,
            )
        except Exception as exc:
            return self._failure(action=IdeaActionType.VERIFY, node_id=node_id, error=exc)


class LeafActionRegistry:
    """Registry of leaf actions.

    Built-ins (`search/visit/think/save/merge`) ship pre-registered and are
    addressable by either their `IdeaActionType` enum value or by name string.
    Custom actions register themselves by name via `register(cls)` or in
    bulk via `install_pack(pack)`.
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})
        # Single source of truth: name → class. Enum lookups go through the
        # enum's `.value` (which equals the name string) into this dict.
        self._by_name: Dict[str, type] = {
            IdeaActionType.SEARCH.value: SearchLeafAction,
            IdeaActionType.VISIT.value: VisitLeafAction,
            IdeaActionType.SAVE.value: SaveLeafAction,
            IdeaActionType.THINK.value: ThinkLeafAction,
            IdeaActionType.MERGE.value: MergeLeafAction,
            IdeaActionType.VERIFY.value: VerifyLeafAction,
            IdeaActionType.PLAN_LIBRARY_SEARCH.value: PlanLibrarySearchLeafAction,
        }
        # Kept for legacy code paths that introspect `._registry` directly.
        self._registry: Dict[IdeaActionType, type] = {
            IdeaActionType.SEARCH: SearchLeafAction,
            IdeaActionType.VISIT: VisitLeafAction,
            IdeaActionType.SAVE: SaveLeafAction,
            IdeaActionType.THINK: ThinkLeafAction,
            IdeaActionType.MERGE: MergeLeafAction,
            IdeaActionType.VERIFY: VerifyLeafAction,
            IdeaActionType.PLAN_LIBRARY_SEARCH: PlanLibrarySearchLeafAction,
        }

    @staticmethod
    def _normalize(key: Any) -> Optional[str]:
        if isinstance(key, IdeaActionType):
            return key.value
        if isinstance(key, str):
            return key
        return None

    def get(self, action_type: Any) -> LeafAction:
        """Resolve an action by enum value or by name string."""
        name = self._normalize(action_type)
        if name is None:
            raise ValueError(f"Unknown action type: {action_type!r}")
        action_cls = self._by_name.get(name)
        if not action_cls:
            raise ValueError(f"Unknown action type: {action_type!r}")
        return action_cls(settings=self.settings)

    def has(self, action_type: Any) -> bool:
        name = self._normalize(action_type)
        return bool(name and name in self._by_name)

    def register(self, cls: type) -> None:
        """Register a `LeafAction` subclass by its `name` ClassVar."""
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Action class {cls.__name__} must declare a non-empty string `name` ClassVar"
            )
        self._by_name[name] = cls

    def install_pack(self, pack: Any) -> List[str]:
        """Register every action class in `pack.ACTION_CLASSES`.

        Returns the list of names that were installed (for callers that want
        to update `allowed_actions` in settings).
        """
        installed: List[str] = []
        action_classes = getattr(pack, "ACTION_CLASSES", None) or []
        for cls in action_classes:
            self.register(cls)
            installed.append(getattr(cls, "name"))
        return installed

    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def menu_lines(self, allowed: Any) -> List[str]:
        """Prompt lines describing the NON-CORE actions among ``allowed``, in ``allowed`` order.

        "Core" means an `IdeaActionType` member: those already have hand-written entries in the
        expansion prompt's ACTIONS block, so re-describing them would both duplicate and
        perturb every existing run. Only registry-only extension actions (whatever
        `register()`/`install_pack()` added) produce a line, and an allowed name nobody
        registered a class for produces nothing — the menu never advertises a dead action.
        """
        core = {member.value for member in IdeaActionType}
        lines: List[str] = []
        seen: Set[str] = set()
        for item in allowed or []:
            name = str(item)
            if name in core or name in seen:
                continue
            action_cls = self._by_name.get(name)
            if action_cls is None:
                continue
            seen.add(name)
            lines.append(action_cls.menu_line())
        return lines


def execute_leaf_action(action: LeafAction, graph: IdeaDag, node_id: str, io: AgentIO):
    return action.execute(graph, node_id, io)
