from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, List, Set
import uuid
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.observation import clean_operation
from agent.app.idea_policies.base import IdeaActionType, DetailKey
from agent.app.idea_policies.action_constants import (
    ActionResultKey,
    PromptKey,
    ContextKey,
    ActionResultBuilder,
    PromptBuilder,
    ContextBuilder,
)


class LeafAction(ABC):
    """Base class for leaf node actions."""
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        """Execute a leaf node action."""
        raise NotImplementedError()

    def _max_observation_chars(self) -> int:
        """Return max observation size."""
        return int(self.settings.get("max_observation_chars", 6000))

    def _timeout_seconds(self, key: str) -> Optional[float]:
        """Return timeout value from settings."""
        value = self.settings.get(key)
        if value is None:
            value = self.settings.get("action_timeout_seconds")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error should be retried."""
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return True
        message = str(error)
        match = re.search(r"status=([0-9]{3})", message)
        if match:
            status = int(match.group(1))
            if status in (401, 403):
                return False
            if status == 429:
                return True
            if status >= 500:
                return True
        error_lower = message.lower()
        bot_blocking_indicators = [
            "forbidden",
            "access denied",
            "cloudflare",
            "bot detection",
            "captcha",
            "blocked",
            "unauthorized",
        ]
        if any(indicator in error_lower for indicator in bot_blocking_indicators):
            return False
        return False

    def _limit_text(self, text: str) -> Dict[str, Any]:
        """Limit text to max observation chars."""
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
        """Create safe copy of node details (primitives only)."""
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
        """Build failure payload with error tracking."""
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
            intent = node.details.get(DetailKey.INTENT.value)
            count = int(node.details.get(DetailKey.COUNT.value, self.settings.get("default_search_count", 10)))
            
            if intent and hasattr(io, 'retrieve_chroma'):
                try:
                    vector_context = await io.retrieve_chroma(
                        topics=[str(intent)],
                        n_results=3,
                        timeout_seconds=self._timeout_seconds("chroma_timeout_seconds"),
                    )
                    self._logger.debug(f"[SEARCH] Retrieved {len(vector_context)} vector DB results for intent: {intent[:50]}")
                except Exception as vec_exc:
                    self._logger.warning(f"[SEARCH] Vector DB query failed: {vec_exc}")
            
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
                    count = int(node.details.get(DetailKey.COUNT.value, self.settings.get("default_search_count", 10)))
                except (ValueError, TypeError):
                    count = self.settings.get("default_search_count", 10)
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


class VisitLeafAction(LeafAction):
    def _is_valid_url(self, candidate: str) -> bool:
        if not candidate or not isinstance(candidate, str):
            return False
        candidate = candidate.strip()
        return candidate.startswith(("http://", "https://"))
    
    def _clean_and_fix_link(self, href: str, base_url: str) -> Optional[str]:
        """
        Clean and fix a link, converting relative URLs to absolute.
        Completes broken/partial links into full URLs where possible.
        :param href: Raw href from HTML.
        :param base_url: Base URL of the current page.
        :return: Cleaned absolute URL or None if invalid.
        """
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
    
    def _filter_and_prioritize_links(self, links: List[str], base_url: str) -> List[str]:
        """
        Filter, clean, and prioritize links.
        :param links: Raw list of links.
        :param base_url: Base URL for resolving relative links.
        :return: Cleaned, deduplicated list of absolute URLs.
        """
        seen: Set[str] = set()
        cleaned_links: List[str] = []
        
        for link in links:
            cleaned = self._clean_and_fix_link(link, base_url)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                cleaned_links.append(cleaned)
        
        return cleaned_links
    
    def _attach_links_to_content(self, content: str, links: List[str], max_links: int = 20) -> str:
        """
        Attach links to the bottom of content for better visibility.
        :param content: Main content text.
        :param links: List of cleaned links.
        :param max_links: Maximum links to attach.
        :return: Content with links appended.
        """
        if not links:
            return content
        
        links_to_attach = links[:max_links]
        links_section = "\n\n--- Links found on this page ---\n"
        for i, link in enumerate(links_to_attach, 1):
            links_section += f"{i}. {link}\n"
        
        if len(links) > max_links:
            links_section += f"\n... and {len(links) - max_links} more links (see 'links' field in action result)\n"
        
        return content + links_section
    
    def _extract_url_from_parents(self, graph: IdeaDag, node: IdeaNode, max_depth: int = 3) -> Optional[str]:
        visited = set()
        queue = [(node, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth or current.node_id in visited:
                continue
            visited.add(current.node_id)
            
            result = current.details.get(DetailKey.ACTION_RESULT.value)
            if result and isinstance(result, dict):
                action_type = result.get("action")

                # Prefer URLs that already came from search results
                if action_type == IdeaActionType.SEARCH.value:
                    search_results = result.get("results", [])
                    if isinstance(search_results, list):
                        for item in search_results[:5]:
                            if isinstance(item, dict):
                                candidate_url = item.get("url") or item.get("link")
                                if candidate_url and self._is_valid_url(candidate_url):
                                    return candidate_url

                # Also allow URLs from previous visit results (their links field)
                if action_type == IdeaActionType.VISIT.value:
                    visit_links = result.get("links", [])
                    if isinstance(visit_links, list):
                        for candidate_url in visit_links:
                            if candidate_url and isinstance(candidate_url, str) and self._is_valid_url(candidate_url):
                                return candidate_url
            
            if current.parent_id:
                parent = graph.get_node(current.parent_id)
                if parent:
                    queue.append((parent, depth + 1))
        
        return None
    
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        url = None
        intent = None
        vector_context = []
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            url = NodeDetailsExtractor.get_url(node.details)
            
            if not url or not self._is_valid_url(url):
                extracted_url = self._extract_url_from_parents(graph, node)
                if extracted_url:
                    url = extracted_url
                    self._logger.info(f"[VISIT] Extracted URL from parent search results: {url[:60]}...")
                else:
                    error_msg = f"Visit node missing valid URL. Node title: '{node.title}'. Details should contain 'url' or 'link' field with a valid HTTP/HTTPS URL."
                    self._logger.error(f"[VISIT] {error_msg}")
                    from agent.app.idea_policies.action_constants import ErrorType
                    return ActionResultBuilder.failure(
                        action=IdeaActionType.VISIT.value,
                        error=error_msg,
                        error_type=ErrorType.INVALID_URL.value,
                        retryable=False,
                        url=url or node.title,
                    )
            
            intent = node.details.get(DetailKey.INTENT.value)
            
            if intent and hasattr(io, 'retrieve_chroma'):
                try:
                    vector_context = await io.retrieve_chroma(
                        topics=[str(intent)],
                        n_results=3,
                        timeout_seconds=self._timeout_seconds("chroma_timeout_seconds"),
                    )
                    self._logger.debug(f"[VISIT] Retrieved {len(vector_context)} vector DB results for intent: {intent[:50]}")
                except Exception as vec_exc:
                    self._logger.warning(f"[VISIT] Vector DB query failed: {vec_exc}")
            
            blocked_reason = graph.is_site_blocked(str(url))
            if blocked_reason:
                self._logger.warning(f"[VISIT] Site blocked: {url} - {blocked_reason}")
                from agent.app.idea_policies.action_constants import ErrorType
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error=f"Site blocked: {blocked_reason}",
                    error_type=ErrorType.BLOCKED_SITE.value,
                    retryable=False,
                    url=url,
                    intent=intent,
                )
            
            timeout_seconds = self._timeout_seconds("fetch_timeout_seconds")
            self._logger.debug(f"[VISIT] url='{url}', intent='{intent}', timeout={timeout_seconds}")
            raw_html = await io.fetch_url(str(url), timeout_seconds=timeout_seconds)
            if not raw_html:
                from agent.app.idea_policies.action_constants import ErrorType
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error="Failed to fetch URL - no content returned",
                    error_type=ErrorType.NETWORK_ERROR.value,
                    retryable=True,
                    url=url,
                )
            self._logger.info(f"[VISIT] {len(raw_html) if raw_html else 0} chars from {url[:60]}...")
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
            
            cleaned_links = self._filter_and_prioritize_links(raw_links, str(url))
            cleaned_link_contexts = {}
            for raw_link in raw_links:
                cleaned = self._clean_and_fix_link(raw_link, str(url))
                if cleaned and raw_link in link_contexts:
                    cleaned_link_contexts[cleaned] = link_contexts[raw_link]
            
            self._logger.debug(f"[VISIT] Extracted {len(raw_links)} raw links, cleaned to {len(cleaned_links)} valid absolute URLs")
            
            max_links_for_llm = int(self.settings.get("max_links_per_visit", 20))
            links_for_llm = cleaned_links[:max_links_for_llm]
            
            content_payload = self._limit_text(cleaned)
            content_for_links = content_payload.get("content") or cleaned or ""
            content_with_links = self._attach_links_to_content(
                content_for_links,
                links_for_llm,
                max_links=max_links_for_llm
            )
            
            if io.telemetry:
                io.telemetry.record_document_seen(
                    source="visit",
                    document={"url": url, "content": content_with_links},
                )
            
            return ActionResultBuilder.success(
                action=IdeaActionType.VISIT.value,
                url=url,
                intent=intent,
                vector_context=[str(doc) for doc in vector_context] if vector_context else [],
                content=content_payload.get("content"),
                content_is_truncated=content_payload.get("is_truncated"),
                content_total_chars=content_payload.get("total_chars"),
                content_full=cleaned,
                content_with_links=content_with_links,
                links=links_for_llm,
                links_full=cleaned_links,
                links_count=len(cleaned_links),
                link_contexts=cleaned_link_contexts,
            )
        except Exception as exc:
            if not url and node:
                url = (
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
            
            if is_bot_block and url:
                graph.mark_site_blocked(str(url), block_reason or "Bot blocking detected")
                self._logger.warning(f"[VISIT] Marking site as blocked: {url} - {block_reason}")
            
            failure = self._failure(
                action=IdeaActionType.VISIT,
                node_id=node_id,
                error=exc,
                context={DetailKey.URL.value: url},
            )
            if url is not None:
                failure[ActionResultKey.URL.value] = url
            if is_bot_block:
                failure[ActionResultKey.RETRYABLE.value] = False
                failure[ActionResultKey.ERROR.value] = f"{block_reason}: {error_str}"
            return failure


class ThinkLeafAction(LeafAction):
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            thinking_content = NodeDetailsExtractor.get_query(node.details, fallback_title=node.title) or ""
            
            if thinking_content and hasattr(io, 'store_chroma'):
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
            return ActionResultBuilder.success(
                action=IdeaActionType.THINK.value,
                node_id=node.node_id,
                title=node.title,
                details=details_copy,
                thinking_content=thinking_content,
            )
        except Exception as exc:
            return self._failure(action=IdeaActionType.THINK, node_id=node_id, error=exc)


class SaveLeafAction(LeafAction):
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
            metadatas = node.details.get(DetailKey.METADATAS.value) or [{} for _ in range(len(docs))]
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


class MergeLeafAction(LeafAction):
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
            
            # Compact merged results to reduce LLM payload size
            from agent.app.idea_policies.action_constants import MergedResultsCompactor
            max_merged_items = int(self.settings.get("max_merged_items_for_llm", 20))
            compacted_merged = MergedResultsCompactor.compact_for_llm(merged_results, max_items=max_merged_items)
            original_size = len(json.dumps(merged_results, ensure_ascii=True))
            compacted_size = len(json.dumps(compacted_merged, ensure_ascii=True))
            self._logger.debug(f"[MERGE] Compacted merged results: {original_size} -> {compacted_size} chars ({100 * compacted_size // max(original_size, 1)}%)")
            
            system_template = self.settings.get("merge_system_prompt", "")
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
                synthesized = json.dumps(compacted_merged, ensure_ascii=True)
                return ActionResultBuilder.success(
                    action=IdeaActionType.MERGE.value,
                    synthesized=synthesized,
                    child_count=len(merged_results),
                )
            
            merged_json = json.dumps(compacted_merged, ensure_ascii=True)
            messages = PromptBuilder.build_messages(
                system_content=system_template,
                user_content=user_template.format(merged_json=merged_json),
            )
            
            model_name = self.settings.get("merge_model") or self.settings.get("final_model")
            json_schema = self.settings.get("merge_json_schema")
            reasoning_effort = self.settings.get("reasoning_effort", "high")
            text_verbosity = self.settings.get("text_verbosity", "medium")
            
            payload = io.build_llm_payload(
                messages=messages,
                json_mode=True,
                model_name=model_name,
                temperature=float(self.settings.get("merge_temperature", 0.3)),
                max_tokens=self.settings.get("merge_max_tokens") if self.settings.get("merge_max_tokens") is not None else None,
                json_schema=json_schema,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
            )
            
            timeout_seconds = self._timeout_seconds("llm_timeout_seconds")
            response = await io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
                timeout_seconds=timeout_seconds,
            )
            
            if not response:
                return ActionResultBuilder.failure(
                    action=IdeaActionType.MERGE.value,
                    error="LLM returned empty response",
                )
            
            try:
                synthesized_data = json.loads(response)
            except json.JSONDecodeError:
                synthesized_data = {"summary": response}
            
            return ActionResultBuilder.success(
                action=IdeaActionType.MERGE.value,
                synthesized=synthesized_data,
                child_count=len(merged_results),
                raw_response=response,
            )
        except Exception as exc:
            return self._failure(action=IdeaActionType.MERGE, node_id=node_id, error=exc)


class LeafActionRegistry:
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})
        self._registry = {
            IdeaActionType.SEARCH: SearchLeafAction,
            IdeaActionType.VISIT: VisitLeafAction,
            IdeaActionType.SAVE: SaveLeafAction,
            IdeaActionType.THINK: ThinkLeafAction,
            IdeaActionType.MERGE: MergeLeafAction,
        }

    def get(self, action_type: IdeaActionType) -> LeafAction:
        action_cls = self._registry.get(action_type)
        if not action_cls:
            raise ValueError(f"Unknown action type: {action_type}")
        return action_cls(settings=self.settings)


def execute_leaf_action(action: LeafAction, graph: IdeaDag, node_id: str, io: AgentIO):
    return action.execute(graph, node_id, io)
