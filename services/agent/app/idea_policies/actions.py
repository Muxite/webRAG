from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import logging
import re
from typing import Any, Dict, Optional
import uuid

from bs4 import BeautifulSoup

from agent.app.idea_dag import IdeaDag
from agent.app.agent_io import AgentIO
from agent.app.observation import clean_operation
from agent.app.idea_policies.base import IdeaActionType, DetailKey


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
        
        return {
            "action": action.value,
            "success": False,
            "node_id": node_id,
            "error": error_str,
            "error_type": error_type,
            "root_cause": root_cause,
            "http_status": http_status,
            "traceback_summary": tb_summary,
            "retryable": self._is_retryable(error),
            "context": context or {},
            "timestamp": None,
        }


class SearchLeafAction(LeafAction):
    """Leaf action for web search."""
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
            query = (
                node.details.get(DetailKey.QUERY.value)
                or node.details.get(DetailKey.PROMPT.value)
                or node.title
            )
            intent = node.details.get(DetailKey.INTENT.value)
            count = int(node.details.get(DetailKey.COUNT.value, self.settings.get("default_search_count", 10)))
            
            # Query vector DB with intent if provided
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
            return {
                "action": IdeaActionType.SEARCH.value,
                "success": True,
                "query": query,
                "intent": intent,
                "vector_context": [str(doc) for doc in vector_context] if vector_context else [],
                "count": count,
                "results": results or [],
            }
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
                failure["query"] = query
            if count is not None:
                failure["count"] = count
            return failure


class VisitLeafAction(LeafAction):
    """Leaf action for visiting URLs and extracting content."""
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        url = None
        intent = None
        vector_context = []
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            url = (
                node.details.get(DetailKey.URL.value)
                or node.details.get(DetailKey.LINK.value)
                or node.title
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
                return {
                    "action": IdeaActionType.VISIT.value,
                    "success": False,
                    "url": url,
                    "intent": intent,
                    "error": f"Site blocked: {blocked_reason}",
                    "error_type": "BlockedSite",
                    "retryable": False,
                }
            
            timeout_seconds = self._timeout_seconds("fetch_timeout_seconds")
            self._logger.debug(f"[VISIT] url='{url}', intent='{intent}', timeout={timeout_seconds}")
            raw_html = await io.fetch_url(str(url), timeout_seconds=timeout_seconds)
            self._logger.info(f"[VISIT] {len(raw_html) if raw_html else 0} chars from {url[:60]}...")
            cleaned = clean_operation(raw_html)
            content_payload = self._limit_text(cleaned)
            soup = BeautifulSoup(raw_html, "html.parser")
            all_links = []
            for tag in soup.find_all("a", href=True):
                href = tag.get("href")
                if href:
                    all_links.append(href)
            
            max_links_for_llm = int(self.settings.get("max_links_per_visit", 20))
            links_for_llm = all_links[:max_links_for_llm]
            
            return {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "url": url,
                "intent": intent,
                "vector_context": [str(doc) for doc in vector_context] if vector_context else [],
                "content": content_payload.get("content"),
                "content_is_truncated": content_payload.get("is_truncated"),
                "content_total_chars": content_payload.get("total_chars"),
                "content_full": cleaned,
                "links": links_for_llm,
                "links_full": all_links,
            }
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
                failure["url"] = url
            if is_bot_block:
                failure["retryable"] = False
                failure["error"] = f"{block_reason}: {error_str}"
            return failure


class ThinkLeafAction(LeafAction):
    """Leaf action with no external call."""
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            details_copy = self._copy_details_safely(node.details)
            return {
                "action": IdeaActionType.THINK.value,
                "success": True,
                "node_id": node.node_id,
                "title": node.title,
                "details": details_copy,
            }
        except Exception as exc:
            return self._failure(action=IdeaActionType.THINK, node_id=node_id, error=exc)


class SaveLeafAction(LeafAction):
    """Leaf action for saving content to ChromaDB."""
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
            return {"action": IdeaActionType.SAVE.value, "success": bool(success), "count": len(docs)}
        except Exception as exc:
            return self._failure(action=IdeaActionType.SAVE, node_id=node_id, error=exc)


class MergeLeafAction(LeafAction):
    """Leaf action for merging child results using LLM."""
    async def execute(self, graph: IdeaDag, node_id: str, io: AgentIO) -> Dict[str, Any]:
        import json
        node = None
        try:
            node = graph.get_node(node_id)
            if not node:
                raise ValueError(f"Unknown node_id: {node_id}")
            
            merged_results = node.details.get(DetailKey.MERGED_RESULTS.value) or []
            if not merged_results:
                return {
                    "action": IdeaActionType.MERGE.value,
                    "success": False,
                    "error": "No merged results to synthesize",
                }
            
            system_template = self.settings.get("merge_system_prompt", "")
            user_template = self.settings.get("merge_user_prompt", "")
            
            if not system_template or not user_template:
                self._logger.warning("No merge prompts found, using simple concatenation")
                synthesized = json.dumps(merged_results, ensure_ascii=True)
                return {
                    "action": IdeaActionType.MERGE.value,
                    "success": True,
                    "synthesized": synthesized,
                    "child_count": len(merged_results),
                }
            
            merged_json = json.dumps(merged_results, ensure_ascii=True)
            messages = [
                {"role": "system", "content": system_template},
                {"role": "user", "content": user_template.format(merged_json=merged_json)},
            ]
            
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
                return {
                    "action": IdeaActionType.MERGE.value,
                    "success": False,
                    "error": "LLM returned empty response",
                }
            
            try:
                synthesized_data = json.loads(response)
            except json.JSONDecodeError:
                synthesized_data = {"summary": response}
            
            return {
                "action": IdeaActionType.MERGE.value,
                "success": True,
                "synthesized": synthesized_data,
                "child_count": len(merged_results),
                "raw_response": response,
            }
        except Exception as exc:
            return self._failure(action=IdeaActionType.MERGE, node_id=node_id, error=exc)


class LeafActionRegistry:
    """Registry for leaf action implementations."""
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
        """Instantiate an action by type."""
        action_cls = self._registry.get(action_type)
        if not action_cls:
            raise ValueError(f"Unknown action type: {action_type}")
        return action_cls(settings=self.settings)


def execute_leaf_action(action: LeafAction, graph: IdeaDag, node_id: str, io: AgentIO):
    """
    Execute a leaf action and return payload.
    :param action: LeafAction instance.
    :param graph: IdeaDag instance.
    :param node_id: Node identifier.
    :param io: AgentIO instance.
    :returns: Execution payload.
    """
    return action.execute(graph, node_id, io)
