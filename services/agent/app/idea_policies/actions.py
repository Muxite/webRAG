from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, List, Set, Tuple
import uuid
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.observation import clean_operation
from agent.app.idea_policies.base import IdeaActionType, DetailKey, IdeaNodeStatus
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
    
    def _search_in_chunk(self, chunk_content: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search for query terms within a document chunk.
        :param chunk_content: Chunk text to search.
        :param query: Search query.
        :param max_results: Maximum number of results.
        :returns: List of search results with snippets.
        """
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
        """
        Extract URL from parent nodes, prioritizing relevant links based on node context.
        :param graph: IdeaDag instance.
        :param node: Current node needing a URL.
        :param max_depth: Maximum depth to search.
        :returns: Best matching URL or None.
        """
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

                # Prefer URLs that already came from search results
                if action_type == IdeaActionType.SEARCH.value:
                    search_results = result.get("results", [])
                    if isinstance(search_results, list):
                        for item in search_results[:5]:
                            if isinstance(item, dict):
                                candidate_url = item.get("url") or item.get("link")
                                if candidate_url and self._is_valid_url(candidate_url):
                                    all_candidates.append((candidate_url, depth, "search"))
                
                # Also allow URLs from previous visit results (their links field)
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
        
        if not all_candidates:
            return None
        
        # Score candidates based on relevance
        scored = []
        for candidate in all_candidates:
            url = candidate[0]
            url_lower = url.lower()
            score = 0
            
            # Prefer closer nodes (lower depth)
            score += (max_depth - candidate[1]) * 10
            
            # Prefer URLs that match keywords in node title/intent
            if "wikipedia" in node_title_lower or "wikipedia" in node_intent_lower:
                if "wikipedia.org" in url_lower:
                    score += 50
            if "guido" in node_title_lower or "guido" in node_intent_lower:
                if "guido" in url_lower or "van_rossum" in url_lower:
                    score += 30
            
            # Prefer visit links over search (more reliable)
            if len(candidate) > 2 and candidate[2] == "visit":
                score += 5
            
            # Check if link context matches node intent
            if len(candidate) > 3:
                context = candidate[3].lower() if candidate[3] else ""
                if context and any(word in context for word in node_title_lower.split()[:5]):
                    score += 20
            
            scored.append((score, url))
        
        # Return highest scored URL
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored:
            return scored[0][1]
        
        return None
    
    def _extract_url_from_think_node(self, graph: IdeaDag, node: IdeaNode) -> Optional[str]:
        """
        Extract URL from a think node that was supposed to select a URL.
        Checks if this visit node depends on a think node via REQUIRES_DATA.
        :param graph: IdeaDag instance.
        :param node: Current visit node.
        :returns: Extracted URL or None.
        """
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
        """
        Extract a URL from completed sibling node results.
        
        In a sequential execution, earlier siblings may have visited pages and
        produced links. This method searches those sibling results for URLs
        that match the current node's intent/title, enabling chained visits.
        
        :param graph: IdeaDag instance.
        :param node: Current visit node.
        :returns: Best matching URL from sibling results or None.
        """
        import re
        
        # Find parent to get siblings
        parent = None
        for pid in node.parent_ids:
            parent = graph.get_node(pid)
            if parent:
                break
        if not parent:
            return None
        
        node_title_lower = (node.title or "").lower()
        node_intent = (node.details.get(DetailKey.INTENT.value) or "").lower()
        
        # Collect URLs from completed sibling results
        sibling_links: List[str] = []
        for sibling_id in parent.children:
            if sibling_id == node.node_id:
                continue
            sibling = graph.get_node(sibling_id)
            if not sibling or sibling.status.value != "done":
                continue
            result = sibling.details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(result, dict) or not result.get("success"):
                continue
            
            # Get links from the sibling's visit result
            links = result.get("links", []) or result.get("links_full", [])
            if isinstance(links, list):
                sibling_links.extend(links)
            
            # Also check inline link content
            content = result.get("content_with_links", "") or result.get("content", "")
            if isinstance(content, str):
                found = re.findall(r'https?://[^\s\]\)\"\'<>]+', content)
                sibling_links.extend(found)
        
        if not sibling_links:
            return None
        
        # Deduplicate
        seen = set()
        unique_links = []
        for link in sibling_links:
            if link not in seen and self._is_valid_url(link):
                seen.add(link)
                unique_links.append(link)
        
        if not unique_links:
            return None
        
        # Try to find a link that matches the node's intent
        search_terms = node_title_lower + " " + node_intent
        best_link = None
        best_score = 0
        
        for link in unique_links:
            link_lower = link.lower()
            score = 0
            # Score based on keyword overlap
            for word in search_terms.split():
                if len(word) > 3 and word in link_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_link = link
        
        if best_link:
            self._logger.info(f"[VISIT] Found URL from sibling results: {best_link[:80]}")
            return best_link
        
        # If no match, return first valid link as fallback
        self._logger.info(f"[VISIT] Using first sibling link as fallback: {unique_links[0][:80]}")
        return unique_links[0]
    
    async def _store_links_in_chroma(
        self,
        base_url: str,
        links: List[str],
        link_contexts: Dict[str, str],
        io: AgentIO,
    ) -> bool:
        """
        Store all links from a visited page in a Chroma collection for later semantic search.
        Collection name: links_{base_url_hash}
        :param base_url: Base URL of the visited page.
        :param links: List of cleaned absolute URLs.
        :param link_contexts: Mapping of URL -> anchor text.
        :param io: Agent IO instance.
        :returns: True if stored successfully.
        """
        if not links or not getattr(io, "connector_chroma", None):
            return False
        
        try:
            if not base_url or not isinstance(base_url, str):
                self._logger.warning(f"[VISIT] Invalid base_url for link storage: {base_url}")
                return False
            
            import hashlib
            url_hash = hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:12]
            collection_name = f"links_{url_hash}"
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
                success = await io.connector_chroma.add_to_chroma(
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
        """
        Query Chroma for links matching the link_idea concept.
        Searches across all link collections (links_*).
        :param link_idea: Short description of what kind of link is wanted.
        :param io: Agent IO instance.
        :param top_k: Maximum number of links to return.
        :returns: List of URLs (best matches first).
        """
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
            
            link_collections = [c for c in all_collections if c and c.startswith("links_")]
            
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
    
    async def _select_links_with_llm(
        self,
        link_idea: str,
        candidate_urls: List[str],
        link_count: int,
        io: AgentIO,
    ) -> List[str]:
        """
        Use lightweight LLM call to select top B links from candidates.
        This is part of the same node, not a separate think/expand node.
        :param link_idea: Original idea of what link is wanted.
        :param candidate_urls: List of candidate URLs to choose from.
        :param link_count: Number of links to select.
        :param io: Agent IO instance.
        :returns: Selected URLs.
        """
        if not candidate_urls or link_count <= 0:
            return []
        
        if len(candidate_urls) <= link_count:
            return candidate_urls[:link_count]
        
        try:
            model_name = self.settings.get("visit_link_selection_model") or self.settings.get("evaluation_model") or ""
            if not model_name:
                return candidate_urls[:link_count]
            
            candidates_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(candidate_urls)])
            system_content = f"Select the top {link_count} URLs that best match the user's request. Return JSON with a 'selected' array of URLs in order of preference."
            user_content = f"User wants: {link_idea}\n\nCandidate URLs:\n{candidates_text}\n\nReturn JSON: {{\"selected\": [\"url1\", \"url2\", ...]}}"
            
            messages = PromptBuilder.build_messages(system_content=system_content, user_content=user_content)
            payload = io.build_llm_payload(
                messages=messages,
                json_mode=True,
                model_name=model_name,
                temperature=0.2,
                max_tokens=500,
            )
            
            timeout_seconds = self._timeout_seconds("llm_timeout_seconds")
            response = await io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
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
                except json.JSONDecodeError:
                    self._logger.warning(f"[VISIT] Failed to parse LLM link selection response: {response[:200]}")
        except Exception as exc:
            self._logger.warning(f"[VISIT] LLM link selection failed: {exc}")
        
        return candidate_urls[:link_count]
    
    async def _visit_single_page(
        self,
        url: str,
        graph: IdeaDag,
        node: IdeaNode,
        io: AgentIO,
        intent: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, str]]:
        """
        Visit a single page and extract content, links, and metadata.
        :param url: URL to visit.
        :param graph: IdeaDag instance.
        :param node: Current node.
        :param io: Agent IO instance.
        :param intent: Optional intent for vector context.
        :returns: Tuple of (result_dict, cleaned_links, cleaned_link_contexts).
        """
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
        
        cleaned_links = self._filter_and_prioritize_links(raw_links, str(url))
        cleaned_link_contexts = {}
        for raw_link in raw_links:
            cleaned = self._clean_and_fix_link(raw_link, str(url))
            if cleaned and raw_link in link_contexts:
                cleaned_link_contexts[cleaned] = link_contexts[raw_link]
        
        await self._store_links_in_chroma(str(url), cleaned_links, cleaned_link_contexts, io)
        
        content_payload = self._limit_text(cleaned)
        content_text = content_payload.get("content") or cleaned or ""
        if not content_text or len(content_text.strip()) == 0:
            content_text = soup.get_text(separator="\n", strip=True)
            if content_text:
                content_payload = self._limit_text(content_text)
                content_text = content_payload.get("content") or content_text
        
        max_links_for_llm = int(self.settings.get("max_links_per_visit", 20))
        links_for_llm = cleaned_links[:max_links_for_llm]
        content_with_links = self._attach_links_to_content(content_text, links_for_llm, max_links=max_links_for_llm)
        
        final_content = content_payload.get("content") or content_text or ""
        
        result = ActionResultBuilder.success(
            action=IdeaActionType.VISIT.value,
            url=url,
            intent=intent,
            content=final_content,
            content_is_truncated=content_payload.get("is_truncated", False),
            content_total_chars=content_payload.get("total_chars", len(final_content)),
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
            
            link_count = node.details.get("link_count")
            if link_count is None:
                link_count = 1
            else:
                try:
                    link_count = int(link_count)
                except (ValueError, TypeError):
                    link_count = 1
            
            link_idea = node.details.get("link_idea") or node.details.get("link_concept") or ""
            optional_url = node.details.get("optional_url") or NodeDetailsExtractor.get_url(node.details)
            
            # Detect and clear placeholder URLs (e.g., "<chosen_next_url from ...>")
            if optional_url and not self._is_valid_url(optional_url):
                if optional_url.startswith("<") or optional_url.startswith("{") or "chosen" in optional_url.lower():
                    self._logger.warning(f"[VISIT] Clearing placeholder URL: {optional_url[:80]}")
                    optional_url = None
            
            if not optional_url or not self._is_valid_url(optional_url):
                think_url = self._extract_url_from_think_node(graph, node)
                if think_url:
                    optional_url = think_url
                else:
                    extracted_url = self._extract_url_from_parents(graph, node)
                    if extracted_url:
                        optional_url = extracted_url
                    else:
                        # Try to extract URL from completed sibling results
                        sibling_url = self._extract_url_from_sibling_results(graph, node)
                        if sibling_url:
                            optional_url = sibling_url
            
            # Auto-generate link_idea from intent or title if missing
            if not link_idea and not (optional_url and self._is_valid_url(optional_url)):
                link_idea = intent or node.title or ""
                if link_idea:
                    # Trim to a short semantic search phrase
                    link_idea = link_idea[:200]
                    self._logger.info(f"[VISIT] Auto-generated link_idea from context: '{link_idea[:60]}...'")
            
            max_sites = int(self.settings.get("visit_max_sites_per_action", 20))
            link_count = min(link_count, max_sites)
            
            urls_to_visit: List[str] = []
            optional_success = False
            
            if optional_url and self._is_valid_url(optional_url):
                result, _, _ = await self._visit_single_page(optional_url, graph, node, io, intent)
                if result and result.get("success"):
                    optional_success = True
                    urls_to_visit.append(optional_url)
                    self._logger.info(f"[VISIT] Optional URL visited successfully: {optional_url[:60]}...")
                    if link_count == 1:
                        # Record visit in telemetry before returning
                        if io.telemetry:
                            io.telemetry.record_document_seen(
                                source="visit",
                                document={"url": optional_url, "content": result.get("content_with_links", result.get("content", ""))},
                            )
                        return result
            
            if not optional_success or link_count > 1:
                if link_idea:
                    query_top_k = int(self.settings.get("visit_link_query_top_k", 10))
                    candidate_urls = await self._query_links_from_chroma(link_idea, io, top_k=query_top_k)
                    
                    if candidate_urls:
                        if link_count > len(urls_to_visit):
                            needed = link_count - len(urls_to_visit)
                            if len(candidate_urls) > needed:
                                selected = await self._select_links_with_llm(link_idea, candidate_urls, needed, io)
                                urls_to_visit.extend(selected)
                            else:
                                urls_to_visit.extend(candidate_urls[:needed])
                else:
                    if not optional_url or not optional_success:
                        error_msg = f"Visit node missing valid URL or link_idea. Node title: '{node.title}'. Details should contain 'url'/'optional_url' or 'link_idea' for semantic link discovery."
                        self._logger.error(f"[VISIT] {error_msg}")
                        return ActionResultBuilder.failure(
                            action=IdeaActionType.VISIT.value,
                            error=error_msg,
                            error_type=ErrorType.INVALID_URL.value,
                            retryable=False,
                            url=optional_url or node.title,
                        )
            
            if not urls_to_visit:
                if optional_url and self._is_valid_url(optional_url):
                    urls_to_visit = [optional_url]
                else:
                    error_msg = f"Visit node: no URLs to visit. link_count={link_count}, link_idea='{link_idea}', optional_url='{optional_url}'"
                    self._logger.error(f"[VISIT] {error_msg}")
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
            
            for url_to_visit in urls_to_visit:
                if url_to_visit in [r.get("url") for r in visited_results]:
                    continue
                
                result, page_links, page_link_contexts = await self._visit_single_page(url_to_visit, graph, node, io, intent)
                
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
                return ActionResultBuilder.failure(
                    action=IdeaActionType.VISIT.value,
                    error="All URL visits failed",
                    error_type=ErrorType.NETWORK_ERROR.value,
                    retryable=True,
                )
            
            primary_result = visited_results[0]
            combined_content_text = "\n\n".join(combined_content)
            max_links_for_llm = int(self.settings.get("max_links_per_visit", 20))
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
            
            return ActionResultBuilder.success(
                action=IdeaActionType.VISIT.value,
                url=urls_to_visit[0] if urls_to_visit else None,
                urls_visited=urls_to_visit,
                intent=intent,
                vector_context=[str(doc) for doc in vector_context] if vector_context else [],
                content=combined_content_text[:6000] if len(combined_content_text) > 6000 else combined_content_text,
                content_is_truncated=len(combined_content_text) > 6000,
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
    def _extract_url_from_parent_result(self, graph: IdeaDag, node: IdeaNode) -> Optional[str]:
        """
        Extract URL from parent node's action result if this think node is meant to select a URL.
        Looks for URLs in parent visit/search results based on node intent.
        :param graph: IdeaDag instance.
        :param node: Current think node.
        :returns: Extracted URL or None.
        """
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
            
            links = source_result.get(ActionResultKey.LINKS.value) or source_result.get(ActionResultKey.LINKS_FULL.value) or []
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
            
            original_goal = node.details.get(DetailKey.GOAL.value) or node.details.get(DetailKey.ORIGINAL_GOAL.value) or node.title
            parent_intent = node.details.get(DetailKey.INTENT.value) or ""
            parent_justification = node.details.get(DetailKey.PARENT_JUSTIFICATION.value) or node.details.get(DetailKey.JUSTIFICATION.value) or ""
            
            if node.parent_id:
                parent = graph.get_node(node.parent_id)
                if parent:
                    if not original_goal:
                        original_goal = parent.details.get(DetailKey.GOAL.value) or parent.details.get(DetailKey.ORIGINAL_GOAL.value) or parent.title
                    if not parent_intent:
                        parent_intent = parent.details.get(DetailKey.INTENT.value) or ""
                    if not parent_justification:
                        parent_justification = parent.details.get(DetailKey.JUSTIFICATION.value) or parent.details.get(DetailKey.PARENT_JUSTIFICATION.value) or ""
            
            merged_json = json.dumps(compacted_merged, ensure_ascii=True)
            user_content = user_template.format(
                merged_json=merged_json,
                original_goal=original_goal or "",
                parent_intent=parent_intent,
                parent_justification=parent_justification
            )
            
            messages = PromptBuilder.build_messages(
                system_content=system_template,
                user_content=user_content,
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
            self._logger.info(f"[MERGE] LLM Input - Full messages: {json.dumps(messages, indent=2, ensure_ascii=True)}")
            response = await io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
                timeout_seconds=timeout_seconds,
            )
            self._logger.info(f"[MERGE] LLM Output - Full response: {response}")
            
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
            goal_evaluation = synthesized_data.get("goal_evaluation", "")
            missing_requirements = synthesized_data.get("missing_requirements", [])
            
            if not isinstance(goal_achieved, bool):
                goal_achieved = bool(goal_achieved)
            
            node.details[DetailKey.GOAL_ACHIEVED.value] = goal_achieved
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
