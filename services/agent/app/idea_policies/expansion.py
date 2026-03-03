from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import ExpansionPolicy, DetailKey, IdeaActionType
from agent.app.idea_dag_settings import load_idea_dag_settings


def _safe_serialize_details(details: Dict[str, Any]) -> str:
    try:
        return json.dumps(details, ensure_ascii=True, default=str)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=True)


class LlmExpansionPolicy(ExpansionPolicy):
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        default_settings = load_idea_dag_settings()
        merged_settings = {**default_settings, **(settings or {})}
        super().__init__(settings=merged_settings)
        self.io = io
        self.model_name = model_name
        self._logger = logging.getLogger(self.__class__.__name__)

    async def expand(self, graph: IdeaDag, node_id: str, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        node = graph.get_node(node_id)
        if not node:
            return []
        messages = self._build_messages(graph, node, memories=memories)
        
        total_prompt_size = sum(len(msg.get("content", "")) for msg in messages)
        if total_prompt_size > 50000:
            self._logger.warning(f"[EXPANSION] Large prompt detected ({total_prompt_size} chars) for node {node_id} - may cause slow expansion")
        
        model_name = self.model_name or self.settings.get("expansion_model")
        json_schema = self.settings.get("expansion_json_schema")
        reasoning_effort = self.settings.get("reasoning_effort", "high")
        text_verbosity = self.settings.get("text_verbosity", "medium")
        max_tokens = self.settings.get("expansion_max_tokens") if self.settings.get("expansion_max_tokens") is not None else None
        
        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=float(self.settings.get("expansion_temperature", 0.4)),
            max_tokens=max_tokens,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        try:
            estimated_tokens = (total_prompt_size // 4) + (max_tokens or 4096)
            self._logger.debug(f"[EXPANSION] Calling LLM for node {node_id} with model={model_name}, prompt={total_prompt_size} chars, max_tokens={max_tokens}, estimated ~{estimated_tokens} total tokens")
            
            default_timeout = self.settings.get("llm_timeout_seconds") or 120
            expansion_timeout = self.settings.get("expansion_timeout_seconds") or default_timeout
            if total_prompt_size > 50000 or estimated_tokens > 10000:
                expansion_timeout = max(expansion_timeout, 180)
            else:
                expansion_timeout = max(expansion_timeout, 120)
            
            self._logger.info(f"[EXPANSION] LLM Input - Full messages: {json.dumps(messages, indent=2, ensure_ascii=True)}")
            content = await self.io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
                timeout_seconds=expansion_timeout,
            )
            self._logger.info(f"[EXPANSION] LLM Output - Full response: {content}")
            candidates, meta = self._parse_candidates(content, graph=graph, parent_node_id=node_id)
            self._logger.info(f"[EXPANSION] Parsed {len(candidates)} candidates from LLM response, meta={meta}")
            if not candidates:
                self._logger.error(f"[EXPANSION] CRITICAL: No candidates parsed from LLM response!")
                self._logger.error(f"[EXPANSION] LLM response length: {len(content) if content else 0} chars")
                self._logger.error(f"[EXPANSION] LLM response preview: {content[:500] if content else 'None'}")
                fallback_candidate = self._create_fallback_candidate(node, graph)
                if fallback_candidate:
                    self._logger.warning(f"[EXPANSION] Created fallback candidate: {fallback_candidate.get('title', 'Unknown')[:60]}...")
                    candidates = [fallback_candidate]
            if meta:
                node.details[DetailKey.EXPANSION_META.value] = meta
            return candidates
        except asyncio.TimeoutError as e:
            self._logger.error(f"[EXPANSION] Timeout during expansion for node {node_id}: {e}")
            self._logger.warning(f"[EXPANSION] Expansion timeout - returning empty candidates. Consider increasing expansion_timeout_seconds.")
            return []
        except Exception as e:
            self._logger.error(f"[EXPANSION] Exception during expansion: {e}", exc_info=True)
            return []

    def _enhance_details_with_inline_links(self, details: Dict[str, Any]) -> Dict[str, Any]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        enhanced = dict(details)
        
        action_result = details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(action_result, dict):
            return enhanced
        
        action = action_result.get(ActionResultKey.ACTION.value)
        if action != IdeaActionType.VISIT.value:
            return enhanced
        
        success = action_result.get(ActionResultKey.SUCCESS.value, False)
        if not success:
            return enhanced
        
        links = action_result.get(ActionResultKey.LINKS.value) or action_result.get(ActionResultKey.LINKS_FULL.value) or []
        if not isinstance(links, list) or len(links) == 0:
            return enhanced
        
        link_contexts = action_result.get(ActionResultKey.LINK_CONTEXTS.value) or {}
        max_links_to_show = int(self.settings.get("max_links_per_visit", 20))
        links_to_show = links[:max_links_to_show]
        
        inline_links_section = []
        for link_url in links_to_show:
            if not isinstance(link_url, str) or not link_url.startswith(("http://", "https://")):
                continue
            
            context_text = ""
            if isinstance(link_contexts, dict) and link_url in link_contexts:
                context = link_contexts[link_url]
                if isinstance(context, str) and context.strip():
                    context_text = context.strip()[:150]
            
            if context_text:
                inline_links_section.append(f"{context_text} [link: {link_url}]")
            else:
                inline_links_section.append(f"[link: {link_url}]")
        
        if inline_links_section:
            enhanced_action_result = dict(action_result)
            enhanced_action_result["_links_inline"] = "\n".join(inline_links_section)
            if len(links) > max_links_to_show:
                enhanced_action_result["_links_inline"] += f"\n... and {len(links) - max_links_to_show} more links (see 'links' field for full list)"
            enhanced[DetailKey.ACTION_RESULT.value] = enhanced_action_result
        
        return enhanced
    
    def _compact_details_for_expansion(self, details: Dict[str, Any]) -> Dict[str, Any]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        compact = dict(details)
        
        action_result = details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(action_result, dict):
            return compact
        
        compact_result = dict(action_result)
        
        large_fields_to_remove = [
            ActionResultKey.CONTENT_FULL.value,
            ActionResultKey.CONTENT_WITH_LINKS.value,
            "content_full",
            "content_with_links",
        ]
        
        for field in large_fields_to_remove:
            if field in compact_result:
                del compact_result[field]
        
        if ActionResultKey.CONTENT.value in compact_result:
            content = compact_result[ActionResultKey.CONTENT.value]
            if isinstance(content, str) and len(content) > 1000:
                compact_result[ActionResultKey.CONTENT.value] = content[:1000] + "... [truncated]"
        
        if ActionResultKey.LINKS_FULL.value in compact_result:
            links_full = compact_result.get(ActionResultKey.LINKS_FULL.value, [])
            if isinstance(links_full, list) and len(links_full) > 20:
                compact_result[ActionResultKey.LINKS_FULL.value] = links_full[:20]
                compact_result["_links_full_truncated"] = f"... and {len(links_full) - 20} more links"
        
        compact[DetailKey.ACTION_RESULT.value] = compact_result
        return compact
    
    def _extract_key_outcome(self, node: IdeaNode) -> Optional[str]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        action = node.details.get(DetailKey.ACTION.value, "")
        if not result.get(ActionResultKey.SUCCESS.value, False):
            error = result.get(ActionResultKey.ERROR.value, "unknown error")
            return f"FAILED: {str(error)[:80]}"
        if action == IdeaActionType.SEARCH.value:
            results = result.get(ActionResultKey.RESULTS.value, [])
            count = len(results) if isinstance(results, list) else 0
            top_urls = []
            if isinstance(results, list):
                for r in results[:3]:
                    if isinstance(r, dict) and r.get("url"):
                        top_urls.append(str(r["url"])[:80])
            if top_urls:
                return f"Found {count} results. Top URLs: {', '.join(top_urls)}"
            return f"Found {count} results"
        if action == IdeaActionType.VISIT.value:
            url = result.get(ActionResultKey.URL.value, "")
            page_title = result.get("page_title", "")
            content_chars = result.get("content_total_chars", 0)
            links_count = result.get("links_count", 0)
            parts = []
            if url:
                parts.append(f"Visited {str(url)[:80]}")
            if page_title:
                parts.append(f"page='{str(page_title)[:50]}'")
            parts.append(f"{content_chars} chars, {links_count} links")
            return ". ".join(parts)
        if action == IdeaActionType.THINK.value:
            return "Internal reasoning completed"
        return None

    def _build_messages(self, graph: IdeaDag, node: IdeaNode, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        max_nodes = int(self.settings.get("expansion_max_context_nodes", 5))
        max_detail_chars = int(self.settings.get("expansion_max_detail_chars", 2000))
        max_children = int(self.settings.get("max_branching", 5))
        if max_children <= 1:
            max_children = 1
        path = graph.path_to_root(node.node_id)
        path = path[:max_nodes]
        
        serialized = []
        for entry in path:
            enhanced_details = self._enhance_details_with_inline_links(entry.details)
            compact_details = self._compact_details_for_expansion(enhanced_details)
            details_text = _safe_serialize_details(compact_details)
            if len(details_text) > max_detail_chars:
                details_text = details_text[:max_detail_chars]
            serialized.append(
                {
                    "node_id": entry.node_id,
                    "title": entry.title,
                    "status": entry.status.value,
                    "score": entry.score,
                    "action": entry.details.get(DetailKey.ACTION.value, "expansion"),
                    "goal": entry.details.get(DetailKey.GOAL.value, ""),
                    "justification": (
                        entry.details.get(DetailKey.JUSTIFICATION.value)
                        or entry.details.get(DetailKey.WHY_THIS_NODE.value)
                        or ""
                    ),
                    "key_outcome": self._extract_key_outcome(entry),
                    "details": details_text,
                }
            )
        allowed = self.settings.get("allowed_actions") or [a.value for a in IdeaActionType]
        allowed_actions = ", ".join(
            str(item) for item in allowed
            if str(item) != IdeaActionType.MERGE.value
        )
        path_json = json.dumps(serialized, ensure_ascii=True)
        
        blocked_sites = graph._blocked_sites if hasattr(graph, "_blocked_sites") else {}
        blocked_sites_list = [f"{domain}: {reason}" for domain, reason in blocked_sites.items()]
        blocked_sites_text = "\n".join(blocked_sites_list) if blocked_sites_list else "None"
        
        errors = []
        for entry in path:
            error = entry.details.get(DetailKey.ACTION_ERROR.value)
            if error:
                errors.append(f"{entry.title}: {error}")
        errors_text = "\n".join(errors) if errors else "None"
        
        memories_text = "None"
        if memories:
            from agent.app.idea_memory import MemoryManager
            temp_mm = MemoryManager(connector_chroma=None, namespace="temp")
            memories_text = temp_mm.format_memories_for_llm(memories, max_chars=4000)
        
        event_log = graph.build_event_log_table(node.node_id, max_events=15)
        event_log_json = json.dumps(event_log) if event_log else json.dumps("No events")
        
        system_template = self.settings.get("expansion_system_prompt")
        user_template = self.settings.get("expansion_user_prompt")
        effective_range = f"exactly {max_children}" if max_children <= 1 else f"2-{max_children}"
        try:
            system = system_template.format(
                allowed_actions=allowed_actions,
                max_children=effective_range,
            ) if system_template else ""
        except KeyError as fmt_err:
            self._logger.error(f"[EXPANSION] System prompt format error (missing key: {fmt_err}) - using raw template")
            system = (system_template or "").replace("{allowed_actions}", str(allowed_actions)).replace("{max_children}", str(effective_range))
        planning_addendum = str(
            self.settings.get(
                "expansion_planning_addendum",
                "Before producing candidates, build an internal plan with target facts, source strategy, and verification steps.",
            )
        ).strip()
        if planning_addendum:
            system = f"{system}\n\n{planning_addendum}" if system else planning_addendum
        format_kwargs = dict(
            path_json=path_json,
            parent_id=node.node_id,
            parent_title=node.title,
            blocked_sites=blocked_sites_text,
            errors=errors_text,
            memories=memories_text,
            event_log=event_log_json,
        )
        try:
            user = user_template.format(**format_kwargs) if user_template else json.dumps(
                {
                    "path": serialized,
                    "parent_id": node.node_id,
                    "parent_title": node.title,
                    "blocked_sites": blocked_sites,
                    "errors": errors,
                    "memories": memories_text,
                    "event_log": event_log,
                },
                ensure_ascii=True,
            )
        except KeyError as fmt_err:
            self._logger.error(f"[EXPANSION] User prompt format error (missing key: {fmt_err}) - using manual substitution")
            user = user_template or ""
            for k, v in format_kwargs.items():
                user = user.replace("{" + k + "}", str(v))
        from agent.app.idea_policies.action_constants import PromptBuilder
        messages = PromptBuilder.build_messages(system_content=system, user_content=user)
        
        total_prompt_size = sum(len(msg.get("content", "")) for msg in messages)
        self._logger.debug(f"[EXPANSION] Prompt size: system={len(system)} chars, user={len(user)} chars, total={total_prompt_size} chars")
        if total_prompt_size > 50000:
            self._logger.warning(f"[EXPANSION] Large prompt detected ({total_prompt_size} chars) - may cause slow expansion. Consider reducing expansion_max_context_nodes or expansion_max_detail_chars")
        
        return messages

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        
        import re
        link_pattern = r'\[link:\s*(https?://[^\]]+)\]'
        match = re.search(link_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        url_pattern = r'https?://[^\s\)\]\>\"\']+'
        match = re.search(url_pattern, text)
        if match:
            url = match.group(0).rstrip('.,;:!?)')
            if url.startswith(("http://", "https://")):
                return url
        
        return None
    
    def _is_url_from_visit(self, graph: IdeaDag, node_id: str) -> bool:
        node = graph.get_node(node_id)
        if not node:
            return False
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        action = NodeDetailsExtractor.get_action(node.details)
        return action == IdeaActionType.VISIT.value
    
    def _extract_url_from_path_context_with_source(self, graph: IdeaDag, node_id: str, candidate_title: str = "") -> tuple[Optional[str], Optional[str]]:
        url = self._extract_url_from_path_context(graph, node_id, candidate_title)
        if not url:
            return None, None
        
        path = graph.path_to_root(node_id)
        candidate_keywords = set()
        if candidate_title:
            import re
            words = re.findall(r'\b\w+\b', candidate_title.lower())
            candidate_keywords = {w for w in words if len(w) > 3}
        
        best_match = None
        best_score = 0
        best_source = None
        first_url = None
        first_source = None
        
        for path_node in reversed(path):
            details = path_node.details or {}
            action_result = details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(action_result, dict):
                continue
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            action_type = action_result.get(ActionResultKey.ACTION.value)
            
            if action_type == IdeaActionType.SEARCH.value:
                results = action_result.get(ActionResultKey.RESULTS.value) or []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            result_url = result.get("url") or result.get("link")
                            if result_url and isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                                if not first_url:
                                    first_url = result_url
                                    first_source = path_node.node_id
                                
                                if candidate_keywords:
                                    result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                                    score = sum(1 for kw in candidate_keywords if kw in result_text)
                                    if score > best_score:
                                        best_score = score
                                        best_match = result_url
                                        best_source = path_node.node_id
            
            elif action_type == IdeaActionType.VISIT.value:
                links_inline = action_result.get("_links_inline")
                if links_inline and isinstance(links_inline, str):
                    for line in links_inline.split('\n'):
                        if '[link:' in line:
                            extracted_url = self._extract_url_from_text(line)
                            if extracted_url:
                                if not first_url:
                                    first_url = extracted_url
                                    first_source = path_node.node_id
                                
                                if candidate_keywords:
                                    line_lower = line.lower()
                                    score = sum(1 for kw in candidate_keywords if kw in line_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = extracted_url
                                        best_source = path_node.node_id
        
        if best_match and best_source:
            return best_match, best_source
        if first_url and first_source:
            return first_url, first_source
        return None, None
    
    def _extract_url_from_path_context(self, graph: IdeaDag, node_id: str, candidate_title: str = "") -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        path = graph.path_to_root(node_id)
        candidate_keywords = set()
        if candidate_title:
            import re
            words = re.findall(r'\b\w+\b', candidate_title.lower())
            candidate_keywords = {w for w in words if len(w) > 3}
        
        best_match = None
        best_score = 0
        first_url = None
        
        for path_node in reversed(path):
            details = path_node.details or {}
            action_result = details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(action_result, dict):
                continue
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            action_type = action_result.get(ActionResultKey.ACTION.value)
            
            if action_type == IdeaActionType.SEARCH.value:
                results = action_result.get(ActionResultKey.RESULTS.value) or []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            result_url = result.get("url") or result.get("link")
                            if result_url and isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                                if not first_url:
                                    first_url = result_url
                                
                                if candidate_keywords:
                                    result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                                    score = sum(1 for kw in candidate_keywords if kw in result_text)
                                    if score > best_score:
                                        best_score = score
                                        best_match = result_url
            
            elif action_type == IdeaActionType.VISIT.value:
                links_inline = action_result.get("_links_inline")
                if links_inline and isinstance(links_inline, str):
                    for line in links_inline.split('\n'):
                        if '[link:' in line:
                            url = self._extract_url_from_text(line)
                            if url:
                                if not first_url:
                                    first_url = url
                                
                                if candidate_keywords:
                                    line_lower = line.lower()
                                    score = sum(1 for kw in candidate_keywords if kw in line_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = url
        
        return best_match if best_match else first_url
    
    def _parse_candidates(self, content: Optional[str], graph: Optional[IdeaDag] = None, parent_node_id: Optional[str] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not content:
            return [], {}
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self._logger.error(f"[EXPANSION] JSON PARSE ERROR: {e}")
            self._logger.error(f"[EXPANSION] Content preview (first 500 chars): {content[:500] if content else 'None'}")
            if content:
                import re
                json_match = re.search(r'\{[^{}]*"candidates"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        self._logger.info(f"[EXPANSION] Extracted JSON from embedded text")
                    except:
                        pass
            if 'data' not in locals():
                return [], {}
        except Exception as e:
            self._logger.error(f"[EXPANSION] PARSE EXCEPTION: {e}", exc_info=True)
            self._logger.error(f"[EXPANSION] Content preview (first 500 chars): {content[:500] if content else 'None'}")
            return [], {}
        
        if 'data' not in locals():
            return [], {}
            
        candidates = data.get("candidates", [])
        if not candidates:
            self._logger.error(f"[EXPANSION] NO CANDIDATES IN RESPONSE!")
            self._logger.error(f"[EXPANSION] Response data keys: {list(data.keys())}")
            self._logger.error(f"[EXPANSION] Full response data: {json.dumps(data, indent=2, ensure_ascii=True)[:1000]}")
        meta = data.get("meta") or {}
        cleaned: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            action = candidate.get(DetailKey.ACTION.value)
            title = candidate.get("title") or ""
            details = candidate.get("details") or {}
            if action:
                details = dict(details)
                details[DetailKey.ACTION.value] = action
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            justification = NodeDetailsExtractor.get_justification(candidate)
            if justification:
                details[DetailKey.JUSTIFICATION.value] = str(justification)

            candidate_goal = candidate.get("goal")
            local_goal: Optional[str] = None
            if isinstance(candidate_goal, str) and candidate_goal.strip():
                local_goal = candidate_goal.strip()
            else:
                existing_goal = details.get(DetailKey.GOAL.value) or details.get(DetailKey.ORIGINAL_GOAL.value)
                if isinstance(existing_goal, str) and existing_goal.strip():
                    local_goal = existing_goal.strip()
                elif isinstance(title, str) and title.strip():
                    local_goal = title.strip()

            if local_goal:
                details[DetailKey.GOAL.value] = details.get(DetailKey.GOAL.value) or local_goal
                if not details.get(DetailKey.ORIGINAL_GOAL.value):
                    details[DetailKey.ORIGINAL_GOAL.value] = local_goal
            
            if action == IdeaActionType.VISIT.value:
                url = (
                    details.get(DetailKey.URL.value)
                    or details.get(DetailKey.LINK.value)
                    or details.get("url")
                    or details.get("link")
                    or details.get("optional_url")
                )
                if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    extracted_url = None
                    source_node_id = None
                    
                    if title:
                        extracted_url = self._extract_url_from_text(title)
                    if not extracted_url and justification:
                        extracted_url = self._extract_url_from_text(str(justification))
                    if not extracted_url and graph and parent_node_id:
                        extracted_url, source_node_id = self._extract_url_from_path_context_with_source(graph, parent_node_id, candidate_title=title)
                    
                    if extracted_url:
                        details[DetailKey.URL.value] = extracted_url
                        if source_node_id:
                            source_node = graph.get_node(source_node_id)
                            if source_node:
                                from agent.app.idea_policies.action_constants import NodeDetailsExtractor
                                source_action = NodeDetailsExtractor.get_action(source_node.details)
                                if source_action == IdeaActionType.THINK.value:
                                    details[DetailKey.REQUIRES_DATA.value] = {
                                        "type": "url_from_think",
                                        "source_node_id": source_node_id
                                    }
                                else:
                                    details[DetailKey.REQUIRES_DATA.value] = {
                                        "type": "urls_from_visit" if self._is_url_from_visit(graph, source_node_id) else "urls_from_search",
                                        "source_node_id": source_node_id
                                    }
                            self._logger.info(f"[EXPANSION] Visit candidate requires data from node {source_node_id}: {extracted_url[:60]}...")
                        self._logger.info(f"[EXPANSION] Proactively extracted URL for visit candidate '{title[:50]}...': {extracted_url[:60]}...")
                    else:
                        self._logger.warning(f"[EXPANSION] Visit candidate missing URL: title='{title[:60]}...', details keys: {list(details.keys())}")
            
            if action == IdeaActionType.SEARCH.value:
                details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_search"}
            
            cleaned.append(
                {
                    "title": str(title),
                    "details": details,
                    "score": candidate.get("score"),
                }
            )
        return cleaned, dict(meta)
    
    def _create_fallback_candidate(self, node: "IdeaNode", graph: Optional["IdeaDag"] = None) -> Optional[Dict[str, Any]]:
        import re
        title = node.title or ""
        mandate = node.details.get("mandate") or ""
        text_to_search = f"{title} {mandate}"
        
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text_to_search)
        
        if urls:
            url = urls[0]
            self._logger.info(f"[EXPANSION] Fallback: Creating visit candidate for URL found in mandate: {url[:60]}...")
            return {
                "title": f"Visit {url}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.URL.value: url,
                    "optional_url": url,
                    DetailKey.JUSTIFICATION.value: "Fallback candidate: URL extracted from mandate",
                },
                "score": None,
            }
        
        text_lower = text_to_search.lower()
        if any(keyword in text_lower for keyword in ["visit", "go to", "fetch", "open", "navigate"]):
            if urls:
                url = urls[0]
                return {
                    "title": f"Visit {url}",
                    "details": {
                        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                        DetailKey.URL.value: url,
                        "optional_url": url,
                        DetailKey.JUSTIFICATION.value: "Fallback candidate: Visit action inferred from mandate",
                    },
                    "score": None,
                }
        
        if any(keyword in text_lower for keyword in ["search", "find", "look for", "query"]):
            query = title[:100] if title else "Search"
            return {
                "title": f"Search: {query}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.QUERY.value: query,
                    DetailKey.JUSTIFICATION.value: "Fallback candidate: Search action inferred from mandate",
                },
                "score": None,
            }
        
        self._logger.warning(f"[EXPANSION] Fallback: Creating generic think node (no URL or search query found)")
        return {
            "title": "Analyze and plan next steps",
            "details": {
                DetailKey.ACTION.value: IdeaActionType.THINK.value,
                DetailKey.JUSTIFICATION.value: "Fallback candidate: Generic think node",
            },
            "score": None,
        }