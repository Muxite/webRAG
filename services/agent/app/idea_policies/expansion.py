from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import ExpansionPolicy, DetailKey, IdeaActionType
from agent.app.idea_dag_settings import load_idea_dag_settings


def _safe_serialize_details(details: Dict[str, Any]) -> str:
    """
    Safely serialize node details to JSON.
    :param details: Details dictionary.
    :returns: JSON string.
    """
    try:
        return json.dumps(details, ensure_ascii=True, default=str)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=True)


class LlmExpansionPolicy(ExpansionPolicy):
    """
    LLM-driven expansion policy for generating candidate nodes.
    :param io: AgentIO instance.
    :param settings: Settings dictionary.
    :param model_name: Optional model override.
    :returns: LlmExpansionPolicy instance.
    """
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        default_settings = load_idea_dag_settings()
        merged_settings = {**default_settings, **(settings or {})}
        super().__init__(settings=merged_settings)
        self.io = io
        self.model_name = model_name
        self._logger = logging.getLogger(self.__class__.__name__)

    async def expand(self, graph: IdeaDag, node_id: str, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Generate candidate child ideas for a node using an LLM.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: List of idea dicts.
        """
        node = graph.get_node(node_id)
        if not node:
            return []
        messages = self._build_messages(graph, node, memories=memories)
        model_name = self.model_name or self.settings.get("expansion_model")
        json_schema = self.settings.get("expansion_json_schema")
        reasoning_effort = self.settings.get("reasoning_effort", "high")
        text_verbosity = self.settings.get("text_verbosity", "medium")
        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=float(self.settings.get("expansion_temperature", 0.4)),
            max_tokens=self.settings.get("expansion_max_tokens") if self.settings.get("expansion_max_tokens") is not None else None,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        try:
            self._logger.debug(f"[EXPANSION] Calling LLM for node {node_id} with model={model_name}")
            content = await self.io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
                timeout_seconds=self.settings.get("llm_timeout_seconds"),
            )
            candidates, meta = self._parse_candidates(content)
            self._logger.info(f"[EXPANSION] {len(candidates)} candidates, meta={meta.get('execute_all_children', False)}")
            if meta:
                node.details[DetailKey.EXPANSION_META.value] = meta
            return candidates
        except Exception as e:
            self._logger.error(f"[EXPANSION] Exception during expansion: {e}", exc_info=True)
            return []

    def _build_messages(self, graph: IdeaDag, node: IdeaNode, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        """
        Build expansion prompt messages.
        :param graph: IdeaDag instance.
        :param node: IdeaNode to expand.
        :returns: Message list.
        """
        max_nodes = int(self.settings.get("expansion_max_context_nodes", 5))
        max_detail_chars = int(self.settings.get("expansion_max_detail_chars", 2000))
        max_children = int(self.settings.get("max_branching", 5))
        path = graph.path_to_root(node.node_id)
        path = path[:max_nodes]
        serialized = []
        for entry in path:
            details_text = _safe_serialize_details(entry.details)
            if len(details_text) > max_detail_chars:
                details_text = details_text[:max_detail_chars]
            serialized.append(
                {
                    "node_id": entry.node_id,
                    "title": entry.title,
                    "status": entry.status.value,
                    "score": entry.score,
                    "details": details_text,
                }
            )
        allowed = self.settings.get("allowed_actions") or [a.value for a in IdeaActionType]
        allowed_actions = ", ".join(str(item) for item in allowed)
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
            memories_text = temp_mm.format_memories_for_llm(memories, max_chars=1500)
        
        event_log = graph.build_event_log_table(node.node_id, max_events=15)
        event_log_json = json.dumps(event_log) if event_log else json.dumps("No events")
        
        system_template = self.settings.get("expansion_system_prompt")
        user_template = self.settings.get("expansion_user_prompt")
        system = system_template.format(
            allowed_actions=allowed_actions,
            max_children=max_children,
        ) if system_template else ""
        user = user_template.format(
            path_json=path_json,
            parent_id=node.node_id,
            parent_title=node.title,
            blocked_sites=blocked_sites_text,
            errors=errors_text,
            memories=memories_text,
            event_log=event_log_json,
        ) if user_template else json.dumps(
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
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse_candidates(self, content: Optional[str]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse candidate list from LLM output.
        :param content: LLM response content.
        :returns: Candidate list.
        """
        if not content:
            return [], {}
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self._logger.warning(f"[EXPANSION] Failed to parse JSON response: {e}. Content preview: {content[:200]}")
            return [], {}
        except Exception as e:
            self._logger.warning(f"[EXPANSION] Unexpected error parsing response: {e}. Content preview: {content[:200]}")
            return [], {}
        candidates = data.get("candidates", [])
        if not candidates:
            self._logger.warning(f"[EXPANSION] No candidates in response. Data keys: {list(data.keys())}")
        meta = data.get("meta") or {}
        cleaned: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            action = candidate.get("action")
            title = candidate.get("title") or ""
            details = candidate.get("details") or {}
            if action:
                details = dict(details)
                details[DetailKey.ACTION.value] = action
            cleaned.append(
                {
                    "title": str(title),
                    "details": details,
                    "score": candidate.get("score"),
                }
            )
        return cleaned, dict(meta)
