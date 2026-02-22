from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import EvaluationPolicy, DetailKey


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


class LlmEvaluationPolicy(EvaluationPolicy):
    """
    LLM-driven evaluation policy for scoring nodes.
    :param io: AgentIO instance used for LLM calls.
    :param settings: Settings dictionary.
    :param model_name: Optional model override.
    :returns: LlmEvaluationPolicy instance.
    """
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        super().__init__(settings=settings)
        self.io = io
        self.model_name = model_name

    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        """
        Score a node based on graph context using an LLM.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Score value.
        """
        node = graph.get_node(node_id)
        if not node:
            return 0.0
        messages = self._build_messages(graph, node)
        model_name = self.model_name or self.settings.get("evaluation_model")
        json_schema = self.settings.get("evaluation_json_schema")
        reasoning_effort = self.settings.get("reasoning_effort", "high")
        text_verbosity = self.settings.get("text_verbosity", "medium")
        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=float(self.settings.get("evaluation_temperature", 0.2)),
            max_tokens=self.settings.get("evaluation_max_tokens") if self.settings.get("evaluation_max_tokens") is not None else None,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        try:
            self._logger.debug(f"[EVALUATION] Calling LLM for node {node_id} with model={model_name}")
            content = await self.io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
                timeout_seconds=self.settings.get("llm_timeout_seconds"),
            )
            self._logger.debug(f"[EVALUATION] LLM response: {content[:200] if content else 'None'}...")
            score, rationale = self._parse_score(content)
            self._logger.debug(f"[EVALUATION] Node {node_id} scored: {score}")
            graph.evaluate(node_id, score)
            node.details[DetailKey.EVALUATION.value] = {"score": score, "rationale": rationale}
            return score
        except Exception as exc:
            self._logger.error(f"[EVALUATION] Exception during evaluation: {exc}", exc_info=True)
            node.details[DetailKey.EVALUATION.value] = {"error": str(exc)}
            return 0.0

    def _build_messages(self, graph: IdeaDag, node: IdeaNode) -> List[Dict[str, str]]:
        """
        Build evaluation prompt messages.
        Includes parent goal to understand what the previous node aimed to solve.
        :param graph: IdeaDag instance.
        :param node: IdeaNode to evaluate.
        :returns: Message list.
        """
        from agent.app.idea_policies.base import DetailKey
        
        max_nodes = int(self.settings.get("evaluation_max_context_nodes", 5))
        max_detail_chars = int(self.settings.get("evaluation_max_detail_chars", 2000))
        path = graph.path_to_root(node.node_id)
        path = path[:max_nodes]
        serialized = []
        parent_goal = None
        
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
            
            # Extract parent goal from parent node
            if entry.node_id == node.parent_id:
                parent_goal = entry.details.get(DetailKey.PARENT_GOAL.value) or entry.title
        
        path_json = json.dumps(serialized, ensure_ascii=True)
        parent_goal_text = parent_goal or "Not specified"
        
        system_template = self.settings.get("evaluation_system_prompt")
        user_template = self.settings.get("evaluation_user_prompt")
        system = system_template.format() if system_template else (
            "You are an evaluation function. Score the candidate node based on the path context. "
            "Return JSON with keys: score (0-1 float) and rationale (short string)."
        )
        user = user_template.format(
            path_json=path_json,
            candidate_id=node.node_id,
            candidate_title=node.title,
            parent_goal=parent_goal_text,
        ) if user_template else json.dumps(
            {
                "path": serialized,
                "candidate_id": node.node_id,
                "candidate_title": node.title,
                "parent_goal": parent_goal_text,
            },
            ensure_ascii=True,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse_score(self, content: Optional[str]) -> tuple[float, str]:
        """
        Parse score and rationale from LLM output.
        :param content: LLM response content.
        :returns: Tuple of score and rationale.
        """
        if not content:
            return 0.0, "empty_response"
        try:
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            rationale = str(data.get("rationale", ""))
            return self._clamp(score), rationale
        except Exception:
            match = re.search(r"([0-9]*\.?[0-9]+)", content)
            score = float(match.group(1)) if match else 0.0
            return self._clamp(score), content.strip()[:200]

    @staticmethod
    def _clamp(score: float) -> float:
        """
        Clamp score to [0,1].
        :param score: Raw score.
        :returns: Clamped score.
        """
        return max(0.0, min(1.0, score))


class LlmBatchEvaluationPolicy(EvaluationPolicy):
    """
    LLM-driven batch evaluation for multiple candidates in one call.
    :param io: AgentIO instance used for LLM calls.
    :param settings: Settings dictionary.
    :param model_name: Optional model override.
    :returns: LlmBatchEvaluationPolicy instance.
    """
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        super().__init__(settings=settings)
        self.io = io
        self.model_name = model_name
        self._logger = logging.getLogger(self.__class__.__name__)

    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        """
        Fallback to single-node evaluation to satisfy interface.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Score value.
        """
        policy = LlmEvaluationPolicy(self.io, settings=self.settings, model_name=self.model_name)
        return await policy.evaluate(graph, node_id)

    async def evaluate_batch(self, graph: IdeaDag, parent_id: str, candidate_ids: List[str]) -> Dict[str, float]:
        """
        Score multiple candidates in one LLM call.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node identifier.
        :param candidate_ids: Candidate node ids.
        :returns: Mapping of node_id to score.
        """
        parent = graph.get_node(parent_id)
        if not parent:
            return {}
        max_candidates = int(self.settings.get("evaluation_batch_max_candidates", 5))
        candidate_ids = candidate_ids[:max_candidates]
        messages, candidate_id_map = self._build_messages(graph, parent, candidate_ids)
        model_name = self.model_name or self.settings.get("evaluation_model")
        json_schema = self.settings.get("evaluation_batch_json_schema")
        reasoning_effort = self.settings.get("reasoning_effort", "high")
        text_verbosity = self.settings.get("text_verbosity", "medium")
        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=float(self.settings.get("evaluation_temperature", 0.2)),
            max_tokens=self.settings.get("evaluation_max_tokens") if self.settings.get("evaluation_max_tokens") is not None else None,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        try:
            self._logger.debug(f"[EVALUATION_BATCH] Calling LLM for {len(candidate_ids)} candidates with model={model_name}")
            content = await self.io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self.settings.get("fallback_model"),
                timeout_seconds=self.settings.get("llm_timeout_seconds"),
            )
            if content:
                self._logger.debug(f"[EVALUATION_BATCH] Full LLM response: {content}")
            else:
                self._logger.warning(f"[EVALUATION_BATCH] LLM returned empty content")
            scores = self._parse_scores(content, candidate_id_map)
            self._logger.debug(f"[EVALUATION_BATCH] Parsed {len(scores)} scores")
            if not scores and content:
                self._logger.warning(f"[EVALUATION_BATCH] Failed to parse scores from response. Content length: {len(content)}, Content: {content[:1000]}")
            for node_id, score in scores.items():
                node = graph.get_node(node_id)
                if not node:
                    self._logger.warning(f"[EVALUATION_BATCH] Skipping unknown node_id: {node_id} (not in graph)")
                    continue
                try:
                    graph.evaluate(node_id, score)
                    node.details[DetailKey.EVALUATION.value] = {"score": score}
                except ValueError as e:
                    self._logger.warning(f"[EVALUATION_BATCH] Failed to evaluate node {node_id}: {e}")
            return scores
        except Exception as exc:
            self._logger.error(f"[EVALUATION_BATCH] Exception during batch evaluation: {exc}", exc_info=True)
            for node_id in candidate_ids:
                node = graph.get_node(node_id)
                if node:
                    node.details[DetailKey.EVALUATION.value] = {"error": str(exc)}
            return {}

    def _build_messages(self, graph: IdeaDag, parent: IdeaNode, candidate_ids: List[str]) -> tuple[List[Dict[str, str]], Dict[str, str]]:
        """
        Build batch evaluation prompt messages.
        :param graph: IdeaDag instance.
        :param parent: Parent node.
        :param candidate_ids: Candidate node ids.
        :returns: Tuple of (message list, candidate_id_map).
        """
        max_nodes = int(self.settings.get("evaluation_max_context_nodes", 5))
        max_detail_chars = int(self.settings.get("evaluation_max_detail_chars", 2000))
        path = graph.path_to_root(parent.node_id)
        path = path[:max_nodes]
        path_serialized = []
        for entry in path:
            details_text = _safe_serialize_details(entry.details)
            if len(details_text) > max_detail_chars:
                details_text = details_text[:max_detail_chars]
            path_serialized.append(
                {
                    "node_id": entry.node_id,
                    "title": entry.title,
                    "status": entry.status.value,
                    "score": entry.score,
                    "details": details_text,
                }
            )
        candidates = []
        candidate_id_map = {}
        for idx, candidate_id in enumerate(candidate_ids, start=1):
            node = graph.get_node(candidate_id)
            if not node:
                continue
            simple_id = str(idx)
            candidate_id_map[simple_id] = candidate_id
            details_text = _safe_serialize_details(node.details)
            if len(details_text) > max_detail_chars:
                details_text = details_text[:max_detail_chars]
            candidates.append(
                {
                    "id": simple_id,
                    "title": node.title,
                    "details": details_text,
                }
            )
        from agent.app.idea_policies.base import DetailKey
        
        path_json = json.dumps(path_serialized, ensure_ascii=True)
        candidates_json = json.dumps(candidates, ensure_ascii=True)
        parent_goal = parent.details.get(DetailKey.PARENT_GOAL.value) or parent.title
        
        system_template = self.settings.get("evaluation_batch_system_prompt")
        user_template = self.settings.get("evaluation_batch_user_prompt")
        system = system_template.format() if system_template else (
            "You are an evaluation function. Score each candidate node based on the path context. "
            "Return JSON with key 'scores' as a list of objects: "
            "{id, score} where id is the candidate's simple identifier (1, 2, 3, etc.) and score is a 0-1 float."
        )
        user = user_template.format(
            path_json=path_json,
            parent_id=parent.node_id,
            parent_goal=parent_goal,
            candidates_json=candidates_json,
        ) if user_template else json.dumps(
            {
                "path": path_serialized,
                "parent_id": parent.node_id,
                "parent_goal": parent_goal,
                "candidates": candidates,
            },
            ensure_ascii=True,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages, candidate_id_map

    def _clamp(self, value: float) -> float:
        """
        Clamp score to [0.0, 1.0] range.
        :param value: Score value.
        :returns: Clamped score.
        """
        return max(0.0, min(1.0, float(value)))

    def _parse_scores(self, content: Optional[str], candidate_id_map: Dict[str, str] = None) -> Dict[str, float]:
        """
        Parse score list from LLM output, converting simple IDs to actual node IDs.
        :param content: LLM response content.
        :param candidate_id_map: Mapping from simple ID (e.g., "1") to actual node_id.
        :returns: Mapping of node_id to score.
        """
        if not content:
            return {}
        if candidate_id_map is None:
            candidate_id_map = {}
        try:
            data = json.loads(content)
            scores = data.get("scores", [])
            output: Dict[str, float] = {}
            for item in scores:
                if not isinstance(item, dict):
                    self._logger.warning(f"[EVALUATION_BATCH] Skipping non-dict item in scores: {item}")
                    continue
                simple_id = item.get("id") or item.get("node_id")
                if not simple_id:
                    self._logger.warning(f"[EVALUATION_BATCH] Missing id/node_id in item: {item}")
                    continue
                simple_id = str(simple_id)
                node_id = candidate_id_map.get(simple_id, simple_id)
                if simple_id in candidate_id_map:
                    self._logger.debug(f"[EVALUATION_BATCH] Mapped simple ID '{simple_id}' to node_id '{node_id}'")
                else:
                    self._logger.debug(f"[EVALUATION_BATCH] Using '{simple_id}' as node_id directly (not in map)")
                score_val = item.get("score")
                if score_val is None:
                    self._logger.warning(f"[EVALUATION_BATCH] Missing score in item: {item}")
                    continue
                output[node_id] = self._clamp(float(score_val))
            return output
        except json.JSONDecodeError as e:
            self._logger.error(f"[EVALUATION_BATCH] JSON decode error: {e}, content: {content[:500]}")
            return {}
        except Exception as e:
            self._logger.error(f"[EVALUATION_BATCH] Parse error: {e}, content: {content[:500]}", exc_info=True)
            return {}