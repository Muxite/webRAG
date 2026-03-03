from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import EvaluationPolicy, DetailKey


def _safe_serialize_details(details: Dict[str, Any]) -> str:
    try:
        return json.dumps(details, ensure_ascii=True, default=str)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=True)


class EvaluationWeights:

    def __init__(
        self,
        no_action_result_base_score: float = 0.1,
        no_action_result_score_cap: float = 0.2,
        search_weight: float = 1.0,
        visit_weight: float = 1.0,
        think_weight: float = 1.0,
        save_weight: float = 1.0,
        default_weight: float = 1.0,
    ):
        self.no_action_result_base_score = float(no_action_result_base_score)
        self.no_action_result_score_cap = float(no_action_result_score_cap)
        self.search_weight = float(search_weight)
        self.visit_weight = float(visit_weight)
        self.think_weight = float(think_weight)
        self.save_weight = float(save_weight)
        self.default_weight = float(default_weight)

    @classmethod
    def from_settings(cls, settings: Optional[Dict[str, Any]]) -> "EvaluationWeights":
        settings = settings or {}
        return cls(
            no_action_result_base_score=settings.get("evaluation_no_action_result_base_score", 0.1),
            no_action_result_score_cap=settings.get("evaluation_no_action_result_score_cap", 0.2),
            search_weight=settings.get("evaluation_weight_search", 1.0),
            visit_weight=settings.get("evaluation_weight_visit", 1.0),
            think_weight=settings.get("evaluation_weight_think", 1.0),
            save_weight=settings.get("evaluation_weight_save", 1.0),
            default_weight=settings.get("evaluation_weight_default", 1.0),
        )

    def apply_action_weight(self, action: Optional[str], score: float) -> float:
        if not action:
            return score * self.default_weight
        action_lower = str(action).lower()
        if action_lower == "search":
            return score * self.search_weight
        if action_lower == "visit":
            return score * self.visit_weight
        if action_lower == "think":
            return score * self.think_weight
        if action_lower == "save":
            return score * self.save_weight
        return score * self.default_weight


class LlmEvaluationPolicy(EvaluationPolicy):
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        super().__init__(settings=settings)
        self.io = io
        self.model_name = model_name
        self.weights = EvaluationWeights.from_settings(settings)

    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        node = graph.get_node(node_id)
        if not node:
            return 0.0
        
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor, ActionResultKey
        from agent.app.idea_policies.base import IdeaActionType
        
        action = NodeDetailsExtractor.get_action(node.details)
        has_action = action and not NodeDetailsExtractor.is_merge_action(node.details)
        has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
        
        if has_action and not has_result:
            action_result = node.details.get(DetailKey.ACTION_RESULT.value)
            if action_result is None:
                self._logger.warning(f"[EVALUATION] Node {node_id} has action '{action}' but no result - penalizing score")
                penalty_score = float(self.weights.no_action_result_base_score)
                graph.evaluate(node_id, penalty_score)
                node.details[DetailKey.EVALUATION.value] = {
                    "score": penalty_score,
                    "rationale": "Action not executed - missing action_result",
                    "penalty": "no_action_result",
                }
                return penalty_score
        
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

            if has_action and not has_result:
                score = min(score, float(self.weights.no_action_result_score_cap))
                rationale = f"{rationale} [PENALTY: action not executed]"

            weighted_score = self.weights.apply_action_weight(action, score)
            weighted_score = self._clamp(weighted_score)

            self._logger.debug(f"[EVALUATION] Node {node_id} scored: {weighted_score}")
            graph.evaluate(node_id, weighted_score)
            node.details[DetailKey.EVALUATION.value] = {"score": weighted_score, "rationale": rationale}
            return weighted_score
        except Exception as exc:
            self._logger.error(f"[EVALUATION] Exception during evaluation: {exc}", exc_info=True)
            node.details[DetailKey.EVALUATION.value] = {"error": str(exc)}
            return 0.0

    def _build_messages(self, graph: IdeaDag, node: IdeaNode) -> List[Dict[str, str]]:
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
        planning_addendum = str(
            self.settings.get(
                "evaluation_planning_addendum",
                "Reward verifiable evidence collection and concrete output fields; penalize vague plans.",
            )
        ).strip()
        if planning_addendum:
            system = f"{system}\n\n{planning_addendum}" if system else planning_addendum
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
        from agent.app.idea_policies.action_constants import PromptBuilder
        return PromptBuilder.build_messages(system_content=system, user_content=user)

    def _parse_score(self, content: Optional[str]) -> tuple[float, str]:
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
        return max(0.0, min(1.0, score))


class LlmBatchEvaluationPolicy(EvaluationPolicy):
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        super().__init__(settings=settings)
        self.io = io
        self.model_name = model_name
        self._logger = logging.getLogger(self.__class__.__name__)
        self.weights = EvaluationWeights.from_settings(settings)

    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        policy = LlmEvaluationPolicy(self.io, settings=self.settings, model_name=self.model_name)
        return await policy.evaluate(graph, node_id)

    async def evaluate_batch(self, graph: IdeaDag, parent_id: str, candidate_ids: List[str]) -> Dict[str, float]:
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
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            for node_id in candidate_ids:
                node = graph.get_node(node_id)
                if not node:
                    continue

                action = NodeDetailsExtractor.get_action(node.details)
                has_action = action and not NodeDetailsExtractor.is_merge_action(node.details)
                has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None

                if has_action and not has_result:
                    if node_id in scores:
                        scores[node_id] = min(scores[node_id], float(self.weights.no_action_result_score_cap))
                    else:
                        scores[node_id] = float(self.weights.no_action_result_base_score)
                        self._logger.warning(f"[EVALUATION_BATCH] Node {node_id} has action '{action}' but no result - penalizing to base score")

                if node_id in scores:
                    scores[node_id] = self.weights.apply_action_weight(action, scores[node_id])
                    scores[node_id] = self._clamp(scores[node_id])

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
        planning_addendum = str(
            self.settings.get(
                "evaluation_planning_addendum",
                "Reward verifiable evidence collection and concrete output fields; penalize vague plans.",
            )
        ).strip()
        if planning_addendum:
            system = f"{system}\n\n{planning_addendum}" if system else planning_addendum
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
        return max(0.0, min(1.0, float(value)))

    def _parse_scores(self, content: Optional[str], candidate_id_map: Dict[str, str] = None) -> Dict[str, float]:
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