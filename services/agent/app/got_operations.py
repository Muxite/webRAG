from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode
    from agent.app.agent_io import AgentIO
    from agent.app.idea_memory import MemoryManager

from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig

_logger = logging.getLogger(__name__)


class GoTOperations:

    def __init__(self, settings: Dict[str, Any], io: AgentIO, memory_manager: Optional[MemoryManager] = None):
        self.settings = settings
        self._cfg = IdeaConfig.from_settings(settings)
        self.io = io
        self.memory_manager = memory_manager
        self._dead_end_count = 0

    async def embed_thought(
        self,
        node_id: str,
        title: str,
        goal: str,
        action_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ) -> bool:
        if not self._cfg.got.embed_on_create:
            return False
        if not self.memory_manager:
            return False

        content_parts = [f"Thought: {title}"]
        if goal:
            content_parts.append(f"Goal: {goal}")
        if action_type:
            content_parts.append(f"Action: {action_type}")
        content = "\n".join(content_parts)

        metadata = {
            "memory_type": "internal_thought",
            "step_type": "thought_node",
            "depth": str(depth),
        }
        if parent_id:
            metadata["parent_id"] = parent_id

        return await self.memory_manager.write_memory(
            content=content,
            node_id=node_id,
            node_title=title,
            action_type=action_type,
            metadata=metadata,
            memory_type="internal_thought",
        )

    async def embed_children(self, graph: IdeaDag, parent_id: str) -> int:
        if not self._cfg.got.embed_on_create:
            return 0
        parent = graph.get_node(parent_id)
        if not parent:
            return 0

        count = 0
        for child_id in parent.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            action = child.details.get(DetailKey.ACTION.value)
            goal = (
                child.details.get(DetailKey.GOAL.value)
                or child.details.get(DetailKey.ORIGINAL_GOAL.value)
                or child.title
            )
            depth = graph.depth(child_id)
            ok = await self.embed_thought(
                node_id=child_id,
                title=child.title,
                goal=goal,
                action_type=action,
                parent_id=parent_id,
                depth=depth,
            )
            if ok:
                count += 1

        if count > 0:
            _logger.debug(f"[GoT:EMBED] Embedded {count} child thoughts for parent {parent_id}")
        return count

    async def try_improve_node(
        self,
        graph: IdeaDag,
        node_id: str,
        model_name: Optional[str] = None,
    ) -> Optional[float]:
        if not self._cfg.got.improve_enabled:
            return None

        node = graph.get_node(node_id)
        if not node or node.score is None:
            return None

        threshold = self._cfg.got.improve_score_threshold
        if node.score >= threshold:
            return None

        max_iters = self._cfg.got.improve_max_iterations
        iteration_count = int(node.details.get("_got_improve_iterations", 0))
        if iteration_count >= max_iters:
            _logger.debug(f"[GoT:IMPROVE] Node {node_id} already improved {iteration_count} times, skipping")
            return None

        evaluation = node.details.get(DetailKey.EVALUATION.value) or {}
        rationale = evaluation.get("rationale", "Low score")

        memories_text = ""
        if self.memory_manager:
            memories = await self.memory_manager.retrieve_relevant_memories(
                query=f"{node.title} {rationale}",
                n_results=5,
            )
            if memories:
                memories_text = self.memory_manager.format_memories_for_llm(memories, max_chars=4000)

        system_prompt = self.settings.get(
            "got_improve_system_prompt",
            "You are a self-refinement function. Given a thought and feedback, produce an improved version. "
            "Return JSON: {\"improved_title\": string, \"improved_details\": object, \"refinement_rationale\": string}.",
        )
        user_content = json.dumps({
            "current_title": node.title,
            "current_score": node.score,
            "scorer_feedback": rationale,
            "action": node.details.get(DetailKey.ACTION.value),
            "goal": node.details.get(DetailKey.GOAL.value) or node.title,
            "relevant_memories": memories_text,
        }, ensure_ascii=True)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        improve_model = model_name or self._cfg.evaluation.model or None
        temperature = self._cfg.got.improve_temperature

        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=improve_model,
            temperature=temperature,
        )

        try:
            response = await self.io.query_llm_with_fallback(
                payload,
                model_name=improve_model,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=self._cfg.timeouts.llm,
            )
            if not response:
                return None

            data = json.loads(response)
            improved_title = data.get("improved_title") or node.title
            improved_details = data.get("improved_details") or {}
            refinement_rationale = data.get("refinement_rationale", "")

            old_title = node.title
            node.title = str(improved_title)

            for key, value in improved_details.items():
                if key not in (
                    DetailKey.ACTION_RESULT.value,
                    DetailKey.MERGED_RESULTS.value,
                    DetailKey.EVALUATION.value,
                ):
                    node.details[key] = value

            node.details["_got_improve_iterations"] = iteration_count + 1
            node.details["_got_last_refinement"] = refinement_rationale
            node.details["_got_pre_improve_score"] = node.score

            node.score = None
            node.details.pop(DetailKey.EVALUATION.value, None)

            _logger.info(
                f"[GoT:IMPROVE] Refined node {node_id}: '{old_title[:40]}' -> '{node.title[:40]}' "
                f"(was {node.details.get('_got_pre_improve_score'):.2f}, iteration {iteration_count + 1})"
            )
            return node.details.get("_got_pre_improve_score")

        except Exception as exc:
            _logger.warning(f"[GoT:IMPROVE] Failed to improve node {node_id}: {exc}")
            return None

    def _adaptive_dedup_threshold(self, graph: IdeaDag) -> float:
        """
        Pick a similarity cutoff based on graph density. Sparse graphs tolerate
        more variety (lower cutoff → less aggressive dedup); dense ones tighten
        the cutoff to keep growth in check. Clamped to [0.75, 0.92].

        :param graph: Current DAG.
        :returns: Threshold in [0.75, 0.92].
        """
        if not self._cfg.got.adaptive_policies:
            return self._cfg.got.dedup_similarity_threshold
        floor = self._cfg.got.dedup_threshold_min
        ceil = self._cfg.got.dedup_threshold_max
        # Use sibling fanout as a density proxy: max children across non-leaf nodes.
        fanout = 0
        for n in graph.iter_depth_first():
            if n.children:
                fanout = max(fanout, len(n.children))
        # 0 siblings → loose (floor); >=8 siblings → tight (ceil); linear in between.
        ratio = min(1.0, fanout / 8.0)
        return round(floor + ratio * (ceil - floor), 3)

    async def is_duplicate_thought(
        self,
        candidate_title: str,
        candidate_goal: str,
        graph: IdeaDag,
    ) -> Tuple[bool, Optional[str]]:
        if not self._cfg.got.dedup_enabled:
            return False, None
        if not self.memory_manager:
            return False, None

        threshold = self._adaptive_dedup_threshold(graph)
        n_query = self._cfg.got.dedup_max_query

        query = f"{candidate_title} {candidate_goal}"
        try:
            memories = await self.memory_manager.retrieve_relevant_memories(
                query=query,
                n_results=n_query,
                memory_type="internal_thought",
            )
            if not memories:
                return False, None

            for mem in memories:
                distance = mem.get("distance", 1.0)
                if isinstance(distance, (int, float)):
                    similarity = 1.0 - distance
                    if similarity >= threshold:
                        existing_node_id = (mem.get("metadata") or {}).get("node_id", "unknown")
                        _logger.info(
                            f"[GoT:DEDUP] Candidate '{candidate_title[:40]}' is duplicate of node {existing_node_id} "
                            f"(similarity={similarity:.3f} >= {threshold})"
                        )
                        return True, existing_node_id

        except Exception as exc:
            _logger.warning(f"[GoT:DEDUP] Dedup check failed: {exc}")

        return False, None

    async def filter_duplicate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        graph: IdeaDag,
    ) -> List[Dict[str, Any]]:
        if not self._cfg.got.dedup_enabled:
            return candidates
        if not self.memory_manager:
            return candidates

        filtered = []
        dedup_count = 0

        for candidate in candidates:
            title = candidate.get("title", "")
            details = candidate.get("details", {})
            goal = (
                details.get(DetailKey.GOAL.value)
                or details.get(DetailKey.ORIGINAL_GOAL.value)
                or title
            )

            is_dup, existing_id = await self.is_duplicate_thought(title, goal, graph)
            if is_dup:
                dedup_count += 1
                continue
            filtered.append(candidate)

        if dedup_count > 0:
            _logger.info(f"[GoT:DEDUP] Filtered {dedup_count} duplicate candidates out of {len(candidates)}")

        return filtered if filtered else candidates[:1]

    def compute_dynamic_beam_width(self, graph: IdeaDag) -> int:
        if not self._cfg.got.dynamic_beam_enabled:
            return self._cfg.engine.max_branching

        beam_min = self._cfg.got.beam_min
        beam_max = self._cfg.got.beam_max

        scores: List[float] = []
        for node in graph.iter_depth_first():
            if node.score is not None and node.parent_id is not None:
                scores.append(float(node.score))

        if not scores:
            return beam_max

        adaptive = self._cfg.got.adaptive_policies
        if adaptive and len(scores) >= 4:
            # Beam widens when scores are spread (uncertain) and narrows when they
            # cluster (converged). Use p25/p75 spread relative to a target band.
            ordered = sorted(scores)
            p25 = ordered[max(0, int(len(ordered) * 0.25) - 1)]
            p75 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))]
            spread = max(0.0, p75 - p25)  # 0..1 in practice
            target_spread = self._cfg.got.beam_target_spread
            ratio = min(1.0, spread / target_spread) if target_spread > 0 else 0.0
            beam = beam_min + int(round(ratio * (beam_max - beam_min)))
            beam = max(beam_min, min(beam_max, beam))
            _logger.debug(
                f"[GoT:BEAM] adaptive p25={p25:.3f} p75={p75:.3f} spread={spread:.3f} -> beam={beam}"
            )
            return beam

        score_high = self._cfg.got.beam_score_high
        score_low = self._cfg.got.beam_score_low
        avg_score = sum(scores) / len(scores)
        if avg_score >= score_high:
            beam = beam_min
        elif avg_score <= score_low:
            beam = beam_max
        else:
            ratio = (score_high - avg_score) / (score_high - score_low)
            beam = beam_min + int(ratio * (beam_max - beam_min))
        beam = max(beam_min, min(beam_max, beam))
        _logger.debug(f"[GoT:BEAM] legacy avg_score={avg_score:.3f} -> beam_width={beam}")
        return beam

    def identify_prune_candidates(self, graph: IdeaDag) -> List[str]:
        if not self._cfg.got.prune_enabled:
            return []

        min_nodes = self._cfg.got.prune_min_nodes_before_prune
        if graph.node_count() < min_nodes:
            return []

        scored: List[float] = []
        for node in graph.iter_depth_first():
            if node.score is not None and node.parent_id is not None:
                scored.append(float(node.score))

        adaptive = self._cfg.got.adaptive_policies
        if adaptive and len(scored) >= 5:
            mean = sum(scored) / len(scored)
            variance = sum((s - mean) ** 2 for s in scored) / len(scored)
            stddev = variance ** 0.5
            stddev_factor = self._cfg.got.prune_stddev_factor
            threshold = max(0.0, mean - stddev_factor * stddev)
        else:
            threshold = self._cfg.got.prune_score_threshold

        prune_ids = []
        for node in graph.iter_depth_first():
            if node.node_id == graph.root_id():
                continue
            if node.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                continue
            if node.score is not None and node.score < threshold:
                has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
                if not has_result:
                    prune_ids.append(node.node_id)

        if prune_ids:
            _logger.info(
                f"[GoT:PRUNE] Identified {len(prune_ids)} low-score nodes for pruning "
                f"(threshold={threshold:.3f}, adaptive={adaptive})"
            )
        return prune_ids

    def prune_nodes(self, graph: IdeaDag, node_ids: List[str]) -> int:
        pruned = 0
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if not node:
                continue
            if node.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED):
                continue
            node.status = IdeaNodeStatus.SKIPPED
            node.details["_got_pruned"] = True
            node.details["_got_prune_reason"] = f"Score {node.score} below threshold"
            pruned += 1

        if pruned > 0:
            _logger.info(f"[GoT:PRUNE] Pruned {pruned} low-score nodes")
        return pruned

    def should_backtrack(self, graph: IdeaDag, current_id: str) -> bool:
        if not self._cfg.got.backtrack_enabled:
            return False

        dead_end_limit = self._cfg.got.backtrack_dead_end_threshold
        low_score = self._cfg.got.backtrack_low_score_threshold

        node = graph.get_node(current_id)
        if not node:
            return False

        consecutive_low = 0
        path = graph.path_to_root(current_id)
        for path_node in path:
            if path_node.score is not None and path_node.score < low_score:
                consecutive_low += 1
            else:
                break

        if consecutive_low >= dead_end_limit:
            _logger.info(
                f"[GoT:BACKTRACK] Dead-end detected: {consecutive_low} consecutive low-score nodes "
                f"at node {current_id} (threshold={dead_end_limit}, low_score<{low_score})"
            )
            self._dead_end_count += 1
            return True

        return False

    def find_backtrack_target(self, graph: IdeaDag, current_id: str) -> Optional[str]:
        low_score = self._cfg.got.backtrack_low_score_threshold
        path = graph.path_to_root(current_id)
        for path_node in path:
            if path_node.node_id == graph.root_id():
                continue
            if path_node.score is not None and path_node.score >= low_score:
                if path_node.parent_id:
                    _logger.info(f"[GoT:BACKTRACK] Backtracking from {current_id} to {path_node.parent_id}")
                    return path_node.parent_id

        return graph.root_id()

    async def hybrid_retrieve(
        self,
        graph: IdeaDag,
        node_id: str,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.memory_manager:
            return []

        vector_results = await self.memory_manager.retrieve_relevant_memories(
            query=query,
            n_results=n_results,
        )

        node = graph.get_node(node_id)
        if not node:
            return vector_results

        path = graph.path_to_root(node_id)
        graph_context = []
        for path_node in path[:5]:
            ar = path_node.details.get(DetailKey.ACTION_RESULT.value)
            if ar and isinstance(ar, dict):
                from agent.app.idea_policies.action_constants import ActionResultExtractor
                if ActionResultExtractor.is_success(ar):
                    content = ar.get("content", "") or ""
                    if content and len(content) > 50:
                        graph_context.append({
                            "content": content[:500],
                            "metadata": {
                                "node_id": path_node.node_id,
                                "node_title": path_node.title,
                                "source": "graph_path",
                            },
                            "distance": 0.0,
                        })

        seen_ids = {m.get("id") for m in vector_results if m.get("id")}
        combined = list(vector_results)
        for gc in graph_context:
            gc_id = gc.get("metadata", {}).get("node_id")
            if gc_id not in seen_ids:
                combined.append(gc)
                seen_ids.add(gc_id)

        return combined[:n_results + len(graph_context)]

    def get_model_for_operation(self, operation: str, default_model: Optional[str] = None) -> Optional[str]:
        if not self._cfg.got.telemetry_routing_enabled:
            return default_model

        if operation in ("score", "evaluate", "evaluation"):
            override = self._cfg.got.telemetry_routing_score_model
            if override:
                return override
            return default_model

        if operation in ("generate", "expand", "expansion"):
            override = self._cfg.got.telemetry_routing_generate_model
            if override:
                return override
            return self._select_cheaper_model(default_model)

        return default_model

    @staticmethod
    def _select_cheaper_model(current_model: Optional[str]) -> Optional[str]:
        from agent.app.model_costs import MODEL_PRICING

        if not current_model or current_model not in MODEL_PRICING:
            return current_model

        current_output_cost = MODEL_PRICING[current_model]["output_per_million"]
        cheaper = None
        cheaper_cost = current_output_cost

        for model_name, pricing in MODEL_PRICING.items():
            if pricing["output_per_million"] < cheaper_cost:
                cheaper = model_name
                cheaper_cost = pricing["output_per_million"]

        if cheaper and cheaper != current_model:
            _logger.debug(f"[GoT:ROUTING] Downgraded {current_model} -> {cheaper} for generate operation")
            return cheaper
        return current_model

    @property
    def dead_end_count(self) -> int:
        return self._dead_end_count
