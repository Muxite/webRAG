from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import asyncio
import hashlib
import logging
import re

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.model_tiers import capability_tier, tier_value
from agent.app.agent_io import AgentIO
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_memory import MemoryManager
from agent.app.idea_policies import (
    DetailKey,
    IdeaActionType,
    LlmExpansionPolicy,
    LlmBatchEvaluationPolicy,
    BestScoreSelectionPolicy,
    ScoreThresholdDecompositionPolicy,
    SimpleMergePolicy,
    LeafActionRegistry,
    NodeDetailsExtractor,
)
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_branch_pair import BranchPair, find_branch_pair, get_completion_path
from agent.app.got_operations import GoTOperations
from agent.app.idea_checkpointer import Checkpointer, create_checkpointer_from_env
from agent.app.idea_policies.data_contracts import ContractRegistry, default_contract_registry
from agent.app.idea_policies.post_expansion_hooks import (
    MandateUrlInjectionHook,
    PostExpansionHook,
    default_post_expansion_hooks,
    extract_mandate,
)
from agent.app.idea_policies.mandate_requirements import parse_mandate_requirements
from agent.app.idea_policies.grounding import evaluate_grounding
from agent.app.idea_policies.candidate_coverage import evaluate_candidate_coverage
from agent.app.idea_policies import plan_library as _plan_library
from agent.app.idea_policies import plan_library_search as _plan_search
from agent.app.plan_library import retrieval as _plan_retrieval
from agent.app.testing import contract_log as _contract_log
from agent.app import idea_chunking
from agent.app import idea_visit_dedup
from agent.app import idea_sequencing
from agent.app import idea_node_state


_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')


def _reformulate_multi_entity_query(query: Optional[str]) -> Optional[str]:
    """OR-join a query's quoted phrases when it bundles 2+ distinct entity names.

    Multiple quoted names AND together and rarely appear on one page. OR-joining
    asks for any of them, which per-entity lookup needs.
    :param query: Query string that produced zero results.
    :return: OR-joined reformulation, or None if fewer than two phrases or already OR-joined.
    """
    if not query or " OR " in query:
        return None
    phrases = _QUOTED_PHRASE_RE.findall(query)
    if len(phrases) < 2:
        return None
    remainder = _QUOTED_PHRASE_RE.sub("", query)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    or_joined = " OR ".join(f'"{p}"' for p in phrases)
    return f"{or_joined} {remainder}" if remainder else or_joined


class IdeaDagEngine:
    def __init__(
        self,
        io: AgentIO,
        settings: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        expansion: Optional[LlmExpansionPolicy] = None,
        evaluation: Optional[LlmBatchEvaluationPolicy] = None,
        selection: Optional[BestScoreSelectionPolicy] = None,
        decomposition: Optional[ScoreThresholdDecompositionPolicy] = None,
        merge: Optional[SimpleMergePolicy] = None,
        actions: Optional[LeafActionRegistry] = None,
        contracts: Optional[ContractRegistry] = None,
        post_expansion_hooks: Optional[List[PostExpansionHook]] = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.settings = {**load_idea_dag_settings(), **(settings or {})}
        self._cfg = IdeaConfig.from_settings(self.settings)
        self._apply_tools_config()
        self._patch_allowed_actions()
        self.io = io
        self.model_name = model_name
        self.actions = actions or LeafActionRegistry(settings=self.settings)
        self.expansion = expansion or LlmExpansionPolicy(
            io=io, settings=self.settings, model_name=model_name, actions=self.actions,
        )
        self.evaluation = evaluation or LlmBatchEvaluationPolicy(io=io, settings=self.settings, model_name=model_name)
        self.selection = selection or BestScoreSelectionPolicy(settings=self.settings)
        self.decomposition = decomposition or ScoreThresholdDecompositionPolicy(settings=self.settings)
        self.merge = merge or SimpleMergePolicy(settings=self.settings)
        self._install_configured_tool_packs()
        self.contracts = contracts or default_contract_registry()
        self.post_expansion_hooks: List[PostExpansionHook] = (
            list(post_expansion_hooks) if post_expansion_hooks is not None else default_post_expansion_hooks()
        )
        self._step_index = 0
        self._step_confidences: List[Dict[str, Any]] = []
        self._leaf_completion_count = 0
        self._memory_manager: Optional[MemoryManager] = None
        self._got: Optional[GoTOperations] = None
        # Built lazily on the first plan-library retrieval (see `_plan_library_corpus`), so a
        # flag-off run never reads the template corpus off disk.
        self._plan_library_corpus_cache: Optional[Any] = None
        self._checkpointer: Optional[Checkpointer] = create_checkpointer_from_env()

    def _apply_tools_config(self) -> None:
        """Seed ``allowed_actions`` from the typed tools config (see ``ToolsConfig``).

        The core action menu is declarative now: ``tools_core_actions`` wins when set, otherwise
        ``ToolsConfig`` carries the run's existing ``allowed_actions`` through verbatim. So this
        rewrites the same list on every run that does not configure it. Byte-identical, and a
        caller's narrowed menu (``debug_runner``, tests) is still respected.
        """
        self.settings["allowed_actions"] = list(self._cfg.tools.core_actions)

    def _install_configured_tool_packs(self) -> None:
        """Install opt-in tool packs armed in ToolsConfig.

        Only configured subset reaches allowed_actions. Imports are local to avoid loading unused packs.
        """
        cfg = self._cfg.tools
        if cfg.sandbox_pack_enabled:
            from agent.app.idea_policies.extra_actions.sandbox_tools import SandboxToolPack
            installed = self.install_action_pack(
                SandboxToolPack(settings=self.settings), allow=False,
            )
            granted = set(cfg.sandbox_pack_actions)
            permitted = [name for name in installed if name in granted]
            if permitted:
                self.allow_actions(permitted)
        if cfg.calculator_pack_enabled:
            from agent.app.idea_policies.extra_actions.calculator_tools import CalculatorToolPack
            self.install_action_pack(CalculatorToolPack(settings=self.settings), allow=True)

    def _patch_allowed_actions(self) -> None:
        """Add ``plan_library_search`` to the action menu. Only when both flags are armed.

        No-op otherwise, so ``allowed_actions`` (and therefore both the dispatch gate and the
        expansion prompt's menu) is byte-identical on every unarmed run, including one whose
        settings never mention the plan library at all.
        """
        cfg = self._cfg.plan_library
        if not (cfg.enabled and cfg.action_enabled):
            return
        name = IdeaActionType.PLAN_LIBRARY_SEARCH.value
        allowed = list(self.settings.get("allowed_actions") or [a.value for a in IdeaActionType])
        if name not in [str(item) for item in allowed]:
            self.settings["allowed_actions"] = allowed + [name]

    def install_action_pack(self, pack: Any, *, allow: bool = True) -> List[str]:
        """Register every action in ``pack`` and (by default) permit it for this run.

        The one supported way to add actions to a live engine, because three things must agree
        for an action to be genuinely usable and they live in three places:
          1. the REGISTRY must know the class (dispatch resolves by name through it);
          2. ``settings["allowed_actions"]`` must contain the name (``_execute_action``'s gate);
          3. the EXPANSION POLICY's own settings copy must contain it too. The policy snapshots
             settings at construction, so assigning a fresh list to ``engine.settings`` after the
             fact updates the dispatch gate while leaving the prompt's action menu stale, and the
             model then never proposes the action it is now allowed to run.

        Pass ``allow=False`` to register without permitting (the higher-stakes packs, e.g. the
        sandbox file/shell tools, are meant to be opted into deliberately, one run at a time).

        :param pack: Anything with an ``ACTION_CLASSES`` list (see ``extra_actions/pack.py``).
        :param allow: Whether to add the installed names to ``allowed_actions``.
        :returns: The names that were installed.
        """
        installed = self.actions.install_pack(pack)
        if allow and installed:
            self.allow_actions(installed)
        return installed

    def allow_actions(self, names: List[str]) -> List[str]:
        """Add ``names`` to ``allowed_actions`` everywhere it is read (engine + expansion policy).

        Returns the resulting allowed list. Idempotent, order-preserving, and it never mutates the
        caller's original list in place (it may be shared).
        """
        allowed = [str(item) for item in (self.settings.get("allowed_actions")
                                          or [a.value for a in IdeaActionType])]
        for name in names:
            if str(name) not in allowed:
                allowed.append(str(name))
        self.settings["allowed_actions"] = allowed
        expansion_settings = getattr(self.expansion, "settings", None)
        if isinstance(expansion_settings, dict):
            expansion_settings["allowed_actions"] = list(allowed)
        return allowed

    async def prepare(self, mandate: str, run_id: Optional[str] = None) -> tuple:
        """Canonical engine/graph setup shared by `run()` and the interactive debugger.

        Computes the memo namespace, wires the per-run memory manager and GoT
        operations, resets per-run instrumentation, and produces the starting
        graph. When ``run_id`` is given and a checkpoint exists it restores the
        saved snapshot (graph/current_id/steps + dead-end + parallel-leaf counters);
        otherwise it constructs a fresh graph rooted at the mandate.

        Returns ``(graph, current_id, steps)``. Callers that do not checkpoint
        (e.g. ``debug_runner``) can ignore ``steps`` and use ``graph`` directly.
        """
        namespace = self._memo_namespace(mandate)
        self.settings[DetailKey.MEMO_NAMESPACE.value] = namespace
        self._memory_manager = MemoryManager(
            connector_chroma=self.io.connector_chroma,
            namespace=namespace,
        )
        self._got = GoTOperations(
            settings=self.settings,
            io=self.io,
            memory_manager=self._memory_manager,
        )
        self._current_mandate = mandate
        # Reset the one-time candidate-coverage budget extension for this run (an engine
        # instance can be reused across runs; the extension must never carry over).
        self._candidate_coverage_extension_applied = False
        # Reset per-run step-confidence instrumentation (opt-in; empty when the flag is off).
        self._step_confidences = []
        self._leaf_completion_count = 0
        root_title = mandate.split("\n\nTask Statement")[0] if "\n\nTask Statement" in mandate else mandate

        graph: Optional[IdeaDag] = None
        current_id: Optional[str] = None
        steps = 0
        if run_id and self._checkpointer:
            try:
                cp = await self._checkpointer.load(run_id)
            except Exception as exc:  # noqa: BLE001. Checkpoint load must never crash a run
                self._logger.warning(f"[RUN] Checkpoint load failed for run_id={run_id}: {exc}")
                cp = None
            if cp and isinstance(cp.get("snapshot"), dict):
                snap = cp["snapshot"]
                try:
                    graph = IdeaDag.from_dict(snap.get("graph") or {})
                    current_id = snap.get("current_id") or graph.root_id()
                    steps = int(cp.get("step_index") or 0) + 1
                    self._step_index = steps
                    self._parallel_leaves_total = int(snap.get("parallel_leaves_total") or 0)
                    if self._got and isinstance(snap.get("got_dead_end_count"), int):
                        self._got.dead_end_count = snap["got_dead_end_count"]
                    self._logger.info(
                        f"[RUN] Resumed run_id={run_id} from checkpoint step={steps - 1}, current_id={current_id}"
                    )
                except Exception as exc:  # noqa: BLE001. Corrupt checkpoint should not block a fresh run
                    self._logger.warning(f"[RUN] Checkpoint restore failed; starting fresh: {exc}")
                    graph = None
                    current_id = None
                    steps = 0

        if graph is None:
            graph = IdeaDag(root_title=root_title, root_details={"mandate": mandate, "memo_namespace": namespace})
            current_id = graph.root_id()
            self._logger.info(f"[RUN] Created graph with root_id={current_id}")
        return graph, current_id, steps

    async def run(self, mandate: str, max_steps: int = 50, run_id: Optional[str] = None, fail_soft: bool = False) -> Dict[str, Any]:
        mandate_short = mandate.split("\n\nTask Statement")[0] if "\n\nTask Statement" in mandate else mandate[:100]
        self._logger.info(f"[RUN] Starting idea DAG engine with mandate: {mandate_short}..., max_steps={max_steps}, run_id={run_id}")
        graph, current_id, steps = await self.prepare(mandate, run_id=run_id)
        # Checkpoint save is a run()-only concern (`run_id` is never passed by the benchmark
        # harness). Wire it as the shared loop's per-step hook so `_run_loop` stays generic.
        on_step = None
        if run_id and self._checkpointer:
            async def on_step(graph, current_id, steps):  # noqa: A002. Mirror caller names
                try:
                    await self._checkpointer.save(
                        run_id,
                        steps - 1,
                        {
                            "graph": graph.to_dict(),
                            "current_id": current_id,
                            "parallel_leaves_total": getattr(self, "_parallel_leaves_total", 0),
                            "got_dead_end_count": getattr(self._got, "dead_end_count", 0) if self._got else 0,
                        },
                    )
                except Exception as exc:  # noqa: BLE001. Checkpoint save must never crash a run
                    self._logger.warning(f"[RUN] Checkpoint save failed at step {steps - 1}: {exc}")

        graph, current_id, steps = await self._run_loop(
            graph, current_id, mandate, max_steps, steps=steps,
            on_step=on_step, fail_soft=fail_soft,
        )
        final_payload = await self.finalize(graph, mandate)
        self._maybe_log_dag(graph, steps, force=True)
        return final_payload

    async def _run_loop(
        self,
        graph: IdeaDag,
        current_id: Optional[str],
        mandate: str,
        max_steps: int,
        steps: int = 0,
        on_step=None,
        fail_soft: bool = False,
    ) -> tuple:
        """Shared budget / step / prune-backtrack / None-handling control loop.

        The single source of truth for the DAG execution loop, called by both `run()`
        (production, ``fail_soft=False``, ``on_step``=checkpoint save) and the benchmark
        harness ``run_test_execution`` (via ``run()``, ``fail_soft=True``). Keeping one
        implementation is what prevents the two call sites from drifting ("bug #4").

        :param on_step: optional ``async (graph, current_id, steps)`` hook run after each
            step's prune/backtrack (checkpoint save in production, unused by the harness).
        :param fail_soft: when True, an exception raised by ``step()`` is logged and the
            loop breaks (so the caller can still finalize with partial state); when False
            (production default) the exception propagates.
        :return: the final ``(graph, current_id, steps)``.
        """
        while True:
            if steps >= max_steps:
                # Step budget exhausted. Grant a one-time, fixed, capped extension IFF the
                # visit-backed candidate-coverage gate is unsatisfied, so the forced re-check
                # actually has room to add the missing visits instead of finalizing
                # under-resolved. Applied at most once (see `_candidate_coverage_extension`).
                _cov_ext = self._candidate_coverage_extension(graph, mandate)
                if _cov_ext <= 0:
                    break
                max_steps += _cov_ext
                current_id = graph.root_id()
                self._logger.info(
                    f"[COVERAGE] Granting one-time +{_cov_ext}-step budget extension "
                    f"(max_steps now {max_steps}) so unchecked candidates can be resolved"
                )
            self._logger.info(f"[RUN] === STEP {steps}/{max_steps} ===")
            try:
                current_id = await self.step(graph, current_id, steps)
            except Exception as exc:  # noqa: BLE001. Fail_soft callers finalize partial state
                if not fail_soft:
                    raise
                self._logger.error(f"[RUN] Step {steps} failed: {exc}", exc_info=True)
                break
            steps += 1
            self._step_index = steps
            self._maybe_log_dag(graph, steps)

            prune_interval = max(1, self._cfg.engine.got_prune_interval_steps)
            if self._got and steps % prune_interval == 0:
                prune_ids = self._got.identify_prune_candidates(graph)
                if prune_ids:
                    self._got.prune_nodes(graph, prune_ids)

            if (
                self._got
                and current_id
                and self._cfg.got.backtrack_enabled
                and self._got.should_backtrack(graph, current_id)
            ):
                target = self._got.find_backtrack_target(graph, current_id)
                if target and target != current_id:
                    self._logger.info(
                        f"[RUN] STEP {steps}: backtrack redirect {current_id[:8]} -> {target[:8]}"
                    )
                    current_id = target

            if (
                self._got
                and self._cfg.got.confidence_early_exit_enabled
                and self._got.should_exit_early(graph, self._step_confidences)
            ):
                self._record_decision(
                    "early_exit", node_id=current_id or graph.root_id(), chosen="finalize",
                    rationale="calibrated high-confidence early exit",
                    metadata={"judged_steps": len(self._step_confidences), "step": steps},
                )
                self._logger.info(
                    f"[RUN] STEP {steps}: calibrated early exit. Finalizing after "
                    f"{len(self._step_confidences)} judged step(s)"
                )
                if on_step is not None:
                    await on_step(graph, current_id, steps)
                break

            if on_step is not None:
                await on_step(graph, current_id, steps)

            if steps == 1:
                root = graph.get_node(graph.root_id())
                if root and not root.children:
                    self._logger.error(f"[RUN] VALIDATION FAILED: Root has no children after step 1!")
                    self._logger.error(f"[RUN] Root status: {root.status.value}, Root details keys: {list(root.details.keys())}")
                    self._logger.error(f"[RUN] Attempting emergency root expansion...")
                    emergency_result = await self._handle_expansion_node(graph, graph.root_id(), steps, None)
                    if emergency_result and root.children:
                        self._logger.info(f"[RUN] Emergency expansion succeeded: {len(root.children)} children created")
                        current_id = emergency_result
                    else:
                        self._logger.error(f"[RUN] Emergency expansion failed - root still has no children")

            if steps == 3:
                action_count = sum(1 for n in graph.iter_depth_first() if NodeDetailsExtractor.get_action(n.details))
                if action_count == 0:
                    self._logger.warning(f"[RUN] VALIDATION WARNING: No actions created after step 3 (total nodes: {graph.node_count()})")

            if current_id is None:
                if self._grounding_replan(graph, mandate, steps, max_steps):
                    current_id = graph.root_id()
                    continue
                _cov_ext = self._candidate_coverage_extension(graph, mandate)
                if _cov_ext > 0:
                    max_steps += _cov_ext
                    current_id = graph.root_id()
                    self._logger.info(
                        f"[COVERAGE] Granting one-time +{_cov_ext}-step budget extension "
                        f"(max_steps now {max_steps}) so unchecked candidates can be resolved"
                    )
                    continue
                self._logger.warning(f"[RUN] Step {steps} returned None, breaking loop")
                break
            else:
                node = graph.get_node(current_id)
                if node and node.status == IdeaNodeStatus.DONE and current_id == graph.root_id():
                    if self._grounding_replan(graph, mandate, steps, max_steps):
                        current_id = graph.root_id()
                        continue
                    break
        return graph, current_id, steps

    async def finalize(self, graph: IdeaDag, mandate: str, pending_check: bool = True) -> Dict[str, Any]:
        """Build the final payload and attach every derived signal.

        Shared by `run()` and the benchmark harness (via `run()`) so both emit an identical
        output shape: `graph`, `pending_nodes_count`, `warning`, `candidate_coverage_*`,
        `got_stats`, `grounded`, `missing_requirements`, `grounding_replans`.
        """
        self._logger.info("[RUN] Finalizing: checking for pending nodes before building payload")

        pending_nodes = self._get_pending_executable_nodes(graph) if pending_check else []
        if pending_nodes:
            pending_ids = [n.node_id for n in pending_nodes]
            self._logger.warning(f"[RUN] GUARDRAIL: {len(pending_nodes)} nodes still pending execution: {pending_ids[:5]}...")
            self._logger.warning(f"[RUN] Cannot finalize with pending nodes. These nodes need action execution:")
            for node in pending_nodes[:5]:
                action = NodeDetailsExtractor.get_action(node.details)
                self._logger.warning(f"[RUN]   - {node.node_id}: {node.title[:60]}... (action={action}, status={node.status.value})")

        # Unconditional pre-finalize candidate-coverage check (opt-in). The mid-loop
        # `_grounding_replan` hook (the only other place this gate runs) fires ONLY when
        # `step()` returns None; a run that exits via step-budget exhaustion
        # (`steps >= max_steps`) never gets that second chance, so a short plan can starve
        # the gate. Guarantee an honest signal here regardless of the exit path: we are
        # out of budget to force another pass at this point, so annotate the final payload
        # when named candidates remain unchecked rather than finalizing as if grounded.
        # Byte-identical when `got_candidate_coverage_enabled` is off.
        _coverage_incomplete = None
        if self._cfg.got.candidate_coverage_enabled:
            try:
                _cov = evaluate_candidate_coverage(graph, mandate)
                if not _cov.satisfied:
                    _coverage_incomplete = _cov
            except Exception as exc:  # noqa: BLE001. The gate must never crash finalize
                self._logger.warning(f"[COVERAGE] pre-finalize check failed: {exc}")

        final_payload = await build_final_payload(
            self.io, self.settings, graph, mandate, self.model_name,
            memory_manager=self._memory_manager,
        )
        final_payload["graph"] = graph.to_dict()
        final_payload["pending_nodes_count"] = len(pending_nodes) if pending_nodes else 0
        if pending_nodes:
            final_payload["warning"] = f"Finalized with {len(pending_nodes)} pending nodes - execution incomplete"

        if _coverage_incomplete is not None:
            final_payload["candidate_coverage_incomplete"] = True
            final_payload["candidate_coverage_missing"] = list(_coverage_incomplete.missing)
            self._logger.warning(
                f"[COVERAGE] Finalizing with unchecked candidates: {_coverage_incomplete.missing}"
            )

        if self._got:
            pruned_count = sum(
                1 for n in graph.iter_depth_first()
                if n.details.get("_got_pruned")
            )
            final_payload["got_stats"] = {
                "dead_ends_detected": self._got.dead_end_count,
                "nodes_pruned": pruned_count,
                "parallel_leaves_total": getattr(self, "_parallel_leaves_total", 0),
            }
            # A6 counter, attached ONLY when armed so the flag-off payload keeps its exact shape.
            if self._cfg.got.confidence_early_exit_enabled:
                final_payload["got_stats"]["early_exits"] = self._got.early_exit_count

        # Opt-in decorrelated per-step confidence trace (empty/absent when the flag is off),
        # surfaced so the E-valuator pilot can consume it as a real verifier-score sequence.
        if self._cfg.got.step_confidence_judge_enabled and self._step_confidences:
            final_payload["step_confidences"] = list(self._step_confidences)

        # Grounding verdict for the final answer (substantiation mandates only). Surfaced
        # in the result so observability/groundedness reflect real visited-page evidence.
        try:
            _req = parse_mandate_requirements(mandate)
            if _req.needs_substantiation:
                _g = evaluate_grounding(graph, _req)
                final_payload["grounded"] = bool(_g.grounded)
                final_payload["missing_requirements"] = _g.missing
                final_payload["grounding_replans"] = int(getattr(self, "_grounding_replans", 0))
                self._record_decision(
                    "finalize", node_id=graph.root_id(), chosen="finalized",
                    grounded=_g.grounded, rationale=_g.reason,
                    metadata={"replans": int(getattr(self, "_grounding_replans", 0)),
                              "missing": _g.missing},
                )
            else:
                self._record_decision("finalize", node_id=graph.root_id(), chosen="finalized")
        except Exception as exc:  # noqa: BLE001. Never crash finalize on grounding
            self._logger.warning(f"[GROUNDING] final grounding check failed: {exc}")

        self._logger.info(f"[RUN] Final payload created, graph has {graph.node_count()} nodes, {len(pending_nodes) if pending_nodes else 0} pending")
        return final_payload

    async def step(self, graph: IdeaDag, current_id: str, step_index: int) -> Optional[str]:
        self._logger.info(f"[STEP {step_index}] Starting step with current_id={current_id}, node_count={graph.node_count()}")
        if graph.node_count() >= self._cfg.engine.max_total_nodes:
            self._logger.warning(f"[STEP {step_index}] Max nodes reached, stopping")
            return None
        
        node = graph.get_node(current_id)
        if not node:
            self._logger.warning(f"[STEP {step_index}] Node {current_id} not found")
            return None
        
        self._logger.info(f"[STEP {step_index}] Processing node: {node.title[:80]}... (status={node.status.value}, children={len(node.children)}, action={NodeDetailsExtractor.get_action(node.details)})")
        
        is_root = current_id == graph.root_id()
        if is_root and not node.children and step_index == 0:
            self._logger.info(f"[STEP {step_index}] ROOT NODE: No children detected, forcing expansion")
            result = await self._handle_expansion_node(graph, current_id, step_index, None)
            if result is None:
                self._logger.error(f"[STEP {step_index}] ROOT EXPANSION FAILED: Expansion returned None")
            elif not node.children:
                self._logger.error(f"[STEP {step_index}] ROOT EXPANSION FAILED: Expansion returned but no children created")
            else:
                self._logger.info(f"[STEP {step_index}] ROOT EXPANSION SUCCESS: Created {len(node.children)} children")
            return result
        
        if NodeDetailsExtractor.is_merge_action(node.details):
            self._logger.info(f"[STEP {step_index}] Node is merge node, handling merge")
            return await self._handle_merge_node(graph, current_id, step_index, None)
        
        is_leaf = node.details.get(DetailKey.IS_LEAF.value, False)
        action = NodeDetailsExtractor.get_action(node.details)
        has_action = action and not NodeDetailsExtractor.is_merge_action(node.details)
        # Re-expansion escape hatch: a completed leaf that was re-expanded now has
        # real children to drive; route it through the children-processing path
        # instead of the leaf handler (which would short-circuit back to parent).
        # Only fires for nodes explicitly marked by `_maybe_reexpand_leaf`.
        reexpanded = bool(node.details.get("_got_reexpanded")) and bool(node.children)
        if (is_leaf or has_action) and not reexpanded:
            self._logger.info(f"[STEP {step_index}] Node is leaf node (is_leaf={is_leaf}, has_action={has_action}, action={action}), executing action")
            return await self._handle_leaf_node(graph, current_id, step_index, None)
        
        if not node.children:
            self._logger.info(f"[STEP {step_index}] Node has no children, expanding into sub-problems")
            result = await self._handle_expansion_node(graph, current_id, step_index, None)
            if result is None:
                self._logger.warning(f"[STEP {step_index}] Expansion returned None for node {current_id}")
            elif not node.children:
                self._logger.warning(f"[STEP {step_index}] Expansion completed but no children created for node {current_id}")
            else:
                self._logger.info(f"[STEP {step_index}] Expansion created {len(node.children)} children")
            return result
        
        # Re-expansion descent: if any child was re-expanded and now owns an
        # in-progress subtree, hand control down to it so step()'s escape hatch
        # drives that subtree. Without this, the children-processing below would
        # treat the re-expanded child (ACTIVE, with an action_result AND its own
        # children) as an executable leaf and re-run its action. Clobbering its
        # status back to DONE and orphaning the freshly-spawned children.
        for child_id in node.children:
            child = graph.get_node(child_id)
            if (
                child
                and child.details.get("_got_reexpanded")
                and child.children
                and child.status not in (
                    IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
            ):
                self._logger.info(
                    f"[STEP {step_index}] REEXPAND DESCENT: routing into re-expanded "
                    f"child {child_id[:8]} to drive its new subtree"
                )
                return child_id

        leaf_children = []
        merge_children = []
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if NodeDetailsExtractor.is_merge_action(child.details):
                merge_children.append(child_id)
            else:
                leaf_children.append(child_id)
        
        all_terminal = all(
            (child := graph.get_node(cid)) and
            child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
            for cid in node.children
        )
        if all_terminal and merge_children:
            self._logger.info(f"[STEP {step_index}] All children complete (incl. merge), marking parent done")
            node.status = IdeaNodeStatus.DONE
            if node.parent_id:
                return node.parent_id
            return None
        
        all_leaves_complete = leaf_children and all(
            (child := graph.get_node(cid)) and
            child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
            for cid in leaf_children
        )
        if all_leaves_complete and not merge_children:
            branch_pair = find_branch_pair(graph, current_id)
            leaf_statuses = {cid: graph.get_node(cid).status.value for cid in leaf_children if graph.get_node(cid)}
            self._logger.info(f"[STEP {step_index}] ALL LEAVES COMPLETE ({len(leaf_children)}): {leaf_statuses}. Creating merge")
            if not self.merge.should_create_merge_node(graph, current_id):
                self._logger.info(
                    f"[STEP {step_index}] NO MERGE NEEDED: node {current_id[:8]} "
                    f"children done, marking DONE"
                )
                node.status = IdeaNodeStatus.DONE
                return node.parent_id if node.parent_id else None
            if branch_pair and branch_pair.needs_merge():
                return await self._handle_merge_creation(graph, current_id, step_index, branch_pair)
        elif not all_leaves_complete and leaf_children:
            pending = {cid: graph.get_node(cid).status.value for cid in leaf_children
                       if graph.get_node(cid) and graph.get_node(cid).status not in
                       (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)}
            self._logger.debug(f"[STEP {step_index}] MERGE GATE: {len(pending)} leaf children still pending: {pending}")
        
        if all_leaves_complete and merge_children:
            for merge_cid in merge_children:
                merge_child = graph.get_node(merge_cid)
                if merge_child and merge_child.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    self._logger.info(f"[STEP {step_index}] All leaves done, executing merge child {merge_cid}")
                    return merge_cid
        
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                continue
            if not self._has_required_data(graph, child):
                required_data = child.details.get(DetailKey.REQUIRES_DATA.value)
                if required_data and isinstance(required_data, dict):
                    source_node_id = required_data.get("source_node_id")
                    if source_node_id:
                        source_node = graph.get_node(source_node_id)
                        # A source that has already RUN (or vanished) and still owes data will
                        # never deliver it. Returning to it just spins the step budget away on
                        # a node the engine has given up on, and letting the dependent run would
                        # silently read some other sibling's results. Retire the dependent
                        # instead so the parent can still merge what did work. (BLOCKED is a
                        # scheduled retry, not a terminal state, so it keeps waiting.)
                        source_spent = source_node is None or source_node.status in (
                            IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED,
                        )
                        if source_spent:
                            child.status = IdeaNodeStatus.SKIPPED
                            child.details[DetailKey.ACTION_ERROR.value] = (
                                f"required data never arrived from source {source_node_id} "
                                f"({source_node.status.value if source_node else 'missing'})"
                            )
                            self._logger.warning(
                                f"[STEP {step_index}] Child {child_id} skipped: source "
                                f"{source_node_id} cannot provide its required data"
                            )
                            continue
                        if source_node.status != IdeaNodeStatus.DONE:
                            self._logger.info(f"[STEP {step_index}] Child {child_id} waiting for data from {source_node_id} - executing source first")
                            return source_node_id

        return await self._handle_intermediate_node(graph, current_id, step_index, None)
    
    async def _handle_merge_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        if node.details.get(DetailKey.ACTION_RESULT.value) is None:
            if self._is_action_ready(node, step_index):
                self._logger.info(f"[STEP {step_index}] Executing merge action for node {node_id}")
                result = await self._execute_action_guarded(graph, node.parent_id or graph.root_id(), node_id)
                if result is not None:
                    await self._apply_action_result(graph, node_id, step_index)
                else:
                    self._logger.warning(f"[STEP {step_index}] Merge action for {node_id} returned None; marking DONE")
                    node.status = IdeaNodeStatus.DONE
        
        parent_id = node.parent_id or graph.root_id()
        if node.status == IdeaNodeStatus.DONE:
            self._logger.info(f"[STEP {step_index}] MERGE COMPLETE for {node_id}: returning to parent {parent_id}")
            return parent_id
        
        if node.status == IdeaNodeStatus.FAILED:
            self._logger.warning(f"[STEP {step_index}] MERGE FAILED for {node_id}: returning to parent {parent_id}")
            return parent_id
        
        return node_id
    
    async def _handle_leaf_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        if not self._has_required_data(graph, node):
            self._logger.warning(f"[STEP {step_index}] Node {node_id} missing required data - cannot execute yet")
            return node.parent_id if node.parent_id else None
        
        has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
        is_blocked_ready = node.status == IdeaNodeStatus.BLOCKED and self._is_action_ready(node, step_index)
        
        if not has_result or is_blocked_ready:
            if self._is_action_ready(node, step_index):
                result = await self._execute_action_guarded(graph, node.parent_id or graph.root_id(), node_id)
                if result is not None:
                    # Shared completion point: records the result and, when the
                    # leaf lands DONE, offers the bounded re-expansion check.
                    await self._apply_action_result(graph, node_id, step_index)

        if node.status == IdeaNodeStatus.DONE:
            should_chunk = self._should_chunk_document(graph, node)
            if should_chunk:
                self._logger.info(f"[STEP {step_index}] Large document detected - creating chunk sub-problems")
                chunk_nodes = await self._create_chunk_subproblems(graph, node)
                if chunk_nodes:
                    return chunk_nodes[0] if chunk_nodes else node.parent_id
        
        # Re-expansion (when enabled) already fired from the shared completion
        # point in `_apply_action_result` above: a leaf that revealed a genuine
        # follow-up is now ACTIVE with children, so it falls through to
        # `return node_id` and the step() escape hatch drives its new children.
        if node.status == IdeaNodeStatus.DONE and node.parent_id:
            return node.parent_id

        return node_id

    async def _maybe_reexpand_leaf(self, graph: IdeaDag, node_id: str, step_index: int) -> bool:
        """Re-expand a completed leaf when its result reveals a real follow-up.

        Returns True if new children were created (caller should stay on this node
        so the normal leaf lifecycle drives the new children). Bounded by
        `got.reexpand_max_iterations` (tracked via `_got_reexpand_count`) and the
        global `max_total_nodes` ceiling.

        Composed of two phases split out so batch callers can parallelize the
        expensive read-only `_reexpand_check` (an independent LLM call per sibling)
        while keeping the mutating `_apply_reexpand` phase sequential. The latter
        re-checks the shared `max_total_nodes` ceiling so concurrent siblings can
        never overshoot it. The single-node path (this method) composes them
        sequentially, so its behavior is byte-identical to the pre-split version.
        """
        verdict = await self._reexpand_check(graph, node_id, step_index)
        if not verdict:
            return False
        return await self._apply_reexpand(graph, node_id, step_index, verdict)

    def _effective_reexpand_max_iterations(self) -> int:
        """The per-lineage re-expansion budget actually in force: the static
        ``got.reexpand_max_iterations`` (flag off, the byte-identical default), or a band picked
        via ``model_tiers.capability_tier(self.model_name)`` when ``reexpand_max_iterations_tiered_
        enabled`` is on. Weak models get more bounded re-expansion attempts, strong models taper
        toward the current default.

        ALL THREE budget-check sites (``_reexpand_check``, ``_apply_reexpand``,
        ``_reexpand_guards_ok``) must read through this single method, not
        ``self._cfg.got.reexpand_max_iterations`` directly: this exact knob was previously found to
        be silently inert at one call site for any value > 1 (see the comment in
        ``_apply_reexpand``). A partial port that missed one of the three sites would reintroduce
        that failure mode for the weak tier specifically.
        """
        if not self._cfg.got.reexpand_max_iterations_tiered_enabled:
            return self._cfg.got.reexpand_max_iterations
        return tier_value(
            capability_tier(self.model_name),
            weak=self._cfg.got.reexpand_max_iterations_weak,
            standard=self._cfg.got.reexpand_max_iterations_standard,
            strong=self._cfg.got.reexpand_max_iterations_strong,
        )

    async def _reexpand_check(self, graph: IdeaDag, node_id: str, step_index: int) -> Optional[Dict[str, Any]]:
        """Read-only re-expansion gate + follow-up LLM check. No graph mutation.

        Returns the positive verdict dict when the node should be re-expanded, else
        None. Being mutation-free (aside from the LLM call) makes this safe to run
        concurrently across sibling leaves via `asyncio.gather`.
        """
        if not self._cfg.got.reexpand_enabled or not self._got:
            return None

        node = graph.get_node(node_id)
        if not node:
            return None
        # Merge nodes reach the shared completion point too; they are never
        # re-expansion candidates (they aggregate branches, they don't reveal
        # a fresh follow-up). Guard explicitly so the centralized call site is
        # safe to fire unconditionally from every path (incl. merge paths).
        if NodeDetailsExtractor.is_merge_action(node.details):
            return None
        if node.children:
            return None

        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if not ActionResultExtractor.is_success(result):
            return None
        # C1a routing: mirrors `_confidence_triggers_reexpand`'s tool-failure gate. If the
        # leaf's "success" is actually a tool failure (empty search results, empty visit
        # content), a fresh subtree would just repeat the same failing fetch. The bounded
        # in-place connector retry (`_maybe_retry_tool_failure`, already run before this
        # point) is the correct recovery, not re-expansion. Without this the follow-up
        # detector could still be asked to judge a tool-failure result and re-expand off
        # it, defeating the suppression the confidence-trigger path already enforces.
        # No-op unless `tool_failure_recovery_enabled` (default off -> byte-identical).
        if (
            self._cfg.action.tool_failure_recovery_enabled
            and ActionResultExtractor.is_tool_failure(result)
        ):
            return None

        max_iters = self._effective_reexpand_max_iterations()
        count = int(node.details.get("_got_reexpand_count", 0))
        if count >= max_iters:
            return None
        if graph.node_count() >= self._cfg.engine.max_total_nodes:
            return None

        verdict = await self._got.check_needs_followup(graph, node_id, model_name=self.model_name)
        if not verdict or not verdict.get("needs_followup"):
            return None
        return verdict

    async def _apply_reexpand(self, graph: IdeaDag, node_id: str, step_index: int, verdict: Dict[str, Any]) -> bool:
        """Mutating expansion given a positive verdict from `_reexpand_check`.

        Must be called sequentially (never inside a gather): it re-reads the
        `max_total_nodes` ceiling immediately before expanding so that when several
        siblings passed the read-only gate concurrently, only as many as the budget
        allows actually expand. The counter/ceiling accounting stays correct because
        there is no `await` between this ceiling read and the graph-growing expansion
        for any *other* node (each apply runs to completion before the next starts).
        """
        node = graph.get_node(node_id)
        if not node:
            return False
        if node.children:
            return False
        # Re-check the ceiling here (not only in _reexpand_check): under the batch
        # path multiple siblings may have cleared the read-only gate before any of
        # them expanded, so enforce the budget at the mutation point.
        if graph.node_count() >= self._cfg.engine.max_total_nodes:
            return False

        max_iters = self._effective_reexpand_max_iterations()
        count = int(node.details.get("_got_reexpand_count", 0))
        if count >= max_iters:
            return False

        self._logger.info(
            f"[STEP {step_index}] REEXPAND: leaf {node_id[:8]} completed with a genuine "
            f"follow-up (iter {count + 1}/{max_iters}): {str(verdict.get('reason', ''))[:80]}"
        )
        node.details["_got_reexpand_reason"] = verdict.get("reason", "")
        # Opt-in (`got_reexpand_corrective_context_enabled`, JSON default False): thread the
        # triggering judge/detector's own reason onto the node as a single-use expansion
        # detail so the re-expansion prompt can course-correct instead of blindly re-targeting
        # the same inadequate source. `_build_messages` consumes-and-clears it, mirroring
        # `HUMAN_FEEDBACK`. No-op (byte-identical) when the flag is off.
        if self._cfg.got.reexpand_corrective_context_enabled:
            reason_text = str(verdict.get("reason", "")).strip()
            if reason_text:
                node.details[DetailKey.REEXPAND_REASON.value] = reason_text
        # Mark before expanding so the step() escape hatch routes this node through
        # the children-processing path on subsequent steps.
        node.details["_got_reexpanded"] = True
        node.status = IdeaNodeStatus.ACTIVE

        before = len(node.children)
        expand_result = await self._handle_expansion_node(graph, node_id, step_index, None)
        if not expand_result or len(node.children) <= before:
            # Expansion produced nothing. Revert markers so the node finalizes normally.
            node.details.pop("_got_reexpanded", None)
            node.status = IdeaNodeStatus.DONE
            self._logger.info(f"[STEP {step_index}] REEXPAND: no children produced for {node_id[:8]}; reverting")
            return False

        node.details["_got_reexpand_count"] = count + 1
        # Propagate the lineage re-expansion budget DOWN to the freshly spawned children.
        # A re-expanded node immediately owns children, so it can never re-expand a second
        # time itself (both re-expand gates skip nodes WITH children). Without this, the
        # `reexpand_max_iterations` knob was inert for any value > 1, because each new child
        # leaf started at count 0 and the lineage only stopped growing when it hit
        # `max_total_nodes`. Seeding each new child with the parent's post-increment count
        # makes `max_iterations` actually govern how many observe->decide->re-expand cycles
        # a single investigative thread may take: a lineage re-expands at most
        # `max_iterations` times end-to-end. With the default `max_iterations=1` a lineage
        # re-expands exactly once (the first leaf), which is the intended single-shot
        # follow-up. Both triggers (the follow-up detector and the confidence loop) route
        # through here, so both honor the same lineage budget.
        for child_id in node.children[before:]:
            child = graph.get_node(child_id)
            if child is not None:
                child.details["_got_reexpand_count"] = count + 1
        self._record_decision(
            "reexpand", node_id=node_id, chosen=f"{len(node.children) - before} follow-up sub-problems",
            rationale=str(verdict.get("reason", ""))[:200],
            metadata={"step": step_index, "iteration": count + 1},
        )
        return True

    def _has_required_data(self, graph: IdeaDag, node: IdeaNode) -> bool:
        requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
        if not requires_data or not isinstance(requires_data, dict):
            return True

        data_source_node_id = requires_data.get("source_node_id")
        if not data_source_node_id:
            return True

        source_node = graph.get_node(data_source_node_id)
        if not source_node:
            return False
        if source_node.status != IdeaNodeStatus.DONE:
            return False

        source_result = source_node.details.get(DetailKey.ACTION_RESULT.value)
        if not source_result:
            return False

        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if not ActionResultExtractor.is_success(source_result):
            return False

        contract = self.contracts.get(requires_data.get("type"))
        if not contract:
            return True
        return contract.is_ready(source_result, source_node)
    
    def _plan_library_corpus(self):
        """The plan-library corpus for this engine instance, loaded once, lazily.

        ``PlanLibrary.__init__`` parses and validates every ``templates/*.json`` off disk and
        runs the manifest drift check, so it is neither free nor cheap to repeat: building one
        per expansion step would re-read the whole corpus at every node of every run. One per
        engine instance is the right grain. A worker handles one mandate at a time (RabbitMQ
        prefetch=1) and the corpus cannot change mid-run. And building it lazily keeps a
        flag-off run from touching the disk at all.
        """
        if self._plan_library_corpus_cache is None:
            self._plan_library_corpus_cache = _plan_retrieval.PlanLibrary()
        return self._plan_library_corpus_cache

    async def _plan_library_resolve(
        self,
        graph: IdeaDag,
        node: IdeaNode,
        step_index: int,
        *,
        call_site: str,
        origin: str,
    ) -> "_plan_search.SearchResolution":
        """One plan-library retrieval attempt for ``node``. The engine's side of it.

        The pipeline itself (query -> rank -> fill -> both logs) lives in
        ``idea_policies.plan_library_search`` because the on-demand leaf action runs exactly
        the same one and a ``LeafAction`` has no engine handle to borrow it from; keeping one
        implementation is what stops the two call sites drifting into different logged fields
        or different fail-toward-silence rules. What belongs HERE is the wiring of the engine's
        collaborators. The per-instance corpus, this run's IO and contracts, the model/timeout
        knobs, plus the outer net: any exception at all (a broken corpus, a connector blowing
        up) becomes an empty resolution, never a degraded run.

        ``RetrievalResult.decision`` IS the confidence gate. The thresholds behind it
        (``retrieval.AUTO_APPLY_THRESHOLD``/``SUGGEST_THRESHOLD``/``MIN_MARGIN``) are
        calibrated against a labelled eval set, so the engine flags govern whether the
        mechanism runs at all, not a second opinion on how confident is confident enough.
        """
        try:
            return await _plan_search.resolve(
                self._plan_library_corpus(),
                graph,
                node,
                io=self.io,
                call_site=call_site,
                origin=origin,
                contracts=self.contracts,
                model_name=self.model_name or self._cfg.expansion.model or None,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=self._cfg.timeouts.expansion or self._cfg.timeouts.llm,
            )
        except Exception as exc:  # noqa: BLE001. The library never breaks a run
            self._logger.warning(f"[STEP {step_index}] PLAN LIBRARY: retrieval failed: {exc}")
            return _plan_search.EMPTY

    async def _plan_library_auto_shortcircuit(
        self, graph: IdeaDag, node: IdeaNode, step_index: int
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """Try to answer this expansion from the plan library instead of the LLM.

        Returns ``(candidates, parent_details)`` when a pre-authored template matched
        confidently and filled. The caller then skips both the memory enrichment and the
        expansion policy entirely, and ``(None, None)`` for every other outcome: no match, a
        suggest-only score, a slot-fill failure, or any exception at all. Fails toward silence
        exactly like ``contract_satisfaction.py`` and like retrieval's own downgrades: a
        library problem may never degrade a run that would otherwise have planned organically.
        """
        resolution = await self._plan_library_resolve(
            graph, node, step_index,
            call_site=_plan_retrieval.CALL_SITE_AUTO,
            origin=_plan_library.ORIGIN_AUTO,
        )
        expansion = resolution.expansion
        if expansion is None or not expansion.candidates:
            result = resolution.retrieval
            self._logger.info(
                f"[STEP {step_index}] PLAN LIBRARY: no template applied "
                f"({getattr(result, 'decision', 'error')}: {getattr(result, 'reason', '')}) "
                f". Expanding organically"
            )
            return None, None
        self._logger.info(
            f"[STEP {step_index}] PLAN LIBRARY: applied template '{expansion.template_id}' "
            f"-> {len(expansion.candidates)} candidates, skipping LLM expansion"
        )
        return list(expansion.candidates), dict(expansion.parent_details)

    async def _handle_expansion_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None

        # Plan library (opt-in, default OFF -> byte-identical): a confidently-matched
        # pre-authored template's filled leaves ARE this node's expansion, so the LLM never
        # invents one. Placed BEFORE the memory/hybrid-retrieve enrichment below so a hit also
        # skips that now-pointless vector-DB work. `candidates is None` on every other path,
        # which is what keeps the flag-off behaviour identical to before.
        candidates = None
        parent_detail_updates = None
        # ON-DEMAND path: `_maybe_plan_library_reexpand` already resolved a template (the leaf
        # action matched, filled and logged it) and left the finished expansion here for us, so
        # both call sites converge on this single expansion entry point rather than each
        # growing children their own way. Single-use. Popped on read; only ever present when
        # both plan-library flags are armed.
        pending = node.details.pop(_plan_library.PLAN_LIBRARY_PENDING, None)
        if isinstance(pending, dict):
            candidates = pending.get("candidates") or None
            parent_detail_updates = pending.get("parent_details") or None
        if candidates is None and self._cfg.plan_library.enabled and self._cfg.plan_library.auto_enabled:
            candidates, parent_detail_updates = await self._plan_library_auto_shortcircuit(
                graph, node, step_index
            )

        if candidates is None:
            memories = []
            if self._memory_manager:
                justification = NodeDetailsExtractor.get_justification(node.details)
                parent_goal = node.details.get(DetailKey.PARENT_GOAL.value) or ""

                query_parts = [node.title]
                if justification:
                    query_parts.append(justification[:100])
                if parent_goal:
                    query_parts.append(parent_goal[:100])
                if hasattr(self, '_current_mandate') and self._current_mandate:
                    query_parts.append(self._current_mandate[:100])

                query = " ".join(query_parts)

                n_internal = self._cfg.memory.expansion_chroma_internal
                n_observations = self._cfg.memory.expansion_chroma_observations
                split_memories = await self._memory_manager.retrieve_memories_split(
                    query=query,
                    node_context={
                        "title": node.title,
                        "action": node.details.get(DetailKey.ACTION.value),
                        "error": node.details.get(DetailKey.ACTION_ERROR.value),
                        "justification": justification,
                    },
                    n_internal=n_internal,
                    n_observations=n_observations,
                )
                memories = split_memories["internal_thoughts"] + split_memories["observations"]

                if self._got:
                    hybrid_extras = await self._got.hybrid_retrieve(
                        graph, node_id, query, n_results=3,
                    )
                    seen_ids = {m.get("id") for m in memories if m.get("id")}
                    for extra in hybrid_extras:
                        if extra.get("id") not in seen_ids:
                            memories.append(extra)
                            seen_ids.add(extra.get("id"))
                self._logger.info(
                    f"[STEP {step_index}] EXPANSION: Retrieved {len(split_memories['internal_thoughts'])} internal thoughts, "
                    f"{len(split_memories['observations'])} observations from vector DB"
                )
                if split_memories["observations"]:
                    obs_preview = "\n".join([str(obs.get("content", "") if isinstance(obs, dict) else obs)[:200] for obs in split_memories["observations"][:3]])
                    self._logger.info(f"[STEP {step_index}] Observations preview:\n{obs_preview}")

            self._logger.info(f"[STEP {step_index}] EXPANSION: Calling expansion policy for node '{node.title[:60]}...'")
            try:
                candidates = await self.expansion.expand(graph, node_id, memories=memories)
                self._logger.info(f"[STEP {step_index}] EXPANSION: Policy returned {len(candidates) if candidates else 0} candidates")
                if not candidates:
                    self._logger.error(f"[STEP {step_index}] EXPANSION FAILED: Expansion policy returned no candidates!")
                    self._logger.error(f"[STEP {step_index}] Node details: {list(node.details.keys())}")
                    self._logger.error(f"[STEP {step_index}] Node title: {node.title}")
                    return None
            except Exception as exc:
                self._logger.error(f"[STEP {step_index}] EXPANSION EXCEPTION: {exc}", exc_info=True)
                return None

        filtered = [
            c for c in candidates
            if c.get("details", {}).get(DetailKey.ACTION.value) != IdeaActionType.MERGE.value
            and c.get("action") != IdeaActionType.MERGE.value
        ]
        if len(filtered) < len(candidates):
            self._logger.info(
                f"[STEP {step_index}] EXPANSION: Stripped {len(candidates) - len(filtered)} "
                f"LLM-generated merge candidates (merge is system-managed)"
            )
        candidates = filtered or candidates[:1]

        if self._got:
            candidates = await self._got.filter_duplicate_candidates(candidates, graph)

        if self._got:
            max_branching = self._got.compute_dynamic_beam_width(graph)
        else:
            max_branching = self._cfg.engine.max_branching
        hard_cap = self._cfg.engine.max_branching
        max_branching = min(max_branching, hard_cap)
        created_children = graph.expand(node_id, candidates[:max_branching])

        # Plan-library post-expand wiring, only for a library-sourced expansion (the flag rides
        # in from the short-circuit; organic candidates never carry these markers, so an
        # unconditional call would merely no-op (see `plan_library_auto_shortcircuit_test`):
        #   * the template's aggregation guidance lands on the PARENT as `details.intent`,
        #     which is the channel `MergeLeafAction` reads as `parent_intent`;
        #   * blueprint-level `depends_on` becomes a real `requires_data` node reference, which
        #     can only be written now that `graph.expand()` has minted the child ids.
        if parent_detail_updates:
            node.details.update(parent_detail_updates)
            linked = _plan_library.link_dependencies(created_children)
            if linked:
                self._logger.info(
                    f"[STEP {step_index}] PLAN LIBRARY: linked {linked} dependent child(ren) "
                    f"onto their producer siblings (sequential execution)"
                )

        parent_goal = node.details.get(DetailKey.GOAL.value) or node.title
        if not node.details.get(DetailKey.GOAL.value):
            node.details[DetailKey.GOAL.value] = parent_goal
        if not node.details.get(DetailKey.ORIGINAL_GOAL.value):
            node.details[DetailKey.ORIGINAL_GOAL.value] = parent_goal
        parent_original_goal = node.details.get(DetailKey.ORIGINAL_GOAL.value) or parent_goal
        parent_justification = NodeDetailsExtractor.get_justification(node.details)
        
        for child_id in node.children[-max_branching:]:
            child = graph.get_node(child_id)
            if not child:
                continue
            
            child.details[DetailKey.PARENT_GOAL.value] = parent_goal
            child_goal = (
                child.details.get(DetailKey.GOAL.value)
                or child.details.get(DetailKey.ORIGINAL_GOAL.value)
                or child.title
            )
            if child_goal:
                if not child.details.get(DetailKey.GOAL.value):
                    child.details[DetailKey.GOAL.value] = child_goal
                if not child.details.get(DetailKey.ORIGINAL_GOAL.value):
                    child.details[DetailKey.ORIGINAL_GOAL.value] = child_goal
            
            if parent_justification:
                child.details[DetailKey.PARENT_JUSTIFICATION.value] = parent_justification
            
            child_justification = NodeDetailsExtractor.get_justification(child.details)
            if not child_justification and parent_justification:
                child.details[DetailKey.WHY_THIS_NODE.value] = f"Parent goal: {parent_goal}. {parent_justification}"
            
            if NodeDetailsExtractor.get_action(child.details) and not NodeDetailsExtractor.is_merge_action(child.details):
                child.details[DetailKey.IS_LEAF.value] = True

            # Contract log (env-gated, no-op unless IDEA_TEST_CONTRACT_LOG): record the
            # contract this leaf was created WITH. Here rather than in the expansion
            # policy, because only now is the candidate a real graph node with an id and a
            # lineage to join its later `contract_resolved` verdict to. The whole block
            # (including the ancestor walk) is skipped when the flag is off. `origin` /
            # `template_id` are read off the child itself, which is what attributes a
            # library-sourced leaf's outcome back to the template that authored it (an
            # LLM-invented leaf carries neither marker and stays "organic").
            if (
                _contract_log.enabled()
                and child.details.get(DetailKey.IS_LEAF.value)
                and child.is_leaf()
            ):
                expect = child.details.get(DetailKey.EXPECT.value)
                if isinstance(expect, str) and expect.strip():
                    _contract_log.record_made(
                        graph, child,
                        origin=child.details.get(_plan_library.PLAN_LIBRARY_ORIGIN) or "organic",
                        template_id=child.details.get(_plan_library.PLAN_LIBRARY_TEMPLATE_ID),
                    )

        # Second half of the plan-library post-expand wiring: every template leaf reads a value
        # off ONE entity's page, so each library-sourced search leaf is followed through into
        # its OWN visit sibling, aimed at its OWN search results. Runs AFTER the loop above so
        # the `node.children[-max_branching:]` slice still sees exactly the expanded leaves (and
        # so each visit can inherit the parent metadata that loop just threaded onto its
        # search). Without it the subtree's only page read is the single, task-blind one
        # `GroundingEvidenceEnforcementHook` injects for the whole parent.
        if parent_detail_updates:
            visits = _plan_library.link_page_visits(graph, created_children)
            if visits:
                self._logger.info(
                    f"[STEP {step_index}] PLAN LIBRARY: followed {len(visits)} search leaf/leaves "
                    f"through into their own page visit (grounding follow-through)"
                )

        self._logger.info(f"[STEP {step_index}] EXPANSION: {len(candidates)} candidates -> {min(len(candidates), max_branching)} children (total nodes: {graph.node_count()})")
        self._record_decision(
            "expansion", node_id=node_id,
            chosen=f"{min(len(candidates), max_branching)} sub-problems",
            alternatives=[
                {"title": str(c.get("title", ""))[:80], "action": c.get("action")}
                for c in (candidates or [])[:8] if isinstance(c, dict)
            ],
            metadata={"step": step_index, "n_candidates": len(candidates or [])},
        )

        mandate = extract_mandate(graph, node_id)
        _telemetry = getattr(self.io, "telemetry", None)
        for hook in self.post_expansion_hooks:
            hook.apply(graph, node_id, step_index, mandate, self._logger, telemetry=_telemetry)

        if self._cfg.engine.semantic_dedup_visits_enabled:
            self._semantic_dedup_visits(graph, node_id, step_index)

        if self._got:
            await self._got.embed_children(graph, node_id)

        return node_id

    def _enforce_visit_nodes_for_mandate_urls(
        self, graph: IdeaDag, node_id: str, step_index: int = 0
    ) -> None:
        """Inject VISIT nodes for explicit URLs named in the mandate.

        Thin delegation onto :class:`MandateUrlInjectionHook` (the canonical home
        for this logic after the god-class breakup). Kept as an engine-level entry
        point because mandate-URL enforcement is part of the engine's test surface.
        """
        mandate = extract_mandate(graph, node_id)
        telemetry = getattr(self.io, "telemetry", None)
        MandateUrlInjectionHook().apply(
            graph, node_id, step_index, mandate, self._logger, telemetry=telemetry
        )

    def _record_decision(self, stage: str, **kwargs) -> None:
        """Proxy a decision onto the telemetry thought-process trace (best-effort)."""
        telemetry = getattr(self.io, "telemetry", None)
        rec = getattr(telemetry, "record_decision", None)
        if callable(rec):
            try:
                rec(stage=stage, **kwargs)
            except Exception:  # noqa: BLE001. Tracing must never crash a run
                pass

    def _candidate_coverage_extension(self, graph: IdeaDag, mandate: str) -> int:
        """Return a ONE-TIME, FIXED, CAPPED step-budget extension (in steps) when the
        visit-backed candidate-coverage gate is unsatisfied; 0 otherwise.

        Design (do not scale or repeat):
        * Triggered ONLY by ``evaluate_candidate_coverage``. A deterministic, visit-backed
          check that ignores node titles and search snippets and the model's own self-report
          of progress/confidence. Nothing the model *says* about itself can earn this budget.
        * FIXED size (``got_candidate_coverage_budget_extension``), never proportional to how
          incomplete coverage is, and applied AT MOST ONCE per run (tracked via
          ``_candidate_coverage_extension_applied``). A scaled or repeatable extension would
          create an incentive gradient to deliberately under-resolve early to "earn" budget; a
          fixed one-time bonus removes that gradient. Returns 0 (no extension) when the gate is
          disabled, the extension is <= 0, it was already applied this run, or coverage is
          already satisfied.
        """
        if not self._cfg.got.candidate_coverage_enabled:
            return 0
        ext = self._cfg.got.candidate_coverage_budget_extension
        if ext <= 0 or getattr(self, "_candidate_coverage_extension_applied", False):
            return 0
        try:
            cov = evaluate_candidate_coverage(graph, mandate)
        except Exception:  # noqa: BLE001. The gate must never crash a run
            return 0
        if cov.satisfied:
            return 0
        self._candidate_coverage_extension_applied = True
        # Re-activate the root so the extended budget re-expands and re-checks the missing
        # candidates (mirrors `_grounding_replan`'s forced-pass behavior).
        root = graph.get_node(graph.root_id())
        if root is not None:
            root.status = IdeaNodeStatus.ACTIVE
        return ext

    def _grounding_replan(self, graph: IdeaDag, mandate: str, steps: int, max_steps: int) -> bool:
        """Soft grounding gate. Returns True if another pass should run.

        If the mandate needs substantiation (navigation / "do not guess") and the graph is
        not grounded, inject follow-through deterministically via the enforcement hooks
        (no extra LLM expansion) and ask the run loop to continue. Bounded by
        `grounding_max_replans` (default 2) so a model that cannot navigate still finalizes
        (flagged ungrounded) rather than hanging.
        """
        telemetry = getattr(self.io, "telemetry", None)

        def _decide(chosen: str, res, **meta) -> None:
            if telemetry is not None and hasattr(telemetry, "record_decision"):
                telemetry.record_decision(
                    stage="grounding", node_id=graph.root_id(), chosen=chosen,
                    rationale=getattr(res, "reason", ""), grounded=getattr(res, "grounded", None),
                    metadata=meta,
                )

        # Deterministic candidate-coverage gate (opt-in). For "branch-eliminate" task
        # shapes (K similarly-named candidates, one distinguishing criterion), require
        # that every named candidate has been touched before finalizing. Otherwise a
        # weak model tends to elect the most familiar one and stop early. Fails open on
        # any non-enumerated mandate. Guarded so behavior is byte-identical when off.
        cov = None
        if self._cfg.got.candidate_coverage_enabled:
            try:
                cov = evaluate_candidate_coverage(graph, mandate)
            except Exception:  # noqa: BLE001. The gate must never crash a run
                cov = None

        try:
            req = parse_mandate_requirements(mandate)
        except Exception:
            req = None
        cov_unsatisfied = cov is not None and not cov.satisfied

        if req is None or not req.needs_substantiation:
            if not cov_unsatisfied:
                return False
            # Coverage gate wants another pass even without substantiation needs; use a
            # placeholder grounding result for telemetry/budget bookkeeping below.
            res = evaluate_grounding(graph, req) if req is not None else None
        else:
            res = evaluate_grounding(graph, req)
            if res.grounded and not cov_unsatisfied:
                _decide("grounded", res, distinct_visits=res.distinct_visits)
                return False

        res_missing = list(getattr(res, "missing", []) or [])
        res_reason = getattr(res, "reason", "")
        if cov_unsatisfied:
            res_missing = res_missing + [f"candidate not checked: {c}" for c in cov.missing]
            cov_reason = "candidate coverage incomplete: " + ", ".join(cov.missing)
            res_reason = f"{res_reason}; {cov_reason}" if res_reason else cov_reason

        replans = getattr(self, "_grounding_replans", 0)
        cap = self._cfg.engine.grounding_max_replans
        if replans >= cap or steps >= max_steps:
            _decide("ungrounded-finalize", res, replans=replans, missing=res_missing)
            return False

        root_id = graph.root_id()
        before = graph.node_count()
        for hook in self.post_expansion_hooks:
            try:
                hook.apply(graph, root_id, steps, mandate, self._logger, telemetry=telemetry)
            except Exception as exc:  # noqa: BLE001. A hook must never crash a run
                self._logger.warning(f"[GROUNDING] hook failed: {exc}")
        injected = graph.node_count() - before
        # No follow-through injected: normally we finalize (nothing would change). But an
        # unsatisfied coverage gate still forces a pass. Re-activating the root lets the
        # executor re-expand and check the remaining candidates. Bounded by the replan cap.
        if injected <= 0 and not cov_unsatisfied:
            _decide("ungrounded-no-followup", res, replans=replans)
            return False

        self._grounding_replans = replans + 1
        root = graph.get_node(root_id)
        if root:
            root.status = IdeaNodeStatus.ACTIVE
        _decide("replan", res, attempt=self._grounding_replans, injected=injected,
                missing=res_missing)
        self._logger.info(
            f"[GROUNDING] Re-plan {self._grounding_replans}/{cap}: injected {injected} "
            f"follow-through node(s) ({res_reason})"
        )
        return True

    def _action_timeout_for(self, action_name: Optional[str]) -> float:
        """Resolve the timeout for an action by name.

        Prefers `{action}_timeout_seconds` (e.g. `visit_timeout_seconds=20`,
        `search_timeout_seconds=15`); falls back to `action_timeout_seconds`
        (default 120).
        """
        fallback = float(self._cfg.timeouts.action)
        if not action_name:
            return fallback
        per_type = getattr(self._cfg.timeouts, action_name, None)
        if per_type is None:
            return fallback
        try:
            return float(per_type)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _url_slug_tokens(url: str) -> List[str]:
        """Pull the trailing path slug from a URL and return its lowercased tokens."""
        return idea_visit_dedup.url_slug_tokens(url)

    def _semantic_dedup_visits(self, graph: IdeaDag, parent_id: str, step_index: int) -> None:
        idea_visit_dedup.semantic_dedup_visits(
            graph,
            parent_id,
            step_index,
            bool(self._cfg.engine.semantic_dedup_require_hook_source),
            self._logger,
        )

    async def _handle_merge_creation(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        if not self.merge.should_create_merge_node(graph, node_id):
            return node_id
        
        node = graph.get_node(node_id)
        
        merge_node_id = self.merge.create_merge_node(graph, node_id)
        if merge_node_id:
            self._logger.info(f"[STEP {step_index}] MERGE CREATED: {merge_node_id} for parent {node_id}")
            merge_node = graph.get_node(merge_node_id)
            if merge_node and node:
                original_goal = (
                    node.details.get(DetailKey.GOAL.value)
                    or node.details.get(DetailKey.ORIGINAL_GOAL.value)
                    or node.title
                )
                merge_node.details[DetailKey.GOAL.value] = original_goal
                merge_node.details[DetailKey.ORIGINAL_GOAL.value] = original_goal
                
            if merge_node and self._is_action_ready(merge_node, step_index):
                if merge_node.details.get("merge_should_skip", False):
                    self._logger.warning(f"[STEP {step_index}] MERGE SKIPPED: Goal not achieved for node {merge_node_id}")
                    merge_node.status = IdeaNodeStatus.SKIPPED
                    merge_node.details["merge_skipped_reason"] = "Goal not achieved according to evaluation"
                    return node_id
                
                result = await self._execute_action_guarded(graph, node_id, merge_node_id)
                if result is not None:
                    await self._apply_action_result(graph, merge_node_id, step_index)
                
                goal_achieved = merge_node.details.get(DetailKey.GOAL_ACHIEVED.value, False)
                if not goal_achieved and merge_node.details.get("merge_should_skip", False):
                    self._logger.warning(f"[STEP {step_index}] MERGE INCOMPLETE: Goal not achieved, marking as incomplete")
                    merge_node.status = IdeaNodeStatus.SKIPPED
                    merge_node.details["merge_skipped_reason"] = merge_node.details.get("goal_evaluation", "Goal not achieved")
                    return node_id
                
                if merge_node.status == IdeaNodeStatus.DONE:
                    completion_path = get_completion_path(graph, merge_node_id)
                    if len(completion_path) > 1:
                        next_id = completion_path[1]
                        self._logger.info(f"[STEP {step_index}] MERGE COMPLETE: Progressing toward completion via {next_id}")
                        return next_id
        
        return merge_node_id or node_id
    
    async def _handle_intermediate_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node or not node.children:
            self._logger.warning(f"[STEP {step_index}] No children to evaluate, stopping")
            return None

        eligible = []
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                continue
            if not self._is_action_ready(child, step_index):
                continue
            eligible.append(child_id)

        if not eligible:
            self._logger.warning(f"[STEP {step_index}] No eligible children")
            return None

        meta = node.details.get(DetailKey.EXPANSION_META.value) or {}
        execute_all = bool(meta.get(DetailKey.EXECUTE_ALL_CHILDREN.value)) if isinstance(meta, dict) else False

        has_dependencies = self._detect_state_dependencies(graph, eligible)
        has_chunk_dependencies = self._detect_chunk_dependencies(graph, eligible)

        if has_dependencies or has_chunk_dependencies:
            self._logger.info(f"[STEP {step_index}] DEPENDENCY DETECTED: Forcing sequential execution")
            execute_all = False

        chunk_nodes = [nid for nid in eligible if self._is_chunk_node(graph, nid)]
        if chunk_nodes and not has_dependencies:
            execute_all = True

        # Auto-parallelize independent leaf-action siblings instead of executing
        # the single best child one-per-step. Only fires when EVERY eligible
        # child is an executable action leaf (has a non-merge action, is ready,
        # not yet executed, and has no children of its own) so we never skip a
        # node that still needs expansion. Gated by auto_parallel_siblings.
        if (
            not execute_all
            and self._cfg.engine.auto_parallel_siblings
            and not has_dependencies
            and len(eligible) > 1
        ):
            all_executable_leaves = all(
                (c := graph.get_node(cid)) is not None
                and not c.children
                and NodeDetailsExtractor.get_action(c.details)
                and not NodeDetailsExtractor.is_merge_action(c.details)
                and c.details.get(DetailKey.ACTION_RESULT.value) is None
                and self._is_action_ready(c, step_index)
                for cid in eligible
            )
            if all_executable_leaves:
                # `parallel_requires_evidence` off (default): unchanged -- every
                # executable-leaf batch is trusted as independent, exactly as before. On:
                # require idea_sequencing.siblings_are_independent's positive evidence (a
                # deterministic, no-LLM check) instead of trusting the heuristic alone.
                independent = True
                reason: Optional[str] = None
                if self._cfg.engine.parallel_requires_evidence:
                    independent, reason = idea_sequencing.siblings_are_independent(
                        graph, eligible, node, self._logger,
                    )
                if independent:
                    self._logger.info(
                        f"[STEP {step_index}] AUTO-PARALLEL: {len(eligible)} independent "
                        f"leaf siblings, executing concurrently"
                    )
                    execute_all = True
                elif reason is not None:
                    self._logger.info(
                        f"[STEP {step_index}] AUTO-PARALLEL SKIPPED ({reason}): "
                        f"{len(eligible)} leaf siblings not batched"
                    )

        if execute_all and self._cfg.engine.allow_execute_all_children and not has_dependencies:
            ready_children = [
                cid for cid in eligible
                if (c := graph.get_node(cid)) and self._is_action_ready(c, step_index)
            ]
            if not ready_children:
                return node_id

            # `defer_unresolved_slots` off (default): unchanged. On: drop any candidate
            # whose details still carry a literal unfilled placeholder from THIS step's
            # batch -- its siblings run, and it becomes eligible again next step. Never
            # drops every candidate (see defer_unresolved_slot_candidates's progress
            # guarantee).
            if self._cfg.engine.defer_unresolved_slots:
                ready_children = idea_sequencing.defer_unresolved_slot_candidates(
                    graph, ready_children, step_index, self._logger,
                )
                if not ready_children:
                    return node_id

            parallel_limit = max(1, self._cfg.engine.parallel_action_limit)
            self._logger.info(
                f"[STEP {step_index}] PARALLEL: Executing {len(ready_children)} children "
                f"(limit={parallel_limit}, skipping evaluation)"
            )
            semaphore = asyncio.Semaphore(parallel_limit)

            async def _run_one(cid: str) -> Optional[Dict[str, Any]]:
                child = graph.get_node(cid)
                action_name = NodeDetailsExtractor.get_action(child.details) if child else None
                timeout_s = self._action_timeout_for(action_name)
                async with semaphore:
                    return await asyncio.wait_for(
                        self._execute_action(graph, node_id, cid),
                        timeout=timeout_s,
                    )

            results = await asyncio.gather(
                *[_run_one(cid) for cid in ready_children],
                return_exceptions=True,
            )

            self._parallel_leaves_total = getattr(self, "_parallel_leaves_total", 0) + len(ready_children)

            done_children: List[str] = []
            for cid, res in zip(ready_children, results):
                if isinstance(res, asyncio.TimeoutError):
                    child = graph.get_node(cid)
                    action_name = NodeDetailsExtractor.get_action(child.details) if child else None
                    timeout_used = self._action_timeout_for(action_name)
                    self._logger.warning(
                        f"[STEP {step_index}] Action timed out after {timeout_used}s "
                        f"(node={cid}, action={action_name})"
                    )
                    if child:
                        child.status = IdeaNodeStatus.FAILED
                        child.details["action_error"] = f"timeout after {timeout_used}s"
                    continue
                if isinstance(res, Exception):
                    self._logger.warning(
                        f"[STEP {step_index}] Action raised (node={cid}): {type(res).__name__}: {res}"
                    )
                    child = graph.get_node(cid)
                    if child:
                        child.status = IdeaNodeStatus.FAILED
                        child.details["action_error"] = f"{type(res).__name__}: {res}"
                    continue
                if res is not None:
                    # Shared completion point: record the result's status bookkeeping
                    # synchronously (no awaits here, so status writes can't interleave).
                    # The bounded re-expansion follow-up check is deferred and run
                    # concurrently across siblings below.
                    self._handle_action_result(graph, cid, step_index)
                    node = graph.get_node(cid)
                    if node and node.status == IdeaNodeStatus.DONE:
                        done_children.append(cid)

            # Opt-in decorrelated per-step confidence judge for the batch-completed siblings
            # (the auto-parallel path bypasses `_apply_action_result`, so wire it here too.
            # the same coverage requirement the re-expansion check has). No-op when the flag
            # is off; per-sibling judge calls run concurrently inside the batch helper.
            if done_children:
                # On-demand plan library for the batch-completed siblings: this path bypasses
                # `_apply_action_result`, so without this an adopted template retrieved by a
                # parallel sibling would never become children. Sequential (never inside a
                # gather): each expansion re-reads the shared `max_total_nodes` ceiling.
                for cid in done_children:
                    await self._maybe_plan_library_reexpand(graph, cid, step_index)
                confidences = await self._maybe_judge_step_confidence_batch(graph, done_children, step_index)
                # F33 contract->action loop for the batch-completed siblings (no-op unless
                # got.contract_reexpand_enabled). Free and deterministic, so it runs for every
                # completed sibling regardless of whether the judge sampled it.
                await self._maybe_contract_reexpand_batch(graph, done_children, step_index)
                # Confidence->action loop for the batch-completed siblings (no-op unless
                # got.step_confidence_reexpand_enabled). Applied sequentially so the shared
                # max_total_nodes ceiling can't be overshot by concurrent siblings. Runs
                # before the follow-up re-expansion block below, whose read-only
                # `_reexpand_check` then skips any sibling already re-expanded here.
                await self._maybe_confidence_reexpand_batch(graph, confidences, step_index)

            # The per-sibling re-expansion checks are independent LLM calls, so run
            # them concurrently (mirrors the action-execution gather above) instead of
            # one blocking await at a time. Mutation-free `_reexpand_check` is safe to
            # gather; the graph-growing `_apply_reexpand` is applied SEQUENTIALLY after
            # so the shared `max_total_nodes` ceiling can't be overshot by concurrent
            # siblings (each apply re-reads the ceiling before expanding).
            if done_children and self._cfg.got.reexpand_enabled and self._got:
                verdicts = await asyncio.gather(
                    *[self._reexpand_check(graph, cid, step_index) for cid in done_children],
                    return_exceptions=True,
                )
                for cid, verdict in zip(done_children, verdicts):
                    if isinstance(verdict, Exception):
                        self._logger.warning(
                            f"[STEP {step_index}] Re-expansion check raised (node={cid}): "
                            f"{type(verdict).__name__}: {verdict}"
                        )
                        continue
                    if verdict:
                        await self._apply_reexpand(graph, cid, step_index, verdict)
            return node_id

        needs_evaluation = any(
            (child := graph.get_node(child_id)) and child.score is None
            for child_id in eligible
        )

        if needs_evaluation:
            self._logger.debug(f"[STEP {step_index}] Evaluating {len(eligible)} children")
            if hasattr(self.evaluation, "evaluate_batch"):
                await self.evaluation.evaluate_batch(graph, node_id, list(eligible))
            else:
                for child_id in eligible:
                    await self.evaluation.evaluate(graph, child_id)

        min_score = self._cfg.engine.min_score_threshold
        allow_unscored = self._cfg.engine.allow_unscored_selection
        scored_eligible = []
        for child_id in eligible:
            child = graph.get_node(child_id)
            if not child:
                continue
            if NodeDetailsExtractor.is_merge_action(child.details):
                scored_eligible.append(child_id)
                continue
            if child.status == IdeaNodeStatus.BLOCKED:
                scored_eligible.append(child_id)
                continue
            if child.score is None and not allow_unscored:
                continue
            if child.score is not None and child.score < min_score:
                continue
            scored_eligible.append(child_id)

        if not scored_eligible:
            self._logger.warning(f"[STEP {step_index}] No eligible children after evaluation")
            return None

        self._logger.debug(f"[STEP {step_index}] Found {len(scored_eligible)} eligible children")
        original_children = list(node.children)
        node.children = scored_eligible
        try:
            if self._cfg.engine.best_first_global:
                selected, parent_id = self._select_best_global(graph, min_score, allow_unscored)
            else:
                selected = self.selection.select(graph, node_id)
                parent_id = node_id
        finally:
            node.children = original_children

        if selected is None:
            self._logger.warning(f"[STEP {step_index}] Selection returned None")
            return None

        best_to_execute = self._reorder_for_sequential(graph, selected, scored_eligible, step_index)
        if best_to_execute and best_to_execute.node_id != selected.node_id:
            self._logger.info(f"[STEP {step_index}] SEQUENTIAL REORDER: {selected.title[:40]}... -> {best_to_execute.title[:40]}...")
            selected = best_to_execute

        self._logger.info(f"[STEP {step_index}] SEQUENTIAL: Executing best child: {selected.title[:50]}...")
        self._record_decision(
            "selection", node_id=selected.node_id, chosen=selected.title,
            score=(selected.details.get(DetailKey.EVALUATION.value) or {}).get("score"),
            rationale=(selected.details.get(DetailKey.EVALUATION.value) or {}).get("rationale", ""),
            metadata={"step": step_index, "action": NodeDetailsExtractor.get_action(selected.details)},
        )
        result = await self._execute_action_guarded(graph, parent_id or node_id, selected.node_id)
        if result is not None:
            # Shared completion point: the sequential-dependency branch now gets
            # the same bounded re-expansion check as every other execution path.
            await self._apply_action_result(graph, selected.node_id, step_index)

        if self._cfg.engine.sequential_prune_siblings:
            for cid in scored_eligible:
                if cid == selected.node_id:
                    continue
                sibling = graph.get_node(cid)
                if sibling and sibling.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    sibling.status = IdeaNodeStatus.SKIPPED
                    # Fix #8: mark so sibling recovery can find these later.
                    sibling.details["__sequential_pruned"] = True
                    sibling.details["__sequential_pruned_parent"] = node_id
            return selected.node_id

        return node_id

    async def _execute_action_guarded(
        self, graph: IdeaDag, parent_id: str, node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Run ``_execute_action`` under the per-action timeout watchdog.

        The per-type action cap (``_action_timeout_for``, default 20s) must bound EVERY
        routing path, not only the auto-parallel gather. Which already wraps its calls
        in ``asyncio.wait_for`` (see ``_run_one``). Leaf, merge, and single-best-child
        selection previously called ``_execute_action`` bare, so on those paths (the
        default engine variant, with auto-parallel off) a slow action (e.g. a ``visit``
        fanning out to many URLs on slow sites, times the in-place tool-retry loop). Could
        wedge the whole cell, since there is no run-level watchdog to catch it. On expiry
        we mark the node FAILED (so the step loop does not re-execute it) and return None,
        matching how the auto-parallel path degrades a timed-out child.
        """
        node = graph.get_node(node_id)
        action_name = NodeDetailsExtractor.get_action(node.details) if node else None
        timeout_s = self._action_timeout_for(action_name)
        try:
            return await asyncio.wait_for(
                self._execute_action(graph, parent_id, node_id), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                f"[ACTION] timed out after {timeout_s}s (node={node_id}, action={action_name})"
            )
            timed_out = graph.get_node(node_id)
            if timed_out:
                timed_out.status = IdeaNodeStatus.FAILED
                timed_out.details["action_error"] = f"timeout after {timeout_s}s"
            return None

    async def _execute_action(self, graph: IdeaDag, parent_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        node = graph.get_node(node_id)
        if not node:
            return None
        action_type = node.details.get(DetailKey.ACTION.value)
        if not action_type:
            return None
        
        existing_node_id = graph.has_executed_action(str(action_type), node.details)
        if existing_node_id and existing_node_id != node_id:
            existing_node = graph.get_node(existing_node_id)
            if existing_node and existing_node.status == IdeaNodeStatus.DONE:
                self._logger.info(f"[ACTION] Skipping duplicate action (already executed by node {existing_node_id})")
                existing_result = existing_node.details.get(DetailKey.ACTION_RESULT.value)
                if existing_result:
                    graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: existing_result})
                    node.status = IdeaNodeStatus.DONE
                    return existing_result
        
        from agent.app.idea_policies.base import IdeaActionType
        if action_type == IdeaActionType.VISIT.value:
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            url = NodeDetailsExtractor.get_url(node.details)
            if url:
                blocked_reason = graph.is_site_blocked(str(url))
                if blocked_reason:
                    self._logger.warning(f"[ACTION] Site blocked, skipping: {url} - {blocked_reason}")
                    result = {
                        "action": action_type,
                        "success": False,
                        "url": url,
                        "error": f"Site blocked: {blocked_reason}",
                        "retryable": False,
                    }
                    graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: result})
                    node.status = IdeaNodeStatus.FAILED
                    return result
        
        # Resolve by NAME through the registry, not by coercing into the IdeaActionType enum.
        # LeafActionRegistry.register()/install_pack() let a caller register a custom action
        # under a name that was never added to the enum, and this is the one place that
        # decides whether such a name is actually reachable from the model's own selection.
        # Coercing through IdeaActionType(str(action_type)) here would raise for any such name
        # and silently fall back to THINK, defeating the registry's own advertised extension
        # point (idea_policies/extra_actions/pack.py's bundled actions hit exactly this).
        action_name = str(action_type)
        allowed = self.settings.get("allowed_actions") or [a.value for a in IdeaActionType]
        try:
            if action_name not in [str(item) for item in allowed]:
                action_name = IdeaActionType.THINK.value
            action = self.actions.get(action_name)
        except (ValueError, KeyError) as exc:
            self._logger.warning(
                "Unknown action_type=%r, defaulting to THINK: %s", action_type, exc,
            )
            action_name = IdeaActionType.THINK.value
            action = self.actions.get(action_name)
        attempts = int(node.details.get(DetailKey.ACTION_ATTEMPTS.value, 0)) + 1
        max_retries = self._cfg.action.max_retries
        graph.update_details(
            node_id,
            {
                DetailKey.ACTION_ATTEMPTS.value: attempts,
                DetailKey.ACTION_MAX_RETRIES.value: max_retries,
            },
        )
        node.status = IdeaNodeStatus.ACTIVE
        result = await action.execute(graph, node_id, self.io)
        # C1a: opt-in bounded in-place retry of a TOOL failure (empty/timeout/HTTP-error fetch,
        # no search results) so a transient failure recovers at the source instead of surfacing
        # as empty grounding. No-op unless `connector_retry_on_failure_enabled`.
        result = await self._maybe_retry_tool_failure(graph, node_id, action, result)
        sanitized_result = self._sanitize_action_result(result) if result else None
        graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: sanitized_result})
        self._record_decision(
            "action", node_id=node_id, chosen=action_name,
            metadata={
                "success": bool(result.get("success")) if isinstance(result, dict) else False,
                "url": (result.get("url") if isinstance(result, dict) else None),
                "link_idea": node.details.get("link_idea"),
            },
        )
        
        if self._memory_manager and result:
            await self._memory_manager.write_node_result(
                node_id=node_id,
                node_title=node.title,
                action_type=str(action_type),
                result=result,
            )
        
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if result and ActionResultExtractor.is_success(result):
            graph.mark_action_executed(node_id, str(action_type), node.details)

        return result

    async def _maybe_retry_tool_failure(self, graph: IdeaDag, node_id: str, action, result):
        """Bounded in-place retry of a TOOL failure (opt-in; no-op unless
        ``connector_retry_on_failure_enabled``).

        A tool failure is a fetch/search that returned no usable evidence. A timeout, an
        HTTP error, or an EMPTY payload (no search results / no extractable content). The
        re-expansion loop is the wrong response to a *transient* one: a fresh subtree just
        repeats the same failing call. Here we simply re-run the SAME action, up to
        ``connector_retry_max_attempts`` times with a small growing backoff, returning the
        first non-failure result (or the last attempt's result if all fail). Fully bounded,
        so it can never hang or loop; permanent failures (``retryable=False`` and not a
        success-with-empty) are not retried. One exception to "same action, same query": an
        empty SEARCH whose query bundles multiple quoted entity names gets OR-joined before
        the retry (see ``_reformulate_search_query_if_multi_entity``). Resending the
        identical AND-shaped query would just fail again.
        """
        if not self._cfg.action.connector_retry_on_failure_enabled:
            return result
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        max_attempts = max(0, int(self._cfg.action.connector_retry_max_attempts))
        backoff = max(0.0, float(self._cfg.action.connector_retry_backoff_seconds))
        attempt = 0
        while (
            attempt < max_attempts
            and self._should_retry_tool_failure(result)
        ):
            attempt += 1
            if backoff > 0:
                await asyncio.sleep(backoff * attempt)
            self._reformulate_search_query_if_multi_entity(graph, node_id, result)
            self._logger.info(
                f"[TOOL-RETRY] node {node_id[:8]} tool_failure; retrying same action "
                f"(attempt {attempt}/{max_attempts})"
            )
            retry_result = await action.execute(graph, node_id, self.io)
            if retry_result is not None:
                result = retry_result
            if not ActionResultExtractor.is_tool_failure(result):
                break
        return result

    def _should_retry_tool_failure(self, result) -> bool:
        """Whether a tool-failure result is worth a bounded in-place retry.

        Retry a transient-ish failure: a success-with-EMPTY payload (search/visit returned
        nothing) OR an explicit failure flagged ``retryable`` (timeout / 5xx / network /
        empty-content). Do NOT retry a permanent failure (``retryable=False``, e.g. HTTP
        403/404/401). Re-running it just burns the budget.
        """
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if not ActionResultExtractor.is_tool_failure(result):
            return False
        if not isinstance(result, dict):
            return True
        if result.get("success"):
            return True  # success-with-empty payload: worth a transient retry
        return ActionResultExtractor.is_retryable(result)

    def _reformulate_search_query_if_multi_entity(self, graph: IdeaDag, node_id: str, result) -> None:
        """Before retrying an empty SEARCH, OR-join a multi-quoted-entity query in place.

        No-op for non-search actions, non-empty results, or a query that isn't multi-entity
        shaped (``_reformulate_multi_entity_query`` returns ``None``). Mutates the node's
        stored query via ``graph.update_details`` so the next ``action.execute`` call in the
        retry loop reads the reformulated query fresh (``SearchLeafAction`` re-fetches the
        node from the graph on every call, it never caches).
        """
        from agent.app.idea_policies.action_constants import ActionResultKey
        if not isinstance(result, dict):
            return
        if result.get(ActionResultKey.ACTION.value) != IdeaActionType.SEARCH.value:
            return
        if result.get(ActionResultKey.RESULTS.value):
            return  # not empty, nothing to reformulate
        node = graph.get_node(node_id)
        if node is None:
            return
        query = NodeDetailsExtractor.get_query(node.details, fallback_title=node.title)
        reformulated = _reformulate_multi_entity_query(query)
        if reformulated:
            self._logger.info(
                f"[TOOL-RETRY] node {node_id[:8]} reformulating multi-entity query: "
                f"{query!r} -> {reformulated!r}"
            )
            graph.update_details(node_id, {DetailKey.QUERY.value: reformulated})

    async def _apply_action_result(self, graph: IdeaDag, node_id: str, step_index: int) -> str:
        """Single shared completion point for every executed action.

        All execution paths (`_handle_leaf_node`, the auto-parallel batch loop,
        the sequential-dependency branch, and the two merge paths) route a
        successful `_execute_action` through here rather than calling
        `_handle_action_result` directly. This records the result's status
        bookkeeping and (when the node lands DONE) offers it the bounded
        re-expansion follow-up check. Centralizing here means the re-expansion
        escape hatch fires regardless of which routing branch drove the node to
        completion, with no per-branch duplication. `_maybe_reexpand_leaf` is a
        no-op unless `got.reexpand_enabled` is set and the node is a genuine
        successful leaf (merge nodes and nodes with children are gated out),
        so flag-off behavior is byte-identical to calling `_handle_action_result`.
        """
        status = self._handle_action_result(graph, node_id, step_index)
        node = graph.get_node(node_id)
        if node and node.status == IdeaNodeStatus.DONE:
            # Contract log (env-gated, no-op unless IDEA_TEST_CONTRACT_LOG): what the
            # deterministic contract check made of this completed leaf. Calls
            # `_evaluate_contract` rather than `_contract_verdict` on purpose. The log
            # audits the contract MECHANISM, so it must observe every completion, including
            # on runs where the re-expansion lever it feeds is switched off.
            if _contract_log.enabled():
                _contract_log.record_resolved(
                    graph, node, self._evaluate_contract(graph, node_id),
                    step_index=step_index,
                )
            # On-demand plan library: an ADOPTED template search becomes real children here
            # (no-op unless both plan-library flags are armed and this node IS such a search).
            # Runs FIRST among the re-expansion triggers: a node it expands then has children,
            # which gates it out of all three below. The retrieved plan wins over an invented
            # follow-up, which is the whole point of having asked for one.
            await self._maybe_plan_library_reexpand(graph, node_id, step_index)
            confidences = await self._maybe_judge_step_confidence(graph, node_id, step_index)
            # F33 contract->action loop: a leaf that did not deliver its contract's required
            # payload/datum/subject re-expands (no-op unless got.contract_reexpand_enabled).
            # Runs FIRST because it is free and better-targeted than the judge; a node it
            # re-expands is then gated out of both paths below (they skip nodes with children).
            await self._maybe_contract_reexpand_batch(graph, [node_id], step_index)
            # Confidence->action loop: a low step-confidence can itself drive re-expansion
            # (no-op unless got.step_confidence_reexpand_enabled). Runs before the follow-up
            # path so a node re-expanded here is gated out of `_maybe_reexpand_leaf`
            # (which skips nodes that already have children).
            await self._maybe_confidence_reexpand_batch(graph, confidences, step_index)
            await self._maybe_reexpand_leaf(graph, node_id, step_index)
        return status

    async def _maybe_judge_step_confidence(
        self, graph: IdeaDag, node_id: str, step_index: int
    ) -> List[Tuple[str, float]]:
        """Single-leaf wrapper over the batch judge (sequential/merge completion paths)."""
        return await self._maybe_judge_step_confidence_batch(graph, [node_id], step_index)

    async def _maybe_judge_step_confidence_batch(
        self, graph: IdeaDag, node_ids: List[str], step_index: int
    ) -> List[Tuple[str, float]]:
        """Emit an opt-in, decorrelated per-step confidence signal for completed leaves.

        No-op unless ``got.step_confidence_judge_enabled`` is set (JSON default False), so
        the default path is byte-identical. Fires once per leaf completion across every
        routing branch (the sequential completion point and the auto-parallel batch).
        ``sample_every`` subsamples (judge every Nth completion) to bound the added call
        volume; the counter advances deterministically in the given order so subsampling is
        reproducible regardless of concurrency. The judge (``GoTOperations.judge_step_confidence``)
        sees only trajectory-visible context (never the grep validators or ground truth). So the
        logged sequence is a genuine E-valuator substrate, decorrelated from the final
        grep-computed label. The per-leaf judge calls run concurrently; this never raises.

        Returns the ``(node_id, confidence)`` pairs actually judged this call so a caller
        can close the confidence->action loop (drive re-expansion off a low score). The
        return value is ignored on the default path.
        """
        judged: List[Tuple[str, float]] = []
        if not self._cfg.got.step_confidence_judge_enabled or not self._got:
            return judged
        sample_every = max(1, int(self._cfg.got.step_confidence_judge_sample_every))
        selected: List[str] = []
        for node_id in node_ids:
            self._leaf_completion_count += 1
            if (self._leaf_completion_count - 1) % sample_every == 0:
                selected.append(node_id)
        if not selected:
            return judged
        verdicts = await asyncio.gather(
            *[self._got.judge_step_confidence(graph, nid, model_name=self.model_name) for nid in selected],
            return_exceptions=True,
        )
        for nid, verdict in zip(selected, verdicts):
            if isinstance(verdict, Exception):
                self._logger.warning(f"[STEP {step_index}] step-confidence judge failed (node={nid}): {verdict}")
                continue
            if not verdict:
                continue
            node = graph.get_node(nid)
            confidence = verdict.get("confidence")
            self._step_confidences.append({
                "step": step_index,
                "node_id": nid,
                "kind": NodeDetailsExtractor.get_action(node.details) if node else None,
                "confidence": confidence,
                "reason": verdict.get("reason", ""),
            })
            if confidence is not None:
                try:
                    judged.append((nid, float(confidence)))
                except (TypeError, ValueError):
                    pass
        return judged

    def _reexpand_guards_ok(self, graph: IdeaDag, node_id: str) -> bool:
        """The trigger-agnostic guards every re-expansion path shares.

        A candidate must be an existing, childless, non-merge leaf with a SUCCESSFUL action
        result, inside both the per-lineage ``reexpand_max_iterations`` budget and the global
        ``max_total_nodes`` ceiling. So termination is guaranteed identically no matter which
        trigger (step-confidence, contract satisfaction, or the follow-up detector) fired.
        No LLM call, no mutation.
        """
        node = graph.get_node(node_id)
        if not node:
            return False
        if NodeDetailsExtractor.is_merge_action(node.details):
            return False
        if node.children:
            return False
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return False
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if not ActionResultExtractor.is_success(result):
            return False
        # C1a routing: if the leaf's "success" is actually a TOOL failure (the fetch/search
        # returned no usable evidence), do NOT re-expand. A fresh subtree would just repeat
        # the failing fetch. The bounded in-place connector retry is the right recovery for
        # that; re-expansion is reserved for a page that genuinely loaded but lacks the answer.
        # No-op unless `tool_failure_recovery_enabled` (default off -> byte-identical).
        if (
            self._cfg.action.tool_failure_recovery_enabled
            and ActionResultExtractor.is_tool_failure(result)
        ):
            return False
        max_iters = self._effective_reexpand_max_iterations()
        count = int(node.details.get("_got_reexpand_count", 0))
        if count >= max_iters:
            return False
        if graph.node_count() >= self._cfg.engine.max_total_nodes:
            return False
        return True

    def _contract_verdict(self, graph: IdeaDag, node_id: str):
        """F33: deterministic contract-satisfaction verdict for a completed leaf.

        Returns ``None`` when the lever is off or the check has NO verdict (the leaf
        retrieved nothing checkable, or its contract text yields neither a measurable datum
        nor an identity token). Callers then fall back to the previous signal. Never raises
        and never mutates; costs no LLM call.

        The lever gate is HERE and the computation is in ``_evaluate_contract`` so the
        contract LOG can observe the verdict independently of whether the re-expansion
        mechanism is switched on. Every caller of this method sees identical behavior.
        """
        if not self._cfg.got.contract_reexpand_enabled:
            return None
        return self._evaluate_contract(graph, node_id)

    def _evaluate_contract(self, graph: IdeaDag, node_id: str):
        """The contract-satisfaction computation itself, ungated by any lever.

        Same contract as ``_contract_verdict`` minus the ``got.contract_reexpand_enabled``
        check: ``None`` for a missing node, a failed check, or NO verdict; otherwise the
        applicable ``ContractSatisfaction``. Callers that drive BEHAVIOR must go through
        ``_contract_verdict``; only pure observers call this directly.
        """
        node = graph.get_node(node_id)
        if not node:
            return None
        root = graph.get_node(graph.root_id())
        mandate = ""
        if root and isinstance(root.details, dict):
            mandate = str(root.details.get("mandate") or "")
        try:
            from agent.app.idea_policies.contract_satisfaction import evaluate_step_contract
            verdict = evaluate_step_contract(node, mandate)
        except Exception as exc:  # noqa: BLE001. A gate must never crash a run
            self._logger.warning(f"[CONTRACT] check failed (node={node_id}): {exc}")
            return None
        return verdict if verdict.applicable else None

    def _confidence_triggers_reexpand(
        self, graph: IdeaDag, node_id: str, confidence: float
    ) -> bool:
        """Gate: does a completed leaf's *low* step-confidence justify re-expanding it?

        This is the confidence->action loop. Unlike ``_reexpand_check`` (which asks a
        separate follow-up-detector LLM whether the result *reveals* a new target), this
        trigger is a genuine "observe the step's trustworthiness, then decide to take
        another step": a leaf the step-judge distrusts (confidence below
        ``got.step_confidence_reexpand_threshold``) is re-investigated by re-expanding it
        into follow-up sub-problems. Reuses ``_reexpand_guards_ok``, so it terminates on
        exactly the same bounds as every other trigger. No LLM call, no mutation.

        F33 corrective (no-op unless ``got.contract_reexpand_enabled``): the judge's number
        is anti-calibrated. A coherent WRONG page scores high and a correct-but-partial page
        scores low. So it may not overrule a leaf whose CONTRACT is demonstrably satisfied
        (the required payload/datum/subject is present in the retrieved evidence). Where the
        contract check has no verdict, the judge still decides, exactly as before.
        """
        if not self._cfg.got.step_confidence_reexpand_enabled:
            return False
        if confidence is None or confidence >= self._cfg.got.step_confidence_reexpand_threshold:
            return False
        if not self._reexpand_guards_ok(graph, node_id):
            return False
        verdict = self._contract_verdict(graph, node_id)
        if verdict is not None and verdict.satisfied:
            self._logger.info(
                f"[CONTRACT] leaf {node_id[:8]} satisfies its contract; ignoring the "
                f"low step-confidence {confidence:.3f} (anti-calibrated trigger suppressed)"
            )
            return False
        return True

    async def _maybe_confidence_reexpand_batch(
        self, graph: IdeaDag, confidences: List[Tuple[str, float]], step_index: int
    ) -> None:
        """Drive bounded re-expansion off low per-step confidence scores.

        No-op unless ``got.step_confidence_reexpand_enabled`` is set (JSON default False),
        so flag-off behavior is byte-identical. Applied SEQUENTIALLY (never inside a
        gather): each ``_apply_reexpand`` re-reads the ``max_total_nodes`` ceiling before
        growing the graph, so concurrent siblings can never overshoot it. Mirroring the
        follow-up re-expansion path. Independent of ``got.reexpand_enabled``: the shared
        ``_apply_reexpand`` machinery is trigger-agnostic, so a run can drive re-expansion
        from confidence alone, from the follow-up detector, or from both.
        """
        if not self._cfg.got.step_confidence_reexpand_enabled or not self._got:
            return
        threshold = self._cfg.got.step_confidence_reexpand_threshold
        for node_id, confidence in confidences:
            if not self._confidence_triggers_reexpand(graph, node_id, confidence):
                continue
            verdict = {
                "needs_followup": True,
                "reason": (
                    f"low step confidence {confidence:.3f} < {threshold:.3f}; "
                    f"re-investigating the distrusted leaf"
                ),
            }
            await self._apply_reexpand(graph, node_id, step_index, verdict)

    async def _maybe_contract_reexpand_batch(
        self, graph: IdeaDag, node_ids: List[str], step_index: int
    ) -> None:
        """F33: drive bounded re-expansion off CONTRACT SATISFACTION, not the judge score.

        No-op unless ``got.contract_reexpand_enabled`` is set (JSON default False), so
        flag-off behavior is byte-identical. A completed retrieval leaf whose deterministic
        contract check reports a missing payload / required datum / subject re-expands: the
        step did not deliver what the task required, which is a signal about task PROGRESS
        rather than about how confident a judge feels. Unlike the confidence loop this needs
        no ``GoTOperations`` and no judge LLM call (it is free). So it fires on every
        completed leaf, including runs with the step-confidence judge off entirely. Applied
        SEQUENTIALLY (never inside a gather): each ``_apply_reexpand`` re-reads the
        ``max_total_nodes`` ceiling before growing the graph.
        """
        if not self._cfg.got.contract_reexpand_enabled:
            return
        for node_id in node_ids:
            verdict = self._contract_verdict(graph, node_id)
            if verdict is None or verdict.satisfied:
                continue
            if not self._reexpand_guards_ok(graph, node_id):
                continue
            await self._apply_reexpand(graph, node_id, step_index, {
                "needs_followup": True,
                "reason": f"step contract unsatisfied: {verdict.reason}",
            })

    async def _maybe_plan_library_reexpand(
        self, graph: IdeaDag, node_id: str, step_index: int
    ) -> bool:
        """Turn a completed, ADOPTED on-demand plan-library search into real children.

        No-op unless ``plan_library.enabled`` and ``plan_library.action_enabled`` are both set
        (JSON default False -> byte-identical), the completed node's action is
        ``plan_library_search``, and its result reports ``adopted=True``. The leaf action is
        read-only by design (it ranks, fills and reports). So this is where the graph actually
        grows, alongside every other re-expansion trigger and under the SAME
        ``_reexpand_guards_ok`` termination bounds.

        The plan is rebuilt from the result's own ``slot_values`` rather than re-retrieved:
        ``candidates_from_template`` is pure, so the rebuild is deterministic and free, whereas
        re-running extraction would spend a second LLM call AND could return values that
        disagree with the ``adopted`` verdict already recorded on the node. Chroma is not
        touched at all.

        Growing the children rides ``_apply_reexpand`` -> ``_handle_expansion_node`` (the
        expansion is handed over as a single-use stash on the node) so the ``_got_reexpanded``
        marker step() routes on, the lineage ``_got_reexpand_count`` budget, the
        ``max_total_nodes`` re-check, the duplicate filter and beam width, the children-metadata
        threading, the ``contract_made`` rows and the post-expansion hooks all stay exactly
        where the automatic path already put them, with nothing re-implemented here.

        Returns True when children were created.
        """
        cfg = self._cfg.plan_library
        if not (cfg.enabled and cfg.action_enabled):
            return False
        node = graph.get_node(node_id)
        if not node:
            return False
        if (
            NodeDetailsExtractor.get_action(node.details)
            != IdeaActionType.PLAN_LIBRARY_SEARCH.value
        ):
            return False
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict) or not result.get("adopted"):
            return False
        if not self._reexpand_guards_ok(graph, node_id):
            return False

        template_id = result.get("adopted_template_id")
        slot_values = result.get("slot_values")
        try:
            template = self._plan_library_corpus().get(template_id)
            if template is None or not isinstance(slot_values, dict):
                self._logger.warning(
                    f"[STEP {step_index}] PLAN LIBRARY: node {node_id[:8]} adopted "
                    f"'{template_id}' but it cannot be rebuilt (template known: "
                    f"{template is not None}, slot values: {isinstance(slot_values, dict)})"
                )
                return False
            expansion = _plan_library.candidates_from_template(
                template,
                slot_values,
                origin=_plan_library.ORIGIN_ACTION,
                contracts=self.contracts,
            )
        except Exception as exc:  # noqa: BLE001. A rebuild failure is silence, not a crash
            self._logger.warning(
                f"[STEP {step_index}] PLAN LIBRARY: rebuilding '{template_id}' failed: {exc}"
            )
            return False
        if not expansion:
            return False

        node.details[_plan_library.PLAN_LIBRARY_PENDING] = {
            "candidates": list(expansion.candidates),
            "parent_details": dict(expansion.parent_details),
        }
        try:
            applied = await self._apply_reexpand(graph, node_id, step_index, {
                "needs_followup": True,
                "reason": (
                    f"plan library template '{expansion.template_id}' adopted on demand "
                    f"({len(expansion.candidates)} sub-problems)"
                ),
            })
        finally:
            # Never leave the stash behind: `_handle_expansion_node` pops it on the happy
            # path, but a node that failed to expand must not carry a plan into its next one.
            node.details.pop(_plan_library.PLAN_LIBRARY_PENDING, None)
        if applied:
            self._logger.info(
                f"[STEP {step_index}] PLAN LIBRARY: on-demand template "
                f"'{expansion.template_id}' -> {len(node.children)} children under "
                f"{node_id[:8]}"
            )
        return applied

    def _handle_action_result(self, graph: IdeaDag, node_id: str, step_index: int) -> str:
        node = graph.get_node(node_id)
        if not node:
            return "missing"
        from agent.app.idea_policies.action_constants import ActionResultKey, ResultStatus
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            node.status = IdeaNodeStatus.FAILED
            return ResultStatus.FAILED.value
        from agent.app.idea_policies.action_constants import ActionResultExtractor, ResultStatus
        from agent.app.idea_policies.base import IdeaActionType
        success = ActionResultExtractor.is_success(result)
        if success:
            action = NodeDetailsExtractor.get_action(node.details)

            if action == IdeaActionType.VISIT.value:
                url = result.get(ActionResultKey.URL.value) or result.get("url") or ""
                content = result.get(ActionResultKey.CONTENT.value) or result.get(ActionResultKey.CONTENT_FULL.value) or result.get("content") or ""
                content_total_chars = result.get(ActionResultKey.CONTENT_TOTAL_CHARS.value) or len(content) if content else 0

                if not content or len(content.strip()) == 0:
                    self._logger.error(
                        f"[VISIT_VALIDATION] Visit marked as success but has no content! "
                        f"Node {node_id}, URL: {url[:80] if url else 'unknown'}, "
                        f"result keys: {list(result.keys())}"
                    )
                    result[ActionResultKey.SUCCESS.value] = False
                    result[ActionResultKey.ERROR.value] = "Visit succeeded but no content was retrieved - this indicates a validation failure"
                    result[ActionResultKey.ERROR_TYPE.value] = "ValidationError"
                    result[ActionResultKey.RETRYABLE.value] = True
                    success = False
                else:
                    self._logger.info(
                        f"[VISIT_SUCCESS] Visit succeeded: Node {node_id}, "
                        f"URL: {url[:80] if url else 'unknown'}, "
                        f"Content: {content_total_chars} chars, "
                        f"Status: DONE"
                    )
                    node.details["visit_content_length"] = content_total_chars
                    node.details["visit_url"] = url

            if success:
                try:
                    action_type = IdeaActionType(action) if action else None
                except ValueError:
                    action_type = None
                if action_type:
                    action_instance = self.actions.get(action_type)
                    contract_name = action_instance.post_execute_provides(node, result)
                    if contract_name:
                        node.details[DetailKey.PROVIDES_DATA.value] = {"type": contract_name}
                        if action_type == IdeaActionType.SEARCH:
                            self._logger.debug(f"[DATA_FLOW] Node {node_id} (search) now provides {contract_name}")

                # Deterministic (LLM-free) value threading for a downstream hop (opt-in, JSON
                # default OFF -> byte-identical when off). See idea_policies/waypoint.py for the
                # extraction design and the wrong-page false-positive measurement that gates it
                # behind a page-identity guard.
                if self._cfg.engine.waypoint_enabled:
                    try:
                        from agent.app.idea_policies import waypoint as _waypoint
                        wp = _waypoint.build_waypoint(node, result, action, node_id, step_index)
                        if wp is not None:
                            node.details["waypoint"] = wp.as_dict()
                    except Exception as exc:  # noqa: BLE001 -- extraction must never fail a leaf
                        self._logger.warning(f"[WAYPOINT] extraction failed (node={node_id}): {exc}")

                node.status = IdeaNodeStatus.DONE
                return ResultStatus.SUCCESS.value
            # Fix #9: VISIT empty-content path flipped `success` to False above.
            # Previously the code fell through to `node.status = DONE` anyway.
            # a latent bug. With `visit_empty_content_retryable` (default True),
            # the node now falls through to the retry/fail branch below.
            if not self._cfg.action.visit_empty_content_retryable:
                # Legacy behavior preserved behind a kill-switch.
                node.status = IdeaNodeStatus.DONE
                return ResultStatus.SUCCESS.value
        retryable = ActionResultExtractor.is_retryable(result)
        node.details[DetailKey.ACTION_RETRYABLE.value] = retryable
        attempts = int(node.details.get(DetailKey.ACTION_ATTEMPTS.value, 0))
        max_retries = self._cfg.action.max_retries
        if retryable and attempts <= max_retries:
            backoff = self._cfg.action.retry_backoff_steps
            next_step = step_index + max(1, backoff)
            node.details[DetailKey.ACTION_COOLDOWN_UNTIL.value] = next_step
            node.status = IdeaNodeStatus.BLOCKED
            return "retry"
        
        from agent.app.idea_policies.action_constants import ActionResultExtractor, ActionResultKey
        error_info = ActionResultExtractor.get_error(result)
        error_type = result.get(ActionResultKey.ERROR_TYPE.value)
        root_cause = result.get(ActionResultKey.ROOT_CAUSE.value)
        http_status = result.get(ActionResultKey.HTTP_STATUS.value)
        traceback_summary = result.get(ActionResultKey.TRACEBACK_SUMMARY.value)
        
        error_details = {
            "error": error_info,
            "error_type": error_type,
            "root_cause": root_cause,
            "http_status": http_status,
            "traceback_summary": traceback_summary,
        }
        node.details[DetailKey.ACTION_ERROR.value] = error_info
        node.details[f"{DetailKey.ACTION_ERROR.value}_details"] = error_details
        node.status = IdeaNodeStatus.FAILED

        # Fix #8: sequential sibling recovery. When the selected sibling in
        # sequential mode fails terminally, un-SKIP the next-best
        # sequential-pruned sibling so the parent can still produce useful
        # output instead of giving up. Gated by setting.
        if self._cfg.engine.sequential_sibling_recovery_enabled:
            self._recover_pruned_sibling(graph, node, step_index)

        return ResultStatus.FAILED.value

    def _recover_pruned_sibling(self, graph: IdeaDag, failed_node: IdeaNode, step_index: int) -> None:
        """Un-SKIP the highest-scored sequential-pruned sibling of a failed node.

        Triggers at most once per parent (tracked via
        `__sequential_recovery_used`). The recovered sibling is set back to
        PENDING with a marker so the next step can pick it up.
        """
        parent_id = failed_node.parent_id
        if not parent_id:
            return
        parent = graph.get_node(parent_id)
        if not parent:
            return
        if parent.details.get("__sequential_recovery_used"):
            return

        candidates: List[IdeaNode] = []
        for cid in parent.children:
            sib = graph.get_node(cid)
            if not sib or sib.node_id == failed_node.node_id:
                continue
            if sib.status != IdeaNodeStatus.SKIPPED:
                continue
            if not sib.details.get("__sequential_pruned"):
                continue
            candidates.append(sib)
        if not candidates:
            return

        candidates.sort(key=lambda n: (n.score if n.score is not None else -1.0), reverse=True)
        winner = candidates[0]
        winner.status = IdeaNodeStatus.PENDING
        winner.details["__sequential_recovery_invoked"] = True
        parent.details["__sequential_recovery_used"] = True
        self._logger.info(
            f"[STEP {step_index}] SEQUENTIAL_RECOVERY: failed node {failed_node.node_id[:8]} -> "
            f"un-SKIPped sibling {winner.node_id[:8]} (score={winner.score})"
        )

    def _reorder_for_sequential(
        self,
        graph: IdeaDag,
        selected: IdeaNode,
        eligible: List[str],
        step_index: int,
    ) -> Optional[IdeaNode]:
        return idea_sequencing.reorder_for_sequential(graph, selected, eligible, step_index)

    def _detect_state_dependencies(self, graph: IdeaDag, candidate_ids: List[str]) -> bool:
        return idea_sequencing.detect_state_dependencies(graph, candidate_ids, self._logger)

    def _should_chunk_document(self, graph: IdeaDag, node: IdeaNode) -> bool:
        return idea_chunking.should_chunk_document(
            graph, node, self._cfg.memory.document_chunk_threshold, self._logger
        )

    async def _create_chunk_subproblems(self, graph: IdeaDag, visit_node: IdeaNode) -> Optional[List[str]]:
        return await idea_chunking.create_chunk_subproblems(
            graph,
            visit_node,
            self._cfg.memory.document_chunk_size,
            self._cfg.memory.document_chunk_overlap,
            self._logger,
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        return idea_chunking.chunk_text(text, chunk_size, chunk_overlap)

    def _detect_chunk_dependencies(self, graph: IdeaDag, candidate_ids: List[str]) -> bool:
        return idea_chunking.detect_chunk_dependencies(graph, candidate_ids)

    def _is_chunk_node(self, graph: IdeaDag, node_id: str) -> bool:
        return idea_chunking.is_chunk_node(graph, node_id)

    def _get_pending_executable_nodes(self, graph: IdeaDag) -> List[IdeaNode]:
        return idea_node_state.get_pending_executable_nodes(graph)
    
    def _is_leaf_node(self, graph: IdeaDag, node, step_index: int) -> bool:
        action_type = node.details.get(DetailKey.ACTION.value)
        if action_type and self._is_action_ready(node, step_index):
            if node.details.get(DetailKey.ACTION_RESULT.value) is None:
                return True
        
        if node.children:
            all_complete = all(
                (child := graph.get_node(child_id)) and 
                child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED, IdeaNodeStatus.SKIPPED)
                for child_id in node.children
            )
            if all_complete:
                return True
        
        return False

    @staticmethod
    def _sanitize_action_result(result: Dict[str, Any]) -> Dict[str, Any]:
        return idea_node_state.sanitize_action_result(result)

    def _is_action_ready(self, node, step_index: int) -> bool:
        return idea_node_state.is_action_ready(node, step_index)

    def _check_and_create_merge_nodes(self, graph: IdeaDag, node_id: str, step_index: int) -> None:
        node = graph.get_node(node_id)
        if not node:
            return
        
        if self.merge.should_create_merge_node(graph, node_id):
            merge_node_id = self.merge.create_merge_node(graph, node_id)
            if merge_node_id:
                self._logger.debug(f"[STEP {step_index}] Created merge node {merge_node_id} for parent {node_id}")
        
        if node.parent_id:
            self._check_and_create_merge_nodes(graph, node.parent_id, step_index)

    def _select_best_global(self, graph: IdeaDag, min_score: float, allow_unscored: bool) -> tuple[Optional[Any], Optional[str]]:
        return idea_node_state.select_best_global(
            graph, min_score, allow_unscored, self._step_index
        )

    def _maybe_log_dag(self, graph: IdeaDag, step_index: int, force: bool = False) -> None:
        if not self._cfg.engine.log_dag_ascii:
            return
        interval = self._cfg.engine.log_dag_step_interval
        if not force and interval <= 0:
            return
        if not force and step_index % interval != 0:
            return
        try:
            from agent.app.idea_dag_log import idea_dag_to_ascii
            self._logger.info("\n%s", idea_dag_to_ascii(graph))
        except Exception:
            return

    @staticmethod
    def _memo_namespace(mandate: str) -> str:
        digest = hashlib.sha256(mandate.encode("utf-8")).hexdigest()[:10]
        return f"idea_dag:{digest}"
