"""DebugSession for the IdeaDAG agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.action_constants import NodeDetailsExtractor

from agent.app.interactive.renderer import Renderer
from agent.app.interactive.controller import Controller, Action, Cmd
from agent.app.interactive.stats import StatsTracker

_log = logging.getLogger(__name__)


class DebugSession:

    def __init__(
        self,
        engine: IdeaDagEngine,
        graph: IdeaDag,
        ctrl: Optional[Controller] = None,
        max_steps: int = 100,
    ):
        self._engine = engine
        self._graph = graph
        self._ctrl = ctrl or Controller()
        self._max_steps = max_steps
        self._step = 0
        self._quit = False
        self._stats = StatsTracker(graph)
        self._out = print

    async def run(self) -> Dict[str, Any]:
        root_id = self._graph.root_id()
        root = self._graph.get_node(root_id)

        self._out(Renderer.banner("agent-debug"))
        self._out(f"  mandate: {root.title[:120]}")
        self._out(Renderer.help_text())

        self._out(Renderer.section("Root Expansion"))
        await self._expand(root_id, depth=0)
        if not self._quit:
            self._print_live()
            await self._walk(root_id, depth=0)

        return self._finish()

    async def _walk(self, parent_id: str, depth: int) -> None:
        if self._halt():
            return
        parent = self._graph.get_node(parent_id)
        if not parent or not parent.children:
            return

        leaves, merges = self._split_children(parent_id)

        for cid in leaves:
            if self._halt():
                return
            child = self._graph.get_node(cid)
            if not child or child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                continue

            self._stats.set_depth(depth + 1)
            self._out(Renderer.section(f"Branch @ depth {depth+1}"))
            self._out(Renderer.node_oneliner(child, depth + 1))

            decision = self._pause(child)
            if decision is None:
                return
            if decision == Action.NEXT:
                await self._autorun(cid, depth + 1)
            else:
                await self._step_into(cid, depth + 1)

        if not self._halt():
            await self._do_merge(parent_id, merges, depth)

    async def _step_into(self, node_id: str, depth: int) -> None:
        if self._halt():
            return
        node = self._graph.get_node(node_id)
        if not node:
            return

        action = NodeDetailsExtractor.get_action(node.details)
        is_leaf = node.details.get(DetailKey.IS_LEAF.value, False)

        if (is_leaf or action) and not NodeDetailsExtractor.is_merge_action(node.details):
            await self._exec_leaf(node_id, depth)
            return

        if not node.children:
            await self._expand(node_id, depth)
        node = self._graph.get_node(node_id)
        if node and node.children and not self._halt():
            await self._walk(node_id, depth)

    async def _expand(self, node_id: str, depth: int) -> None:
        node = self._graph.get_node(node_id)
        if not node:
            return
        self._out(f"  expanding: {node.title[:90]}")
        await self._engine_step(node_id)
        self._print_live()
        node = self._graph.get_node(node_id)
        if node and node.children:
            self._out(f"  -> {len(node.children)} children:")
            self._out(Renderer.children_list(self._graph, node_id))
        else:
            self._out("  -> no children produced.")

    async def _exec_leaf(self, node_id: str, depth: int) -> None:
        node = self._graph.get_node(node_id)
        if not node:
            return
        action = NodeDetailsExtractor.get_action(node.details) or "?"
        self._out(f"{'  ' * depth}> {action}: {node.title[:72]}")
        await self._engine_step(node_id)
        node = self._graph.get_node(node_id)
        if node:
            self._out(Renderer.result_card(node))
            if node.status == IdeaNodeStatus.FAILED:
                self._out(f"{'  ' * depth}  FAILED")
        self._print_live()

    async def _do_merge(self, parent_id: str, merge_ids: List[str], depth: int) -> None:
        parent = self._graph.get_node(parent_id)
        if not parent:
            return

        leaf_ids = [
            cid for cid in parent.children
            if not NodeDetailsExtractor.is_merge_action(
                (self._graph.get_node(cid) or IdeaNode(node_id="", title="")).details
            )
        ]
        all_done = all(
            (c := self._graph.get_node(cid)) and
            c.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
            for cid in leaf_ids
        )
        if not all_done:
            return

        self._out(Renderer.section(f"Merge: {parent.title[:55]}"))

        if not merge_ids:
            await self._engine_step(parent_id)
            parent = self._graph.get_node(parent_id)
            merge_ids = [
                cid for cid in (parent.children if parent else [])
                if NodeDetailsExtractor.is_merge_action(
                    (self._graph.get_node(cid) or IdeaNode(node_id="", title="")).details
                )
            ]

        for mid in merge_ids:
            mnode = self._graph.get_node(mid)
            if not mnode:
                continue
            self._out(Renderer.merge_preview(mnode))
            self._out("  merging...")
            await self._engine_step(mid)
            mnode = self._graph.get_node(mid)
            if mnode:
                self._out(Renderer.result_card(mnode))
        self._print_live()

    async def _autorun(self, node_id: str, depth: int) -> None:
        node = self._graph.get_node(node_id)
        label = node.title[:55] if node else "?"
        self._out(f"{'  ' * depth}>> auto: {label}")
        current = node_id
        safety = 0
        while current and safety < self._max_steps and not self._halt():
            safety += 1
            self._step += 1
            self._stats.tick(depth)
            try:
                nxt = await self._engine.step(self._graph, current, self._step)
            except Exception as exc:
                self._out(f"{'  ' * depth}  error: {exc}")
                break
            if nxt is None:
                break

            if nxt == node_id:
                n = self._graph.get_node(node_id)
                if n and all(
                    (c := self._graph.get_node(cid)) and
                    c.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
                    for cid in n.children
                ):
                    break
            ancestors = set()
            w = self._graph.get_node(node_id)
            if w and w.parent_id:
                ancestors.add(w.parent_id)
            if nxt in ancestors:
                break
            current = nxt

        self._out(f"{'  ' * depth}  done. subtree:")
        self._out(Renderer.subtree(self._graph, node_id, max_depth=3))
        self._print_live()

    def _pause(self, node: IdeaNode, label: str = "") -> Optional[Action]:
        tag = label or node.title[:35]
        while True:
            cmd = self._ctrl.ask(label=tag)

            if cmd.action == Action.QUIT:
                self._quit = True
                return None

            if cmd.action == Action.HELP:
                self._out(Renderer.help_text())
                continue

            if cmd.action == Action.INFO:
                self._handle_info(cmd, node)
                continue

            if cmd.action == Action.PRINT:
                self._out(Renderer.result_card(node))
                continue

            if cmd.action == Action.LIST:
                self._out(Renderer.children_list(self._graph, node.node_id))
                continue

            if cmd.action == Action.GRAPH:
                self._out(Renderer.ascii_dag(self._graph))
                continue

            return cmd.action

    def _handle_info(self, cmd: Cmd, node: IdeaNode) -> None:
        arg = cmd.arg.strip().lower()
        if arg == "nodes":
            self._out(Renderer.subtree(self._graph, self._graph.root_id(), max_depth=10))
        elif arg == "stats":
            self._out(Renderer.stats_panel(self._stats))
        else:
            self._out(Renderer.node_card(node))

    def _split_children(self, parent_id: str) -> tuple[List[str], List[str]]:
        parent = self._graph.get_node(parent_id)
        if not parent:
            return [], []
        leaves, merges = [], []
        for cid in parent.children:
            child = self._graph.get_node(cid)
            if not child:
                continue
            if NodeDetailsExtractor.is_merge_action(child.details):
                merges.append(cid)
            else:
                leaves.append(cid)
        return leaves, merges

    def _halt(self) -> bool:
        return self._quit or self._step >= self._max_steps

    async def _engine_step(self, node_id: str) -> Optional[str]:
        if self._step >= self._max_steps:
            self._out("  max steps reached")
            return None
        self._step += 1
        self._stats.tick()
        try:
            return await self._engine.step(self._graph, node_id, self._step)
        except Exception as exc:
            self._out(f"  engine error: {exc}")
            _log.error(f"engine step error: {exc}", exc_info=True)
            return None

    def _print_live(self) -> None:
        self._out(Renderer.stats_panel(self._stats))
        dag_str = Renderer.ascii_dag(self._graph)
        if dag_str and len(dag_str) < 3000:
            self._out(dag_str)

    def _finish(self) -> Dict[str, Any]:
        self._out(Renderer.banner("Session Complete"))
        self._out(Renderer.stats_panel(self._stats))
        self._out(Renderer.ascii_dag(self._graph))
        return {
            "graph": self._graph,
            "steps": self._step,
            "quit_early": self._quit,
            "stats": self._stats.snapshot(),
        }
