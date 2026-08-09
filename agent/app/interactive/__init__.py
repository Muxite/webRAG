"""
agent-debug: stepping debugger for the IdeaDAG agent.

Walk the reasoning graph depth-first. Pause at expansions and merges.
Inspect nodes, view live stats and ASCII graph.
"""

from agent.app.interactive.renderer import Renderer
from agent.app.interactive.controller import Controller, Action
from agent.app.interactive.session import DebugSession
from agent.app.interactive.stats import StatsTracker

__all__ = ["Renderer", "Controller", "Action", "DebugSession", "StatsTracker"]
