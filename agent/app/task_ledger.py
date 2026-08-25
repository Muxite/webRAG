"""
Run-scoped requirement ledger (observe-only).

A :class:`TaskLedger` is the "what does this mandate actually require, and how much of it is
backed by real evidence right now" record for a single run. It is compiled once when the graph
root exists and refreshed at finalize, and in this mode it does nothing else: it creates no
nodes, touches no scores, and appears in no prompt. Everything it reports is telemetry, gated
by ``RunPolicy.ledger_mode`` (``"off"`` by default, ``"observe"`` to record).

Design notes
------------
* The requirement set is NOT a new parse. It is exactly
  :func:`idea_policies.candidate_coverage.evaluate_candidate_coverage`'s enumeration and exactly
  its notion of "resolved" (a candidate whose name matches a SUCCESSFULLY-VISITED page). Two
  code paths that both answer "is this run's roster covered?" would drift, and the gate is the
  one the engine already acts on, so the ledger is a *view* of it rather than a second opinion.
  ``requirements_supported == requirements_total`` therefore agrees with
  ``CandidateCoverageResult.satisfied`` (i.e. with the payload's
  ``candidate_coverage_incomplete``) by construction, and ``task_ledger_test`` pins that.

* Fails OPEN in the same sense the gate does: a mandate that enumerates nothing compiles to an
  empty, trivially-satisfied ledger rather than an error or an absent record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.app.idea_policies.candidate_coverage import evaluate_candidate_coverage


@dataclass
class TaskLedger:
    """The enumerated requirements of one mandate plus their current evidence backing.

    :param entities: Every candidate the mandate enumerates, in mandate order.
    :param requirements_supported: How many of them a successfully-visited page resolves.
    :param unresolved_entities: The remainder, in mandate order.
    :param task_metadata: Opaque, caller-supplied run context, carried verbatim. Reserved for
        the enforce-mode successor (which needs to know *which* task it is ledgering); nothing
        reads it in observe mode.
    """

    entities: List[str] = field(default_factory=list)
    requirements_supported: int = 0
    unresolved_entities: List[str] = field(default_factory=list)
    task_metadata: Optional[Dict[str, Any]] = None
    #: The mandate this ledger was compiled from, so :meth:`refresh` can re-run the same check.
    mandate: str = ""

    @property
    def requirements_total(self) -> int:
        """Number of enumerated requirements (``0`` for an un-enumerated mandate)."""
        return len(self.entities)

    @classmethod
    def compile(
        cls,
        mandate: str,
        task_metadata: Optional[Dict[str, Any]],
        graph,
    ) -> "TaskLedger":
        """Build a ledger for ``mandate`` against ``graph``'s current evidence.

        Called once per run, normally while the graph is still just a root (so
        ``requirements_supported`` starts at 0), but correct against a partially-executed graph
        too — restored checkpoints and the interactive debugger both compile mid-run.
        """
        ledger = cls(mandate=mandate or "", task_metadata=task_metadata)
        ledger.refresh(graph)
        return ledger

    def refresh(self, graph) -> "TaskLedger":
        """Recompute the requirement set and its backing against ``graph`` as it stands now.

        Idempotent and cheap enough to call per node completion later; today it runs at compile
        and at finalize. Returns ``self`` so a caller can chain into :meth:`to_dict`.
        """
        cov = evaluate_candidate_coverage(graph, self.mandate)
        self.entities = list(cov.named)
        self.unresolved_entities = list(cov.missing)
        self.requirements_supported = len(cov.resolved)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot, suitable for a node ``details`` value or a result payload."""
        return {
            "entities": list(self.entities),
            "requirements_total": self.requirements_total,
            "requirements_supported": self.requirements_supported,
            "unresolved_entities": list(self.unresolved_entities),
        }
