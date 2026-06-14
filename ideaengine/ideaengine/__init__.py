"""
ideaengine — a Graph-of-Thought DAG executor for LLM research tasks.

Public API (v0.1, flat namespace):

    from ideaengine import (
        IdeaDagEngine,             # core engine
        load_idea_dag_settings,    # config loader
        AgentIO,                   # I/O facade for LLM / search / HTTP / vector store
        Solver, IdeaEngineSolver,  # solver protocol + engine adapter
        build_final_payload,       # final synthesis function
        MemoryManager,             # ChromaDB-backed memory
        GoTOperations,             # GoT mechanics (dedup, beam, prune, backtrack)
        ContractRegistry, DataContract, default_contract_registry,
        load_default_prompts, apply_default_prompts,
    )

See `docs/IDEA_ENGINE.md` for the architecture deep-dive and
`docs/PLUGINS.md` for how to add custom actions.
"""

from ideaengine.idea_engine import IdeaDagEngine
from ideaengine.idea_dag_settings import load_idea_dag_settings
from ideaengine.agent_io import AgentIO
from ideaengine.solver import Solver, IdeaEngineSolver, SolverResult
from ideaengine.idea_finalize import build_final_payload
from ideaengine.idea_memory import MemoryManager
from ideaengine.got_operations import GoTOperations
from ideaengine.idea_policies.data_contracts import (
    ContractRegistry,
    DataContract,
    default_contract_registry,
)
from ideaengine.prompts.loader import (
    load_default_prompts,
    apply_default_prompts,
)

__version__ = "0.1.0"

__all__ = [
    "IdeaDagEngine",
    "load_idea_dag_settings",
    "AgentIO",
    "Solver",
    "IdeaEngineSolver",
    "SolverResult",
    "build_final_payload",
    "MemoryManager",
    "GoTOperations",
    "ContractRegistry",
    "DataContract",
    "default_contract_registry",
    "load_default_prompts",
    "apply_default_prompts",
    "__version__",
]
