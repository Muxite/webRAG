"""Memory tool + durable store.

Two backends behind one interface:
- FileMemoryStore — a durable append-only JSONL bound to a STABLE identity key (not a
  per-task prompt hash), with keyword recall. Fully local, inspectable, serialized
  single-writer. Used in tests and as a zero-dependency default.
- VectorMemoryStore (P1) — reuse agent/app/idea_memory.MemoryManager + the
  local MiniLM embeddings + a persistent Chroma volume, same identity-keyed namespace.
  Swaps in behind the same MemoryStore protocol; nothing else changes.

The key fix vs today's engine: memory is keyed to a durable IDENTITY, queried at task
start, so facts survive across sessions (today it's an ephemeral per-task collection).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .base import ToolContext, ToolResult


class MemoryStore(Protocol):
    def remember(self, text: str, meta: Dict[str, Any] | None = None) -> str: ...
    def recall(self, query: str, k: int = 3) -> List[str]: ...


@dataclass
class FileMemoryStore:
    """Durable JSONL memory bound to a stable identity. Single-writer (serialized)."""
    path: Path
    identity: str = "default"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("identity") == self.identity:
                out.append(r)
        return out

    def remember(self, text: str, meta: Dict[str, Any] | None = None) -> str:
        rows = self._rows_all()
        mid = f"mem{len(rows) + 1}"
        row = {"id": mid, "identity": self.identity, "text": text,
               "meta": meta or {}, "t": round(time.time(), 3)}
        with open(self.path, "a", encoding="utf-8") as fh:      # append = serialized single write
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return mid

    def _rows_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [line for line in self.path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _toks(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

    def recall(self, query: str, k: int = 3) -> List[str]:
        q = self._toks(query)
        scored = []
        for r in self._rows():
            overlap = len(q & self._toks(r["text"]))
            if overlap:
                scored.append((overlap, r["t"], r["text"]))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [t for _, _, t in scored[:k]]


class VectorBackend(Protocol):
    def add(self, collection: str, ids: List[str], docs: List[str], metas: List[dict]) -> None: ...
    def query(self, collection: str, query: str, k: int) -> List[str]: ...


class InMemoryBackend:
    """Test/default backend: keyword recall, no external service. Mimics the vector
    store's add/query shape so VectorMemoryStore logic is exercised without Chroma."""
    def __init__(self) -> None:
        self._c: Dict[str, List[tuple]] = {}

    def add(self, collection, ids, docs, metas) -> None:
        self._c.setdefault(collection, []).extend(zip(ids, docs, metas))

    def query(self, collection, query, k) -> List[str]:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        rows = self._c.get(collection, [])
        scored = [(len(q & set(re.findall(r"[a-z0-9]+", d.lower()))), d) for _, d, _ in rows]
        scored = [(s, d) for s, d in scored if s]
        scored.sort(reverse=True)
        return [d for _, d in scored[:k]]


class ChromaBackend:
    """Production backend: local MiniLM embeddings via IdeaEngine's ConnectorChroma
    (durable volume + serialized writes configured at deploy). Guarded/lazy import;
    validated live in P1 — the InMemoryBackend covers logic in unit tests."""
    def __init__(self, chroma: Any = None) -> None:
        self._chroma = chroma
        if self._chroma is None:
            from agent.app.connector_chroma import ConnectorChroma  # services on path at runtime
            self._chroma = ConnectorChroma()

    def add(self, collection, ids, docs, metas) -> None:
        fn = getattr(self._chroma, "add_to_chroma")
        _maybe_run(fn(collection, ids, docs, metas))

    def query(self, collection, query, k) -> List[str]:
        res = _maybe_run(self._chroma.query_chroma(collection, [query], k))
        docs = (res or {}).get("documents") or []
        return docs[0] if docs and isinstance(docs[0], list) else [d for d in docs if isinstance(d, str)]


def _maybe_run(x):
    """Run a coroutine to completion if the connector is async; else pass through."""
    import asyncio
    import inspect
    if inspect.isawaitable(x):
        try:
            return asyncio.get_event_loop().run_until_complete(x)
        except RuntimeError:
            return asyncio.new_event_loop().run_until_complete(x)
    return x


@dataclass
class VectorMemoryStore:
    """Durable, identity-scoped semantic memory. Fixes the engine's per-task prompt-hash
    namespace: the collection is keyed to a STABLE identity so facts persist across
    unrelated sessions. Same MemoryStore protocol as FileMemoryStore — swaps in freely."""
    identity: str
    backend: VectorBackend
    collection_prefix: str = "agentmem"
    _n: int = 0

    @property
    def collection(self) -> str:
        return f"{self.collection_prefix}_{self.identity}"

    def remember(self, text: str, meta: Dict[str, Any] | None = None) -> str:
        self._n += 1
        mid = f"mem{self._n}"
        self.backend.add(self.collection, [mid], [text], [meta or {}])   # serialized single write
        return mid

    def recall(self, query: str, k: int = 3) -> List[str]:
        return self.backend.query(self.collection, query, k)


class MemoryTool:
    name = "memory"

    def execute(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        store: MemoryStore = ctx.memory
        if store is None:
            return ToolResult(False, "memory unavailable", error="no_store")
        op = request.get("op")
        if op == "remember":
            text = request.get("text", "").strip()
            if not text:
                return ToolResult(False, "remember: empty text", error="empty")
            mid = store.remember(text)
            return ToolResult(True, f"remembered: “{text[:80]}”", data={"id": mid})
        if op == "recall":
            query = request.get("query", "").strip()
            hits = store.recall(query, k=int(request.get("k", 3)))
            if not hits:
                return ToolResult(True, f"recall({query!r}): nothing found", data={"hits": []})
            joined = " | ".join(h[:80] for h in hits)
            return ToolResult(True, f"recall({query!r}): {joined}", data={"hits": hits})
        return ToolResult(False, f"unknown memory op {op!r}", error="unknown_op")
