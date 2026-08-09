"""
Step-level checkpointing for the GoT engine.

Snapshots the serialized DAG plus minimal engine state after each step so a
crashed run can resume where it left off. Two backends: Redis (production)
and File (dev/tests).
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Checkpointer(ABC):
    """
    Abstract checkpointer. Backends implement save/load/list.
    """

    @abstractmethod
    async def save(self, run_id: str, step_index: int, snapshot: Dict[str, Any]) -> None:
        """
        Persist a snapshot for the given run + step.

        :param run_id: Run identifier (idempotency key).
        :param step_index: 0-based step number.
        :param snapshot: Serialized graph + engine state.
        :returns: None.
        """

    @abstractmethod
    async def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest snapshot for a run, if any.

        :param run_id: Run identifier.
        :returns: Latest snapshot or None.
        """

    @abstractmethod
    async def list_runs(self) -> List[str]:
        """
        Enumerate run ids with stored checkpoints.

        :returns: List of run ids.
        """

    async def delete(self, run_id: str) -> None:
        """
        Remove all snapshots for a run. Default no-op for backends that
        only keep the latest snapshot.

        :param run_id: Run identifier.
        :returns: None.
        """
        return None


class FileCheckpointer(Checkpointer):
    """
    Stores snapshots as ./.checkpoints/<run_id>/<step>.json plus latest.json.
    """

    def __init__(self, root_dir: str = ".checkpoints") -> None:
        """
        :param root_dir: Directory under which snapshots are stored.
        """
        self.root_dir = root_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self.root_dir, run_id)

    async def save(self, run_id: str, step_index: int, snapshot: Dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)
        payload = {
            "run_id": run_id,
            "step_index": step_index,
            "saved_at": time.time(),
            "snapshot": snapshot,
        }
        step_path = os.path.join(run_dir, f"{step_index:04d}.json")
        latest_path = os.path.join(run_dir, "latest.json")
        with open(step_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(latest_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    async def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        latest_path = os.path.join(self._run_dir(run_id), "latest.json")
        if not os.path.exists(latest_path):
            return None
        try:
            with open(latest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Failed to load checkpoint %s: %s", run_id, exc)
            return None

    async def list_runs(self) -> List[str]:
        if not os.path.isdir(self.root_dir):
            return []
        return [d for d in os.listdir(self.root_dir) if os.path.isdir(self._run_dir(d))]

    async def delete(self, run_id: str) -> None:
        import shutil

        run_dir = self._run_dir(run_id)
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)


class RedisCheckpointer(Checkpointer):
    """
    Stores snapshots as JSON under euglena:checkpoint:<run_id> with a TTL.
    Only retains the latest snapshot per run (one key per run).
    """

    KEY_PREFIX = "euglena:checkpoint"
    INDEX_KEY = "euglena:checkpoint:_index"

    def __init__(self, redis_client: Any, ttl_seconds: int = 86400) -> None:
        """
        :param redis_client: An async Redis client (e.g. from connector_redis).
        :param ttl_seconds: TTL applied to each checkpoint key.
        """
        self.client = redis_client
        self.ttl_seconds = ttl_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def _key(self, run_id: str) -> str:
        return f"{self.KEY_PREFIX}:{run_id}"

    async def save(self, run_id: str, step_index: int, snapshot: Dict[str, Any]) -> None:
        payload = {
            "run_id": run_id,
            "step_index": step_index,
            "saved_at": time.time(),
            "snapshot": snapshot,
        }
        encoded = json.dumps(payload)
        await self.client.set(self._key(run_id), encoded, ex=self.ttl_seconds)
        await self.client.sadd(self.INDEX_KEY, run_id)
        await self.client.expire(self.INDEX_KEY, self.ttl_seconds)

    async def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        raw = await self.client.get(self._key(run_id))
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.warning("Corrupt checkpoint for %s: %s", run_id, exc)
            return None

    async def list_runs(self) -> List[str]:
        members = await self.client.smembers(self.INDEX_KEY)
        if not members:
            return []
        return [m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m) for m in members]

    async def delete(self, run_id: str) -> None:
        await self.client.delete(self._key(run_id))
        await self.client.srem(self.INDEX_KEY, run_id)


def create_checkpointer_from_env(redis_client: Any = None) -> Optional[Checkpointer]:
    """
    Build a Checkpointer based on environment variables.

    Env vars:
      IDEA_CHECKPOINT_ENABLED — "1"/"true" to enable (default off).
      IDEA_CHECKPOINT_BACKEND — "redis" | "file" (default file).
      IDEA_CHECKPOINT_DIR — root dir for file backend (default ".checkpoints").
      IDEA_CHECKPOINT_TTL_SECONDS — TTL for redis backend (default 86400).

    :param redis_client: Optional pre-constructed Redis client for redis backend.
    :returns: Checkpointer instance or None when disabled.
    """
    if (os.environ.get("IDEA_CHECKPOINT_ENABLED") or "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    backend = (os.environ.get("IDEA_CHECKPOINT_BACKEND") or "file").strip().lower()
    if backend == "redis":
        if redis_client is None:
            logging.getLogger(__name__).warning(
                "IDEA_CHECKPOINT_BACKEND=redis but no Redis client provided; falling back to file"
            )
        else:
            ttl = int(os.environ.get("IDEA_CHECKPOINT_TTL_SECONDS", "86400"))
            return RedisCheckpointer(redis_client, ttl_seconds=ttl)
    return FileCheckpointer(os.environ.get("IDEA_CHECKPOINT_DIR") or ".checkpoints")
