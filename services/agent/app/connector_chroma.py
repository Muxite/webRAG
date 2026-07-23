import asyncio
import math
import os
import tempfile
import time
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.connector_base import ConnectorBase


class ConnectorChroma(ConnectorBase):
    """
    ChromaDB connector using the native ``AsyncHttpClient``.

    All methods are truly async and non-blocking.  The connection is lazily
    initialized on first use and retried automatically if ChromaDB is
    temporarily unavailable.

    :param connector_config: Shared connector configuration with chroma_url.
    """

    PARALLEL_BATCH_SIZE = 50
    # Consecutive per-op failures tolerated before we tear the client down and force
    # ONE bounded re-init. A single transient lock/timeout must NOT rebuild the client:
    # rebuild churn opens fresh connections, which makes the SQLite contention WORSE.
    FAILURE_TEARDOWN_THRESHOLD = 3

    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self._chroma = None
        self.chroma_api_ready = False
        # collection-name -> Collection, memoized to save a round-trip on every
        # add/query. Bound to the current client; cleared on rebuild/teardown.
        self._collections: Dict[str, Any] = {}
        self._consecutive_failures = 0
        # True when the active client is the async HttpClient; False for the sync
        # embedded PersistentClient (whose ops are dispatched via asyncio.to_thread).
        self._is_async_client = True
        # Lazily-built embedding function (None => chroma's default ONNX-CPU EF).
        self._embedding_function: Any = None
        self._embedding_function_built = False
        self._embedded_path: Optional[str] = None

    def _embedding_function_or_none(self) -> Any:
        """Resolve the collection embedding function once, honoring ``chroma_embed_device``.

        Returns None for the CPU path (chroma uses its built-in ONNX MiniLM). For
        ``cuda``/``auto`` it builds a SentenceTransformers EF on the GPU; any failure
        (no torch, no CUDA, model download blocked) falls back to None so embedding
        never hard-breaks — it just stays on CPU.

        :returns: An embedding function instance or None.
        """
        if self._embedding_function_built:
            return self._embedding_function
        self._embedding_function_built = True
        device = (self.config.chroma_embed_device or "cpu").lower()
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if device == "cpu":
            self._embedding_function = None
            return None
        try:
            from chromadb.utils import embedding_functions as ef
            self._embedding_function = ef.SentenceTransformerEmbeddingFunction(
                model_name=self.config.chroma_embed_model, device=device
            )
            self.logger.info(
                "Chroma embedding on GPU (device=%s, model=%s)",
                device, self.config.chroma_embed_model,
            )
        except Exception as e:
            self.logger.warning(
                "GPU embedding unavailable (%s); falling back to CPU default EF", e
            )
            self._embedding_function = None
        return self._embedding_function

    def _resolve_embedded_path(self) -> str:
        """Persist dir for embedded mode: the configured path, else a fresh temp dir
        (per-process isolation + clean per-run memory)."""
        if self.config.chroma_embedded_path:
            os.makedirs(self.config.chroma_embedded_path, exist_ok=True)
            return self.config.chroma_embedded_path
        if not self._embedded_path:
            self._embedded_path = tempfile.mkdtemp(prefix="chroma_embedded_")
        return self._embedded_path

    async def _bounded(self, awaitable: Any, op: str) -> Any:
        """Await a raw ChromaDB coroutine under a hard timeout.

        The AsyncHttpClient await can hang forever when the shared SQLite server is
        stuck on a write lock; ``asyncio.wait_for`` bounds it to ``chroma_op_timeout``
        and cancels the stuck request. (Embedding is computed synchronously before the
        awaited HTTP call, so the loop is not starved while the timer runs — the timeout
        genuinely fires on the network hang.)

        :param awaitable: The raw chroma coroutine.
        :param op: Operation name for logging.
        :returns: The coroutine result.
        :raises asyncio.TimeoutError: When the op exceeds ``chroma_op_timeout``.
        """
        return await asyncio.wait_for(awaitable, timeout=self.config.chroma_op_timeout)

    async def _op(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a client/collection method under the op timeout, dispatching by client
        kind: async HttpClient methods are awaited directly; sync PersistentClient
        methods run in a worker thread (``asyncio.to_thread``) so the CPU-bound embedding
        happens OFF the event loop. Both are bounded by ``chroma_op_timeout``.
        """
        if self._is_async_client:
            awaitable = method(*args, **kwargs)
        else:
            awaitable = asyncio.to_thread(lambda: method(*args, **kwargs))
        return await asyncio.wait_for(awaitable, timeout=self.config.chroma_op_timeout)

    def _on_op_success(self) -> None:
        """Reset the consecutive-failure counter after any successful op."""
        self._consecutive_failures = 0

    def _on_op_failure(self) -> None:
        """Record a per-op failure; tear down (forcing a bounded re-init on the next
        op) only after ``FAILURE_TEARDOWN_THRESHOLD`` consecutive failures — never on
        a single transient error, which would churn connections and worsen contention.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.FAILURE_TEARDOWN_THRESHOLD:
            self.chroma_api_ready = False
            self._collections.clear()
            self._consecutive_failures = 0

    async def _try_init_chroma(self) -> bool:
        """Connect to ChromaDB: shared async HttpClient (``http`` mode) or a
        per-process sync PersistentClient (``embedded`` mode)."""
        if (self.config.chroma_mode or "http") == "embedded":
            return await self._try_init_embedded()
        return await self._try_init_http()

    async def _try_init_http(self) -> bool:
        """Attempt to connect to ChromaDB via AsyncHttpClient and verify heartbeat."""
        chroma_url = self.config.chroma_url
        if not chroma_url:
            self.logger.warning("Chroma URL not set")
            return False
        try:
            raw = chroma_url.replace("http://", "").replace("https://", "")
            parts = raw.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 and parts[1] else 8000
            self._is_async_client = True
            self._chroma = await self._bounded(
                chromadb.AsyncHttpClient(
                    host=host,
                    port=port,
                    ssl=False,
                    settings=Settings(anonymized_telemetry=False),
                ),
                "connect",
            )
            await self._bounded(self._chroma.heartbeat(), "heartbeat")
            # Fresh client → any memoized collections are bound to a dead client.
            self._collections.clear()
            self._consecutive_failures = 0
            self.logger.info("ChromaDB OPERATIONAL (async)")
            self.chroma_api_ready = True
            return True
        except Exception as e:
            self.logger.warning(f"ChromaDB connection failed: {e}")
            self.chroma_api_ready = False
            return False

    async def _try_init_embedded(self) -> bool:
        """Build a per-process embedded PersistentClient (its own SQLite file → no
        cross-subprocess write-lock contention). Built in a thread; ops are dispatched
        the same way, so the loop is never blocked by SQLite or embedding."""
        try:
            path = self._resolve_embedded_path()
            self._is_async_client = False
            self._chroma = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: chromadb.PersistentClient(
                        path=path, settings=Settings(anonymized_telemetry=False)
                    )
                ),
                timeout=self.config.chroma_op_timeout,
            )
            self._collections.clear()
            self._consecutive_failures = 0
            self.logger.info("ChromaDB OPERATIONAL (embedded @ %s)", path)
            self.chroma_api_ready = True
            return True
        except Exception as e:
            self.logger.warning(f"ChromaDB embedded init failed: {e}")
            self.chroma_api_ready = False
            return False

    async def init_chroma(self) -> bool:
        """
        Initialize or verify the ChromaDB connection.
        :returns: True if ready.
        """
        if self.chroma_api_ready:
            return True
        # Bounded re-init: the old (max_attempts=10, base_delay=default_delay=2,
        # max_delay=60) summed to ~302s of silent backoff per subprocess that lost its
        # client under contention — a self-amplifying storm. A couple of fast attempts
        # is enough: a healthy server answers immediately; a dead one should fail open
        # in ~1s so the run degrades rather than freezing to the cell cap.
        retry = Retry(
            func=self._try_init_chroma,
            max_attempts=self.config.chroma_init_attempts,
            base_delay=self.config.retry_base_delay,
            max_delay=2.0,
            name="ChromaDBinit",
            jitter=self.config.jitter_seconds,
        )
        success = await retry.run()
        if not success:
            self.logger.error("ChromaDB failed to initialize after retries.")
        return success

    async def _ensure_ready(self) -> bool:
        """Ensure ChromaDB is initialized before any operation."""
        if self.chroma_api_ready:
            return True
        return await self.init_chroma()

    async def get_or_create_collection(self, collection: str) -> Any:
        """
        Get or create a named collection. Uses default embedding (all-MiniLM-L6-v2).
        :param collection: Collection name.
        :returns: ChromaDB collection object, or None on failure.
        """
        cached = self._collections.get(collection)
        if cached is not None:
            return cached
        if not await self._ensure_ready():
            self.logger.warning("ChromaDB not ready.")
            return None
        try:
            create_kwargs: Dict[str, Any] = {"name": collection}
            embed_fn = self._embedding_function_or_none()
            if embed_fn is not None:
                create_kwargs["embedding_function"] = embed_fn
            coll = await self._op(self._chroma.get_or_create_collection, **create_kwargs)
            self._collections[collection] = coll
            self._on_op_success()
            return coll
        except Exception as e:
            self._on_op_failure()
            self.logger.error(f"Failed to create/get collection '{collection}': {e}")
            return None

    async def delete_collection(self, collection: str) -> bool:
        """
        Delete a collection by name.
        :param collection: Collection name.
        :returns: True if deleted.
        """
        if not await self._ensure_ready():
            return False
        try:
            if self._chroma is None:
                return False
            await self._op(self._chroma.delete_collection, name=collection)
            self._collections.pop(collection, None)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to delete collection '{collection}': {e}")
            return False

    async def list_collections(self) -> List[str]:
        """
        List all collection names.
        :returns: List of collection name strings.
        """
        if not await self._ensure_ready():
            return []
        try:
            if self._chroma is None:
                return []
            collections = await self._op(self._chroma.list_collections)
            return [col.name for col in collections] if collections else []
        except Exception as e:
            self.logger.warning(f"Failed to list collections: {e}")
            return []

    async def add_to_chroma(
        self,
        collection: str,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> bool:
        """
        Add documents to a collection.
        :param collection: Target collection name.
        :param ids: Unique IDs for each document.
        :param metadatas: Metadata dicts per document.
        :param documents: Document text strings.
        :returns: True on success.
        """
        if not await self._ensure_ready():
            return False
        started_at = time.perf_counter()
        try:
            coll = await self.get_or_create_collection(collection)
            if coll is None:
                return False
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]
            self._record_io(
                direction="in",
                operation="chroma_add",
                payload={"collection": collection, "count": len(documents)},
            )
            await self._op(coll.add, ids=ids, metadatas=sanitized_metadatas, documents=documents)
            self._on_op_success()
            self._record_timing(
                name="chroma_add", started_at=started_at, success=True,
                payload={"collection": collection, "count": len(documents)},
            )
            self._record_io(
                direction="out", operation="chroma_add",
                payload={"collection": collection, "count": len(documents), "success": True},
            )
            return True
        except Exception as e:
            self._on_op_failure()
            self.logger.error(f"Failed to add to collection '{collection}': {e}")
            self._record_timing(
                name="chroma_add", started_at=started_at, success=False,
                payload={"collection": collection, "count": len(documents)},
                error=str(e),
            )
            self._record_io(
                direction="out", operation="chroma_add",
                payload={"collection": collection, "count": len(documents), "success": False},
                error=str(e),
            )
            return False

    async def add_to_chroma_parallel(
        self,
        collection: str,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        batch_size: Optional[int] = None,
    ) -> bool:
        """
        Add a large document set in size-bounded batches, stored SEQUENTIALLY.

        This used to fan the batches out with ``asyncio.gather``, but the target is a
        SQLite-backed Chroma server with a single global write lock: concurrent writes
        to it do not go faster — they serialize on that lock and, under the barrage's
        multi-subprocess load, surface as ``database is locked`` / ``unable to open
        database file (code 14)`` errors (A7 measured a 31.8% ``chroma_add`` error
        rate). Sequential batches keep the total payload per request bounded without
        adding write concurrency the backend cannot honor. Falls back to a single
        ``add_to_chroma`` when the input fits one batch.

        :param collection: Target collection name.
        :param ids: Unique IDs for each document.
        :param metadatas: Metadata dicts per document.
        :param documents: Document text strings.
        :param batch_size: Documents per batch (default PARALLEL_BATCH_SIZE).
        :returns: True if all batches succeeded.
        """
        if not documents:
            return False
        bs = batch_size or self.PARALLEL_BATCH_SIZE
        total = len(documents)
        if total <= bs:
            return await self.add_to_chroma(collection, ids, metadatas, documents)

        n_batches = math.ceil(total / bs)
        failures = 0
        for i in range(n_batches):
            start = i * bs
            end = start + bs
            ok = await self.add_to_chroma(
                collection=collection,
                ids=ids[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end],
            )
            if not ok:
                failures += 1
        if failures:
            self.logger.warning(
                f"add_to_chroma_parallel: {failures}/{n_batches} batch(es) failed"
            )
        return failures == 0

    async def query_chroma(
        self,
        collection: str,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Query a collection for nearest neighbors.
        :param collection: Collection name.
        :param query_texts: Query strings.
        :param n_results: Max results per query.
        :param where: Optional metadata filter.
        :returns: ChromaDB result dict or None on failure.
        """
        if not await self._ensure_ready():
            return None
        started_at = time.perf_counter()
        try:
            coll = await self.get_or_create_collection(collection)
            if coll is None:
                return None
            self._record_io(
                direction="in", operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "n_results": n_results, "where": where},
            )
            query_kwargs: Dict[str, Any] = {"query_texts": query_texts, "n_results": n_results}
            if where:
                query_kwargs["where"] = where
            results = await self._op(coll.query, **query_kwargs)
            self._on_op_success()
            self._record_timing(
                name="chroma_query", started_at=started_at, success=True,
                payload={"collection": collection, "queries": len(query_texts)},
            )
            self._record_io(
                direction="out", operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "success": True},
            )
            return results
        except Exception as e:
            self._on_op_failure()
            self.logger.error(f"ChromaDB query failed for collection {collection}: {e}")
            self._record_timing(
                name="chroma_query", started_at=started_at, success=False,
                payload={"collection": collection, "queries": len(query_texts)},
                error=str(e),
            )
            self._record_io(
                direction="out", operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "success": False},
                error=str(e),
            )
            return None

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce metadata values to ChromaDB-compatible types (str, int, float, bool).
        :param metadata: Raw metadata dict.
        :returns: Sanitized metadata dict.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                import json
                sanitized[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
