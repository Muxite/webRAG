import asyncio
import math
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

    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self._chroma = None
        self.chroma_api_ready = False

    async def _try_init_chroma(self) -> bool:
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
            self._chroma = await chromadb.AsyncHttpClient(
                host=host,
                port=port,
                ssl=False,
                settings=Settings(anonymized_telemetry=False),
            )
            await self._chroma.heartbeat()
            self.logger.info("ChromaDB OPERATIONAL (async)")
            self.chroma_api_ready = True
            return True
        except Exception as e:
            self.logger.warning(f"ChromaDB connection failed: {e}")
            self.chroma_api_ready = False
            return False

    async def init_chroma(self) -> bool:
        """
        Initialize or verify the ChromaDB connection.
        :returns: True if ready.
        """
        if self.chroma_api_ready:
            return True
        retry = Retry(
            func=self._try_init_chroma,
            max_attempts=10,
            base_delay=self.config.default_delay,
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
        if not await self._ensure_ready():
            self.logger.warning("ChromaDB not ready.")
            return None
        try:
            return await self._chroma.get_or_create_collection(name=collection)
        except Exception as e:
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
            await self._chroma.delete_collection(name=collection)
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
            collections = await self._chroma.list_collections()
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
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]
            self._record_io(
                direction="in",
                operation="chroma_add",
                payload={"collection": collection, "count": len(documents)},
            )
            await coll.add(ids=ids, metadatas=sanitized_metadatas, documents=documents)
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
            self.logger.error(f"Failed to add to collection '{collection}': {e}")
            self.chroma_api_ready = False
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
        Add documents in parallel batches using asyncio.gather.

        Splits the input into chunks of ``batch_size`` and stores them
        concurrently.  Falls back to a single ``add_to_chroma`` call
        when the total count is at or below the batch size.

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
        tasks = []
        for i in range(n_batches):
            start = i * bs
            end = start + bs
            tasks.append(
                self.add_to_chroma(
                    collection=collection,
                    ids=ids[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents[start:end],
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = 0
        for r in results:
            if isinstance(r, Exception):
                self.logger.error(f"Parallel chroma batch failed: {r}")
                failures += 1
            elif not r:
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
            self._record_io(
                direction="in", operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "n_results": n_results, "where": where},
            )
            query_kwargs: Dict[str, Any] = {"query_texts": query_texts, "n_results": n_results}
            if where:
                query_kwargs["where"] = where
            results = await coll.query(**query_kwargs)
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
            self.logger.error(f"ChromaDB query failed for collection {collection}: {e}")
            self.chroma_api_ready = False
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
