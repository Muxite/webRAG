import time
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.connector_base import ConnectorBase


class ConnectorChroma(ConnectorBase):
    """
    Manages a connection to ChromaDB and exposes common operations.
    """

    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self._chroma = None
        self.chroma_api_ready = False

    async def _try_init_chroma(self) -> bool:
        """
        Single attempt to initialize HttpClient and heartbeat Chroma.
        :return bool: True on success, False on failure.
        """
        chroma_url = self.config.chroma_url
        if not chroma_url:
            self.logger.warning("Chroma URL not set")
            return False

        try:
            raw = chroma_url.replace("http://", "").replace("https://", "")
            parts = raw.split(":")

            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 and parts[1] else 8000

            self._chroma = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=False,
                settings=Settings(anonymized_telemetry=False),
            )

            self._chroma.heartbeat()
            self.logger.info("ChromaDB OPERATIONAL")
            self.chroma_api_ready = True
            return True

        except Exception as e:
            self.logger.warning(f"ChromaDB connection failed: {e}")
            self.chroma_api_ready = False
            return False

    async def init_chroma(self) -> bool:
        """
        Initialize or verify the ChromaDB connection. Sets self.chroma_api_ready.
        :return bool: True on success, False on failure.
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
        """
        Ensure ChromaDB is initialized.
        :returns: True if ready, False otherwise.
        """
        if self.chroma_api_ready:
            return True
        return await self.init_chroma()

    async def get_or_create_collection(self, collection: str) -> Any:
        """
        Create a ChromaDB collection or get it if it already exists.
        Uses default embedding function (all-MiniLM-L6-v2) for efficiency.
        :param collection: Name of the collection to create or get.
        :return chromadb.Collection: ChromaDB collection object.
        """
        if not await self._ensure_ready():
            self.logger.warning("ChromaDB not ready.")
            return None
        try:
            # ChromaDB uses all-MiniLM-L6-v2 by default, which is efficient
            # No need to specify embedding function - uses default
            coll = self._chroma.get_or_create_collection(
                name=collection,
            )
            return coll
        except Exception as e:
            self.logger.error(f"Failed to create/get collection '{collection}': {e}")
            return None

    async def add_to_chroma(
        self,
        collection: str,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> bool:
        """
        Add documents to a ChromaDB collection.
        :param collection: ChromaDB collection name.
        :param ids: List of unique IDs for each document.
        :param metadatas: List of metadata dicts for each document.
        :param documents: List of document texts.
        :return bool: True on success, False on failure.
        """
        if not await self._ensure_ready():
            self.logger.warning("ChromaDB not ready.")
            return False
        try:
            coll = await self.get_or_create_collection(collection)
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]
            started_at = time.perf_counter()
            self._record_io(
                direction="in",
                operation="chroma_add",
                payload={"collection": collection, "count": len(documents)},
            )
            coll.add(
                ids=ids,
                metadatas=sanitized_metadatas,
                documents=documents,
            )
            self._record_timing(
                name="chroma_add",
                started_at=started_at,
                success=True,
                payload={"collection": collection, "count": len(documents)},
            )
            self._record_io(
                direction="out",
                operation="chroma_add",
                payload={"collection": collection, "count": len(documents), "success": True},
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add to collection '{collection}': {e}")
            self.chroma_api_ready = False
            self._record_timing(
                name="chroma_add",
                started_at=started_at,
                success=False,
                payload={"collection": collection, "count": len(documents)},
                error=str(e),
            )
            self._record_io(
                direction="out",
                operation="chroma_add",
                payload={"collection": collection, "count": len(documents), "success": False},
                error=str(e),
            )
            return False

    async def query_chroma(
        self,
        collection: str,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Query a ChromaDB collection for nearest neighbors.
        :param collection: ChromaDB collection name.
        :param query_texts: List of query texts.
        :param n_results: Number of results to return.
        :param where: Optional where clause for metadata filtering (e.g., {"memory_type": "observation"}).
        :return List of results, or None on failure.
        """
        if not await self._ensure_ready():
            self.logger.warning("ChromaDB not ready.")
            return None

        try:
            coll = await self.get_or_create_collection(collection)
            started_at = time.perf_counter()
            self._record_io(
                direction="in",
                operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "n_results": n_results, "where": where},
            )
            query_kwargs = {
                "query_texts": query_texts,
                "n_results": n_results,
            }
            if where:
                query_kwargs["where"] = where
            results = coll.query(**query_kwargs)
            self._record_timing(
                name="chroma_query",
                started_at=started_at,
                success=True,
                payload={"collection": collection, "queries": len(query_texts)},
            )
            self._record_io(
                direction="out",
                operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "success": True},
            )
            return results
        except Exception as e:
            self.logger.error(f"ChromaDB query failed for collection {collection}: {e}")
            self.chroma_api_ready = False
            self._record_timing(
                name="chroma_query",
                started_at=started_at,
                success=False,
                payload={"collection": collection, "queries": len(query_texts)},
                error=str(e),
            )
            self._record_io(
                direction="out",
                operation="chroma_query",
                payload={"collection": collection, "queries": len(query_texts), "success": False},
                error=str(e),
            )
            return None

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all metadata values are ChromaDB compatible types.
        :param metadata: Metadata dict to sanitize.
        :return Dict of sanitized metadata.
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
