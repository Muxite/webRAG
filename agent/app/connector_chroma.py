import logging
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
from shared.connector_config import ConnectorConfig
from shared.retry import Retry


class ConnectorChroma:
    """
    Manages a connection to ChromaDB and exposes common operations.

    This connector relies on a local sentence-transformers model via LocalEmbeddingFunction for embeddings.
    """

    def __init__(self, connector_config: ConnectorConfig):
        self.config = connector_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chroma = None
        self.chroma_api_ready = False

    async def _try_init_chroma(self) -> bool:
        """
        Single attempt to initialize HttpClient and heartbeat Chroma.
        Returns True on success, False on failure.
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

            self.chroma = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=False,
                settings=Settings(anonymized_telemetry=False),
            )

            self.chroma.heartbeat()
            self.logger.info("ChromaDB OPERATIONAL")
            self.chroma_api_ready = True
            return True

        except Exception as e:
            self.logger.warning(f"ChromaDB connection failed: {e}")
            self.chroma_api_ready = False
            return False

    async def init_chroma(self) -> bool:
        """
        Initialize or verify the ChromaDB connection.
        Sets self.chroma_api_ready.
        """
        if self.chroma_api_ready:
            return True
        retry = Retry(
            func=self._try_init_chroma,
            max_attempts=10,
            delay=self.config.default_delay,
            name="ChromaDBinit",
            jitter=self.config.jitter_seconds,
        )
        success = await retry.run()
        if not success:
            self.logger.error("ChromaDB failed to initialize after retries.")
        return success

    async def get_or_create_collection(self, collection: str) -> Any:
        """
        Create a ChromaDB collection or get it if it already exists.
        The collection will use our local embedding function.
        """
        if not await self.init_chroma():
            self.logger.warning("ChromaDB not ready.")
            return None
        try:
            coll = self.chroma.get_or_create_collection(
                name=collection,
            )
            self.logger.info(f"Collection '{collection}' ready (cached in map)")
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
        """
        if not self.chroma_api_ready:
            self.logger.warning("ChromaDB not ready.")
            return False
        try:
            coll = await self.get_or_create_collection(collection)
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]
            coll.add(
                ids=ids,
                metadatas=sanitized_metadatas,
                documents=documents,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add to collection '{collection}': {e}")
            return False

    async def query_chroma(self, collection: str, query_texts: List[str], n_results: int = 3) -> Optional[Dict[str, Any]]:
        """
        Query a ChromaDB collection for nearest neighbors.
        """
        if not self.chroma_api_ready:
            self.logger.warning("ChromaDB not ready.")
            return None

        try:
            coll = await self.get_or_create_collection(collection)
            results = coll.query(
                query_texts=query_texts,
                n_results=n_results,
            )
            return results
        except Exception as e:
            self.logger.error(f"ChromaDB query failed for collection {collection}: {e}")
            return None

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all metadata values are ChromaDB compatible types.
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
