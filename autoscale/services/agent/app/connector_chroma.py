import logging
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.local_embedding_function import LocalEmbeddingFunction


class ConnectorChroma:
    """
    ChromaDB connector for vector storage and retrieval.
    Handles connection lifecycle and document operations with client-side embeddings.
    """

    def __init__(self, connector_config: ConnectorConfig):
        """
        Initialize connector.
        :param connector_config: Configuration
        """
        self.config = connector_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._chroma = None
        self.chroma_api_ready = False
        try:
            self._embedding_function = LocalEmbeddingFunction()
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {e}")
            self._embedding_function = None

    async def _try_init_chroma(self) -> bool:
        """
        Single attempt to initialize HttpClient and heartbeat ChromaDB.
        
        :return: True on success, False on failure.
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
            self.chroma_api_ready = True
            return True

        except Exception as e:
            self.logger.warning(f"ChromaDB connection failed: {e}")
            self.chroma_api_ready = False
            return False

    async def init_chroma(self) -> bool:
        """
        Initialize connection with retry logic.
        :returns Bool: true on success
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

    async def get_or_create_collection(self, collection: str) -> Any:
        """
        Create or get ChromaDB collection. Embeddings computed client-side.
        
        :param collection: Collection name.
        :return: Collection object or None on failure.
        """
        if not await self.init_chroma():
            return None
        try:
            coll = self._chroma.get_or_create_collection(name=collection)
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
        Add documents to collection.
        :param collection: Collection name
        :param ids: Document IDs
        :param metadatas: Metadata per document
        :param documents: Document texts
        :returns Bool: true on success
        """
        if not self.chroma_api_ready:
            self.logger.warning("ChromaDB not ready.")
            return False
        if not self._embedding_function:
            self.logger.error("Embedding function not available.")
            return False
        if not documents:
            return True
        try:
            coll = await self.get_or_create_collection(collection)
            if coll is None:
                return False
            embeddings = self._embedding_function.embed(documents)
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]
            coll.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=sanitized_metadatas,
                documents=documents
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add to collection '{collection}': {e}")
            return False

    async def query_chroma(self, collection: str, query_texts: List[str], n_results: int = 3) -> Optional[Dict[str, Any]]:
        """
        Query ChromaDB collection for similar documents. Computes embeddings client-side.
        
        :param collection: Collection name.
        :param query_texts: Query texts.
        :param n_results: Number of results per query. Defaults to 3.
        :return: Query results dict with documents/metadatas/distances, or None on failure.
        """
        if not self.chroma_api_ready:
            return None
        if not self._embedding_function:
            self.logger.error("Embedding function not available.")
            return None
        if not query_texts:
            return None

        try:
            coll = await self.get_or_create_collection(collection)
            if coll is None:
                return None
            query_embeddings = self._embedding_function.embed(query_texts)
            try:
                results = coll.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                )
            except TypeError:
                results = coll.query(
                    embeddings=query_embeddings,
                    n_results=n_results,
                )
            return results
        except Exception as e:
            self.logger.error(f"ChromaDB query failed for collection {collection}: {e}")
            return None

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata to ChromaDB-compatible types.
        :param metadata: Metadata dict
        :returns Dict: sanitized metadata
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
