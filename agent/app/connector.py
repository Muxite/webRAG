import os
import logging
import aiohttp
from typing import Optional, Dict, Any
from redis.asyncio import Redis
import chromadb
from chromadb.config import Settings
import asyncio
from shared.retry import Retry
import math
import random

# TODO, add circuit breaker pattern failure resolution, ensure Connector always returns something meaningful
# TODO, add intelligent HealthChecks embedded in all service-calling methods, check if X seconds since last successful call
# TODO: detect failed calls and return viable messages with differentiation so they are usable

class Connector:
    """
    Class that manages connections to external services. Handles retry logic, jittering
    Services will return None if they fail.

    Those are:
    1. Redis connection (async)
    2. aiohttp session
    3. LLM client
    4. ChromaDB client
    """

    def __init__(self, worker_type: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.worker_type = worker_type
        self.redis_url = os.environ.get("REDIS_URL")
        self.redis_ready = False
        self.default_timeout = int(os.environ.get("DEFAULT_TIMEOUT", "10"))
        self.cold_start_time = float(os.environ.get("COLD_START_SECONDS", "90"))
        self.jitter_seconds = float(os.environ.get("JITTER_SECONDS", "0.1"))

        self.redis: Optional[Redis] = None
        if not self.redis_url:
            self.logger.warning(f"No Redis URL set")

        self.chroma = None
        self.chroma_url = os.environ.get("CHROMA_URL")
        if not self.chroma_url:
            self.logger.warning("No Chroma URL set")
        self.chroma_api_ready = False

        self.llm_url = f"{os.environ.get('MODEL_API_URL')}/v1/chat/completions"
        if not self.llm_url:
            self.logger.warning(f"No LLM URL set")
        self.llm_api_ready = False


        self.session: Optional[aiohttp.ClientSession] = None

    async def init_redis(self) -> bool:
        """
        Initialize or verify the Redis connection.
        Returns True only after a successful ping; returns False for transient failures.
        Raises for missing/malformed configuration that should not be retried.
        """
        if not self.redis_url:
            raise ValueError("REDIS_URL not set")
        if self.redis is None:
            try:
                self.redis = Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=self.default_timeout,  # per-call bound
                )
            except Exception as e:
                self.logger.error(f"Redis client creation failed: {e}")
                return False
        try:
            await asyncio.wait_for(self.redis.ping(), timeout=self.default_timeout)
        except Exception as e:
            self.logger.warning(f"Redis ping failed: {e}")
            try:
                await self.redis.close()
            except Exception:
                pass
            self.redis = None
            self.redis_ready = False
            return False
        self.redis_ready = True
        return True

    async def init_http_session(self) -> bool:
        """
        Ensure an aiohttp session exists with explicit timeouts.
        Always returns True if a usable session is present after the call.
        """
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.default_timeout,
                connect=self.default_timeout,
                sock_read=self.default_timeout,
            )
            try:
                self.session = aiohttp.ClientSession(timeout=timeout)
            except Exception as e:
                self.logger.error(f"HTTP session creation failed: {e}")
                return False
        return True

    async def init_chroma(self) -> bool:
        """
        Initialize or verify the ChromaDB connection.
        Sets self.chroma_api_ready.
        """
        if self.chroma_api_ready:
            return True

        try:
            self.chroma = chromadb.HttpClient(
                host=self.chroma_url.replace("http://", "")
                .replace("https://", "").split(":")[0],
                port=int(self.chroma_url.split(":")[-1]) if ":" in self.chroma_url.split("//")[-1] else 8000,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )

            self.chroma.heartbeat()
            self.logger.info("ChromaDB OPERATIONAL")
            self.chroma_api_ready = True
            return True

        except Exception as e:
            self.logger.warning(f"ChromaDB connection failed: {e}")
            self.chroma_api_ready = False
            return False

    async def init_llm(self) -> bool:
        """
        Initialize or verify the LLM completions call such that it matches the OpenAI format.
        Sets self.llm_api_ready.
        """
        if self.llm_api_ready:
            return True

        await self.init_http_session()
        if not self.llm_url:
            self.logger.warning("LLM URL not set")
            return False

        test_payload = {
            "model": "llama",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
            "stream": False,
        }

        try:
            async with self.session.post(self.llm_url, json=test_payload, timeout=self.default_timeout) as resp:
                if resp.status != 200:
                    self.logger.warning(f"LLM health probe POST failed with {resp.status}")
                    return False
                result = await resp.json()
        except Exception as e:
            self.logger.warning(f"LLM health probe exception: {e}")
            return False

        try:
            _ = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            self.logger.warning(f"LLM health probe returned unexpected structure: {result}")
            return False

        self.logger.info("LLM OPERATIONAL")
        self.llm_api_ready = True
        return True

    async def post_json(self, url, payload, retries=6, **kwargs) -> Optional[Dict[str, Any]]:
        await self.init_http_session()
        for attempt in range(1, retries + 1):
            try:
                async with self.session.post(
                        url, json=payload, timeout=self.default_timeout, **kwargs
                ) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        self.logger.warning(f"POST {url} failed with {resp.status}: {text}")
                    if resp.status in (401, 403, 404):
                        self.logger.error(f"POST {url}: permanent error {resp.status}, aborting retries")
                        return None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                self.logger.warning(f"POST {url} failed attempt {attempt}/{retries}: {e}")
            wait_secs = min(60, 2 ** attempt)
            self.logger.info(f"Retrying POST after {wait_secs}s...")
            await asyncio.sleep(wait_secs)
        self.logger.error(f"POST {url} failed after {retries} attempts")
        return None

    async def create_or_get_collection(self, collection: str, metadata: dict = None) -> bool:
        """
        Create a ChromaDB collection or get it if it already exists.
        :param collection: ChromaDB collection name
        :param metadata: Optional metadata for the collection
        :return: True if successful, False otherwise
        """
        if not self.chroma_api_ready:
            self.logger.warning("ChromaDB not ready.")
            return False

        try:
            self.chroma.get_or_create_collection(
                name=collection,
                metadata=metadata
            )
            self.logger.info(f"Collection '{collection}' ready")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create/get collection '{collection}': {e}")
            return False

    async def add_to_chroma(self,
                            collection: str,
                            ids: list[str],
                            metadatas: list[dict],
                            documents: list[str]) -> bool:
        """
        Add documents or embeddings to a ChromaDB collection.
        :param collection: ChromaDB collection name
        :param ids: List of document IDs
        :param metadatas: List of metadata dictionaries
        :param documents: List of documents
        :return: True if successful, False otherwise
        """
        if not self.chroma_api_ready:
            self.logger.warning("ChromaDB not ready.")
            return False

        try:
            coll = self.chroma.get_collection(name=collection)

            coll.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add to collection '{collection}': {e}")
            return False

    async def query_chroma(self, collection: str, query_texts: list[str], n_results: int = 5) -> Optional[dict]:
        """
        Query a ChromaDB collection for nearest neighbors.
        :param collection: ChromaDB collection name
        :param query_texts: List of query texts
        :param n_results: Number of results to return
        :return: Dictionary of results
        """
        if not self.chroma_api_ready:
            self.logger.warning("ChromaDB not ready.")
            return None

        try:
            coll = self.chroma.get_collection(name=collection)

            results = coll.query(
                query_texts=query_texts,
                n_results=n_results
            )

            return results

        except Exception as e:
            self.logger.error(f"ChromaDB query failed for collection {collection}: {e}")
            return None

    async def query_llm(self, payload) -> Optional[str]:
        """
        Send a chat completion request to the LLM API.
        :param payload: the full JSON payload for the chat completion request.
            model, messages, temperature, max_tokens,...
        :return: Response text or None if request failed or response is malformed.
        """
        if not self.llm_api_ready or not self.llm_url:
            self.logger.warning("LLM not ready or URL missing.")
            return None

        result = await self.post_json(self.llm_url, payload, retries=3)
        if not result:
            self.logger.error("LLM query failed.")
            return None

        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            self.logger.warning(f"Unexpected LLM response structure: {result}")
            return None

    async def await_all_connections_ready(self) -> bool:
        """
        Open all connections concurrently and wait until they are ready.
        """

        await self.init_http_session()
        attempts = int(self.cold_start_time / self.default_timeout + 2)

        redis_retry = Retry(
            func=self.init_redis,
            max_attempts=attempts,
            delay=self.default_timeout,
            jitter=self.jitter_seconds,
            name="Redis initialization"
        )
        chroma_retry = Retry(
            func=self.init_chroma,
            max_attempts=attempts,
            delay=self.default_timeout,
            jitter=self.jitter_seconds,
            name="ChromaDB initialization"
        )
        llm_retry = Retry(
            func=self.init_llm,
            max_attempts=attempts,
            delay=self.default_timeout,
            jitter=self.jitter_seconds,
            name="LLM initialization"
        )

        results = await asyncio.gather(
            redis_retry.run(),
            chroma_retry.run(),
            llm_retry.run()
        )

        if not all(results):
            self.logger.error("One or more connections failed to initialize")
            return False
        else:
            self.logger.info("All connections ready")
            return True

    async def close_connections(self):
        """
        close all connections
        """
        if self.session:
            if not self.session.closed:
                try:
                    await self.session.close()
                except Exception as e:
                    self.logger.error(f"Error closing HTTP session: {e}")
            self.session = None

        if self.redis:
            try:
                await self.redis.aclose()
            except Exception as e:
                self.logger.error(f"Error closing Redis: {e}")
            self.redis = None
        self.llm_api_ready = False
        self.chroma_api_ready = False

    async def __aenter__(self):
        """Support async context manager."""
        await self.await_all_connections_ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        await self.close_connections()

    def get_redis(self) -> Redis:
        if self.redis is None:
            raise RuntimeError("Redis is not initialized. Call open_connections() first.")
        return self.redis

    def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("HTTP session not initialized. Call open_connections() first.")
        return self.session
