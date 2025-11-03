import os
import logging
import aiohttp
from typing import Optional, Dict, Any, List
from redis.asyncio import Redis
import chromadb
from chromadb.config import Settings
import asyncio
from shared.retry import Retry
from shared.request_result import RequestResult
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

    HTTP_STATUS_CODES = {
        200: "OK - Request succeeded",
        201: "Created - Resource created successfully",
        202: "Accepted - Request accepted for processing",
        204: "No Content - Success but no content to return",

        301: "Moved Permanently - Resource permanently moved",
        302: "Found - Resource temporarily moved",
        304: "Not Modified - Cached version still valid",

        400: "Bad Request - Invalid syntax or parameters",
        401: "Unauthorized - Authentication required or failed",
        403: "Forbidden - Server refuses to authorize request",
        404: "Not Found - Resource doesn't exist",
        405: "Method Not Allowed - HTTP method not supported",
        408: "Request Timeout - Server timed out waiting for request",
        409: "Conflict - Request conflicts with current state",
        422: "Unprocessable Entity - Semantic errors in request",
        429: "Too Many Requests - Rate limit exceeded",

        500: "Internal Server Error - Generic server error",
        502: "Bad Gateway - Invalid response from upstream server",
        503: "Service Unavailable - Server temporarily unavailable",
        504: "Gateway Timeout - Upstream server timed out",
    }

    PERMANENT_ERROR_CODES = {401, 403, 404, 405, 422, 500, 502, 503, 504}

    def __init__(self, worker_type: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.worker_type = worker_type
        self.redis_url = os.environ.get("REDIS_URL")
        self.redis_ready = False
        self.default_timeout = int(os.environ.get("DEFAULT_TIMEOUT", "2"))
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

    async def request(
            self,
            method: str,
            url: str,
            retries: int = 4,
            **kwargs
    ):
        """
        Generic request with exponential backoff retry logic.

        :param method: HTTP method (POST or GET)
        :param url: Target URL
        :param retries: Maximum retry attempts
        :param kwargs: Additional arguments for session.request()
        :return: RequestResult object containing the status and data (could be an error message)
        """
        await self.init_http_session()

        kwargs.pop('retries', None)
        last_exc = None
        last_status = None

        for attempt in range(1, retries + 1):
            try:

                async with self.session.request(
                        method=method,
                        url=url,
                        timeout=self.default_timeout,
                        **kwargs
                ) as resp:
                    await resp.read()
                    last_status = resp.status
                    if resp.status in Connector.PERMANENT_ERROR_CODES:
                        return RequestResult(
                            status=resp.status,
                            error=True,
                            data=Connector.HTTP_STATUS_CODES[resp.status]
                        )

                    if resp.status == 200:
                        return RequestResult(
                            status=resp.status,
                            error=False,
                            data=resp
                        )

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exc = e
                self.logger.warning(f"{method} {url} attempt {attempt}/{retries}: {e}")

            if attempt < retries:
                wait_secs = min(60, self.default_timeout * 2 ** attempt)
                await asyncio.sleep(wait_secs)

        return RequestResult(
            status=last_status,
            error=True,
            data=f"Request failed after {retries} attempts: {last_exc}"
        )

    async def query_search(self, query: str, count: int = 10) -> Optional[List[Dict[str, str]]]:
        """
        Send a search request to the configured Search API endpoint.
        :param query: Search query string
        :param count: Number of results to return (default 10)
        :return: List of search results or None if request failed or bad response
        """
        search_api_key = os.environ.get("SEARCH_API_KEY")
        if not search_api_key:
            self.logger.warning("Search API key not set.")
            return None

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": search_api_key
        }
        params = {
            "q": query,
            "count": count
        }

        result = await self.request("GET", url, retries=3, headers=headers, params=params)

        if result.error:
            self.logger.error(f"Search API query failed with {result.status}, {result.data}")
            return None

        try:
            web_results = (await result.data.json())["web"]["results"]
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", "")
                }
                for item in web_results
            ]
        except (KeyError, TypeError, IndexError):
            self.logger.warning(f"Unexpected Search API response structure: {result}")
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

    async def query_chroma(self, collection: str, query_texts: list[str], n_results: int = 3) -> Optional[dict]:
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
        :requires: self.llm_api_ready
        :param payload: the full JSON payload for the chat completion request.
            model, messages, temperature, max_tokens,...
        :return: Response text or None if request failed or response is malformed.
        """
        if not self.llm_api_ready or not self.llm_url:
            self.logger.warning("LLM not ready or URL missing.")
            return None

        result = await self.request("POST", self.llm_url, retries=3, json=payload)
        if result.error:
            self.logger.error(f"LLM query failed with {result.status}, {result.data}")
            return None

        try:
            json = (await result.data.json())["choices"][0]["message"]["content"]
            return json
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
