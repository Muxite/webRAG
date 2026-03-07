import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_browser import ConnectorBrowser, BROWSER_FALLBACK_STATUSES
from agent.app.observation import clean_operation
from agent.app.telemetry import TelemetrySession

_logger = logging.getLogger(__name__)


class AgentIO:
    """
    Unified async interface for LLM, web search, HTTP, ChromaDB, and browser operations.

    All methods are async. Telemetry is recorded automatically when a session
    is attached. Timeouts are configurable per call.

    ``visit`` tries aiohttp first; on 403/401 it falls back to the headless
    Chrome browser connector (if provided) to bypass bot protection.

    :param connector_llm: LLM connector.
    :param connector_search: Search API connector.
    :param connector_http: HTTP connector (aiohttp).
    :param connector_chroma: ChromaDB connector.
    :param connector_browser: Optional headless Chrome connector for bot-blocked sites.
    :param telemetry: Optional telemetry session.
    :param collection_name: ChromaDB collection for memory isolation.
    """
    def __init__(
        self,
        connector_llm: ConnectorLLM,
        connector_search: ConnectorSearch,
        connector_http: ConnectorHttp,
        connector_chroma: ConnectorChroma,
        connector_browser: Optional[ConnectorBrowser] = None,
        telemetry: Optional[TelemetrySession] = None,
        collection_name: str = "agent_memory",
    ) -> None:
        self.connector_llm = connector_llm
        self.connector_search = connector_search
        self.connector_http = connector_http
        self.connector_chroma = connector_chroma
        self.connector_browser = connector_browser
        self.collection_name = collection_name
        self.telemetry = telemetry
        self._attach_telemetry()

    def _attach_telemetry(self) -> None:
        if self.connector_llm:
            self.connector_llm.set_telemetry(self.telemetry)
        if self.connector_search:
            self.connector_search.set_telemetry(self.telemetry)
        if self.connector_http:
            self.connector_http.set_telemetry(self.telemetry)
        if self.connector_chroma:
            self.connector_chroma.set_telemetry(self.telemetry)
        if self.connector_browser:
            self.connector_browser.set_telemetry(self.telemetry)

    def set_telemetry(self, telemetry: Optional[TelemetrySession]) -> None:
        self.telemetry = telemetry
        self._attach_telemetry()

    def clear_telemetry(self) -> None:
        self.telemetry = None
        self._attach_telemetry()

    async def query_llm(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """
        Send a payload to the LLM and return the response text.
        :param payload: LLM request payload.
        :param model_name: Override model for this call.
        :param timeout_seconds: Optional timeout.
        :returns: Response text or None.
        """
        started_at = time.perf_counter()
        success = False
        error_text = None
        try:
            response = await self._with_timeout(
                self.connector_llm.query_llm(payload, model_name=model_name),
                timeout_seconds,
            )
            success = response is not None
            return response
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            if self.telemetry:
                self.telemetry.record_timing(
                    name="llm_call",
                    started_at=started_at,
                    success=success,
                    payload={"model": model_name or payload.get("model") or self.connector_llm.get_model()},
                    error=error_text,
                )

    async def query_llm_with_fallback(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """
        Query the LLM; retry with fallback_model if the primary fails.
        :returns: Response text or None.
        """
        primary_error: Optional[Exception] = None
        content: Optional[str] = None
        try:
            content = await self.query_llm(payload, model_name=model_name, timeout_seconds=timeout_seconds)
        except Exception as exc:
            primary_error = exc
        if content:
            return content
        if fallback_model and fallback_model.strip():
            fallback_name = fallback_model.strip()
            if not model_name or fallback_name != model_name:
                return await self.query_llm(payload, model_name=fallback_name, timeout_seconds=timeout_seconds)
        if primary_error:
            raise primary_error
        return content

    def pop_last_llm_usage(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the last LLM usage record (tokens, cost).
        :returns: Usage dict or None.
        """
        return self.connector_llm.pop_last_usage()

    def build_llm_payload(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool,
        model_name: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a normalized LLM request payload.
        :returns: Payload dict ready for ``query_llm``.
        """
        return self.connector_llm.build_payload(
            messages=messages,
            json_mode=json_mode,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )

    async def search(
        self,
        query: str,
        count: int = 10,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Web search via Brave Search API.
        :param query: Search query string.
        :param count: Max results.
        :returns: List of dicts with title, url, description.
        """
        started_at = time.perf_counter()
        try:
            results = await self._with_timeout(
                self.connector_search.query_search(query, count=count),
                timeout_seconds,
            )
        except Exception as exc:
            if self.telemetry:
                self.telemetry.record_timing(
                    name="search",
                    started_at=started_at,
                    success=False,
                    payload={"query": query, "result_count": 0},
                    error=str(exc),
                )
            raise
        if self.telemetry and results:
            for item in results:
                self.telemetry.record_document_seen(
                    source="search",
                    document={
                        "query": query,
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "description": item.get("description"),
                    },
                )
        if self.telemetry:
            self.telemetry.record_timing(
                name="search",
                started_at=started_at,
                success=results is not None,
                payload={"query": query, "result_count": len(results or [])},
            )
        return results

    async def visit(self, url: str, timeout_seconds: Optional[float] = None) -> str:
        """
        Fetch a URL, clean the HTML, return extracted text.

        Uses headless Chrome browser connector by default. Falls back to aiohttp
        if browser is not available.

        :param url: Target URL.
        :param timeout_seconds: Optional per-call timeout.
        :returns: Cleaned page text.
        :raises RuntimeError: On HTTP failure after all attempts.
        """
        started_at = time.perf_counter()
        used_browser = False
        result = None

        if self.connector_browser:
            _logger.info(f"Using headless Chrome to fetch {url}")
            browser_result = await self._with_timeout(
                self.connector_browser.fetch_page(url),
                timeout_seconds,
            )
            if not browser_result.error:
                result = browser_result
                used_browser = True
            else:
                _logger.warning(f"Browser fetch failed for {url}: {browser_result.data}, falling back to HTTP")
        
        if not result or result.error:
            _logger.info(f"Falling back to aiohttp for {url}")
            result = await self._with_timeout(
                self.connector_http.request("GET", url, retries=3),
                timeout_seconds,
            )

        if result.error:
            error_text = f"HTTP visit failed: {url} status={result.status}"
            if self.telemetry:
                self.telemetry.record_timing(
                    name="visit",
                    started_at=started_at,
                    success=False,
                    payload={"url": url, "status": result.status, "used_browser": used_browser},
                    error=error_text,
                )
            raise RuntimeError(error_text)

        resp = result.data
        try:
            if isinstance(resp, dict):
                text_body = json.dumps(resp)
            else:
                text_body = resp
            cleaned = clean_operation(text_body)
            summary = cleaned if cleaned else "[No main content found]"
        except Exception as exc:
            error_text = str(exc)
            if self.telemetry:
                self.telemetry.record_timing(
                    name="visit",
                    started_at=started_at,
                    success=False,
                    payload={"url": url, "status": result.status, "used_browser": used_browser},
                    error=error_text,
                )
            raise

        if self.telemetry:
            self.telemetry.record_document_seen(
                source="visit",
                document={"url": url, "content": summary},
            )
            self.telemetry.record_timing(
                name="visit",
                started_at=started_at,
                success=True,
                payload={"url": url, "status": result.status, "used_browser": used_browser},
            )
        return summary

    async def fetch_url(
        self,
        url: str,
        retries: int = 3,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Fetch raw content from a URL (no HTML cleaning).

        Uses headless Chrome browser connector by default. Falls back to aiohttp
        if browser is not available.

        :param url: Target URL.
        :param retries: Number of aiohttp retries (if fallback needed).
        :param timeout_seconds: Optional per-call timeout.
        :returns: Raw response text or JSON string.
        :raises RuntimeError: On HTTP failure after all attempts.
        """
        result = None

        if self.connector_browser:
            _logger.info(f"Using headless Chrome to fetch {url}")
            browser_result = await self._with_timeout(
                self.connector_browser.fetch_page(url),
                timeout_seconds,
            )
            if not browser_result.error:
                result = browser_result
            else:
                _logger.warning(f"Browser fetch failed for {url}: {browser_result.data}, falling back to HTTP")
        
        if not result or result.error:
            _logger.info(f"Falling back to aiohttp for {url}")
            result = await self._with_timeout(
                self.connector_http.request("GET", url, retries=retries),
                timeout_seconds,
            )
        if result.error:
            raise RuntimeError(f"HTTP fetch failed: {url} status={result.status}")
        resp = result.data
        if isinstance(resp, dict):
            return json.dumps(resp)
        return str(resp)

    async def store_chroma(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        Store documents in ChromaDB. Auto-fills title/topics metadata if absent.
        :param documents: Document texts.
        :param metadatas: Per-document metadata.
        :param ids: Unique document IDs.
        :returns: True on success.
        """
        if not documents:
            return False
        prepared_metadatas: List[Dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            if metadata is None:
                metadata = {}
            prepared = dict(metadata)
            if "title" not in prepared:
                words = doc.split()
                prepared["title"] = " ".join(words[:8]) if words else "untitled"
            if "topics" not in prepared:
                prepared["topics"] = prepared.get("title", "")
            prepared_metadatas.append(prepared)
        started_at = time.perf_counter()
        try:
            success = await self._with_timeout(
                self.connector_chroma.add_to_chroma(
                    collection=self.collection_name,
                    ids=ids,
                    metadatas=prepared_metadatas,
                    documents=documents,
                ),
                timeout_seconds,
            )
        except Exception as exc:
            if self.telemetry:
                self.telemetry.record_timing(
                    name="chroma_store",
                    started_at=started_at,
                    success=False,
                    payload={"count": len(documents), "collection": self.collection_name},
                    error=str(exc),
                )
            raise
        if self.telemetry:
            self.telemetry.record_chroma_store(
                {
                    "collection": self.collection_name,
                    "count": len(documents),
                    "ids": ids,
                    "metadatas": prepared_metadatas,
                    "documents": documents,
                }
            )
            self.telemetry.record_timing(
                name="chroma_store",
                started_at=started_at,
                success=bool(success),
                payload={"count": len(documents), "collection": self.collection_name},
            )
        return bool(success)

    async def retrieve_chroma(
        self,
        topics: List[str],
        n_results: int = 3,
        timeout_seconds: Optional[float] = None,
        memory_type: Optional[str] = None,
    ) -> List[str]:
        """
        Semantic search in ChromaDB.
        :param topics: Query strings.
        :param n_results: Max results per query.
        :param memory_type: Filter by ``internal_thought`` or ``observation``.
        :returns: List of matching document strings.
        """
        if not topics:
            return []
        
        started_at = time.perf_counter()
        where = None
        if memory_type:
            where = {"memory_type": memory_type}
        
        try:
            results = await self._with_timeout(
                self.connector_chroma.query_chroma(
                    collection=self.collection_name,
                    query_texts=topics,
                    n_results=n_results,
                    where=where,
                ),
                timeout_seconds,
            )
        except Exception as exc:
            if self.telemetry:
                self.telemetry.record_timing(
                    name="chroma_retrieve",
                    started_at=started_at,
                    success=False,
                    payload={"count": 0, "collection": self.collection_name, "memory_type": memory_type},
                    error=str(exc),
                )
            raise
        all_docs: List[str] = []
        if results and "documents" in results:
            for doc_list in results["documents"]:
                all_docs.extend(doc_list)
        if self.telemetry:
            self.telemetry.record_chroma_retrieve(
                {
                    "collection": self.collection_name,
                    "topics": topics,
                    "count": len(all_docs),
                    "memory_type": memory_type,
                    "documents": all_docs,
                }
            )
            self.telemetry.record_timing(
                name="chroma_retrieve",
                started_at=started_at,
                success=True,
                payload={"count": len(all_docs), "collection": self.collection_name, "memory_type": memory_type},
            )
        return all_docs
    
    async def retrieve_chroma_split(
        self,
        topics: List[str],
        n_internal: int = 3,
        n_observations: int = 3,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """
        Retrieve documents split into internal_thoughts and observations.
        :returns: Dict with ``internal_thoughts`` and ``observations`` string lists.
        """
        if not topics:
            return {"internal_thoughts": [], "observations": []}
        
        started_at = time.perf_counter()
        internal_thoughts = []
        observations = []
        
        try:
            internal_results = await self._with_timeout(
                self.connector_chroma.query_chroma(
                    collection=self.collection_name,
                    query_texts=topics,
                    n_results=n_internal,
                    where={"memory_type": "internal_thought"},
                ),
                timeout_seconds,
            )
            if internal_results and "documents" in internal_results:
                for doc_list in internal_results["documents"]:
                    internal_thoughts.extend(doc_list)
        except Exception as exc:
            _logger.warning(f"Failed to retrieve internal thoughts: {exc}")
        
        try:
            observation_results = await self._with_timeout(
                self.connector_chroma.query_chroma(
                    collection=self.collection_name,
                    query_texts=topics,
                    n_results=n_observations,
                    where={"memory_type": "observation"},
                ),
                timeout_seconds,
            )
            if observation_results and "documents" in observation_results:
                for doc_list in observation_results["documents"]:
                    observations.extend(doc_list)
        except Exception as exc:
            _logger.warning(f"Failed to retrieve observations: {exc}")
        
        if self.telemetry:
            self.telemetry.record_chroma_retrieve(
                {
                    "collection": self.collection_name,
                    "topics": topics,
                    "internal_thoughts_count": len(internal_thoughts),
                    "observations_count": len(observations),
                    "documents": internal_thoughts + observations,
                }
            )
            self.telemetry.record_timing(
                name="chroma_retrieve_split",
                started_at=started_at,
                success=True,
                payload={
                    "count": len(internal_thoughts) + len(observations),
                    "collection": self.collection_name,
                    "internal_thoughts": len(internal_thoughts),
                    "observations": len(observations),
                },
            )
        return {"internal_thoughts": internal_thoughts, "observations": observations}

    async def _with_timeout(self, coro, timeout_seconds: Optional[float]):
        if timeout_seconds is None:
            return await coro
        timeout_value = float(timeout_seconds)
        if timeout_value <= 0:
            return await coro
        return await asyncio.wait_for(coro, timeout=timeout_value)
