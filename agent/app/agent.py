import logging
import json
from typing import Optional, Dict, List
from app.connector_config import ConnectorConfig
from app.connector_llm import ConnectorLLM
from app.connector_search import ConnectorSearch
from app.connector_http import ConnectorHttp
from app.tick_output import TickOutput, ActionType
from app.prompt_builder import PromptBuilder
from app.prompt_builder import build_payload
from app.observation import clean_operation
from app.connector_chroma import ConnectorChroma
from shared.pretty_log import pretty_log
from urllib.parse import urlparse
import asyncio


class Agent:
    """
    Autonomous RAG agent that executes a tick-based reasoning loop.
    Handles graceful degradation when external services fail.
    """

    def __init__(self, mandate: str, max_ticks: int = 50):
        """
        Initialize the agent with a mandate and tick limit.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mandate = mandate
        self.max_ticks = max_ticks
        self.current_tick = 0

        self.history: List[str] = []
        self.notes: str = ""
        self.deliverables: List[str] = []
        self.pending_data_topics: List[str] = []
        self.retrieved_context: List[str] = []
        self.observations: str = ""

        self.config = ConnectorConfig()
        self.llm_connector = ConnectorLLM(self.config)
        self.search_connector = ConnectorSearch(self.config)
        self.http_connector = ConnectorHttp(self.config)
        self.chroma_connector = ConnectorChroma(self.config)

        self.collection_name = "agent_memory"

    async def initialize(self) -> bool:
        """Initialize all connectors and setup dependencies."""
        self.logger.info("Initializing agent connectors...")

        search_ready = await self.search_connector.init_search_api()
        chroma_ready = await self.chroma_connector.init_chroma()
        llm_ready = self.llm_connector.llm_api_ready

        return all([llm_ready, search_ready, chroma_ready])

    async def run(self) -> Dict:
        """
        The primary event loop of the agent.
        1. Build prompt with current observations.
        2. Receive and parse LLM response.
        3. Execute action and update internal memory fields accordingly
        4. Store context in ChromaDB.
        5. Retrieve context from ChromaDB.
        Repeat

        :return: Dictionary of full agent results.
        """
        if not await self.initialize():
            return {"success": False, "error": "Initialization failed", "deliverables": []}

        self.logger.info(f"Agent started: {self.mandate}\n")


        while self.current_tick < self.max_ticks:
            await asyncio.sleep(1)
            self.current_tick += 1
            self.logger.info(f"=== TICK {self.current_tick}/{self.max_ticks} ===")

            prompt_builder = PromptBuilder(
                mandate=self.mandate,
                short_term_summary=self.history,
                notes=self.notes,
                retrieved_long_term=self.retrieved_context,
                observations=self.observations
            )
            prompt = build_payload(prompt_builder.build_messages(), True)
            pretty_log(prompt_builder.get_summary(), logger=self.logger, indents=1)

            llm_response = await self.llm_connector.query_llm(prompt)
            if llm_response is None:
                self.logger.error("LLM query failed, terminating.")
                self.logger.info(prompt)
                continue
            try:
                tick_output = TickOutput(json.loads(llm_response))
                pretty_log(tick_output.get_summary(), logger=self.logger, indents=1)
            except Exception as e:
                self.logger.error(f"Failed to parse LLM output: {e}")
                self.logger.debug(llm_response)
                self.logger.debug(prompt)
                continue

            self.observations = ""
            self._apply_tick_output(tick_output)
            await self._do_action(tick_output)
            chroma_topics = tick_output.show_requested_data_topics()
            if chroma_topics:
                self.logger.info(f"ChromaDB: Retrieving context for topics: {chroma_topics}")
                self.retrieved_context = await self._retrieve_chroma(chroma_topics)
                self.logger.info(f"ChromaDB: Retrieved context: {self.retrieved_context}")
            else:
                self.retrieved_context = []

            await self._store_chroma(
                documents=tick_output.get_vector_documents(),
                metadatas=tick_output.get_vector_metadatas(),
                ids=tick_output.get_vector_ids()
            )

            if tick_output.show_next_action()[0] == ActionType.EXIT:
                self.logger.info("Agent exit action received, stopping.")
                break

        await self.http_connector.__aexit__(None, None, None)
        return {
            "success": True,
            "ticks_completed": self.current_tick,
            "deliverables": self.deliverables,
            "history": self.history
        }

    def _apply_tick_output(self, tick_output: TickOutput):
        """
        Update local memory fields based on tick output.
        :param tick_output:
        """
        if tick_output.show_history():
            self.history.append(f"[Tick {self.current_tick}] {tick_output.show_history()}")
        if tick_output.show_notes():
            self.notes = tick_output.show_notes()
        if tick_output.deliverable().strip():
            self.deliverables.append(tick_output.deliverable())

    async def _do_action(self, tick_output: TickOutput):
        action, param = tick_output.show_next_action()
        if action == ActionType.SEARCH:
            self.logger.info(f"Searching for '{param}'...")
            await self._perform_web_search(param)
        elif action == ActionType.VISIT:
            self.logger.info(f"Visiting '{param}'...")
            await self._perform_visit(param)
        elif action == ActionType.THINK:
            self.logger.info("Agent is thinking...")
            pass
        elif action == ActionType.EXIT:
            self.logger.info("Agent exit action received, stopping.")
            pass

    async def _perform_web_search(self, query: str, count: int = 10):
        """
        Run a web search using the Connector and update self.observations.
        :param query: Topic/question to search for.
        :param count: Number of search results to include.
        """
        search_results = await self.search_connector.query_search(query, count=count)
        if not search_results:
            result = "[Search API unavailable or failed]"
        else:
            result = "\n".join(
                f"- {item.get('url', '')} ({item.get('url', '')})\n{item.get('description', '')}"
                for item in search_results
            )
        self.observations += f"\nWeb search for '{query}':\n{result}\n"

    async def _perform_visit(self, url: str):
        """
        Visit a URL and extract text to update self.observations.
        :param url: URL to visit.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            self.logger.error(f"Invalid URL provided: '{url}'")
            self.observations += f"\n[Could not fetch URL: Invalid URL] {url}\n"
            return

        result = await self.http_connector.request("GET", url, retries=3)
        if result.error:
            self.observations += f"\n[Could not fetch URL: {result.status}] {url}\n"
            return
        resp = result.data
        try:
            if isinstance(resp, dict):
                text_body = json.dumps(resp)
            else:
                text_body = resp
            cleaned = clean_operation(text_body)
            summary = cleaned if cleaned else "[No main content found]"
        except Exception as e:
            self.logger.warning(f"Failed to extract body from {url}: {e}")
            summary = "[No main content found]"

        self.observations += f"\nVisited {url}:\n{summary}\n"

    async def _store_chroma(self, ids: List[str], metadatas: List[Dict], documents: List[str]):
        if not self.chroma_connector.chroma_api_ready or not documents:
            return
        await self.chroma_connector.add_to_chroma(
            collection=self.collection_name,
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )

    async def _retrieve_chroma(self, topics: List[str]) -> List[str]:
        if not self.chroma_connector.chroma_api_ready or not topics:
            return []
        results = await self.chroma_connector.query_chroma(
            collection=self.collection_name,
            query_texts=topics,
            n_results=3,
        )
        all_docs = []
        if results and "documents" in results:
            for doc_list in results["documents"]:
                all_docs.extend(doc_list)
        return all_docs

    async def __aenter__(self):
        await self.search_connector.__aenter__()
        await self.http_connector.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.search_connector.__aexit__(exc_type, exc_val, exc_tb)
        await self.http_connector.__aexit__(exc_type, exc_val, exc_tb)
