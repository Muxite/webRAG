import logging
import json
from typing import Dict, List
from shared.connector_config import ConnectorConfig
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.tick_output import TickOutput, ActionType
from agent.app.prompt_builder import PromptBuilder
from agent.app.prompt_builder import build_payload
from agent.app.observation import clean_operation
from agent.app.connector_chroma import ConnectorChroma
from shared.pretty_log import pretty_log
from urllib.parse import urlparse
import asyncio
from shared.models import FinalResult


class Agent:
    """
    Autonomous RAG agent that executes a tick-based reasoning loop.
    Handles graceful degradation when external services fail.
    """

    def __init__(self, mandate: str, max_ticks: int = 50):
        """
        Initialize the agent with a mandate and tick limit.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self.mandate = mandate
        self.max_ticks = max_ticks
        self.current_tick = 0

        self.history: List[str] = []
        self.notes: List[str] = []
        self.deliverables: List[str] = []
        self.pending_data_topics: List[str] = []
        self.retrieved_context: List[str] = []
        self.observations: str = ""

        self.config = ConnectorConfig()
        self.connector_llm = ConnectorLLM(self.config)
        self.connector_search = ConnectorSearch(self.config)
        self.connector_http = ConnectorHttp(self.config)
        self.connector_chroma = ConnectorChroma(self.config)

        self.collection_name = "agent_memory"

    async def initialize(self) -> bool:
        """Initialize all connectors and setup dependencies."""
        self._logger.info("Initializing agent connectors...")

        search_ready = await self.connector_search.init_search_api()
        chroma_ready = await self.connector_chroma.init_chroma()
        llm_ready = self.connector_llm.llm_api_ready

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

        self._logger.info(f"Agent started: {self.mandate}\n")


        while self.current_tick < self.max_ticks:
            await asyncio.sleep(1)
            self.current_tick += 1
            self._logger.info(f"=== TICK {self.current_tick}/{self.max_ticks} ===")

            prompt_builder = PromptBuilder(
                mandate=self.mandate,
                short_term_summary=self.history,

                notes=(self.notes[-1] if self.notes else ""),
                retrieved_long_term=self.retrieved_context,
                observations=self.observations
            )
            prompt = build_payload(prompt_builder.build_messages(), True)
            pretty_log(prompt_builder.get_summary(), logger=self._logger, indents=1)

            llm_response = await self.connector_llm.query_llm(prompt)
            if llm_response is None:
                self._logger.error("LLM query failed after retries; breaking to final output.")
                self._logger.debug(prompt)
                break
            try:
                tick_output = TickOutput(json.loads(llm_response))
                pretty_log(tick_output.get_summary(), logger=self._logger, indents=1)
            except Exception as e:
                self._logger.error(f"Failed to parse LLM output: {e}")
                self._logger.debug(llm_response)
                self._logger.debug(prompt)
                continue

            self.observations = ""
            self._apply_tick_output(tick_output)
            await self._do_action(tick_output)
            chroma_topics = tick_output.show_requested_data_topics()
            if chroma_topics:
                self._logger.info(f"ChromaDB: Retrieving context for topics: {chroma_topics}")
                self.retrieved_context = await self._retrieve_chroma(chroma_topics)
                self._logger.info(f"ChromaDB: Retrieved context: {self.retrieved_context}")
            else:
                self.retrieved_context = []

            await self._store_chroma(
                documents=tick_output.get_vector_documents(),
                metadatas=tick_output.get_vector_metadatas(),
                ids=tick_output.get_vector_ids()
            )

            if tick_output.show_next_action()[0] == ActionType.EXIT:
                self._logger.info("Agent exit action received, stopping.")
                break

        final_output: FinalResult = await self._final_output()
        try:
            return final_output.model_dump()
        except Exception:
            return final_output.dict()

    def _apply_tick_output(self, tick_output: TickOutput):
        """
        Update local memory fields based on tick output.
        :param tick_output: TickOutput object containing agent's current state.
        """
        if tick_output.show_history():
            self.history.append(f"[Tick {self.current_tick}] {tick_output.show_history()}")
        if tick_output.show_notes():
            self.notes.append(tick_output.show_notes())
        if tick_output.deliverable().strip():
            self.deliverables.append(tick_output.deliverable())

    async def _do_action(self, tick_output: TickOutput):
        """
        Perform the action specified by the tick output.
        :param tick_output:
        """
        action, param = tick_output.show_next_action()
        if action == ActionType.SEARCH:
            self._logger.info(f"Searching for '{param}'...")
            await self._perform_web_search(param)
        elif action == ActionType.VISIT:
            self._logger.info(f"Visiting '{param}'...")
            await self._perform_visit(param)
        elif action == ActionType.THINK:
            self._logger.info("Agent is thinking...")
            pass
        elif action == ActionType.EXIT:
            self._logger.info("Agent exit action received, stopping.")
            pass

    async def _perform_web_search(self, query: str, count: int = 10):
        """
        Run a web search using the Connector and update self.observations.
        :param query: Topic/question to search for.
        :param count: Number of search results to include.
        """
        search_results = await self.connector_search.query_search(query, count=count)
        obs = PromptBuilder.build_web_search_observation(query, search_results)
        self.observations += obs

    async def _perform_visit(self, url: str):
        """
        Visit a URL and extract text to update self.observations.
        :param url: URL to visit.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            self._logger.error(f"Invalid URL provided: '{url}'")
            self.observations += PromptBuilder.build_invalid_url_observation(url)
            return

        result = await self.connector_http.request("GET", url, retries=3)
        if result.error:
            self.observations += PromptBuilder.build_visit_error_observation(url, result.status)
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
            self._logger.warning(f"Failed to extract body from {url}: {e}")
            summary = "[No main content found]"

        self.observations += PromptBuilder.build_visit_observation(url, summary)

    async def _store_chroma(self, ids: List[str], metadatas: List[Dict], documents: List[str]):
        if not self.connector_chroma.chroma_api_ready or not documents:
            return
        await self.connector_chroma.add_to_chroma(
            collection=self.collection_name,
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )

    async def _retrieve_chroma(self, topics: List[str]) -> List[str]:
        """
        Accesses ChromaDB to retrieve documents for a list of topics.
        :param topics: List of sentences or ideas to search for.
        :return: List of documents (str) that match the topics.
        """
        if not self.connector_chroma.chroma_api_ready or not topics:
            return []
        results = await self.connector_chroma.query_chroma(
            collection=self.collection_name,
            query_texts=topics,
            n_results=3,
        )
        all_docs = []
        if results and "documents" in results:
            for doc_list in results["documents"]:
                all_docs.extend(doc_list)
        return all_docs

    async def _final_output(self) -> FinalResult:
        """
        Generate final output that combines accumulated deliverables and notes into a final answer to the users mandate.
        keys: final_deliverable, action_summary, success
        :return: Dictionary with final deliverable and action summary.
        """
        final_messages = PromptBuilder.build_final_messages(
            mandate=self.mandate,
            history=self.history,
            notes=self.notes,
            deliverables=self.deliverables,
            retrieved_context=self.retrieved_context,
        )
        final_prompt = build_payload(final_messages, True)
        self._logger.info("=== GENERATING FINAL OUTPUT ===")

        llm_response = await self.connector_llm.query_llm(final_prompt)
        if llm_response is None:
            self._logger.error("Final output LLM query failed.")

            deliverable_text = "\n\n".join(d for d in self.deliverables if d.strip())
            if not deliverable_text:
                deliverable_text = (
                    "We could not complete the full mandate due to unavailable LLM service. "
                )
            return FinalResult(
                final_deliverable=deliverable_text,
                action_summary="",
                success=False,
            )

        try:
            final_output = json.loads(llm_response)
            self._logger.info("Final output generated successfully")
            return FinalResult(
                final_deliverable=final_output.get("deliverable", ""),
                action_summary=final_output.get("summary", ""),
                success=True,
            )
        except Exception as e:
            self._logger.error(f"Failed to parse final output: {e}")
            return FinalResult(
                final_deliverable="",
                action_summary="",
                success=False,
            )

    async def __aenter__(self):
        await self.connector_search.__aenter__()
        await self.connector_http.__aenter__()
        await self.connector_llm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connector_search.__aexit__(exc_type, exc_val, exc_tb)
        await self.connector_http.__aexit__(exc_type, exc_val, exc_tb)
        await self.connector_llm.__aexit__(exc_type, exc_val, exc_tb)
