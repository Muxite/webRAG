import json
from typing import Dict, List, Optional, Callable, Any
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.tick_output import TickOutput, ActionType
from agent.app.prompt_builder import TickPromptBuilder, FinalPromptBuilder, ObservationBuilder
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.connector_chroma import ConnectorChroma
from shared.pretty_log import pretty_log
from urllib.parse import urlparse
import asyncio
import logging
import os
from shared.models import FinalResult
from agent.app.trace_recorder import TraceRecorder
from agent.app.agent_io import AgentIO


class Agent:
    """
    Autonomous RAG agent that executes a tick-based reasoning loop.
    Handles graceful degradation when external services fail.
    """

    def __init__(
        self,
        mandate: str,
        max_ticks: int = 50,
        connector_llm: ConnectorLLM = None,
        connector_search: ConnectorSearch = None,
        connector_http: ConnectorHttp = None,
        connector_chroma: ConnectorChroma = None,
        model_name: str | None = None,
        tracer: Optional[TraceRecorder] = None,
        agent_io: Optional[AgentIO] = None,
        model_selector: Optional[Callable[[str, int], Optional[str]]] = None,
        idea_dag_settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent with a mandate and tick limit.
        Connectors are injected via dependency injection for reuse across multiple mandates.
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

        self.connector_llm = connector_llm
        self.connector_search = connector_search
        self.connector_http = connector_http
        self.connector_chroma = connector_chroma
        self.model_name = model_name
        self.tracer = tracer
        self.model_selector = model_selector
        self.idea_dag_settings = idea_dag_settings

        self.io = agent_io or AgentIO(
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
        )

        self.collection_name = "agent_memory"
        self.blocked_count = 0
        self.blocked_limit = int(os.environ.get("AGENT_BLOCKED_LIMIT", "3"))
        self.hard_stop_reason: Optional[str] = None
        self.metrics = {
            "ticks": 0,
            "searches": 0,
            "search_results": 0,
            "visits": 0,
            "visited_urls": [],
            "observations_chars": 0,
            "observations_bytes": 0,
            "prompt_chars": 0,
            "prompt_bytes": 0,
            "llm_output_chars": 0,
            "llm_output_bytes": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "llm_calls": 0,
            "llm_inputs": [],
            "llm_outputs": [],
            "cache_docs_added": 0,
            "cache_doc_chars": 0,
            "cache_doc_bytes": 0,
            "retrieved_chunks": 0,
            "retrieved_chars": 0,
            "retrieved_bytes": 0,
        }

    def _track_text(self, field: str, text: str) -> None:
        """
        Track text size in chars and bytes.
        :param field: Metric prefix to update.
        :param text: Text to measure.
        :return: None
        """
        if text is None:
            return
        value = str(text)
        self.metrics[f"{field}_chars"] = self.metrics.get(f"{field}_chars", 0) + len(value)
        self.metrics[f"{field}_bytes"] = self.metrics.get(f"{field}_bytes", 0) + len(value.encode("utf-8"))

    async def initialize(self) -> bool:
        """Verify that all required connectors are available."""
        self._logger.info("Verifying agent connectors...")

        if not self.connector_llm or not self.connector_search or not self.connector_http or not self.connector_chroma:
            self._logger.error("Missing required connectors")
            return False

        if self.model_name:
            self.connector_llm.set_model(self.model_name)
        if self.tracer:
            self.tracer.record(
                "init",
                {
                    "mandate": self.mandate,
                    "max_ticks": self.max_ticks,
                    "model": self.connector_llm.get_model() if self.connector_llm else None,
                    "blocked_limit": self.blocked_limit,
                },
            )

        llm_ready = self.connector_llm.llm_api_ready
        search_ready = self.connector_search.search_api_ready
        chroma_ready = self.connector_chroma.chroma_api_ready

        return all([llm_ready, search_ready, chroma_ready])

    def set_model_name(self, model_name: str | None) -> None:
        """
        Update the model name used for subsequent LLM calls.
        :param model_name: Model identifier to set.
        :return: None
        """
        if model_name and model_name.strip():
            self.model_name = model_name.strip()
            if self.connector_llm:
                self.connector_llm.set_model(self.model_name)

    def _select_model_for_call(self, purpose: str) -> Optional[str]:
        """
        Select the model name for a specific call.
        :param purpose: Label for the call context.
        :returns: Model name or None.
        """
        if self.model_selector:
            return self.model_selector(purpose, self.current_tick)
        return self.model_name

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

        use_idea_dag_env = os.environ.get("AGENT_USE_IDEA_DAG", "").lower()
        settings = self.idea_dag_settings or load_idea_dag_settings()
        use_idea_dag = use_idea_dag_env in ("1", "true", "yes", "on")
        if use_idea_dag_env == "":
            use_idea_dag = bool(settings.get("enable_idea_dag", False))

        if use_idea_dag:
            engine = IdeaDagEngine(
                io=self.io,
                model_name=self._select_model_for_call("idea_dag"),
                settings=settings,
            )
            result = await engine.run(self.mandate, max_steps=self.max_ticks)
            result["metrics"] = self.metrics
            return result

        self._logger.info(f"Agent started: {self.mandate}\n")

        final_payload = None
        while self.current_tick < self.max_ticks:
            await asyncio.sleep(1)
            self.current_tick += 1
            self.metrics["ticks"] = self.current_tick
            self._logger.info(f"=== TICK {self.current_tick}/{self.max_ticks} ===")
            if self.tracer:
                self.tracer.record(
                    "tick_start",
                    {"tick": self.current_tick, "max_ticks": self.max_ticks},
                )

            prompt_builder = TickPromptBuilder(
                mandate=self.mandate,
                short_term_summary=self.history,

                notes=(self.notes[-1] if self.notes else ""),
                retrieved_long_term=self.retrieved_context,
                observations=self.observations,
                current_tick=self.current_tick,
                max_ticks=self.max_ticks,
            )
            messages = prompt_builder.build_messages()
            prompt = self.io.build_llm_payload(
                messages=messages,
                json_mode=True,
                model_name=self._select_model_for_call("tick"),
            )
            user_message = prompt_builder._build_user_message()
            self._track_text("prompt", user_message)
            self._track_text("prompt", prompt_builder.SYSTEM_INSTRUCTIONS)
            self.metrics["llm_inputs"].append(messages)
            if self.tracer:
                self.tracer.record(
                    "llm_input",
                    {
                        "tick": self.current_tick,
                        "messages": messages,
                        "system": prompt_builder.SYSTEM_INSTRUCTIONS,
                        "user": user_message,
                    },
                )
            pretty_log(prompt_builder.get_summary(), logger=self._logger, indents=1)

            llm_response = await self.io.query_llm(prompt, model_name=self._select_model_for_call("tick"))
            if llm_response is None:
                self._logger.error("LLM query failed after retries; breaking to final output.")
                self._logger.debug(prompt)
                break
            self._track_text("llm_output", llm_response)
            self.metrics["llm_outputs"].append(llm_response)
            if self.tracer:
                self.tracer.record(
                    "llm_output",
                    {"tick": self.current_tick, "content": llm_response},
                )
            usage = self.io.pop_last_llm_usage()
            if usage:
                self.metrics["llm_calls"] += 1
                self.metrics["llm_prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.metrics["llm_completion_tokens"] += usage.get("completion_tokens", 0)
                self.metrics["llm_total_tokens"] += usage.get("total_tokens", 0)
                if self.tracer:
                    self.tracer.record(
                        "llm_usage",
                        {"tick": self.current_tick, "usage": usage},
                    )
            try:
                tick_output = TickOutput(json.loads(llm_response))
                pretty_log(tick_output.get_summary(), logger=self._logger, indents=1)
            except Exception as e:
                self._logger.error(f"Failed to parse LLM output: {e}")
                self._logger.debug(llm_response)
                self._logger.debug(prompt)
                continue

            self._logger.info(
                "Tick state",
                extra={
                    "tick": self.current_tick,
                    "history_count": len(self.history),
                    "notes_len": len(self.notes[-1]) if self.notes else 0,
                    "observations_len": len(self.observations or ""),
                    "retrieved_context_count": len(self.retrieved_context),
                },
            )
            if self.notes:
                self._logger.info("Notes", extra={"tick": self.current_tick, "notes": self.notes[-1]})
            if self.history:
                self._logger.info("History", extra={"tick": self.current_tick, "history": self.history[-3:]})
            if self.observations:
                self._logger.info("Observations", extra={"tick": self.current_tick, "observations": self.observations[:2000]})
            if self.retrieved_context:
                self._logger.info(
                    "Retrieved context",
                    extra={"tick": self.current_tick, "retrieved": self.retrieved_context[:5]},
                )

            action, param = tick_output.show_next_action()
            self._logger.info(
                "Tick decision",
                extra={
                    "tick": self.current_tick,
                    "next_action": str(action),
                    "param": param,
                    "cache_update": len(tick_output.get_vector_documents()),
                    "cache_retrieve": len(tick_output.show_requested_data_topics()),
                },
            )

            self.observations = ""
            self._apply_tick_output(tick_output)
            await self._do_action(tick_output)
            if self.hard_stop_reason:
                self._logger.warning(f"Hard stop: {self.hard_stop_reason}")
                break
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
            if self.tracer:
                self.tracer.record(
                    "tick_end",
                    {
                        "tick": self.current_tick,
                        "history_update": tick_output.show_history(),
                        "note_update": tick_output.show_notes(),
                        "deliverable": tick_output.deliverable(),
                        "next_action": tick_output.show_next_action(),
                        "cache_retrieve": tick_output.show_requested_data_topics(),
                        "cache_update_count": len(tick_output.get_vector_documents()),
                    },
                )

            if tick_output.show_next_action()[0] == ActionType.EXIT or self.current_tick >= self.max_ticks:
                self._logger.info("Agent exit action received, stopping.")
                final_payload = await self._final_output()
                break

        if final_payload is None:
            final_payload = await self._final_output()
        try:
            payload = final_payload.model_dump()
        except Exception:
            payload = final_payload.dict()
        payload["metrics"] = self.metrics
        if self.tracer:
            self.tracer.record("final_output", payload)
        return payload

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
        try:
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
        except Exception as exc:
            self._logger.warning(f"Action failed: {action} {param} ({exc})")
            obs = ObservationBuilder.build_exception_observation(str(action), str(exc))
            self.observations += obs
            self._track_text("observations", obs)
            if "403" in str(exc) or "blocked" in str(exc).lower():
                self.blocked_count += 1
                if self.blocked_count >= self.blocked_limit:
                    summary = (
                        f"Stopping early after repeated blocks. "
                        f"Attempts blocked: {self.blocked_count}. Last error: {exc}"
                    )
                    self.notes.append(summary)
                    self.deliverables.append(
                        f"Stopped early due to repeated blocks. {summary}"
                    )
                    self.hard_stop_reason = summary
            if self.tracer:
                self.tracer.record(
                    "action_error",
                    {"action": str(action), "param": param, "error": str(exc)},
                )

    async def _perform_web_search(self, query: str, count: int = 10):
        """
        Run a web search using the Connector and update self.observations.
        :param query: Topic/question to search for.
        :param count: Number of search results to include.
        """
        self.metrics["searches"] += 1
        try:
            search_results = await self.io.search(query, count=count)
        except Exception as exc:
            error_text = f"Search failed for '{query}': {exc}"
            obs = ObservationBuilder.build_exception_observation("search", error_text)
            self.observations += obs
            self._track_text("observations", obs)
            return
        if search_results:
            self.metrics["search_results"] += len(search_results)
        obs = ObservationBuilder.build_web_search_observation(query, search_results)
        self.observations += obs
        self._track_text("observations", obs)
        if self.tracer:
            self.tracer.record(
                "search_observation",
                {"query": query, "results_count": len(search_results or [])},
            )

    async def _perform_visit(self, url: str):
        """
        Visit a URL and extract text to update self.observations.
        :param url: URL to visit.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            self._logger.error(f"Invalid URL provided: '{url}'")
            obs = ObservationBuilder.build_invalid_url_observation(url)
            self.observations += obs
            self._track_text("observations", obs)
            return

        try:
            summary = await self.io.visit(url)
        except Exception as e:
            self._logger.warning(f"Failed to extract body from {url}: {e}")
            summary = "[No main content found]"

        obs = ObservationBuilder.build_visit_observation(url, summary)
        self.observations += obs
        self._track_text("observations", obs)
        self.metrics["visits"] += 1
        self.metrics["visited_urls"].append(url)
        if self.tracer:
            self.tracer.record("visit_observation", {"url": url})

    async def _store_chroma(self, ids: List[str], metadatas: List[Dict], documents: List[str]):
        if not self.connector_chroma.chroma_api_ready or not documents:
            return
        self.metrics["cache_docs_added"] += len(documents)
        for doc in documents:
            self._track_text("cache_doc", doc)
        await self.io.store_chroma(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        if self.tracer:
            self.tracer.record(
                "cache_store",
                {"count": len(documents), "metadatas": metadatas},
            )

    async def _retrieve_chroma(self, topics: List[str]) -> List[str]:
        """
        Accesses ChromaDB to retrieve documents for a list of topics.
        :param topics: List of sentences or ideas to search for.
        :return: List of documents (str) that match the topics.
        """
        if not self.connector_chroma.chroma_api_ready or not topics:
            return []
        all_docs = await self.io.retrieve_chroma(topics, n_results=3)
        self.metrics["retrieved_chunks"] += len(all_docs)
        for doc in all_docs:
            self._track_text("retrieved", doc)
        if self.tracer:
            self.tracer.record(
                "cache_retrieve",
                {"topics": topics, "count": len(all_docs)},
            )
        return all_docs

    async def _final_output(self) -> FinalResult:
        """
        Generate final output that combines accumulated deliverables and notes into a final answer to the users mandate.
        keys: final_deliverable, action_summary, success
        :return: Dictionary with final deliverable and action summary.
        """
        final_messages = FinalPromptBuilder(
            mandate=self.mandate,
            history=self.history,
            notes=self.notes,
            deliverables=self.deliverables,
            retrieved_context=self.retrieved_context,
        ).build_messages()
        final_prompt = self.io.build_llm_payload(
            messages=final_messages,
            json_mode=True,
            model_name=self._select_model_for_call("final"),
        )
        self.metrics["llm_inputs"].append(final_messages)
        if self.tracer:
            self.tracer.record("final_llm_input", {"messages": final_messages})
        self._logger.info("=== GENERATING FINAL OUTPUT ===")

        llm_response = await self.io.query_llm(final_prompt, model_name=self._select_model_for_call("final"))
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
            self._track_text("llm_output", llm_response)
            self.metrics["llm_outputs"].append(llm_response)
            usage = self.io.pop_last_llm_usage()
            if usage:
                self.metrics["llm_calls"] += 1
                self.metrics["llm_prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.metrics["llm_completion_tokens"] += usage.get("completion_tokens", 0)
                self.metrics["llm_total_tokens"] += usage.get("total_tokens", 0)
            deliverable = final_output.get("deliverable", "")
            summary = final_output.get("summary", "")
            if not isinstance(deliverable, str):
                deliverable = json.dumps(deliverable, ensure_ascii=True)
            if not isinstance(summary, str):
                summary = json.dumps(summary, ensure_ascii=True)
            return FinalResult(
                final_deliverable=deliverable,
                action_summary=summary,
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
        """Enter async context manager, ensuring connectors are ready."""
        if self.connector_search:
            await self.connector_search.__aenter__()
        if self.connector_http:
            await self.connector_http.__aenter__()
        if self.connector_llm:
            await self.connector_llm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager, cleaning up connector resources.
        
        Note: Connectors are shared and reused across multiple Agent instances,
        so we do NOT close them here. They are managed at the InterfaceAgent level.
        """
        # Do not close shared connectors - they are reused across multiple agents
        # The InterfaceAgent manages their lifecycle
        pass
