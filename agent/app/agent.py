import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from app.prompt_builder import PromptBuilder
from app.connector import Connector
from app.tick_output import TickOutput, ActionType


class Agent:
    """
    An autonomous RAG agent that executes a single reasoning tick.

    It manages the agent's memory components, constructs the LLM prompt,
    calls the LLM API, and updates its state based on the LLM's output.
    It creates and manages its own Connector instance.
    """

    def __init__(
            self,
            mandate: str,
            worker_type: str,
            logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Agent's state and dependencies.

        :param str mandate: The primary directive for the agent.
        :param Optional[logging.Logger] logger: Logger instance.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._connector = Connector(worker_type=worker_type)
        self._mandate = mandate
        self._notes: str = ""
        self._short_term_summary: List[str] = []
        self._retrieved_long_term: List[str] = []
        self._observations: str = ""
        self.is_running = True

    def set_retrieved_context(self, context: List[str]):
        """Update the RAG chunks retrieved from the vector DB for the current tick."""
        self._retrieved_long_term = context

    def set_observations(self, observations: str):
        """Set the new external input (e.g., user text, search results, HTML)."""
        self._observations = observations

    async def _call_llm(self, builder_obj: 'PromptBuilder') -> Dict[str, Any]:
        """
        Makes the asynchronous API call to the LLM endpoint using a PromptBuilder object.

        :param PromptBuilder builder_obj: A PromptBuilder instance configured with the current tick's memory.
        :return Dict[str, Any]: The parsed JSON response from the LLM.
        """
        if not self._connector.llm_api_ready:
            self.logger.error("LLM API not ready. Cannot call model.")
            if not await self._connector.init_llm():
                raise RuntimeError("LLM API not ready.")

        payload = builder_obj.build_payload()
        llm_url = self._connector.llm_url
        session = self._connector.get_session()

        self.logger.info(f"Calling LLM at {llm_url}...")

        try:
            async with session.post(llm_url, json=payload, timeout=self._connector.default_timeout * 2) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    content_str = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

                    try:
                        llm_output_dict = json.loads(content_str)
                        return llm_output_dict
                    except json.JSONDecodeError as e:
                        self.logger.error(f"LLM output was not valid JSON: {e}\nRaw output: {content_str}")
                        return {}
                else:
                    text = await resp.text()
                    self.logger.error(f"LLM call failed with status {resp.status}: {text}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error during LLM API call: {e}")
            return {}

    def _update_state(self, tick_output: 'TickOutput'):
        """
        Updates the agent's internal memory state based on the TickOutput.
        """
        if tick_output.history_update:
            self._short_term_summary.append(tick_output.history_update)

        self._notes = tick_output.note_update

        if tick_output.next_action[0] == ActionType.EXIT:
            self.is_running = False
            self.logger.info("Agent received EXIT action. Halting.")

    async def act(self) -> 'TickOutput':
        """
        Execute one full reasoning tick of the agent, calling the LLM and then
        executing the specified action.
        """

        builder = PromptBuilder(
            mandate=self._mandate,
            short_term_summary=self._short_term_summary,
            notes=self._notes,
            retrieved_long_term=self._retrieved_long_term,
            observations=self._observations,
        )

        llm_output_dict = await self._call_llm(builder)
        tick_output = TickOutput(llm_output_dict)
        self._update_state(tick_output)

        # 4. Handle Long-Term Memory (LTM) Update
        # In a full agent, this is where the orchestrator would save new facts to the Vector DB.
        # We log it here for demonstration, but the agent itself doesn't own the DB.
        if tick_output.cache_update:
            self.logger.info(f"Preparing {len(tick_output.cache_update)} facts for LTM cache.")
            # NOTE: A real implementation would call an external vector DB client here.
            # e.g., self._vector_db_client.insert_records(tick_output.to_vector_records())

        # 5. Execute the Next Action
        action_type, action_param = tick_output.show_next_action()

        # Reset ephemeral inputs BEFORE executing the action. The execution step
        # will then populate the inputs (observations/context) for the *next* tick.
        self._retrieved_long_term = []
        self._observations = ""

        if action_type == ActionType.EXIT:
            self.logger.info("ACTION: EXIT requested. Agent program terminating.")
            # self.is_running is already set to False in _update_state

        elif action_type == ActionType.SEARCH:
            self.logger.info(f"ACTION: SEARCHing the web for: '{action_param}'")
            # In a real setup, this would call an external tool.
            # self._observations = await self._execute_search(action_param)
            self._observations = f"Search results for '{action_param}' retrieved..."

        elif action_type == ActionType.VISIT:
            self.logger.info(f"ACTION: VISITING URL: '{action_param}'")
            # In a real setup, this would call an HTTP client and HTML parser.
            # self._observations = await self._execute_visit(action_param)
            self._observations = f"Content from {action_param} fetched and summarized..."

        elif action_type == ActionType.THINK:
            self.logger.info("ACTION: THINK requested. No external tool executed.")
            # The agent relies on the 'data' request being fulfilled by the orchestrator
            # before the *next* tick, giving it time to reason internally.
            pass

        # 6. Return the TickOutput
        # The orchestrator uses this object to:
        # a) Execute the LTM update (cache_update)
        # b) Fulfill the LTM retrieval request ('data' topics)
        return tick_output

    async def __aenter__(self):
        """Support async context manager for connection setup."""
        await self._connector.await_all_connections_ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        await self._connector.close_connections()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Return the agent's current memory state.
        """
        return {
            "mandate": self._mandate,
            "notes": self._notes,
            "short_term_summary": self._short_term_summary,
            "is_running": self.is_running,
        }