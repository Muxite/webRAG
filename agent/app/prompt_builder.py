from typing import Optional, List, Dict, Any

class PromptBuilder:
    """
    Construct an OpenAI formatted JSON prompt for one LLM call.
    The agent should maintain the following memory:
      - short_term (a small summary of recent actions)
      - notes (freeform non-deterministic notes)
      - long_term (tagged string dumps pulled from cache)
    """

    SYSTEM_INSTRUCTIONS = (
        "You are an autonomous agent. You work in ticks, info from previous ticks can be passed to the current tick."
        "On each tick, you receive:\n"
        "- MANDATE: Your immutable primary directive from the user. This is your end goal.\n"
        "- HISTORY: An enumerated list of your actions on this branch. A new line is added every tick.\n"
        "- NOTE MEMORY: free-form notes you have been carrying. Passed from tick to tick. Store future plans here.\n"
        "- RETRIEVED CONTEXT: Facts or data retrieved from a cache."
        " These are the result of last tick's action.\n"
        "- OBSERVATIONS: new input / page / HTML content for this tick.\n\n"
        "You MUST output valid JSON with exactly these top-level keys:\n"
        "* log "
        "* history_update: A very short summary of what will happen this tick. This will be appended.\n"
        "* note_update: Your new note memory, important info for next tick to know.\n"
        "* cache_update: dict - new long-term facts or data to store persistently (shared across agents)."
        "Very useful for preliminary research after getting the mandate. Use a tag that is easy to search for."
        "Ex: {tree definition: a tree is a...}\n"
        "* next_action: Your next action. \n"
        "No extra keys. Do not include commentary or explanation outside the JSON."
    )
    def __init__(
        self,
        mandate: Optional[str] = None,
        short_term_summary: Optional[str] = None,
        notes: Optional[str] = None,
        retrieved_long_term: Optional[str] = None,
        observations: Optional[str] = None
    ):
        """
        :param mandate: The immutable primary directive from the user, representing the agent's end goal.

        :param short_term_summary: A concise summary of recent actions in this branch, providing context for the current tick.

        :param notes: Free-form notes carried from previous ticks, useful for storing future plans and considerations.

        :param retrieved_long_term: Facts or data retrieved from the long-term cache, providing persistent knowledge relevant to the task.

        :param observations: New input, page, or HTML content for this tick, offering fresh data for processing.

        Initializes the agent's internal state with the provided parameters, setting up the context for
        decision-making and action in the current tick.
        """

        self.mandate = mandate or ""
        self.short_term_summary = short_term_summary or ""
        self.notes = notes or ""
        self.retrieved_long_term = retrieved_long_term or ""
        self.observations = observations or ""


    def _build_user_message(self) -> str:
        """Assemble the content of the user message."""
        parts: List[str] = []

        if self.mandate:
            parts.append("MANDATE:\n" + self.mandate.strip())

        if self.short_term_summary:
            parts.append("SHORT TERM MEMORY (recent history):\n" + self.short_term_summary.strip())

        if self.notes:
            parts.append("NOTES:\n" + self.notes.strip())

        if self.retrieved_long_term:
            parts.append("RETRIEVED LONG-TERM CONTEXT:\n" + self.retrieved_long_term.strip())

        if self.observations:
            parts.append("OBSERVATIONS:\n" + self.observations.strip())

        return "\n\n".join(parts)

    def build_messages(self) -> List[Dict[str, str]]:
        """
        Returns the list of messages to send to OpenAI (or compatible) model.
        """
        system_msg = {"role": "system", "content": self.SYSTEM_INSTRUCTIONS}
        user_msg = {"role": "user", "content": self._build_user_message()}
        return [system_msg, user_msg]

    def build_payload(self) -> Dict[str, Any]:
        """
        Build the full payload dict (model, messages, plus optional llm_kwargs
        such as temperature, max_tokens, etc.).
        """
        return {
            "model": "llama",
            "messages": self.build_messages(),
            "temperature": 0.4,
            "max_tokens": 3200,
            "response_format": {"type": "json_object"}
        }