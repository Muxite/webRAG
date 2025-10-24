from typing import Optional, List, Dict, Any


class PromptBuilder:
    """
    Construct a JSON prompt for one LLM call in a tick-based RAG agent.

    The agent maintains the following memory components:
      - short_term: recent tick summaries (enumerated)
      - notes: persistent freeform scratchpad text passed from tick to tick
      - long_term: retrieved RAG chunks from a vector database (semantic cache)
    """

    SYSTEM_INSTRUCTIONS = (
        "You are an autonomous agent reasoning in discrete ticks, where each tick is one step of thought or action. "
        "You persist memory between ticks through NOTES and cached data. You can deliver 'deliverable' in each tick. "
        "Deliver whatever matches your mandate, trickling deliverable every few ticks is highly encouraged. "
        "Do not deliver unfinished products, Ex: if you are returning a link, write down your goals, visit the link, "
        "read the content, and then deliver the link once you are sure it is good. \n\n"

        "INPUT STRUCTURE:\n"
        "- MANDATE: Your immutable directive or mission.\n"
        "- SHORT TERM MEMORY: Summary of recent actions in this reasoning branch.\n"
        "- NOTES: Your personal scratchpad for reasoning, include hypotheses, plans, partial results, or reminders. "
        "Everything written here will be passed unchanged into the next tick, so organize it usefully.\n"
        "- RETRIEVED CONTEXT: Information fetched from your vector database based on the previous tick’s 'data' request.\n"
        "- OBSERVATIONS: New external input, such as user text or HTML, from this tick.\n\n"

        "SYSTEM RULES:\n"
        "- If you need specific knowledge, list it in the 'data' field as a short, comma-separated set of topics "
        "(ex: 'fish, ocean ecosystem').\n"
        "- The system will retrieve relevant semantic chunks and include them in 'RETRIEVED CONTEXT' on the next tick.\n"
        "- Summarize observations and facts/data in the 'cache_update'. Use the format {topic : summary} for as many "
        "important pieces of info as you would like. These will later be accessible by the 'data' field.\n"
        "Recommended Strategy: make many searches at the start, and cache_update as many comprehensible facts. "
        "The cache is extremely fast, so do not worry about having too much data.\n\n"
        
        "OUTPUT FORMAT:\n"
        "You must output valid JSON with EXACTLY these top-level keys:\n"
        "* history_update : one-sentence summary of this tick’s action\n"
        "* note_update : updated notes or reasoning to persist into the next tick\n"
        "* cache_update : dictionary of long-term facts or definitions to store persistently\n"
        "* next_action : description of your next goal or operation\n"
        "* data : comma-separated topics to retrieve for the next tick\n"
        "* deliverable : a final output to be sent to the user based on your mandate, a viable product or nothing.\n"
        "Do not include any commentary or keys outside this JSON.\n\n"
        
        "INFO ON ACTION:\n"
        "The agent must output 'next_action': 'ACTION_NAME, PARAM'\n"
        "Allowed actions (case sensitive):\n"
        "* think : reason internally, param unused. You can use this to buy time and search your database.\n"
        "* search : search internet for a search term. Useful for preliminary data gathering.\n"
        "Ex: 'next_action': 'search, how to implement BFS'\n"
        "* visit : visits a link directly. Very useful if you are on a webpage and want to investigate further.\n"
        "Ex: 'next_action': 'visit, https://en.wikipedia.org/wiki/Main_Page'\n"
        "* exit : ends the program, param unused. PROGRAM WILL END, USE ONLY WHEN FINISHED EVERYTHING."

    )

    def __init__(
        self,
        mandate: Optional[str] = None,
        short_term_summary: Optional[List[str]] = None,
        notes: Optional[str] = None,
        retrieved_long_term: Optional[List[str]] = None,
        observations: Optional[str] = None,
    ):
        """
        Initializes the agent's context for one tick.
        """
        self._mandate = mandate or ""
        self._short_term_summary = short_term_summary or []
        self._notes = notes or ""
        self._retrieved_long_term = retrieved_long_term or []
        self._observations = observations or ""

    def set_mandate(self, text: str):
        """Set the agent's mandate."""
        self._mandate = text.strip()

    def add_history_entry(self, summary: str):
        """Add a single summary entry to the short-term memory."""
        self._short_term_summary.append(summary.strip())

    def update_notes(self, new_notes: str):
        """Replace or append to the freeform note scratchpad."""
        self._notes = new_notes.strip()

    def add_retrieved_context(self, chunk: str):
        """Add a single RAG chunk retrieved from the vector database."""
        self._retrieved_long_term.append(chunk.strip())

    def update_observations(self, new_obs: str):
        """Replace or append to the observation text."""
        self._observations = new_obs.strip()


    def _format_section(self, title: str, content: str) -> str:
        """
        Format a section of the user message.
        :param str title: Section title.
        :param str content: Section content.
        :return str: Formatted section.
        """
        return f"{title}:\n{content.strip()}"

    def _build_user_message(self) -> str:
        """
        Assemble the full user message with easily readable, distinct sections.
        :return str: The contents of the user message.
        """
        parts: List[str] = []

        if self._mandate:
            parts.append(self._format_section("MANDATE", self._mandate))

        if self._short_term_summary:
            joined_history = "\n".join(
                f"{i + 1}. {entry}" for i, entry in enumerate(self._short_term_summary)
            )
            parts.append(self._format_section("SHORT TERM MEMORY (recent history)", joined_history))

        if self._notes.strip():
            parts.append(self._format_section("NOTES", self._notes))

        if self._retrieved_long_term:
            joined_chunks = "\n".join(
                f"[{i + 1}] {chunk}" for i, chunk in enumerate(self._retrieved_long_term)
            )
            parts.append(self._format_section("RETRIEVED LONG-TERM CONTEXT", joined_chunks))

        if self._observations.strip():
            parts.append(self._format_section("OBSERVATIONS", self._observations))

        return "\n\n".join(parts)


    def build_messages(self) -> List[Dict[str, str]]:
        """
        Returns a message list compatible with OpenAI-style chat completions.
        :return list: List of dicts, each with "role" and "content" keys.
        """
        return [
            {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": self._build_user_message()},
        ]

    def build_payload(self) -> Dict[str, Any]:
        """
        Build the final dict payload for API call.
        :return dict: OpenAI-compatible JSON payload.
        """
        return {
            "model": "llama",
            "messages": self.build_messages(),
            "temperature": 0.4,
            "max_tokens": 3200,
            "response_format": {"type": "json_object"},
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Return current memory state for debugging or display.
        :return dict: Memory state.
        """
        return {
            "mandate": self._mandate,
            "short_term_summary": self._short_term_summary,
            "notes": self._notes,
            "retrieved_long_term": self._retrieved_long_term,
            "observations": self._observations,
        }
