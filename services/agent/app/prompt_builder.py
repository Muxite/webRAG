from typing import Optional, List, Dict, Any


def build_payload(messages: list, json_mode: bool) -> Dict[str, Any]:
    """
    Constructs the final JSON payload for the API call, can be switched for different LLM APIs.
    Model specific parameters like temperature, tokens, and response_format.
    """
    payload = {
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 4096,
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


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
        "- SHORT TERM MEMORY: Summary of recent actions in this reasoning branch. BE DESCRIPTIVE\n"
        "- NOTES: Your scratchpad for reasoning. You wrote these last tick and passed them here because "
        "they are meaningful. "
        "- RETRIEVED CONTEXT: Information fetched from your vector database based on the previous tick’s 'cache_retrieved' request.\n"
        "- OBSERVATIONS: New external input, such as user text or HTML, from this tick.\n\n"

        "SYSTEM RULES:\n"
        "- If you need specific knowledge, list it in the 'cache_retrieved' field as a short, comma-separated set of topics "
        "(ex: 'fish, ocean ecosystem').\n"
        "- The system will retrieve relevant semantic chunks and include them in 'cache_retrieved' on the next tick.\n"
        "Each paragraph should summarize one complete idea or piece of information."
        "Write as many paragraphs as you find informative and relevant. Each paragraph will be "
        "separately stored for future retrieval. These will later be accessible by the 'cache_retrieved' field.\n"
        "Recommended Strategy: make many searches at the start, and cache_update as many comprehensible facts. "
        "The cache is extremely fast, so do not worry about having too much data.\n\n"
        
        "OUTPUT FORMAT:\n"
        "You must output valid JSON with EXACTLY these top-level keys:\n"
        "* history_update : two or three sentence summary of this tick’s action, including what action is taken "
        "key findings and observations made based on the data recevied, and how this advances the mandate.\n"
        "* note_update : updated notes or reasoning to persist into the next tick. 3 to 5 sentences is approprate."
        "This becomes notes in the next input. Explain how your tick advances the goal of the mandate, next steps,"
        "hypotheses, decisions made, relevant context that next ticks will need.\n"
         "* cache_update : (list of dicts) detailed, focused, data to store in semantic database."
        "Contents must fit a topic, concept, idea, or key note.\n"
        "  Format: [\n"
        "    {'document': 'DOCUMENT_TEXT', 'metadata': {'KEY1': 'VALUE1', 'KEY2': 'VALUE2'}},\n"
        "    {'document': 'DOCUMENT_TEXT', 'metadata': {'KEY1': 'VALUE1', 'KEY2': 'VALUE2'}}\n"
        "  ]\n"
        "  - Each entry must have 'document' (60 word detailed summary) and 'metadata'\n"
        "  - Metadata values MUST be scalars (strings, numbers, booleans). Do NOT use lists or nested dicts.\n"
        "  - Use comma-separated strings for multi-value fields (ex: 'topics': 'AI, robotics, machine learning').\n"
        "* next_action : the next action you will take\n"
        "* cache_retrieve : list of sentences to query and retrieve from storage\n"
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

    @staticmethod
    def build_web_search_observation(query: str, search_results: Optional[List[Dict[str, str]]]) -> str:
        """
        Build the observation string for a web search.
        If results are None or empty, returns an appropriate failure message.
        """
        if not search_results:
            result_str = "[Search API unavailable or failed]"
        else:
            result_str = "\n".join(
                f"- {item.get('url', '')} ({item.get('url', '')})\n{item.get('description', '')}"
                for item in search_results
            )
        return f"\nWeb search for '{query}':\n{result_str}\n"

    @staticmethod
    def build_invalid_url_observation(url: str) -> str:
        return f"\n[Could not fetch URL: Invalid URL] {url}\n"

    @staticmethod
    def build_visit_error_observation(url: str, status: Optional[int]) -> str:
        status_text = status if status is not None else "Unknown"
        return f"\n[Could not fetch URL: {status_text}] {url}\n"

    @staticmethod
    def build_visit_observation(url: str, summary: str) -> str:
        return f"\nVisited {url}:\n{summary}\n"

    FINAL_SYSTEM_INSTRUCTIONS = (
        "You are a synthesis agent that has run a project following the user's mandate "
        "for a number of ticks. Given the mandate, execution history, notes, deliverables, "
        "and retrieved context, generate a comprehensive final deliverable and concise action summary."
        "Your answer should be at least 50 words per tick. If you have 20 ticks, write about 1000 words."
        " Return JSON with keys: 'deliverable' and 'summary'."
    )

    @classmethod
    def build_final_messages(
        cls,
        mandate: str,
        history: List[str],
        notes: List[str],
        deliverables: List[Any],
        retrieved_context: List[Any],
    ) -> List[Dict[str, str]]:
        """
        Build a message list for producing the final output. Mirrors FinalOutputBuilder.
        """
        import json as _json

        def _format_section(title: str, content: str) -> str:
            sep = "=" * len(title)
            return f"{title}\n{sep}\n{content}"

        parts: List[str] = []
        if mandate:
            parts.append(_format_section("MANDATE", mandate))
        if history:
            joined_history = "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(history))
            parts.append(_format_section("EXECUTION HISTORY", joined_history))
        if notes:
            joined_notes = "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(notes))
            parts.append(_format_section("NOTES", joined_notes))
        if deliverables:
            formatted_deliverables = _json.dumps(deliverables, indent=2)
            parts.append(_format_section("DELIVERABLES", formatted_deliverables))
        if retrieved_context:
            joined_context = "\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(retrieved_context))
            parts.append(_format_section("RETRIEVED CONTEXT", joined_context))

        user_message = "\n\n".join(parts)
        return [
            {"role": "system", "content": cls.FINAL_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ]

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
            "observations": self._observations[:1024],
        }

