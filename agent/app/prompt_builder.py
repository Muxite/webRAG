from typing import Optional, List, Dict, Any


class BasePromptBuilder:
    """
    Base class for building prompt messages.
    """
    SYSTEM_INSTRUCTIONS: str = ""

    def build_messages(self) -> List[Dict[str, str]]:
        """
        Returns a message list compatible with OpenAI-style chat completions.
        :return list: List of dicts, each with "role" and "content" keys.
        """
        return [
            {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": self._build_user_message()},
        ]

    def _build_user_message(self) -> str:
        """
        Build the user message content.
        :return str: User content string.
        """
        raise NotImplementedError()

    @staticmethod
    def _format_section(title: str, content: str) -> str:
        """
        Format a section of the user message.
        :param str title: Section title.
        :param str content: Section content.
        :return str: Formatted section.
        """
        return f"{title}:\n{content.strip()}"

    @staticmethod
    def _format_block(title: str, content: str) -> str:
        """
        Format a section with a title and underline.
        :param str title: Section title.
        :param str content: Section content.
        :return str: Formatted section string.
        """
        sep = "=" * len(title)
        return f"{title}\n{sep}\n{content}"


class TickPromptBuilder(BasePromptBuilder):
    """
    Construct a JSON prompt for one LLM call in a tick-based RAG agent.
    """
    SYSTEM_INSTRUCTIONS = (
        "You are a cost-optimized autonomous agent operating in discrete ticks. "
        "Each tick is a small step: decide, act, store, retrieve. "
        "Use tick count to plan and update notes with a brief plan. "
        "Minimize generated text each tick; reading is cheaper than writing. "
        "Use the vector database aggressively as primary memory. "
        "It is acceptable to spend multiple ticks only retrieving or only storing facts. "
        "Final deliverables can be deferred; small partials are ok when useful.\n\n"

        "INPUTS:\n"
        "- MANDATE: immutable goal.\n"
        "- TICK STATUS: current tick and max ticks.\n"
        "- SHORT TERM MEMORY: brief recent summaries.\n"
        "- NOTES: tiny scratchpad, keep minimal.\n"
        "- RETRIEVED CONTEXT: vector DB hits from prior cache_retrieve.\n"
        "- OBSERVATIONS: new external input (may be long, full pages are ok).\n\n"

        "MEMORY STRATEGY:\n"
        "- Always check vector memory before searching the internet.\n"
        "- Use NOTES as pointers to memory (what to retrieve next); keep them short.\n"
        "- Anything not written to NOTES or stored in vector DB will be lost next tick.\n"
        "- Rely on cache_update/cache_retrieve every tick to reduce context growth.\n"
        "- Store small, atomic facts with strong titles and tags in metadata.\n"
        "- You can query memory like a search engine; use precise queries.\n"
        "- If a visit returns 403/blocked, do not retry that site; pivot to other sources.\n"
        "- When blocked, log the block in notes and shift strategy or revisit planning.\n"
        "- If you determine you are stuck (blocked with no alternatives), use exit and return what you have.\n"
        "- If unclear, do more searches/visits rather than long reasoning.\n\n"

        "OUTPUT FORMAT (JSON only, exact keys):\n"
        "* history_update: 1-2 concise sentences about what happened and why.\n"
        "* note_update: 1-2 short sentences; keep minimal, prefer cache_update.\n"
        "* cache_update: list of dicts for vector DB storage.\n"
        "  Format: [{'document': '...', 'metadata': {'KEY': 'VALUE'}}]\n"
        "  - document: ~40-80 words focused on a single fact/idea.\n"
        "  - metadata values must be scalars; use comma-separated strings for multi-values.\n"
        "  - include metadata.title and metadata.topics for better retrieval.\n"
        "* next_action: 'ACTION, PARAM'\n"
        "* cache_retrieve: list of short query sentences.\n"
        "* deliverable: empty unless you have a usable partial or final output.\n"
        "No extra keys or commentary.\n\n"

        "ACTIONS (case sensitive):\n"
        "* think : internal reasoning, no param.\n"
        "* search : web search. Example: 'search, query'.\n"
        "* visit : visit URL. Example: 'visit, https://...'.\n"
        "* exit : end program, only when finished."
    )

    def __init__(
        self,
        mandate: Optional[str] = None,
        short_term_summary: Optional[List[str]] = None,
        notes: Optional[str] = None,
        retrieved_long_term: Optional[List[str]] = None,
        observations: Optional[str] = None,
        current_tick: Optional[int] = None,
        max_ticks: Optional[int] = None,
    ):
        """
        Initializes the agent's context for one tick.
        """
        self._mandate = mandate or ""
        self._short_term_summary = short_term_summary or []
        self._notes = notes or ""
        self._retrieved_long_term = retrieved_long_term or []
        self._observations = observations or ""
        self._current_tick = current_tick
        self._max_ticks = max_ticks

    def set_mandate(self, text: str):
        """
        Set the agent's mandate.
        """
        self._mandate = text.strip()

    def add_history_entry(self, summary: str):
        """
        Add a single summary entry to the short-term memory.
        """
        self._short_term_summary.append(summary.strip())

    def update_notes(self, new_notes: str):
        """
        Replace or append to the freeform note scratchpad.
        """
        self._notes = new_notes.strip()

    def add_retrieved_context(self, chunk: str):
        """
        Add a single RAG chunk retrieved from the vector database.
        """
        self._retrieved_long_term.append(chunk.strip())

    def update_observations(self, new_obs: str):
        """
        Replace or append to the observation text.
        """
        self._observations = new_obs.strip()

    def _build_user_message(self) -> str:
        """
        Assemble the full user message with easily readable, distinct sections.
        :return str: The contents of the user message.
        """
        parts: List[str] = []

        if self._mandate:
            parts.append(self._format_section("MANDATE", self._mandate))

        if self._current_tick is not None and self._max_ticks is not None:
            parts.append(
                self._format_section(
                    "TICK STATUS",
                    f"{self._current_tick} / {self._max_ticks}"
                )
            )

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


class FinalPromptBuilder(BasePromptBuilder):
    """
    Build messages for final output synthesis.
    """
    SYSTEM_INSTRUCTIONS = (
        "You are a synthesis agent. Produce the final deliverable using the mandate, "
        "execution history, final notes, accumulated deliverables, and retrieved context. "
        "Be accurate and complete, but avoid unnecessary verbosity. "
        "Return JSON with keys: 'deliverable' and 'summary'."
    )

    def __init__(
        self,
        mandate: str,
        history: List[str],
        notes: List[str],
        deliverables: List[Any],
        retrieved_context: List[Any],
    ):
        """
        Initialize final output builder data.
        """
        self._mandate = mandate
        self._history = history
        self._notes = notes
        self._deliverables = deliverables
        self._retrieved_context = retrieved_context

    def _build_user_message(self) -> str:
        """
        Assemble the full user message with easily readable, distinct sections.
        :return str: The contents of the user message.
        """
        import json as _json

        parts: List[str] = []
        if self._mandate:
            parts.append(self._format_block("MANDATE", self._mandate))
        if self._history:
            joined_history = "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(self._history))
            parts.append(self._format_block("EXECUTION HISTORY", joined_history))
        if self._notes:
            joined_notes = "\n".join(f"{i + 1}. {entry}" for i, entry in enumerate(self._notes))
            parts.append(self._format_block("NOTES", joined_notes))
        if self._deliverables:
            formatted_deliverables = _json.dumps(self._deliverables, indent=2)
            parts.append(self._format_block("DELIVERABLES", formatted_deliverables))
        if self._retrieved_context:
            joined_context = "\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(self._retrieved_context))
            parts.append(self._format_block("RETRIEVED CONTEXT", joined_context))

        return "\n\n".join(parts)


class ObservationBuilder:
    """
    Observation formatting helpers.
    """
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
        """
        Build the observation string for invalid URLs.
        """
        return f"\n[Could not fetch URL: Invalid URL] {url}\n"

    @staticmethod
    def build_visit_error_observation(url: str, status: Optional[int]) -> str:
        """
        Build the observation string for visit errors.
        """
        status_text = status if status is not None else "Unknown"
        return f"\n[Could not fetch URL: {status_text}] {url}\n"

    @staticmethod
    def build_visit_observation(url: str, summary: str) -> str:
        """
        Build the observation string for a visit.
        """
        return f"\nVisited {url}:\n{summary}\n"

    @staticmethod
    def build_exception_observation(action: str, error: str) -> str:
        """
        Build an observation string for action exceptions.
        :param action: Action name that failed.
        :param error: Exception message.
        :return: Observation string.
        """
        return f"\n[Action error: {action}] {error}\n"
