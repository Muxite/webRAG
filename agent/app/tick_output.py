from typing import Optional, Dict, List
from enum import IntEnum


class ActionType(IntEnum):
    THINK = 0
    SEARCH = 1
    VISIT = 2
    EXIT = 3


class TickOutput:
    """
    Represents one LLM tick's output in the RAG reasoning loop.

    The `cache_update` content is what gets stored into the vector database.
    The `data` field specifies what information should be retrieved next tick.
    """

    DEFAULT_OUTPUT = {
        "history_update": "",
        "note_update": "",
        "cache_update": {},
        "next_action": "think",
        "data": "",
        "deliverable": ""
    }

    def __init__(self, output_dict: Dict[str, any]):
        merged = {**self.DEFAULT_OUTPUT, **(output_dict or {})}

        self.raw = merged
        self.history_update: str = str(merged.get("history_update", ""))
        self.note_update: str = str(merged.get("note_update", ""))
        self._data_raw: str = str(merged.get("data", ""))
        self._next_action_raw: str = str(merged.get("next_action", "think"))
        self._deliverable: str = str(merged.get("deliverable", ""))

        raw_cache = merged.get("cache_update", {})
        if not isinstance(raw_cache, dict):
            try:
                raw_cache = dict(raw_cache)
            except Exception:
                raw_cache = {}
        self.cache_update: Dict[str, str] = raw_cache
        self.next_action = self._parse_next_action()
        self.data_topics: List[str] = self._parse_data_topics()
        self.corrections: List[str] = []
        self._validate_fields()

    def _validate_fields(self):
        """
        Detect and record any self-corrections for reporting/debugging purposes.
        #TODO implement system to report corrections/errors back to LLM via optional key.
        """
        if not self.history_update:
            self.corrections.append("Missing history_update, defaulted to empty string.")
        if not self.note_update:
            self.corrections.append("Missing note_update, defaulted to empty string.")
        if not isinstance(self.cache_update, dict):
            self.corrections.append("Invalid cache_update type, replaced with {}.")
        if not isinstance(self._next_action_raw, str):
            self.corrections.append("next_action was non-string, reset to 'think'.")

    def _parse_next_action(self):
        """
        Parse the 'next_action' field into a clean string.
        """
        raw = self._next_action_raw
        if not isinstance(raw, str):
            return ActionType.THINK, None

        parts = [p.strip() for p in raw.split(",", 1) if p.strip()]
        action_name = parts[0].lower() if parts else "think"
        param = parts[1] if len(parts) > 1 else None

        mapping = {
            "think": ActionType.THINK,
            "search": ActionType.SEARCH,
            "visit": ActionType.VISIT,
            "exit": ActionType.EXIT,
        }
        return mapping.get(action_name, ActionType.THINK), param

    def show_next_action(self):
        """
        :return enum: The next action to take.
        """
        return self.next_action

    def _parse_data_topics(self) -> List[str]:
        """
        Split the 'data' field by commas and clean up whitespace.
        :return List: List of topic strings compatible with a vector database.
        """
        if not isinstance(self._data_raw, str) or not self._data_raw.strip():
            return []
        return [topic.strip() for topic in self._data_raw.split(",") if topic.strip()]

    def show_requested_data_topics(self) -> List[str]:
        """
        :return List: List of topic strings requested by the agent for the next tick.
        """
        return self.data_topics

    def show_history(self) -> Optional[str]:
        """
        :return: Full enumerated history of the agent.
        """
        return self.history_update

    def show_notes(self) -> Optional[str]:
        """
        :return: The notes for this tick of the agent.
        """
        return self.note_update

    def show_cache_update(self) -> Dict:
        """
        A dictionary containing summarized parts of the previous observations to be memorized in the vector database.
        :return: The new updates to be added to the vector database.
        """
        return self.cache_update

    def to_vector_records(self) -> List[Dict[str, str]]:
        """
        Converts cache_update into a list of {tag, content} records ready for vector DB insertion.
        :return List: List of dicts ready for vector DB insertion.
        """
        return [{"tag": tag, "content": content}
                for tag, content in self.cache_update.items() if content.strip()]

    def deliverable(self) -> str:
        """
        :return: The deliverable for this tick of the agent.
        """
        return self._deliverable

    def summary(self) -> Dict[str, any]:
        """
        :return dict: A summary of the tick's output.
        """
        return {
            "history": self.history_update,
            "notes": self.note_update,
            "cache": self.cache_update,
            "next_action": self.next_action,
            "requested_data": self.data_topics,
        }
