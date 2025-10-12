from typing import Optional, Dict, List


class TickOutput:
    """
    Represents one LLM tick's output in the RAG reasoning loop.

    Expected JSON format:
    {
        "history_update": ...,
        "note_update": ...,
        "cache_update": {
            "topic_or_tag": "some fact"
        },
        "next_action": ...,
        "data": "a, b, ..."
    }

    The `cache_update` content is what gets stored into the vector database.
    The `data` field specifies what information should be retrieved next tick.
    """

    def __init__(self, output_dict: Dict[str, any]):
        self.raw = output_dict or {}

        self.history_update: Optional[str] = self.raw.get("history_update")
        self.note_update: Optional[str] = self.raw.get("note_update")
        self.cache_update: Optional[Dict] = self.raw.get("cache_update", {})
        self._next_action_raw: Optional[str] = self.raw.get("next_action")
        self._data_raw: Optional[str] = self.raw.get("data")

        self.next_action: Optional[str] = self._parse_next_action()
        self.data_topics: List[str] = self._parse_data_topics()


    def _parse_next_action(self) -> Optional[str]:
        """
        Parse the 'next_action' field into a clean string.
        """
        if not self._next_action_raw:
            return None
        return self._next_action_raw.strip()

    def show_next_action(self) -> Optional[str]:
        """
        :return string: The next action to take.
        """
        return self.next_action

    def _parse_data_topics(self) -> List[str]:
        """
        Split the 'data' field by commas and clean up whitespace.
        :return List: List of topic strings compatible with a vector database.
        """
        if not self._data_raw:
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
        return [{"tag": tag, "content": content} for tag, content in self.cache_update.items() if content.strip()]

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
