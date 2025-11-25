from typing import Optional, Dict, List, Any
from enum import IntEnum
import hashlib


class ActionType(IntEnum):
    THINK = 0
    SEARCH = 1
    VISIT = 2
    EXIT = 3


def _to_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _parse_cache_update(value: Any) -> Dict[str, str]:
    """
    Strictly parse cache_update:
    - Only accept a dict; otherwise return {}.
    - Coerce keys/values to str, trim, drop empties.
    """
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        ks = _to_str(k)
        vs = _to_str(v)
        if ks and vs:
            out[ks] = vs
    return out


class TickOutput:
    """
    Represents one LLM tick's output in the RAG reasoning loop.

    The cache_update content is what gets stored into the vector database.
    The cache_retrieved field specifies what information should be retrieved next tick.
    """

    DEFAULT_OUTPUT = {
        "history_update": "",
        "note_update": "",
        "cache_update": [],
        "next_action": "think",
        "cache_retrieve": [],
        "deliverable": ""
    }

    def __init__(self, output_dict: Dict[str, any]):
        merged = {**self.DEFAULT_OUTPUT, **(output_dict or {})}

        self.raw = merged
        self.corrections: List[str] = []
        self.history_update: str = str(merged.get("history_update", ""))
        self.note_update: str = str(merged.get("note_update", ""))
        self._next_action_raw: str = str(merged.get("next_action", "think"))
        self._deliverable: str = str(merged.get("deliverable", ""))

        self.cache_update = self._parse_cache_update(merged.get("cache_update", []))
        self.cache_retrieved: List[str] = merged.get("cache_retrieve", [])
        self.next_action = self._parse_next_action()
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

    def show_requested_data_topics(self) -> List[str]:
        """
        :return List: List of topic strings requested by the agent for the next tick.
        """
        return self.cache_retrieved

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

    def _parse_cache_update(self, list_of_dicts) -> List[Any]:
        if not isinstance(list_of_dicts, list):
            return []
        records = []
        for entry in list_of_dicts:
            if not isinstance(entry, dict):
                continue
            doc = entry.get("document", "").strip()
            metadata = entry.get("metadata", {})
            if not (doc and metadata):
                self.corrections.append(f"Invalid cache_update entry: {entry}")
                continue
            hashid = hashlib.sha256(f"{doc}{metadata}".encode()).hexdigest()
            records.append({
                "documents": doc,
                "metadatas": metadata,
                "ids": hashid
            })
        return records

    def get_vector_documents(self) -> List[str]:
        return [rec["documents"] for rec in self.cache_update]

    def get_vector_metadatas(self) -> List[Any]:
        return [rec["metadatas"] for rec in self.cache_update]

    def get_vector_ids(self):
        return [rec["ids"] for rec in self.cache_update]

    def deliverable(self) -> str:
        """
        :return: The deliverable for this tick of the agent.
        """
        return self._deliverable

    def get_summary(self) -> Dict[str, any]:
        """
        :return dict: A summary of the tick's output.
        """
        return {
            "history": self.history_update,
            "notes": self.note_update,
            "next_action": self.next_action,
            "cache_update": self.cache_update,
            "cache_retrieved": self.cache_retrieved,
        }
