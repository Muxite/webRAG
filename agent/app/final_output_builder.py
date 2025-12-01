from typing import List, Dict, Any
import json


class FinalOutputBuilder:
    """
    Builder for assembling a comprehensive final deliverable message
    from project execution artifacts.
    """

    def __init__(
            self,
            mandate: str,
            history: List[str],
            notes: List[str],
            deliverables: List[Any],
            retrieved_context: List[Any],
    ):
        self._mandate = mandate
        self._history = history
        self._notes = notes
        self._deliverables = deliverables
        self._retrieved_context = retrieved_context
        self._SYSTEM_INSTRUCTIONS = (
            "You are a synthesis agent that has run a project following the user's mandate "
            "for a number of ticks. Given the mandate, execution history, notes, deliverables, "
            "and retrieved context, generate a comprehensive final deliverable and concise action summary."
            "Your answer should be at least 50 words per tick. If you have 20 ticks, write about 1000 words."
            " Return JSON with keys: 'deliverable' and 'summary'."
        )

    @staticmethod
    def _format_section(title: str, content: str) -> str:
        """
        Format a section with a title and content.

        :param title: Section title
        :param content: Section content
        :return: Formatted section string
        """
        separator = "=" * len(title)
        return f"{title}\n{separator}\n{content}"

    def _build_user_message(self) -> str:
        """
        Assemble the full user message with easily readable, distinct sections.
        Good for humans and also AI.

        :return: The contents of the user message
        """
        parts: List[str] = []

        if self._mandate:
            parts.append(self._format_section("MANDATE", self._mandate))

        if self._history:
            joined_history = "\n".join(
                f"{i + 1}. {entry}" for i, entry in enumerate(self._history)
            )
            parts.append(self._format_section("EXECUTION HISTORY", joined_history))

        if self._notes:
            joined_notes = "\n".join(
                f"{i + 1}. {entry}" for i, entry in enumerate(self._notes)
            )
            parts.append(self._format_section("NOTES", joined_notes))

        if self._deliverables:
            formatted_deliverables = json.dumps(self._deliverables, indent=2)
            parts.append(self._format_section("DELIVERABLES", formatted_deliverables))

        if self._retrieved_context:
            joined_context = "\n".join(
                f"[{i + 1}] {chunk}" for i, chunk in enumerate(self._retrieved_context)
            )
            parts.append(self._format_section("RETRIEVED CONTEXT", joined_context))

        return "\n\n".join(parts)

    def build_messages(self) -> List[Dict[str, str]]:
        """
        Build a message list compatible with OpenAI-style chat completions.

        :return: List of message dicts with "role" and "content" keys
        """
        return [
            {"role": "system", "content": self._SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": self._build_user_message()},
        ]