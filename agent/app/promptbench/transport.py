import dataclasses
import time
from typing import Any, List, Optional


@dataclasses.dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cached_prompt_tokens


class ScriptedLLM:
    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.prompts: List[str] = []

    def complete(self, prompt: str, model: str = "") -> Completion:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("ScriptedLLM response queue is exhausted")
        text = self._responses.pop(0)
        # Token counts are approximated via word split, sufficient for test mocking.
        prompt_tokens = len(prompt.split()) if prompt else 0
        completion_tokens = len(text.split()) if text else 0
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=0,
            latency_s=0.0,
            model=model,
        )
