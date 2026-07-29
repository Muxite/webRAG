"""LLM interface for the orchestrator.

The orchestrator depends only on a callable ``LLM(prompt, ...) -> str``. This keeps
the whole control loop testable with a ScriptedLLM (no model, no network) — the same
discipline that let schema_check.py be proven before any live run. The OllamaLLM
adapter (local, OpenAI-compatible) is the real backend; it is importable without a
running server and only touches the network when actually called.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Protocol


class LLM(Protocol):
    def __call__(self, prompt: str, *, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 256,
                 choices: Optional[List[str]] = None) -> str: ...


class ScriptedLLM:
    """Returns pre-authored completions in order. Raises if over-drawn so a test that
    triggers an unexpected extra call fails loudly instead of hanging."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self._i = 0
        self.calls: List[str] = []

    def __call__(self, prompt: str, *, system=None, temperature=0.0, max_tokens=256,
                 choices=None) -> str:
        self.calls.append(prompt)
        if self._i >= len(self._responses):
            raise AssertionError(f"ScriptedLLM exhausted after {self._i} calls; "
                                 f"unexpected prompt:\n{prompt[:400]}")
        r = self._responses[self._i]
        self._i += 1
        return r


class OllamaLLM:
    """Local OpenAI-compatible chat backend (Ollama/vLLM/llama.cpp). ``choices`` uses
    the ``json_schema`` enum path (Ollama enforces it — proven in the format-stress
    run) so the router literally cannot emit an illegal action token."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: str = "ollama") -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("MODEL_API_URL")
                         or "http://localhost:11435/v1").rstrip("/")
        self.api_key = api_key

    def __call__(self, prompt: str, *, system=None, temperature=0.0, max_tokens=256,
                 choices=None) -> str:
        import httpx  # lazy: importable without httpx present until actually called
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if choices:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "choice",
                                "schema": {"type": "object",
                                           "properties": {"choice": {"type": "string", "enum": choices}},
                                           "required": ["choice"], "additionalProperties": False}},
            }
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                       headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"] or ""
        if choices:
            try:
                return json.loads(content).get("choice", content)
            except Exception:
                return content
        return content
