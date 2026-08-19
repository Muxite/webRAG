"""OpenAI-compatible HTTP client returning usage and latency.

Kept separate from ``transport.py`` so that module stays import-safe and
network-free for the offline test suite. Serves both endpoints the benchmark
uses -- badmodel-ollama on :11435 and OpenRouter -- since both speak the same
chat-completions shape.

Cached prompt tokens are reported alongside, never subtracted from,
``prompt_tokens``. A provider cache hit that shrinks billed input would
otherwise read as a token saving caused by the prompt shape, which it is not.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from agent.app.promptbench.transport import Completion


class HttpLLM:
    def __init__(self, base_url: str, api_key: str = "ollama", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, prompt: str, *, model: str, temperature: float = 0.0,
                 max_tokens: int = 400) -> Completion:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.load(r)
        latency = time.perf_counter() - t0

        text = ((body.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cached_prompt_tokens=int(details.get("cached_tokens", 0)),
            latency_s=latency,
            model=model,
        )


def openrouter_key_usage(api_key: str, timeout: float = 20.0) -> Optional[float]:
    """Cumulative USD spent on this key, straight from the provider.

    Baseline-delta against this is the only spend guard that a timeout or an
    in-flight request cannot fool -- a locally-summed estimate misses calls
    whose response never came back.
    """
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return float((json.load(r).get("data") or {}).get("usage", 0.0))
    except (urllib.error.URLError, ValueError, KeyError, TypeError):
        return None
