"""One bounded re-prompt that turns a malformed LLM JSON answer into a usable object.

Extracted verbatim from ``LeafAction`` (which still exposes the same two methods, now as
thin delegates) so callers that are NOT leaf actions -- the evidence store's claim
extraction, for one -- can reuse the exact same repair call and the exact same telemetry
events rather than growing a second, drifting copy.

Every entry point fails OPEN: a repair that is empty, unparseable, non-object or that
raises returns ``None``, and the caller falls through to whatever fallback it had before.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.app.idea_policies.action_constants import PromptBuilder

#: Caps for the strings echoed back into a JSON-repair prompt. Both are generous
#: enough to carry the real shape/instruction and small enough that the repair call
#: can never cost more than the original one.
JSON_REPAIR_INSTRUCTION_CHARS = 4000
JSON_REPAIR_RESPONSE_CHARS = 4000

_DEFAULT_LOGGER = logging.getLogger(__name__)


def record_malformed_llm_action(
    io: Any,
    site: str,
    parse_error: Exception,
    repaired: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Count one malformed-JSON action response on the telemetry event stream.

    ``malformed_llm_action`` fires once per detected parse failure regardless of what
    the repair attempt does (it measures how often the bug happens);
    ``malformed_llm_action_repaired`` fires once per repair outcome so the recovery
    rate is a division of two event counts.
    """
    log = logger or _DEFAULT_LOGGER
    telemetry = getattr(io, "telemetry", None)
    if not telemetry:
        return
    try:
        if repaired is None:
            telemetry.record_event(
                "malformed_llm_action", {"site": site, "error": str(parse_error)[:200]}
            )
        else:
            telemetry.record_event(
                "malformed_llm_action_repaired", {"site": site, "repaired": bool(repaired)}
            )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break an action
        log.debug(f"[REPAIR] telemetry record failed at {site}: {exc}")


async def repair_malformed_json(
    io: Any,
    *,
    site: str,
    messages: List[Dict[str, str]],
    malformed_text: str,
    parse_error: Exception,
    fallback_model: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    timeout_seconds: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """One bounded re-prompt when an LLM response that had to be JSON is not JSON.

    Call sites used to fail OPEN straight to a heuristic default the moment
    ``json.loads`` raised -- the model's answer was thrown away without ever being asked
    to fix its own formatting. This makes ONE extra call carrying the original
    instruction, the malformed text, and the parse error, and returns the corrected
    object.

    ``json_schema``, when given, is passed straight through to ``build_llm_payload`` as real
    JSON-schema-constrained decoding (``run_policy_constrained_decoding_enabled``'s effect,
    already gated by the caller to configs where this is safe -- see that flag's docstring in
    ``idea_policies/config.py``) instead of leaving the repair call's shape to a plain
    ``json_object`` hint. ``None`` (the default) is today's exact behaviour.

    Returns ``None`` when the repair call is empty/unparseable/raises, so every caller
    falls through to exactly the fallback it used before. Always on: the worst case is
    one extra call and the same fallback as today.
    """
    log = logger or _DEFAULT_LOGGER
    record_malformed_llm_action(io, site, parse_error, logger=log)
    repaired = False
    try:
        original_instruction = "\n\n".join(
            str(m.get("content") or "") for m in (messages or []) if m.get("content")
        )[:JSON_REPAIR_INSTRUCTION_CHARS]
        repair_messages = PromptBuilder.build_messages(
            system_content=(
                "You are a JSON repair function. The previous response to the request below "
                "was not valid JSON. Return the SAME content as a single valid JSON object "
                "matching the shape the original request asked for. Output JSON only -- no "
                "prose, no markdown fences, no explanation. Invent nothing: preserve the "
                "original response's content and only fix its structure."
            ),
            user_content=(
                f"ORIGINAL REQUEST:\n{original_instruction}\n\n"
                f"MALFORMED RESPONSE:\n{malformed_text[:JSON_REPAIR_RESPONSE_CHARS]}\n\n"
                f"JSON PARSE ERROR:\n{parse_error}\n\n"
                "Corrected JSON:"
            ),
        )
        payload = io.build_llm_payload(
            messages=repair_messages,
            json_mode=True,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )
        response = await io.query_llm_with_fallback(
            payload,
            model_name=model_name,
            fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )
        if response:
            data = json.loads(response)
            # Every call site reads a JSON OBJECT; valid JSON of any other shape
            # is no more usable than the malformed text, so it counts as a failed
            # repair rather than a recovery the caller then quietly discards.
            if isinstance(data, dict):
                repaired = True
                log.info(f"[REPAIR] Recovered a malformed JSON response at {site}")
                return data
            log.warning(
                f"[REPAIR] JSON repair call at {site} returned a non-object ({type(data).__name__})"
            )
        else:
            log.warning(f"[REPAIR] JSON repair call at {site} returned nothing")
    except json.JSONDecodeError as exc:
        log.warning(f"[REPAIR] JSON repair call at {site} was also unparseable: {exc}")
    except Exception as exc:  # noqa: BLE001 - repair is best-effort; fall back as before
        log.warning(f"[REPAIR] JSON repair call at {site} failed: {exc}")
    finally:
        record_malformed_llm_action(io, site, parse_error, repaired=repaired, logger=log)
    return None
