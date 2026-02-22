from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.app.idea_dag import IdeaDag
from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import DetailKey
from agent.app.prompt_builder import FinalPromptBuilder

_logger = logging.getLogger(__name__)


async def build_final_payload(
    io: AgentIO,
    settings: Dict[str, Any],
    graph: IdeaDag,
    mandate: str,
    model_name: Optional[str],
) -> Dict[str, Any]:
    """
    Generate a final answer using merged data.
    :param io: AgentIO instance.
    :param settings: Settings dictionary.
    :param graph: IdeaDag instance.
    :param mandate: Mandate text.
    :param model_name: Optional model override.
    :returns: Final payload.
    """
    root = graph.get_node(graph.root_id())
    merged = []
    if root:
        merged = root.details.get(DetailKey.MERGED_RESULTS.value) or []
    _logger.info(f"[FINALIZE] Building final payload with {len(merged)} merged results, graph has {graph.node_count()} nodes")
    if len(merged) == 0:
        _logger.warning(f"[FINALIZE] WARNING: No merged results! This may cause empty LLM responses.")
        _logger.warning(f"[FINALIZE] Root node details keys: {list(root.details.keys()) if root else 'no root'}")
    system_template = settings.get("final_system_prompt")
    user_template = settings.get("final_user_prompt")
    if not system_template or not user_template:
        _logger.error(f"[FINALIZE] Missing prompt templates! system_template={bool(system_template)}, user_template={bool(user_template)}")
    merged_json = json.dumps(merged, ensure_ascii=True)
    if system_template and user_template:
        final_messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_template.format(mandate=mandate, merged_json=merged_json)},
        ]
    else:
        final_messages = FinalPromptBuilder(
            mandate=mandate,
            history=[],
            notes=[],
            deliverables=merged,
            retrieved_context=[],
        ).build_messages()
    model_name = model_name or settings.get("final_model")
    _logger.info(f"[FINALIZE] Calling LLM for final synthesis with model={model_name}")
    system_content = final_messages[0].get("content", "") if final_messages else ""
    user_content = final_messages[1].get("content", "") if len(final_messages) > 1 else ""
    _logger.info(f"[FINALIZE] System prompt length: {len(system_content)}, content preview: {system_content[:200]}...")
    _logger.info(f"[FINALIZE] User prompt length: {len(user_content)}, content preview: {user_content[:200]}...")
    _logger.info(f"[FINALIZE] Merged JSON length: {len(merged_json)}, content: {merged_json[:500]}...")
    if not system_content.strip() or not user_content.strip():
        _logger.warning(f"[FINALIZE] WARNING: Empty prompts detected! system_len={len(system_content)}, user_len={len(user_content)}")
    json_schema = settings.get("final_json_schema")
    reasoning_effort = settings.get("reasoning_effort", "high")
    text_verbosity = settings.get("text_verbosity", "medium")
    payload = io.build_llm_payload(
        messages=final_messages,
        json_mode=True,
        model_name=model_name,
        temperature=float(settings.get("final_temperature", 0.3)),
        max_tokens=settings.get("final_max_tokens") if settings.get("final_max_tokens") is not None else None,
        json_schema=json_schema,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )
    _logger.info(f"[FINALIZE] Payload keys: {list(payload.keys())}, model={payload.get('model')}, max_tokens={payload.get('max_tokens') or payload.get('max_completion_tokens')}")
    final_timeout = settings.get("final_timeout_seconds") or settings.get("llm_timeout_seconds") or 120
    merged_size = len(merged_json)
    if merged_size > 5000:
        final_timeout = max(final_timeout, int(merged_size / 100))
    _logger.info(f"[FINALIZE] Using timeout: {final_timeout}s (merged JSON size: {merged_size} chars)")
    response = await io.query_llm_with_fallback(
        payload,
        model_name=model_name,
        fallback_model=settings.get("fallback_model"),
        timeout_seconds=final_timeout,
    )
    _logger.info(f"[FINALIZE] LLM response length: {len(response) if response else 0}")
    _logger.debug(f"[FINALIZE] LLM response: {response[:500] if response else 'None'}...")
    if not response:
        return {"final_deliverable": "", "action_summary": "", "success": False}
    try:
        data = json.loads(response)
        return {
            "final_deliverable": data.get("deliverable", ""),
            "action_summary": data.get("summary", ""),
            "success": True,
        }
    except Exception:
        return {"final_deliverable": response, "action_summary": "", "success": True}
