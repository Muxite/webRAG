from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.app.idea_dag import IdeaDag
from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
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
    
    # Compact merged results to reduce LLM payload size
    from agent.app.idea_policies.action_constants import MergedResultsCompactor
    max_merged_items = int(settings.get("max_merged_items_for_llm", 20))
    compacted_merged = MergedResultsCompactor.compact_for_llm(merged, max_items=max_merged_items)
    original_size = len(json.dumps(merged, ensure_ascii=True))
    compacted_size = len(json.dumps(compacted_merged, ensure_ascii=True))
    _logger.info(f"[FINALIZE] Compacted merged results: {original_size} -> {compacted_size} chars ({100 * compacted_size // max(original_size, 1)}%)")
    
    system_template = settings.get("final_system_prompt")
    user_template = settings.get("final_user_prompt")
    if not system_template or not user_template:
        _logger.error(f"[FINALIZE] Missing prompt templates! system_template={bool(system_template)}, user_template={bool(user_template)}")
    merged_json = json.dumps(compacted_merged, ensure_ascii=True)
    tool_runtime_clause = (
        "Runtime capability: this agent has tool-mediated web access via search and visit actions. "
        "Do not claim you cannot browse, cannot access the web, or need user permission to search. "
        "Use only evidence present in merged results; if data is missing there, state what is missing without capability disclaimers."
    )
    if system_template and user_template:
        final_messages = [
            {"role": "system", "content": f"{system_template}\n\n{tool_runtime_clause}"},
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
        deliverable = data.get("deliverable", "")
        action_summary = data.get("summary", "")
        
        goal_achieved = False
        if root:
            goal_achieved = root.details.get(DetailKey.GOAL_ACHIEVED.value, False)
            if not goal_achieved:
                merge_nodes = [n for n in graph.iter_depth_first() if NodeDetailsExtractor.is_merge_action(n.details)]
                for merge_node in merge_nodes:
                    if merge_node.details.get(DetailKey.GOAL_ACHIEVED.value, False):
                        goal_achieved = True
                        break
        
        failed_nodes = [n for n in graph.iter_depth_first() if n.status.value == "failed"]
        has_critical_failures = len(failed_nodes) > 0
        
        success = goal_achieved and not has_critical_failures and bool(deliverable.strip())
        
        return {
            "final_deliverable": deliverable,
            "action_summary": action_summary,
            "success": success,
            "goal_achieved": goal_achieved,
            "has_failures": has_critical_failures,
        }
    except Exception as e:
        _logger.warning(f"[FINALIZE] Failed to parse response: {e}")
        return {"final_deliverable": response, "action_summary": "", "success": False}
