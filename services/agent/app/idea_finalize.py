from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.app.idea_dag import IdeaDag
from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.prompt_builder import FinalPromptBuilder
from agent.app.idea_memory import MemoryManager

_logger = logging.getLogger(__name__)


def _compact_action_result(ar: dict, action: str) -> dict:
    """Strip large content fields from action results to keep merged_json small.
    Full visit content is captured separately in visit_content."""
    compact = {}
    large_fields = {
        "content", "content_full", "content_with_links",
        "links_full", "link_contexts", "_links_inline",
    }
    for k, v in ar.items():
        if k in large_fields:
            if k == "content" and isinstance(v, str):
                compact[k] = v[:1000] + "..." if len(v) > 1000 else v
            elif k in ("links_full", "link_contexts"):
                continue
            else:
                continue
        else:
            compact[k] = v
    return compact


def _collect_leaf_results_fallback(graph: IdeaDag) -> list:
    from agent.app.idea_policies.base import IdeaNodeStatus, IdeaActionType
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    results = []
    for node in graph.iter_depth_first():
        if node.node_id == graph.root_id():
            continue
        action = node.details.get(DetailKey.ACTION.value)
        if not action:
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not ar or not isinstance(ar, dict):
            continue
        # Only include successful action results in the fallback merged list.
        # Failures are still visible in node_summary / event_log; keeping merged
        # clean improves final synthesis reliability.
        if not ActionResultExtractor.is_success(ar):
            continue
        if action == IdeaActionType.MERGE.value:
            results.append({"node": node.title, "action": action, "result": _compact_action_result(ar, action)})
            continue
        compact_ar = _compact_action_result(ar, action)
        entry = {"node": node.title, "action": action, "result": compact_ar}
        results.append(entry)
    return results


def _collect_all_visit_content(graph: IdeaDag, max_chars_per_visit: int = 15000) -> str:
    from agent.app.idea_policies.base import IdeaActionType
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    sections = []
    total_chars = 0
    max_total = 80000

    for node in graph.iter_depth_first():
        action = node.details.get(DetailKey.ACTION.value)
        if action != IdeaActionType.VISIT.value:
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not ar or not isinstance(ar, dict):
            continue
        if not ActionResultExtractor.is_success(ar):
            continue
        url = ar.get("url", "") or ""
        title = ar.get("title", "") or ""
        content = ar.get("content", "") or ""
        if not content:
            continue

        content_trimmed = content[:max_chars_per_visit]
        section = f"--- URL: {url}\n"
        if title:
            section += f"Title: {title}\n"
        section += f"Content ({len(content)} chars):\n{content_trimmed}\n"

        if total_chars + len(section) > max_total:
            remaining = max_total - total_chars
            if remaining > 500:
                sections.append(section[:remaining] + "\n[... truncated due to total size limit ...]")
            break
        sections.append(section)
        total_chars += len(section)

    if not sections:
        return ""
    return "\n".join(sections)


def _build_fallback_deliverable(graph: IdeaDag, merged: list) -> str:
    """
    Construct a deliverable from graph data when the LLM finalize call fails.
    :param graph: The idea DAG.
    :param merged: Collected merged/leaf results.
    :returns: Best-effort deliverable text.
    """
    from agent.app.idea_policies.base import IdeaActionType, IdeaNodeStatus
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    sections = []

    for node in graph.iter_depth_first():
        if node.node_id == graph.root_id():
            continue
        action = node.details.get(DetailKey.ACTION.value)
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not ar or not isinstance(ar, dict) or not ActionResultExtractor.is_success(ar):
            continue

        if action == IdeaActionType.MERGE.value:
            synth = ar.get("synthesized")
            if isinstance(synth, dict):
                summary = synth.get("summary", "")
                if summary:
                    sections.append(summary)
            elif isinstance(synth, str):
                sections.append(synth)
        elif action == IdeaActionType.VISIT.value:
            url = ar.get("url", "")
            content = ar.get("content", "")
            if content:
                sections.append(f"Source: {url}\n{content[:3000]}")
        elif action == IdeaActionType.SEARCH.value:
            results = ar.get("results", [])
            if isinstance(results, list):
                for r in results[:5]:
                    if isinstance(r, dict):
                        title = r.get("title", "")
                        url = r.get("url", "")
                        snippet = r.get("snippet", "")
                        if title or snippet:
                            sections.append(f"{title} ({url}): {snippet}")

    return "\n\n".join(sections) if sections else ""


def _build_node_summary_table(graph: IdeaDag) -> str:
    from agent.app.idea_policies.action_constants import ActionResultKey, ActionResultExtractor
    from agent.app.idea_policies.base import IdeaActionType

    lines = []
    for node in graph.iter_depth_first():
        action = node.details.get(DetailKey.ACTION.value, "root")
        status = node.status.value
        ar = node.details.get(DetailKey.ACTION_RESULT.value)

        outcome = ""
        if ar and isinstance(ar, dict):
            if ActionResultExtractor.is_success(ar):
                if action == IdeaActionType.SEARCH.value:
                    results = ActionResultExtractor.get_results(ar)
                    if results:
                        snippets = []
                        for r in results[:5]:
                            title = r.get("title", "") if isinstance(r, dict) else str(r)
                            url = r.get("url", "") if isinstance(r, dict) else ""
                            if title:
                                snippets.append(f"{title[:120]} ({url[:150]})" if url else title[:150])
                        outcome = f"found {len(results)} results: " + "; ".join(snippets)
                    else:
                        outcome = "ok"
                elif action == IdeaActionType.VISIT.value:
                    url = ActionResultExtractor.get_url(ar) or ""
                    content_len = len(ar.get("content", "") or "")
                    title = ar.get("title", "") or ""
                    outcome = f"visited {url[:300]} ({content_len} chars)"
                    if title:
                        outcome += f" title={title[:200]}"
                elif action == IdeaActionType.MERGE.value:
                    merged = ar.get("merged", [])
                    summary = ar.get("summary", "")
                    outcome = f"merged {len(merged)} items"
                    if summary:
                        outcome += f": {str(summary)[:500]}"
                elif action == IdeaActionType.SAVE.value:
                    outcome = "saved"
                elif action == IdeaActionType.THINK.value:
                    content = ar.get("content", "") or ""
                    outcome = content[:500].replace("\n", " ")
                else:
                    outcome = "ok"
            else:
                error = ActionResultExtractor.get_error(ar, default="unknown error")
                outcome = f"FAILED: {str(error)[:300]}"
        elif status == "done":
            outcome = "completed"
        elif status == "failed":
            outcome = "failed"
        elif status == "pending":
            outcome = "not executed"

        line = f"[{status:>8}] {action:>7} | {node.title[:300]}"
        if outcome:
            line += f" -> {outcome}"
        lines.append(line)

    return "\n".join(lines) if lines else "No nodes in graph."


async def _retrieve_final_chroma_context(
    memory_manager: Optional[MemoryManager],
    mandate: str,
    graph: IdeaDag,
    n_results: int = 20,
) -> str:
    if not memory_manager:
        return ""

    seen_ids = set()
    all_memories: List[Dict[str, Any]] = []

    async def _query(query_text: str, n: int) -> None:
        if not query_text or not query_text.strip():
            return
        try:
            mems = await memory_manager.retrieve_relevant_memories(
                query=query_text[:1000],
                n_results=n,
            )
            for mem in mems:
                mem_id = mem.get("id")
                if mem_id and mem_id in seen_ids:
                    continue
                if mem_id:
                    seen_ids.add(mem_id)
                all_memories.append(mem)
        except Exception as exc:
            _logger.warning(f"[FINALIZE] Chroma query failed: {exc}")

    await _query(mandate, min(n_results, 10))

    from agent.app.idea_policies.base import IdeaActionType
    merge_queries = []
    for node in graph.iter_depth_first():
        action = node.details.get(DetailKey.ACTION.value)
        if action == IdeaActionType.MERGE.value:
            ar = node.details.get(DetailKey.ACTION_RESULT.value)
            if ar and isinstance(ar, dict):
                summary = ar.get("summary", "")
                if summary:
                    merge_queries.append(str(summary)[:600])
    for mq in merge_queries[:3]:
        await _query(mq, 5)

    titles = []
    for node in graph.iter_depth_first():
        if node.node_id == graph.root_id():
            continue
        titles.append(node.title)
    if titles:
        titles_query = " ".join(titles)[:500]
        await _query(titles_query, 8)

    url_queries = []
    for node in graph.iter_depth_first():
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if ar and isinstance(ar, dict):
            url = ar.get("url", "")
            if url and isinstance(url, str) and url.startswith("http"):
                url_queries.append(url)
    for uq in url_queries[:5]:
        await _query(uq, 3)

    unique = []
    final_seen = set()
    for mem in all_memories:
        content = mem.get("content", "")
        content_key = content[:200]
        if content_key in final_seen:
            continue
        final_seen.add(content_key)
        unique.append(mem)
        if len(unique) >= n_results:
            break

    if not unique:
        return ""

    _logger.info(f"[FINALIZE] Retrieved {len(unique)} unique chroma memories")
    return memory_manager.format_memories_for_llm(unique, max_chars=80000)


async def build_final_payload(
    io: AgentIO,
    settings: Dict[str, Any],
    graph: IdeaDag,
    mandate: str,
    model_name: Optional[str],
    memory_manager: Optional[MemoryManager] = None,
) -> Dict[str, Any]:
    root = graph.get_node(graph.root_id())

    merged = []
    if root:
        merged = root.details.get(DetailKey.MERGED_RESULTS.value) or []
    _logger.info(f"[FINALIZE] {len(merged)} merged results, {graph.node_count()} nodes")
    if len(merged) == 0:
        _logger.warning("[FINALIZE] No merged results, collecting leaf fallback")
        merged = _collect_leaf_results_fallback(graph)
        _logger.info(f"[FINALIZE] Fallback collected {len(merged)} leaf results")

    node_summary = _build_node_summary_table(graph)
    event_log = graph.build_event_log_table(graph.root_id(), max_events=100)
    visit_content = _collect_all_visit_content(graph)

    n_final_chroma = int(settings.get("final_chroma_results", 15))
    chroma_context = await _retrieve_final_chroma_context(
        memory_manager=memory_manager,
        mandate=mandate,
        graph=graph,
        n_results=n_final_chroma,
    )

    # Compact merged results — strip large raw content fields
    # (full visit content is provided separately via visit_content)
    compacted_merged = []
    for item in merged:
        if isinstance(item, dict):
            result = item.get("result")
            if isinstance(result, dict):
                item = dict(item)
                item["result"] = _compact_action_result(result, item.get("title", ""))
            compacted_merged.append(item)
        else:
            compacted_merged.append(item)
    merged_json = json.dumps(compacted_merged, ensure_ascii=True)

    # Cap individual components to prevent token overflow
    max_prompt_chars = int(settings.get("final_max_prompt_chars", 200000))
    total_raw = len(merged_json) + len(node_summary) + len(chroma_context) + len(visit_content)
    if total_raw > max_prompt_chars:
        _logger.warning(f"[FINALIZE] Prompt too large ({total_raw}c), trimming to {max_prompt_chars}c")
        # Priority: node_summary > visit_content > chroma > merged_json
        budget = max_prompt_chars
        budget -= len(node_summary)  # keep full summary (usually small)
        budget -= len(event_log)
        if budget < 10000:
            budget = 10000
        # Split remaining budget: 50% visit_content, 25% chroma, 25% merged
        vc_budget = int(budget * 0.50)
        ch_budget = int(budget * 0.25)
        mj_budget = int(budget * 0.25)
        if len(visit_content) > vc_budget:
            visit_content = visit_content[:vc_budget] + "\n[... visit content truncated ...]"
        if len(chroma_context) > ch_budget:
            chroma_context = chroma_context[:ch_budget] + "\n[... chroma context truncated ...]"
        if len(merged_json) > mj_budget:
            merged_json = merged_json[:mj_budget] + "... ]"

    _logger.info(
        f"[FINALIZE] merged={len(merged_json)}c node_summary={len(node_summary)}c "
        f"event_log={len(event_log)}c chroma={len(chroma_context)}c visit_content={len(visit_content)}c"
    )

    system_template = settings.get("final_system_prompt")
    user_template = settings.get("final_user_prompt")

    tool_runtime_clause = (
        "Runtime capability: this agent has tool-mediated web access via search and visit actions. "
        "Do not claim you cannot browse, cannot access the web, or need user permission to search. "
        "Use only evidence present in the data below; if data is missing, state what is missing "
        "without capability disclaimers."
    )

    if system_template and user_template:
        final_messages = [
            {"role": "system", "content": f"{system_template}\n\n{tool_runtime_clause}"},
            {
                "role": "user",
                "content": user_template.format(
                    mandate=mandate,
                    merged_json=merged_json,
                    node_summary=node_summary,
                    event_log=event_log,
                    chroma_context=chroma_context,
                    visit_content=visit_content,
                ),
            },
        ]
    else:
        final_messages = FinalPromptBuilder(
            mandate=mandate,
            history=[],
            notes=[],
            deliverables=merged,
            retrieved_context=[chroma_context] if chroma_context else [],
        ).build_messages()

    model_name = model_name or settings.get("final_model")
    _logger.info(f"[FINALIZE] LLM call model={model_name}")

    system_content = final_messages[0].get("content", "") if final_messages else ""
    user_content = final_messages[1].get("content", "") if len(final_messages) > 1 else ""
    _logger.info(f"[FINALIZE] system={len(system_content)}c user={len(user_content)}c")
    if not system_content.strip() or not user_content.strip():
        _logger.warning("[FINALIZE] Empty prompts detected")

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
    _logger.info(
        f"[FINALIZE] payload model={payload.get('model')} "
        f"max_tokens={payload.get('max_tokens') or payload.get('max_completion_tokens')}"
    )

    final_timeout = settings.get("final_timeout_seconds") or settings.get("llm_timeout_seconds") or 180
    total_prompt_size = len(merged_json) + len(node_summary) + len(chroma_context) + len(visit_content)
    if total_prompt_size > 5000:
        final_timeout = max(final_timeout, int(total_prompt_size / 60))
    final_timeout = min(final_timeout, 600)
    _logger.info(f"[FINALIZE] timeout={final_timeout}s prompt_material={total_prompt_size}c")

    response = await io.query_llm_with_fallback(
        payload,
        model_name=model_name,
        fallback_model=settings.get("fallback_model"),
        timeout_seconds=final_timeout,
    )
    _logger.info(f"[FINALIZE] response={len(response) if response else 0}c")
    _logger.debug(f"[FINALIZE] response preview: {response[:500] if response else 'None'}")
    if not response:
        _logger.warning("[FINALIZE] LLM returned empty response, constructing fallback deliverable")
        fallback = _build_fallback_deliverable(graph, merged)
        return {"final_deliverable": fallback, "action_summary": "Fallback: LLM finalize call failed", "success": bool(fallback.strip())}

    try:
        data = json.loads(response)
        deliverable = data.get("deliverable", "")
        action_summary = data.get("summary", "")

        goal_achieved = False
        if root:
            goal_achieved = root.details.get(DetailKey.GOAL_ACHIEVED.value, False)
            if not goal_achieved:
                merge_nodes = [
                    n for n in graph.iter_depth_first()
                    if NodeDetailsExtractor.is_merge_action(n.details)
                ]
                for merge_node in merge_nodes:
                    if merge_node.details.get(DetailKey.GOAL_ACHIEVED.value, False):
                        goal_achieved = True
                        break

        from agent.app.idea_policies.base import IdeaActionType
        critical_actions = {IdeaActionType.SEARCH.value, IdeaActionType.VISIT.value, IdeaActionType.MERGE.value}
        failed_nodes = [n for n in graph.iter_depth_first() if n.status.value == "failed"]
        critical_failures = [
            n for n in failed_nodes
            if n.details.get(DetailKey.ACTION.value) in critical_actions
        ]
        has_critical_failures = len(critical_failures) > 0
        success = bool(deliverable.strip()) and (goal_achieved or not has_critical_failures)

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
