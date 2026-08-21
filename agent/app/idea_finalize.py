from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.app.answer_vote import vote_key, majority_vote
from agent.app.model_tiers import (
    capability_tier,
    local_model_size_band,
    size_band_value,
    tier_value,
)

from agent.app.idea_dag import IdeaDag
from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.shape_classifier import classify_answer_shape
from agent.app.prompt_builder import FinalPromptBuilder
from agent.app.idea_memory import MemoryManager

_logger = logging.getLogger(__name__)


def _compact_action_result(ar: dict, action: str) -> dict:
    """Strip large content fields from action results to keep merged_json small.

    The stripping is lossy on purpose: ``content`` survives here only as a 1000-char
    preview. Visit bodies are re-read (from ``content_full``, bounded per visit and in
    total) by ``_collect_all_visit_content`` into the separate ``visit_content`` block,
    so a visit's page text still reaches the final prompt. Every other large field
    (``content_with_links``, ``links_full``, ``link_contexts``, ``_links_inline``) is
    dropped outright and is NOT recovered anywhere else.
    """
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


def _is_superseded(node: Any) -> bool:
    """True for a node whose degenerate expansion guess was re-planned away (F6/F7 repair).

    The graph is append-only: `_maybe_reexpand_fallback_parent` marks the bad guess
    `SKIPPED` + ``FALLBACK_SUPERSEDED`` and adds the real children alongside it rather than
    removing it. The guess was already EXECUTED, so its action result is present and
    successful, and every selector below picks content on exactly that basis — which would
    feed the retried-away output to the final synthesis next to the retry's own. Content
    selectors therefore skip these nodes.

    Only ``got.reexpand_fallback_nodes_enabled`` ever stamps the marker, so this is a no-op
    on the default path. Deliberately NOT applied to `_visited_sources` /
    `_has_grounded_evidence`: the superseded node really did open its page, and dropping it
    there could make the grounding gate refuse an otherwise-grounded run.
    """
    return bool(node.details.get(DetailKey.FALLBACK_SUPERSEDED.value))


def _collect_leaf_results_fallback(graph: IdeaDag) -> list:
    from agent.app.idea_policies.base import IdeaNodeStatus, IdeaActionType
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    results = []
    for node in graph.iter_depth_first():
        if node.node_id == graph.root_id():
            continue
        if _is_superseded(node):
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
        if _is_superseded(node):
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not ar or not isinstance(ar, dict):
            continue
        if not ActionResultExtractor.is_success(ar):
            continue
        url = ar.get("url", "") or ""
        title = ar.get("title", "") or ""
        # ``content`` is the connector's already-truncated payload; ``content_full`` is the
        # same page uncut and is present on the same dict (see VisitLeafAction's result
        # builder). Prefer it, exactly as `_has_page_evidence` below already does: this
        # block applies its OWN `max_chars_per_visit` bound, so reading the pre-truncated
        # field only threw page text away twice.
        content = ar.get("content_full") or ar.get("content") or ""
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


def _visited_sources(graph: IdeaDag, cap: int = 25) -> List[Dict[str, str]]:
    """The distinct pages the agent actually read, as ``[{"url","title"}]``.

    Surfaced to the caller so an outside user can spot-check provenance instead of trusting
    the prose alone. Deduped by normalized URL, first-seen order, capped to keep the result
    payload small. Mirrors the successful-VISIT selection used by ``_collect_all_visit_content``.
    """
    from agent.app.idea_policies.base import IdeaActionType
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    sources: List[Dict[str, str]] = []
    seen: set = set()
    for node in graph.iter_depth_first():
        if node.details.get(DetailKey.ACTION.value) != IdeaActionType.VISIT.value:
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not ar or not isinstance(ar, dict) or not ActionResultExtractor.is_success(ar):
            continue
        url = (ar.get("url") or "").strip()
        if not url:
            continue
        key = url.split("#", 1)[0].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        sources.append({"url": url, "title": (ar.get("title") or "").strip()})
        if len(sources) >= cap:
            break
    return sources


_URL_RE = re.compile(r'https?://[^\s<>"\']+')


def _clean_url(raw: str) -> str:
    """Trim trailing punctuation a URL collects when scanned out of prose/markdown, WITHOUT eating
    a balanced path paren (``.../Mercury_(element)`` / ``.../Beloved_(novel)`` stay intact). Mirrors
    the scanner in scripts/prewarm_fixtures so a cited URL normalizes identically to the visited one;
    a path paren must survive here because ``_norm_cite`` compares the two and a strict-prefix
    mismatch would false-flag a correctly-grounded citation as UNVERIFIED."""
    u = str(raw or "").strip()
    trail = ".,;:'\"<>"
    while True:
        while u and u[-1] in trail:
            u = u[:-1]
        if u.endswith(")") and u.count("(") < u.count(")"):
            u = u[:-1]
            continue
        break
    return u


def _norm_cite(url: str) -> str:
    """Normalize a URL for citation matching (drop scheme/fragment/trailing slash, lowercase).

    Cleans trailing prose punctuation FIRST (incl. balanced-paren handling) so the comparison key is
    stable whether or not a ``#fragment`` follows a path paren."""
    s = _clean_url(url).lower()
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.split("#", 1)[0].rstrip("/")


def _unverified_citations(text: str, sources: List[Dict[str, str]], cap: int = 20) -> List[str]:
    """URLs cited in the answer that are NOT among the pages the agent actually opened.

    The finalize prompt asks for citations but nothing enforces them, so an LLM can attribute a
    claim to a page it never read. This surfaces those so a consumer can distrust them. Compares
    against the visited ``sources`` only (search-result URLs the agent never opened still count
    as unverified — they were not read)."""
    if not text:
        return []
    visited = {_norm_cite(s.get("url", "")) for s in (sources or [])}
    out: List[str] = []
    seen: set = set()
    for match in _URL_RE.findall(text):
        key = _norm_cite(match)
        if not key or key in visited or key in seen:
            continue
        seen.add(key)
        out.append(_clean_url(match))
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# Hard grounding gate (opt-in; kills Mode-2 pure-parametric hallucination)
# ---------------------------------------------------------------------------
#
# The census found runs that opened ZERO pages and still emitted a confident answer —
# some fabricating their own provenance ("accessed via memory retrieval") — at ~35% of
# deepseek-baseline and ~13% of nano-baseline runs. Nothing downstream distinguishes such
# an answer from a researched one. This gate does: on a grounded-research mandate with no
# successfully-opened page, the answer is banner-flagged, its unverifiable URLs are
# stripped, and success/goal_achieved are forced False. It is a validity gate, not a
# quality lever — it deliberately does NOT re-run the model or spend a token.

_UNGROUNDED_BANNER = (
    "**Insufficient grounded evidence.** No source page was successfully retrieved for this "
    "task, so nothing below is grounded in evidence this agent actually read. Any specific "
    "value or citation in it is unverified and may be fabricated.\n\n"
)
_STRIPPED_URL_MARKER = "[unverified source removed]"


def _has_grounded_evidence(graph: IdeaDag) -> bool:
    """True when at least one node successfully opened a page and got something back.

    Deliberately the SAME notion of evidence the ``sources`` provenance list uses (a
    successful VISIT), plus a content-only fallback for a connector that returns body text
    without echoing the URL — so the gate can never fire on a run that did read a page.
    """
    from agent.app.idea_policies.base import IdeaActionType
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    for node in graph.iter_depth_first():
        if node.details.get(DetailKey.ACTION.value) != IdeaActionType.VISIT.value:
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(ar, dict) or not ActionResultExtractor.is_success(ar):
            continue
        if (ar.get("url") or "").strip() or (ar.get("content") or ar.get("content_full") or ""):
            return True
    return False


def _strip_urls(text: str, urls: List[str]) -> str:
    """Replace each unverifiable URL in ``text`` with an explicit marker."""
    out = text
    for url in urls:
        if url:
            out = out.replace(url, _STRIPPED_URL_MARKER)
    return out


def _apply_grounding_gate(
    payload: Dict[str, Any], graph: IdeaDag, mandate: str, sources: List[Dict[str, str]]
) -> None:
    """Refuse to present an ungrounded answer as a researched one (mutates ``payload``).

    No-op unless the run is a grounded-research task that opened no page. Never raises and
    never drops the model's text (a reader can still see what it claimed) — it prefixes the
    refusal banner, strips citations that were never opened, and marks the run unsuccessful.
    """
    from agent.app.idea_policies.grounding import requires_grounded_answer

    try:
        if _has_grounded_evidence(graph) or sources:
            return
        if not requires_grounded_answer(mandate, graph):
            return
    except Exception as exc:  # noqa: BLE001 — the gate must never crash finalize
        _logger.warning(f"[GROUNDING-GATE] check failed: {exc}")
        return

    deliverable = str(payload.get("final_deliverable", "") or "")
    unverified = _unverified_citations(deliverable, sources)
    if unverified:
        deliverable = _strip_urls(deliverable, unverified)
    payload["final_deliverable"] = _UNGROUNDED_BANNER + deliverable
    payload["success"] = False
    payload["goal_achieved"] = False
    payload["grounded"] = False
    payload["grounding_gate"] = "refused-ungrounded"
    if unverified:
        payload["stripped_citations"] = unverified
    _logger.warning(
        "[GROUNDING-GATE] zero opened pages on a grounded-research mandate: "
        f"refusing to present the answer as grounded (stripped {len(unverified)} citation(s))"
    )


def _looks_truncated(response: Optional[str], max_completion_tokens: int) -> bool:
    """Heuristic: the finalize completion reached ~the token cap (no finish_reason available).

    Uses ~4 chars/token on the raw completion string; flags at >=90% of the cap so a silently
    cut-off answer is marked rather than presented as complete."""
    if not (max_completion_tokens and response):
        return False
    return (len(response) / 4.0) >= 0.9 * max_completion_tokens


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
        if _is_superseded(node):
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
        # The row carries the node's OUTCOME (search titles/URLs, visit length, think text),
        # so a superseded guess leaks its content here as well; drop the row outright.
        if _is_superseded(node):
            continue
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
    rank_by_similarity: bool = False,
) -> str:
    """Pool memories from four query batches (mandate, merge summaries, node titles, URLs).

    ``rank_by_similarity`` decides what the ``n_results`` cap keeps. OFF (shipped) keeps the
    first ``n_results`` in the order the batches were ISSUED, so a distant hit from the
    mandate query outranks a near one from the URL query purely because it was asked first
    (ASSUMPTION_AUDIT.md T3-3). ON sorts by the similarity the retrieval already computed,
    stably, so equal-similarity rows keep their arrival order and the OFF path stays the
    exact historical one.
    """
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
        if _is_superseded(node):
            continue
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
        if node.node_id == graph.root_id() or _is_superseded(node):
            continue
        titles.append(node.title)
    if titles:
        titles_query = " ".join(titles)[:500]
        await _query(titles_query, 8)

    url_queries = []
    for node in graph.iter_depth_first():
        if _is_superseded(node):
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if ar and isinstance(ar, dict):
            url = ar.get("url", "")
            if url and isinstance(url, str) and url.startswith("http"):
                url_queries.append(url)
    for uq in url_queries[:5]:
        await _query(uq, 3)

    pooled = all_memories
    if rank_by_similarity:
        pooled = sorted(all_memories, key=lambda m: -float(m.get("similarity") or 0.0))

    unique = []
    final_seen = set()
    for mem in pooled:
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


def _deliverable_of(response: str) -> str:
    """The terminal-answer text to vote on: the parsed ``deliverable`` field, else the raw
    response (a non-JSON completion still carries the answer text)."""
    try:
        return str(json.loads(response).get("deliverable", "") or response)
    except Exception:  # noqa: BLE001
        return response


async def _vote_finalize_response(
    io: AgentIO,
    *,
    final_messages: list,
    model_name: Optional[str],
    max_tokens: Optional[int],
    json_schema: Any,
    reasoning_effort: Optional[str],
    text_verbosity: Optional[str],
    fallback_model: Optional[str],
    timeout_seconds: int,
    k: int,
) -> Optional[str]:
    """C1b: run k INDEPENDENT finalize extractions and return the response whose terminal answer
    wins an approximator-stripped majority vote (tie-break toward the temp-0 anchor).

    The first sample is anchored at temperature 0 (the deterministic best read); the remaining
    k-1 add mild diversity (temp 0.3) to surface alternatives. Voting normalizes only the answer
    VALUE via ``vote_key`` (approximator/units/case stripped) — it never rounds or fuzzy-matches,
    so distinct values never merge. Returns the winning full response string (so summary/citations
    stay consistent), or ``None`` if every sample was empty/failed (caller falls back to a single
    call). Only reached when the flag is on and k>=2, so the default path never calls this.
    """
    temps = [0.0] + [0.3] * (k - 1)

    async def _one(temp: float) -> Optional[str]:
        payload = io.build_llm_payload(
            messages=final_messages,
            json_mode=True,
            model_name=model_name,
            temperature=temp,
            max_tokens=max_tokens,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        return await io.query_llm_with_fallback(
            payload, model_name=model_name, fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )

    outcomes = await asyncio.gather(*[_one(t) for t in temps], return_exceptions=True)
    # Keep valid responses in temp order so index 0 is the temp-0 anchor when it succeeded.
    candidates = [r for r in outcomes if isinstance(r, str) and r.strip()]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    winner = majority_vote(candidates, key_fn=lambda r: vote_key(_deliverable_of(r)), anchor_index=0)
    return winner


# ---------------------------------------------------------------------------
# Post-synthesis reconcile chain (opt-in; attacks Mode-1 "right page, wrong value")
# ---------------------------------------------------------------------------
#
# Re-expansion fixes the wrong PAGE. These passes fix the wrong VALUE extracted from
# the RIGHT page: after the finalize draft is produced, an answer-shaped task can run
# one or more corrective LLM passes over the SAME raw evidence, each taking the prior
# stage's deliverable as its new draft and each failing OPEN (keep the prior draft on
# empty/error/timeout, never regress to nothing). All default OFF -> byte-identical.

_RECOMPUTE_SYSTEM = (
    "You are a re-derivation checker in a research system. You are given a DRAFT answer and the "
    "RAW EVIDENCE it was drawn from. Do exactly this:\n"
    "1. List the exact source values/quantities you will use, each with a VERBATIM quote from the "
    "evidence and its source URL.\n"
    "2. Re-derive the answer step by step using ONLY those listed values.\n"
    "3. If your re-derivation differs from the draft, return the CORRECTED answer; otherwise confirm "
    "the draft.\n"
    "Rely ONLY on the evidence below, never on prior knowledge. Return JSON "
    "{deliverable, summary}: the deliverable is the final (corrected or confirmed) answer including "
    "the listed values, their verbatim quotes and URLs, and the derivation."
)
_RECOMPUTE_USER = "MANDATE:\n{mandate}\n\nDRAFT ANSWER:\n{draft}\n\nRAW EVIDENCE:\n{evidence}"

_VERIFY_SYSTEM = (
    "You are a support checker in a research system. You are given an ANSWER and the RAW EVIDENCE. "
    "Quote the exact passage from the evidence that SUPPORTS the answer. If NO passage in the "
    "evidence supports it, the answer is UNSUPPORTED — return instead what the evidence ACTUALLY "
    "says. Rely ONLY on the evidence, never on prior knowledge. Return JSON {deliverable, summary}: "
    "the deliverable is the supported answer (with its verbatim supporting quote and URL), or the "
    "evidence-supported correction."
)
_VERIFY_USER = "MANDATE:\n{mandate}\n\nANSWER TO VERIFY:\n{draft}\n\nRAW EVIDENCE:\n{evidence}"

# K decorrelated framings of the core question. The point is that a systematically-biased reader
# misreads ONE framing but not the others, so a disagreement surfaces the correct value — unlike
# k-vote, which re-runs the identical prompt and reproduces the same misread.
_VARIATION_ANGLES = (
    "Rephrase the question in your own words first, then answer it directly.",
    "Ignore any assumptions. Locate the single specific value/quantity the question asks for, "
    "verbatim, in the evidence, and report exactly that value.",
    "Work backward from the evidence: find the passage that DEFINITIVELY answers the question, "
    "quote it, then state the answer it implies.",
)
_VARIATION_SYSTEM = (
    "You answer a research question strictly from the RAW EVIDENCE. {angle} Quote the exact "
    "supporting passage and its source URL. Rely ONLY on the evidence, never on prior knowledge; if "
    "the evidence does not contain the answer, say so. Return JSON {{deliverable, summary}}: the "
    "deliverable is the specific answer with its verbatim quote and URL."
)
_VARIATION_USER = "QUESTION:\n{mandate}\n\nRAW EVIDENCE:\n{evidence}"

_COLLATE_SYSTEM = (
    "You reconcile several independently-derived answers to the SAME question into the single "
    "best-supported answer. Each candidate carries its own evidence quote. If they agree, return "
    "that answer. If they DISAGREE, choose the one whose verbatim quote most directly supports it, "
    "and say why. Rely ONLY on the provided candidates and their quotes. Return JSON "
    "{deliverable, summary}."
)
_COLLATE_USER = "QUESTION:\n{mandate}\n\nCANDIDATE ANSWERS:\n{candidates}"


def _split_deliverable_summary(response: str) -> tuple[str, str]:
    """Best-effort ``(deliverable, summary)`` from a finalize-shaped response string."""
    try:
        data = json.loads(response)
        return str(data.get("deliverable", "") or ""), str(data.get("summary", "") or "")
    except Exception:  # noqa: BLE001
        return response, ""


def _valid_reconcile_response(response: Optional[str]) -> Optional[str]:
    """Return ``response`` iff it parses as ``{deliverable, summary}`` with a non-empty deliverable,
    else ``None``. Guarantees a reconcile stage never replaces a parseable draft with garbage —
    the downstream ``json.loads`` in :func:`build_final_payload` stays valid."""
    if not response or not response.strip():
        return None
    try:
        data = json.loads(response)
    except Exception:  # noqa: BLE001
        return None
    deliverable = data.get("deliverable")
    if not (isinstance(deliverable, str) and deliverable.strip()):
        return None
    return response


async def _reconcile_llm_call(
    io: AgentIO,
    messages: list,
    *,
    model_name: Optional[str],
    json_schema: Any,
    reasoning_effort: Optional[str],
    text_verbosity: Optional[str],
    max_tokens: Optional[int],
    fallback_model: Optional[str],
    timeout_seconds: int,
) -> Optional[str]:
    """One corrective/reconcile call, temp-0 for determinism. Returns a validated finalize-shaped
    response string, or ``None`` on empty/parse-failure/exception (fail-open)."""
    payload = io.build_llm_payload(
        messages=messages,
        json_mode=True,
        model_name=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        json_schema=json_schema,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )
    try:
        response = await io.query_llm_with_fallback(
            payload, model_name=model_name, fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 — a reconcile pass must never crash finalize
        _logger.warning(f"[FINALIZE] reconcile call failed: {exc}")
        return None
    return _valid_reconcile_response(response)


async def _recompute_pass(
    io: AgentIO, *, mandate: str, evidence: str, draft_response: str, **call_kwargs: Any
) -> Optional[str]:
    draft_deliverable, _ = _split_deliverable_summary(draft_response)
    messages = [
        {"role": "system", "content": _RECOMPUTE_SYSTEM},
        {"role": "user", "content": _RECOMPUTE_USER.format(
            mandate=mandate, draft=draft_deliverable, evidence=evidence)},
    ]
    return await _reconcile_llm_call(io, messages, **call_kwargs)


async def _verify_pass(
    io: AgentIO, *, mandate: str, evidence: str, draft_response: str, **call_kwargs: Any
) -> Optional[str]:
    draft_deliverable, _ = _split_deliverable_summary(draft_response)
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {"role": "user", "content": _VERIFY_USER.format(
            mandate=mandate, draft=draft_deliverable, evidence=evidence)},
    ]
    return await _reconcile_llm_call(io, messages, **call_kwargs)


async def _variation_ensemble(
    io: AgentIO, *, mandate: str, evidence: str, k: int, **call_kwargs: Any
) -> Optional[str]:
    """Answer K decorrelated framings independently, then run ONE collate/reconcile call.

    Returns the reconciled finalize-shaped response, or ``None`` (keep the prior draft) if fewer
    than two framings produced a valid answer or the collate call failed."""
    k = max(1, int(k or 1))
    angles = [_VARIATION_ANGLES[i % len(_VARIATION_ANGLES)] for i in range(k)]

    async def _one(angle: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": _VARIATION_SYSTEM.format(angle=angle)},
            {"role": "user", "content": _VARIATION_USER.format(mandate=mandate, evidence=evidence)},
        ]
        return await _reconcile_llm_call(io, messages, **call_kwargs)

    outcomes = await asyncio.gather(*[_one(a) for a in angles], return_exceptions=True)
    answers = [r for r in outcomes if isinstance(r, str) and r.strip()]
    if len(answers) < 2:
        # Nothing to reconcile — keep the prior draft rather than swap in one un-collated framing.
        return None

    blocks = []
    for i, ans in enumerate(answers, 1):
        deliverable, summary = _split_deliverable_summary(ans)
        block = f"[Candidate {i}] {deliverable}"
        if summary:
            block += f"\n  supporting evidence: {summary}"
        blocks.append(block)
    collate_messages = [
        {"role": "system", "content": _COLLATE_SYSTEM},
        {"role": "user", "content": _COLLATE_USER.format(
            mandate=mandate, candidates="\n\n".join(blocks))},
    ]
    return await _reconcile_llm_call(io, collate_messages, **call_kwargs)


async def _reconcile_finalize_response(
    io: AgentIO,
    *,
    mandate: str,
    evidence: str,
    draft_response: str,
    cfg_final: Any,
    call_kwargs: Dict[str, Any],
) -> str:
    """Run the enabled reconcile passes in order (variations->collate, recompute, verify).

    Each stage takes the prior stage's deliverable as its new draft and fails OPEN — a stage that
    returns ``None`` leaves the running draft untouched, so we never regress to nothing. Only the
    stages whose flag is set run; the shape gate is applied by the caller."""
    response = draft_response

    if cfg_final.final_variations_enabled:
        new = await _variation_ensemble(
            io, mandate=mandate, evidence=evidence, k=cfg_final.final_variations_k, **call_kwargs
        )
        if new:
            _logger.info("[FINALIZE] variation ensemble reconciled the draft")
            response = new

    if cfg_final.final_recompute_enabled:
        new = await _recompute_pass(
            io, mandate=mandate, evidence=evidence, draft_response=response, **call_kwargs
        )
        if new:
            _logger.info("[FINALIZE] recompute pass revised the draft")
            response = new

    if cfg_final.final_verify_enabled:
        new = await _verify_pass(
            io, mandate=mandate, evidence=evidence, draft_response=response, **call_kwargs
        )
        if new:
            _logger.info("[FINALIZE] verify pass revised the draft")
            response = new

    return response


async def build_final_payload(
    io: AgentIO,
    settings: Dict[str, Any],
    graph: IdeaDag,
    mandate: str,
    model_name: Optional[str],
    memory_manager: Optional[MemoryManager] = None,
) -> Dict[str, Any]:
    cfg = IdeaConfig.from_settings(settings)
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
    sources = _visited_sources(graph)

    n_final_chroma = cfg.final.chroma_results
    chroma_context = await _retrieve_final_chroma_context(
        memory_manager=memory_manager,
        mandate=mandate,
        graph=graph,
        n_results=n_final_chroma,
        rank_by_similarity=cfg.memory.final_context_rank_by_similarity,
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
    max_prompt_chars = cfg.final.max_prompt_chars
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

    model_name = model_name or cfg.final.model
    _logger.info(f"[FINALIZE] LLM call model={model_name}")

    system_content = final_messages[0].get("content", "") if final_messages else ""
    user_content = final_messages[1].get("content", "") if len(final_messages) > 1 else ""
    _logger.info(f"[FINALIZE] system={len(system_content)}c user={len(user_content)}c")
    if not system_content.strip() or not user_content.strip():
        _logger.warning("[FINALIZE] Empty prompts detected")

    json_schema = settings.get("final_json_schema")
    reasoning_effort = cfg.generation.reasoning_effort
    text_verbosity = cfg.generation.text_verbosity

    payload = io.build_llm_payload(
        messages=final_messages,
        json_mode=True,
        model_name=model_name,
        temperature=cfg.final.temperature,
        max_tokens=cfg.final.max_tokens,
        json_schema=json_schema,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )
    _logger.info(
        f"[FINALIZE] payload model={payload.get('model')} "
        f"max_tokens={payload.get('max_tokens') or payload.get('max_completion_tokens')}"
    )

    final_timeout = cfg.timeouts.final or cfg.timeouts.llm or 180
    total_prompt_size = len(merged_json) + len(node_summary) + len(chroma_context) + len(visit_content)
    if total_prompt_size > 5000:
        final_timeout = max(final_timeout, int(total_prompt_size / 60))
    final_timeout = min(final_timeout, 600)
    _logger.info(f"[FINALIZE] timeout={final_timeout}s prompt_material={total_prompt_size}c")

    # Post-synthesis reconcile chain (opt-in, default OFF). Only computed when a pass is enabled,
    # and only for answer-shaped tasks (computation/count/argmax/disambiguation/single_value) —
    # narrative/open-ended tasks skip it (a re-derived single value there is meaningless). This gate
    # also decides whether the variation ensemble supersedes the k-vote path below (they overlap).
    _reconcile_enabled = (
        cfg.final.final_recompute_enabled
        or cfg.final.final_verify_enabled
        or cfg.final.final_variations_enabled
    )
    _answer_shape = classify_answer_shape(mandate) if _reconcile_enabled else None
    _shape_ok = (
        _reconcile_enabled
        and _answer_shape is not None
        and _answer_shape in cfg.final.final_recompute_shapes
    )
    _variations_will_run = cfg.final.final_variations_enabled and _shape_ok

    # C1b: opt-in approximator-stripped k-sample vote for the terminal answer. Default (flag off
    # or k<2) is the single call below, byte-identical. When enabled with k>=2, the voted response
    # replaces it (falling back to the single call only if every sample failed).
    response = None
    _vote_k = int(getattr(cfg.final, "native_vote_k", 1) or 1)
    if cfg.final.native_vote_k_tiered_enabled:
        _vote_k = tier_value(
            capability_tier(model_name),
            weak=cfg.final.native_vote_k_weak,
            standard=cfg.final.native_vote_k_standard,
            strong=cfg.final.native_vote_k_strong,
        )
        # Refine the flat weak band by LOCAL model size where the tag encodes one. An unparseable
        # tag or a priced model yields band None -> `unknown=_vote_k` -> exactly the tiered value
        # computed above, so this layer can only ever narrow, never reinterpret, the tiering.
        if cfg.final.native_vote_k_size_band_enabled:
            _vote_k = size_band_value(
                local_model_size_band(model_name),
                tiny=cfg.final.native_vote_k_local_tiny,
                small=cfg.final.native_vote_k_local_small,
                medium=cfg.final.native_vote_k_local_medium,
                large=cfg.final.native_vote_k_local_large,
                unknown=_vote_k,
            )
    if cfg.final.native_vote_k_enabled and _vote_k >= 2:
        if _variations_will_run:
            # k-vote re-runs the identical prompt; the variation ensemble is its decorrelated
            # superset. Prefer variations and skip the (overlapping, measured net-negative) k-vote.
            _logger.info("[FINALIZE] variation ensemble enabled -> skipping k-vote (overlap)")
        else:
            response = await _vote_finalize_response(
                io,
                final_messages=final_messages,
                model_name=model_name,
                max_tokens=cfg.final.max_tokens,
                json_schema=json_schema,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                fallback_model=cfg.generation.fallback_model,
                timeout_seconds=final_timeout,
                k=_vote_k,
            )
    if response is None:
        response = await io.query_llm_with_fallback(
            payload,
            model_name=model_name,
            fallback_model=cfg.generation.fallback_model,
            timeout_seconds=final_timeout,
        )
    _logger.info(f"[FINALIZE] response={len(response) if response else 0}c")
    _logger.debug(f"[FINALIZE] response preview: {response[:500] if response else 'None'}")
    if not response:
        _logger.warning("[FINALIZE] LLM returned empty response, constructing fallback deliverable")
        fallback = _build_fallback_deliverable(graph, merged)
        # `is_fallback_deliverable` marks the degraded shape structurally: without it a caller
        # or grader has to string-match `action_summary` to tell a stitched-together fallback
        # from a real model answer. Set on this path only — absent means a normal finalize,
        # matching how the other optional detail keys here (`truncated`, `unverified_citations`)
        # are emitted only when they apply.
        payload = {"final_deliverable": fallback, "action_summary": "Fallback: LLM finalize call failed", "success": bool(fallback.strip()), "sources": sources, "is_fallback_deliverable": True}
        if cfg.final.require_grounding:
            _apply_grounding_gate(payload, graph, mandate, sources)
        return payload

    # Post-synthesis reconcile chain (opt-in): correct a "right page, wrong value" draft over the
    # SAME raw evidence. Runs only for answer-shaped tasks; every stage fails open (keeps the prior
    # draft). Bounded by the finalize timeout. Skipped entirely (byte-identical) when no pass is on.
    if _shape_ok:
        _logger.info(
            f"[FINALIZE] reconcile chain (shape={_answer_shape}): "
            f"variations={cfg.final.final_variations_enabled} "
            f"recompute={cfg.final.final_recompute_enabled} verify={cfg.final.final_verify_enabled}"
        )
        response = await _reconcile_finalize_response(
            io,
            mandate=mandate,
            evidence=visit_content,
            draft_response=response,
            cfg_final=cfg.final,
            call_kwargs=dict(
                model_name=model_name,
                json_schema=json_schema,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                max_tokens=cfg.final.max_tokens,
                fallback_model=cfg.generation.fallback_model,
                timeout_seconds=final_timeout,
            ),
        )

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
        allow_partial_success = cfg.final.allow_partial_success
        if allow_partial_success:
            success = bool(deliverable.strip())
        else:
            success = bool(deliverable.strip()) and (goal_achieved or not has_critical_failures)

        unverified = _unverified_citations(deliverable, sources)
        truncated = _looks_truncated(response, int(getattr(cfg.final, "max_tokens", 0) or 0))
        if truncated and deliverable.strip():
            # The answer hit the length cap — tell the reader instead of letting it look complete.
            deliverable = deliverable.rstrip() + "\n\n_[Answer may be incomplete: it reached the length limit.]_"

        payload = {
            "final_deliverable": deliverable,
            "action_summary": action_summary,
            "success": success,
            "goal_achieved": goal_achieved,
            "has_failures": has_critical_failures,
            "sources": sources,
        }
        if unverified:
            payload["unverified_citations"] = unverified
        if truncated:
            payload["truncated"] = True
        # F31: last thing before the payload leaves — an answer with zero opened pages on a
        # grounded-research mandate is refused rather than presented as a researched result.
        if cfg.final.require_grounding:
            _apply_grounding_gate(payload, graph, mandate, sources)
        return payload
    except Exception as e:
        _logger.warning(f"[FINALIZE] Failed to parse response: {e}")
        payload = {"final_deliverable": response, "action_summary": "", "success": False, "sources": sources}
        # A parse failure on a near-cap response is itself a strong truncation signal.
        if _looks_truncated(response, int(getattr(cfg.final, "max_tokens", 0) or 0)):
            payload["truncated"] = True
        if cfg.final.require_grounding:
            _apply_grounding_gate(payload, graph, mandate, sources)
        return payload
