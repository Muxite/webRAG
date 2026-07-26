from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import ExpansionPolicy, DetailKey, IdeaActionType
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.shape_classifier import classify_shape
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.llm_backends import json_instruction_from_response_format


def _safe_serialize_details(details: Dict[str, Any]) -> str:
    try:
        return json.dumps(details, ensure_ascii=True, default=str)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=True)


# --- malformed-expansion-JSON repair ---------------------------------------------------------
# A cheap executor model that emits prose-wrapped, fence-wrapped or TRUNCATED JSON used to fall
# through a single non-nesting regex (``\{[^{}]*"candidates"[^{}]*\}``) that cannot match once the
# candidates array contains objects — i.e. always. The plan silently became EMPTY (no children ->
# a tool-free run). These helpers salvage the object instead: brace-balanced extraction first,
# then a bounded "close the open brackets at the last complete element" repair for truncation.
# Every path fails safe (``None`` -> the caller returns an empty plan), never raises.
_JSON_REPAIR_MAX_STARTS = 32     # candidate '{' offsets tried for a complete object
_JSON_REPAIR_MAX_CUTS = 64       # truncation cut points tried, longest first
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads_lenient(text: str) -> Optional[Any]:
    """``json.loads`` with the two forgiving retries an LLM payload usually needs.

    ``strict=False`` tolerates raw control characters inside strings (a model pasting a page
    excerpt with literal newlines); the trailing-comma strip covers the other common slip.
    """
    for attempt in (text, _TRAILING_COMMA_RE.sub(r"\1", text)):
        try:
            return json.loads(attempt, strict=False)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def _scan_json(text: str, start: int) -> tuple[Optional[int], List[str], List[int]]:
    """String-aware bracket scan of the JSON value beginning at ``text[start]``.

    :returns: ``(end, stack, cuts)`` where ``end`` is the index AFTER the matching closing
        bracket (``None`` if the value never closes, i.e. truncated), ``stack`` is the still-open
        bracket list at the end of the text, and ``cuts`` are offsets just past a COMPLETE nested
        value — the only places a truncated fragment may safely be cut before auto-closing.
    """
    stack: List[str] = []
    cuts: List[int] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if stack:
                    cuts.append(i + 1)
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None, stack, cuts  # unbalanced -> not a value we can trust
            stack.pop()
            if not stack:
                return i + 1, stack, cuts
            cuts.append(i + 1)
        elif ch in "0123456789truefalsn." and stack:
            # End of a bare literal (number / true / false / null) is a safe cut too.
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in ("", ",", " ", "\t", "\r", "\n", "}", "]"):
                cuts.append(i + 1)
    return None, stack, cuts


def _autoclose(fragment: str) -> Optional[str]:
    """Close the brackets left open by a truncated ``fragment`` (``None`` if it can't be closed)."""
    end, stack, _ = _scan_json(fragment, 0)
    if end is not None or not stack:
        return None
    return fragment + "".join(reversed(stack))


def _repair_json_object(content: str, required_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of a JSON object from a malformed LLM response.

    Order: (1) every ``{`` offset is tried as the start of a brace-balanced object — this is what
    handles fences/prose around an otherwise VALID object with nested structures; (2) if nothing
    complete parses the payload is treated as TRUNCATED and repaired by cutting back to the last
    complete nested value and closing the open brackets; (3) a bare top-level array is accepted as
    the ``required_key`` list (a frequent cheap-model shape slip). Returns ``None`` when the text is
    genuinely unrepairable — the caller degrades to an empty plan rather than raising.
    """
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):  # drop a markdown fence so a bare array still starts at offset 0
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    starts = [i for i, ch in enumerate(text) if ch == "{"][:_JSON_REPAIR_MAX_STARTS]

    # An object that parses but lacks ``required_key`` is only a fallback, and only from the
    # OUTERMOST offset — a later ``{`` is an inner element (e.g. one candidate), never the plan.
    best: Optional[Dict[str, Any]] = None
    for start in starts:
        end, _stack, _cuts = _scan_json(text, start)
        if end is None:
            continue
        data = _loads_lenient(text[start:end])
        if not isinstance(data, dict):
            continue
        if required_key is None or required_key in data:
            return data
        if best is None and start == starts[0]:
            best = data

    for start in starts:
        end, _stack, cuts = _scan_json(text, start)
        if end is not None:
            continue  # already tried as a complete object above
        for cut in reversed(cuts[-_JSON_REPAIR_MAX_CUTS:]):
            closed = _autoclose(text[start:cut])
            if closed is None:
                continue
            data = _loads_lenient(closed)
            if isinstance(data, dict) and (required_key is None or required_key in data):
                return data

    # A bare top-level array of candidates ("[{...}, {...}]" with no wrapper object). Only when the
    # payload actually STARTS with the array — an inner array under some other key is that key's
    # data, not the plan, and re-labelling it would invent candidates.
    bracket = text.find("[")
    if required_key and bracket == 0:
        end, _stack, cuts = _scan_json(text, bracket)
        slices = [text[bracket:end]] if end is not None else [
            _autoclose(text[bracket:cut]) for cut in reversed(cuts[-_JSON_REPAIR_MAX_CUTS:])
        ]
        for candidate in slices:
            data = _loads_lenient(candidate) if candidate else None
            if isinstance(data, list) and data:
                return {required_key: data}
    return best


# Opt-in expansion addendum (``expansion_expect_contract_enabled``): borrows the compiled
# path's leaf discipline — one atomic fact per leaf, read off an authoritative page (never
# guessed from memory), reported as an EXACT value alongside its source URL. Injected into
# the expansion system prompt only when the flag is on (default path is byte-identical).
_EXPECT_CONTRACT_ADDENDUM = (
    "MEASURABLE OUTPUT CONTRACT: for every LEAF candidate (a concrete search/visit/think "
    "sub-task that resolves ONE fact), add an \"expect\" field at the candidate's top level: "
    "a single line naming the EXACT value to report AND requiring its source URL alongside "
    "it. Keep each leaf to ONE atomic fact read off an authoritative page — never guessed "
    "from memory. Example: \"expect\": \"the exact founding year (e.g. 1861) AND the source "
    "URL it was read from\". Omit \"expect\" for non-leaf or aggregation candidates."
)


# IDEA_TEST_REASONING_EXEMPLAR: per-run toggle that injects a general reasoning
# demonstration (a task-shape few-shot) into the expansion system prompt, so a
# cheap executor model learns HOW to reason through a chain/mixed/parallel task
# without leaking any answer. Follows the IDEA_TEST_GOT_REEXPAND convention: unset
# or "none" means byte-identical prompt behavior. Exemplars are read from disk once
# and cached per name for the process lifetime.
#
# DISPROVEN, kept only for reference — do not reach for this as a default. Live R=3
# validation (ADAPTIVE_DISTILLATION_HANDOFF.md Phase 1) found narrative exemplars
# unreliable on weak models: they backfired twice on the branch_eliminate/mixed
# shape (the model copied the exemplar's surface structure, not its intent, and
# under-decomposed the task further) and a seeming win on the parallel shape
# dissolved to noise at R=3. Prefer `IDEA_TEST_REASONING_RULES` (a flat imperative
# checklist) instead, which this project's research found narrative few-shot
# reasoning demonstrations are unusually sensitive to being copied for shape, not
# intent (see RESEARCH_NOTES.md, "Why narrative few-shot exemplars backfired").
_EXEMPLAR_NAMES = ("chain", "mixed", "parallel")
_EXEMPLAR_DIR = Path(__file__).resolve().parent.parent / "reasoning_exemplars"
_EXEMPLAR_CACHE: Dict[str, str] = {}
_EXEMPLAR_WARNED: set = set()


def _load_reasoning_exemplar() -> str:
    """Return the labeled few-shot exemplar block for IDEA_TEST_REASONING_EXEMPLAR,
    or "" when unset/none/invalid. Reads and caches the .md content once per name."""
    name = os.environ.get("IDEA_TEST_REASONING_EXEMPLAR", "").strip().lower()
    if not name or name == "none":
        return ""
    if name not in _EXEMPLAR_NAMES:
        if name not in _EXEMPLAR_WARNED:
            _EXEMPLAR_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Ignoring invalid IDEA_TEST_REASONING_EXEMPLAR=%r "
                "(expected one of %s or 'none')",
                name,
                ", ".join(_EXEMPLAR_NAMES),
            )
        return ""
    if name not in _EXEMPLAR_WARNED:
        _EXEMPLAR_WARNED.add(name)
        logging.getLogger("LlmExpansionPolicy").warning(
            "[EXPANSION] IDEA_TEST_REASONING_EXEMPLAR=%r is a DISPROVEN mechanism "
            "(see ADAPTIVE_DISTILLATION_HANDOFF.md Phase 1) — it backfired on the "
            "hardest task shape in live R=3 validation. Kept for reference only; "
            "prefer IDEA_TEST_REASONING_RULES instead.",
            name,
        )
    if name in _EXEMPLAR_CACHE:
        return _EXEMPLAR_CACHE[name]
    try:
        text = (_EXEMPLAR_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    except OSError as exc:
        if name not in _EXEMPLAR_WARNED:
            _EXEMPLAR_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Could not read reasoning exemplar %r: %s", name, exc
            )
        _EXEMPLAR_CACHE[name] = ""
        return ""
    block = (
        "## Reference reasoning pattern (a general demonstration, not this task's answer)\n\n"
        f"{text}\n\n"
        "## Now apply that reasoning pattern to the current task below.\n"
    )
    _EXEMPLAR_CACHE[name] = block
    return block


# IDEA_TEST_REASONING_RULES: per-run toggle that injects a FLAT IMPERATIVE checklist
# (not a narrative) into the expansion system prompt, to enforce a discipline (e.g.
# "check ALL candidates before electing a survivor") that a weak executor tends to
# skip. Fully independent of IDEA_TEST_REASONING_EXEMPLAR; either can be set alone.
# Follows the same convention: unset/none/invalid means byte-identical prompt behavior.
# Rule files are read from disk once and cached per name for the process lifetime.
_RULES_NAMES = ("branch_eliminate",)
_RULES_DIR = Path(__file__).resolve().parent.parent / "reasoning_rules"
_RULES_CACHE: Dict[str, str] = {}
_RULES_WARNED: set = set()


def _read_rules_block(name: str) -> str:
    """Read, cache and wrap the rule-checklist .md for ``name`` (already validated as a
    member of ``_RULES_NAMES``). Returns "" and warns once if the file is unreadable."""
    if name in _RULES_CACHE:
        return _RULES_CACHE[name]
    try:
        text = (_RULES_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    except OSError as exc:
        if name not in _RULES_WARNED:
            _RULES_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Could not read reasoning rules %r: %s", name, exc
            )
        _RULES_CACHE[name] = ""
        return ""
    block = (
        "## Mandatory reasoning rules (follow every rule literally)\n\n"
        f"{text}\n"
    )
    _RULES_CACHE[name] = block
    return block


def _load_reasoning_rules() -> str:
    """Return the imperative rule-checklist block for IDEA_TEST_REASONING_RULES,
    or "" when unset/none/invalid. Reads and caches the .md content once per name."""
    name = os.environ.get("IDEA_TEST_REASONING_RULES", "").strip().lower()
    if not name or name == "none":
        return ""
    if name not in _RULES_NAMES:
        if name not in _RULES_WARNED:
            _RULES_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Ignoring invalid IDEA_TEST_REASONING_RULES=%r "
                "(expected one of %s or 'none')",
                name,
                ", ".join(_RULES_NAMES),
            )
        return ""
    return _read_rules_block(name)


def _auto_reasoning_rules(mandate: str) -> str:
    """Auto-select a rule-checklist block from the mandate's classified shape.

    Only invoked when IDEA_TEST_REASONING_RULES is UNSET (the manual override always
    wins). Uses the deterministic ``classify_shape``. Today only ``branch_eliminate``
    has a matching rule file, so a correctly-classified ``chain``/``parallel_merge``
    mandate intentionally yields NO block (documented gap — no placeholder files are
    fabricated). Fails open to "" for unclassified mandates."""
    shape = classify_shape(mandate or "")
    if not shape:
        return ""
    if shape not in _RULES_NAMES or not (_RULES_DIR / f"{shape}.md").exists():
        logging.getLogger("LlmExpansionPolicy").info(
            "[EXPANSION] Auto-classified mandate shape=%s but no reasoning rule file "
            "exists yet; skipping auto-injection.",
            shape,
        )
        return ""
    logging.getLogger("LlmExpansionPolicy").info(
        "[EXPANSION] Auto-selected reasoning rules for classified shape=%s", shape
    )
    return _read_rules_block(shape)


class LlmExpansionPolicy(ExpansionPolicy):
    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        default_settings = load_idea_dag_settings()
        merged_settings = {**default_settings, **(settings or {})}
        super().__init__(settings=merged_settings)
        self._cfg = IdeaConfig.from_settings(merged_settings)
        self.io = io
        self.model_name = model_name
        self._logger = logging.getLogger(self.__class__.__name__)

    async def expand(self, graph: IdeaDag, node_id: str, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        node = graph.get_node(node_id)
        if not node:
            return []
        messages = self._build_messages(graph, node, memories=memories)

        # The expansion schema's candidate ``details`` is a free-form, per-action
        # object. Strict structured output (OpenAI/Azure) rejects any object lacking
        # ``additionalProperties: false``, which we cannot add without forbidding the
        # action keys (query/url/mandate/...). So convey the candidate shape as a
        # text instruction and drop to ``json_object`` mode (provider-agnostic — no
        # model-name special-casing).
        json_schema = self.settings.get("expansion_json_schema")
        # Opt-in: when the measurable-output contract is enabled, use the schema variant
        # that advertises the optional per-candidate ``expect`` field, so the text schema
        # hint tells the model it may declare a leaf's measurable target. Default path is
        # byte-identical (the plain schema from settings).
        if self._cfg.expansion.expect_contract_enabled:
            from agent.app.idea_dag_schemas import EXPANSION_JSON_SCHEMA_WITH_EXPECT
            json_schema = EXPANSION_JSON_SCHEMA_WITH_EXPECT
        schema_hint = (
            json_instruction_from_response_format({"type": "json_schema", "json_schema": json_schema})
            if json_schema
            else None
        )
        if schema_hint:
            messages = self._inject_schema_hint(messages, schema_hint)

        total_prompt_size = sum(len(msg.get("content", "")) for msg in messages)
        if total_prompt_size > 50000:
            self._logger.warning(f"[EXPANSION] Large prompt detected ({total_prompt_size} chars) for node {node_id} - may cause slow expansion")

        model_name = self.model_name or self._cfg.expansion.model
        reasoning_effort = self._cfg.generation.reasoning_effort
        text_verbosity = self._cfg.generation.text_verbosity
        max_tokens = self._cfg.expansion.max_tokens

        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=self._cfg.expansion.temperature,
            max_tokens=max_tokens,
            json_schema=None,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        try:
            estimated_tokens = (total_prompt_size // 4) + (max_tokens or 4096)
            self._logger.debug(f"[EXPANSION] Calling LLM for node {node_id} with model={model_name}, prompt={total_prompt_size} chars, max_tokens={max_tokens}, estimated ~{estimated_tokens} total tokens")
            
            default_timeout = self._cfg.timeouts.llm or 120
            expansion_timeout = self._cfg.timeouts.expansion or default_timeout
            if total_prompt_size > 50000 or estimated_tokens > 10000:
                expansion_timeout = max(expansion_timeout, 180)
            else:
                expansion_timeout = max(expansion_timeout, 120)
            
            try:
                preview = json.dumps(messages, indent=2, ensure_ascii=True)
            except Exception:
                preview = str(messages)
            if len(preview) > 2000:
                preview = preview[:2000] + "... [truncated]"
            self._logger.debug(f"[EXPANSION] LLM Input preview: {preview}")
            content = await self.io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=expansion_timeout,
            )
            output_preview = content[:2000] + "... [truncated]" if isinstance(content, str) and len(content) > 2000 else content
            self._logger.debug(f"[EXPANSION] LLM Output preview: {output_preview}")
            candidates, meta = self._parse_candidates(content, graph=graph, parent_node_id=node_id)
            self._logger.info(f"[EXPANSION] Parsed {len(candidates)} candidates from LLM response, meta={meta}")
            if not candidates:
                self._logger.error(f"[EXPANSION] CRITICAL: No candidates parsed from LLM response!")
                self._logger.error(f"[EXPANSION] LLM response length: {len(content) if content else 0} chars")
                self._logger.error(f"[EXPANSION] LLM response preview: {content[:500] if content else 'None'}")
                fallback_candidate = self._create_fallback_candidate(node, graph)
                if fallback_candidate:
                    self._logger.warning(f"[EXPANSION] Created fallback candidate: {fallback_candidate.get('title', 'Unknown')[:60]}...")
                    candidates = [fallback_candidate]
            if meta:
                node.details[DetailKey.EXPANSION_META.value] = meta
            return candidates
        except asyncio.TimeoutError as e:
            self._logger.error(f"[EXPANSION] Timeout during expansion for node {node_id}: {e}")
            self._logger.warning(f"[EXPANSION] Expansion timeout - returning empty candidates. Consider increasing expansion_timeout_seconds.")
            return []
        except Exception as e:
            self._logger.error(f"[EXPANSION] Exception during expansion: {e}", exc_info=True)
            return []

    def _inject_schema_hint(self, messages: List[Dict[str, str]], hint: str) -> List[Dict[str, str]]:
        """Append the candidate-shape instruction to the system message.

        The expansion schema cannot ride on the wire as strict structured output
        (free-form ``details``), so its shape is folded into the prompt instead.
        Operates on a copy and keeps the message count stable (system + user);
        if there is no system message, one is prepended.
        """
        out = [dict(msg) for msg in messages]
        for msg in out:
            if msg.get("role") == "system":
                existing = msg.get("content") or ""
                msg["content"] = f"{existing}\n\n{hint}" if existing else hint
                return out
        out.insert(0, {"role": "system", "content": hint})
        return out

    def _enhance_details_with_inline_links(self, details: Dict[str, Any]) -> Dict[str, Any]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        enhanced = dict(details)
        
        action_result = details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(action_result, dict):
            return enhanced
        
        action = action_result.get(ActionResultKey.ACTION.value)
        if action != IdeaActionType.VISIT.value:
            return enhanced
        
        success = action_result.get(ActionResultKey.SUCCESS.value, False)
        if not success:
            return enhanced
        
        links = action_result.get("links") or action_result.get("links_full") or []
        if not isinstance(links, list) or len(links) == 0:
            return enhanced
        
        link_contexts = action_result.get(ActionResultKey.LINK_CONTEXTS.value) or {}
        max_links_to_show = self._cfg.action.max_links_per_visit
        links_to_show = links[:max_links_to_show]
        
        inline_links_section = []
        for link_url in links_to_show:
            if not isinstance(link_url, str) or not link_url.startswith(("http://", "https://")):
                continue
            
            context_text = ""
            if isinstance(link_contexts, dict) and link_url in link_contexts:
                context = link_contexts[link_url]
                if isinstance(context, str) and context.strip():
                    context_text = context.strip()[:150]
            
            if context_text:
                inline_links_section.append(f"{context_text} [link: {link_url}]")
            else:
                inline_links_section.append(f"[link: {link_url}]")
        
        if inline_links_section:
            enhanced_action_result = dict(action_result)
            enhanced_action_result["_links_inline"] = "\n".join(inline_links_section)
            if len(links) > max_links_to_show:
                enhanced_action_result["_links_inline"] += f"\n... and {len(links) - max_links_to_show} more links (see 'links' field for full list)"
            enhanced[DetailKey.ACTION_RESULT.value] = enhanced_action_result
        
        return enhanced
    
    def _compact_details_for_expansion(self, details: Dict[str, Any]) -> Dict[str, Any]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        compact = dict(details)
        
        action_result = details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(action_result, dict):
            return compact
        
        compact_result = dict(action_result)
        
        large_fields_to_remove = [
            ActionResultKey.CONTENT_FULL.value,
            ActionResultKey.CONTENT_WITH_LINKS.value,
            "content_full",
            "content_with_links",
        ]
        
        for field in large_fields_to_remove:
            if field in compact_result:
                del compact_result[field]
        
        if ActionResultKey.CONTENT.value in compact_result:
            content = compact_result[ActionResultKey.CONTENT.value]
            if isinstance(content, str) and len(content) > 1000:
                compact_result[ActionResultKey.CONTENT.value] = content[:1000] + "... [truncated]"
        
        if "links_full" in compact_result:
            links_full = compact_result.get("links_full", [])
            if isinstance(links_full, list) and len(links_full) > 20:
                compact_result["links_full"] = links_full[:20]
                compact_result["_links_full_truncated"] = f"... and {len(links_full) - 20} more links"
        
        compact[DetailKey.ACTION_RESULT.value] = compact_result
        return compact
    
    def _extract_key_outcome(self, node: IdeaNode) -> Optional[str]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        action = node.details.get(DetailKey.ACTION.value, "")
        if not result.get(ActionResultKey.SUCCESS.value, False):
            error = result.get(ActionResultKey.ERROR.value, "unknown error")
            return f"FAILED: {str(error)[:80]}"
        if action == IdeaActionType.SEARCH.value:
            results = result.get(ActionResultKey.RESULTS.value, [])
            count = len(results) if isinstance(results, list) else 0
            top_urls = []
            if isinstance(results, list):
                for r in results[:3]:
                    if isinstance(r, dict) and r.get("url"):
                        top_urls.append(str(r["url"])[:80])
            if top_urls:
                return f"Found {count} results. Top URLs: {', '.join(top_urls)}"
            return f"Found {count} results"
        if action == IdeaActionType.VISIT.value:
            url = result.get(ActionResultKey.URL.value, "")
            page_title = result.get("page_title", "")
            content_chars = result.get("content_total_chars", 0)
            links_count = result.get("links_count", 0)
            parts = []
            if url:
                parts.append(f"Visited {str(url)[:80]}")
            if page_title:
                parts.append(f"page='{str(page_title)[:50]}'")
            parts.append(f"{content_chars} chars, {links_count} links")
            return ". ".join(parts)
        if action == IdeaActionType.THINK.value:
            return "Internal reasoning completed"
        return None

    def _build_messages(self, graph: IdeaDag, node: IdeaNode, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        max_nodes = self._cfg.expansion.max_context_nodes
        max_detail_chars = self._cfg.expansion.max_detail_chars
        max_children = self._cfg.engine.max_branching
        if max_children <= 1:
            max_children = 1
        path = graph.path_to_root(node.node_id)
        path = path[:max_nodes]
        
        serialized = []
        for entry in path:
            enhanced_details = self._enhance_details_with_inline_links(entry.details)
            compact_details = self._compact_details_for_expansion(enhanced_details)
            details_text = _safe_serialize_details(compact_details)
            if len(details_text) > max_detail_chars:
                details_text = details_text[:max_detail_chars]
            serialized.append(
                {
                    "node_id": entry.node_id,
                    "title": entry.title,
                    "status": entry.status.value,
                    "score": entry.score,
                    "action": entry.details.get(DetailKey.ACTION.value, "expansion"),
                    "goal": entry.details.get(DetailKey.GOAL.value, ""),
                    "justification": (
                        entry.details.get(DetailKey.JUSTIFICATION.value)
                        or entry.details.get(DetailKey.WHY_THIS_NODE.value)
                        or ""
                    ),
                    "key_outcome": self._extract_key_outcome(entry),
                    "details": details_text,
                }
            )
        allowed = self.settings.get("allowed_actions") or [a.value for a in IdeaActionType]
        allowed_actions = ", ".join(
            str(item) for item in allowed
            if str(item) != IdeaActionType.MERGE.value
        )
        path_json = json.dumps(serialized, ensure_ascii=True)
        
        blocked_sites = graph._blocked_sites if hasattr(graph, "_blocked_sites") else {}
        blocked_sites_list = [f"{domain}: {reason}" for domain, reason in blocked_sites.items()]
        blocked_sites_text = "\n".join(blocked_sites_list) if blocked_sites_list else "None"
        
        errors = []
        for entry in path:
            error = entry.details.get(DetailKey.ACTION_ERROR.value)
            if error:
                errors.append(f"{entry.title}: {error}")
        errors_text = "\n".join(errors) if errors else "None"
        
        memories_text = "None"
        if memories:
            from agent.app.idea_memory import MemoryManager
            temp_mm = MemoryManager(connector_chroma=None, namespace="temp")
            memories_text = temp_mm.format_memories_for_llm(memories, max_chars=4000)
        
        event_log = graph.build_event_log_table(node.node_id, max_events=15)
        event_log_json = json.dumps(event_log) if event_log else json.dumps("No events")
        
        system_template = self.settings.get("expansion_system_prompt")
        user_template = self.settings.get("expansion_user_prompt")
        effective_range = f"exactly {max_children}" if max_children <= 1 else f"2-{max_children}"
        try:
            system = system_template.format(
                allowed_actions=allowed_actions,
                max_children=effective_range,
            ) if system_template else ""
        except KeyError as fmt_err:
            self._logger.error(f"[EXPANSION] System prompt format error (missing key: {fmt_err}) - using raw template")
            system = (system_template or "").replace("{allowed_actions}", str(allowed_actions)).replace("{max_children}", str(effective_range))
        planning_addendum = str(
            self.settings.get(
                "expansion_planning_addendum",
                "Before producing candidates, build an internal plan with target facts, source strategy, and verification steps.",
            )
        ).strip()
        if planning_addendum:
            system = f"{system}\n\n{planning_addendum}" if system else planning_addendum
        # Opt-in measurable-output contract: append the leaf ``expect`` discipline only when
        # ``expansion_expect_contract_enabled`` is set. Default path is byte-identical.
        if self._cfg.expansion.expect_contract_enabled:
            system = f"{system}\n\n{_EXPECT_CONTRACT_ADDENDUM}" if system else _EXPECT_CONTRACT_ADDENDUM
        # Optional prompt prefixes, ordered top-to-bottom: reasoning exemplar (a
        # narrative demonstration) then the imperative rule checklist, then the existing
        # system template. The two env vars are fully independent — either may be set alone.
        prefix_blocks = []
        exemplar_block = _load_reasoning_exemplar()
        if exemplar_block:
            prefix_blocks.append(exemplar_block)
        rules_block = _load_reasoning_rules()
        if not rules_block and not os.environ.get("IDEA_TEST_REASONING_RULES", "").strip():
            # Env var UNSET: fall back to deterministic auto-classification of the root
            # mandate. Manual IDEA_TEST_REASONING_RULES (set) always takes priority.
            rules_block = _auto_reasoning_rules(self._root_mandate(graph))
        if rules_block:
            prefix_blocks.append(rules_block)
        # Single-use human steer injected via the interactive debugger (agent-debug
        # `f`/`feedback`). Surface it once, clearly labeled as a human steer, then
        # consume-and-clear it so it never appears on a later expansion. No-op /
        # byte-identical prompt when no feedback was ever injected.
        if DetailKey.HUMAN_FEEDBACK.value in node.details:
            # Consume-and-clear: remove the key from the live graph node so it never
            # surfaces on a later expansion (of this or any other node). A blank
            # value is cleared without surfacing, so it can't pollute later prompts.
            feedback = node.details.pop(DetailKey.HUMAN_FEEDBACK.value, None)
            if isinstance(feedback, str) and feedback.strip():
                prefix_blocks.append(
                    "HUMAN STEER (operator guidance for THIS expansion only; not part of "
                    f"the task itself):\n{feedback.strip()}"
                )
        # Corrective re-expansion context (opt-in via `got_reexpand_corrective_context_enabled`;
        # engine only writes this detail when the flag is on, so the default path never sees
        # it). Surfaces the triggering confidence-judge/follow-up-detector reason once, then
        # consume-and-clear so it never leaks into a later, unrelated expansion of this or any
        # other node — mirroring the HUMAN_FEEDBACK single-use pattern above.
        if DetailKey.REEXPAND_REASON.value in node.details:
            reexpand_reason = node.details.pop(DetailKey.REEXPAND_REASON.value, None)
            if isinstance(reexpand_reason, str) and reexpand_reason.strip():
                prefix_blocks.append(
                    "CORRECTIVE CONTEXT (the previous step at this node was inadequate for "
                    f"this reason):\n{reexpand_reason.strip()}\n"
                    "Target a source that actually provides the required attribute; do not "
                    "re-read the same insufficient page."
                )
        if prefix_blocks:
            prefix = "\n".join(prefix_blocks)
            system = f"{prefix}\n{system}" if system else prefix
        format_kwargs = dict(
            path_json=path_json,
            parent_id=node.node_id,
            parent_title=node.title,
            blocked_sites=blocked_sites_text,
            errors=errors_text,
            memories=memories_text,
            event_log=event_log_json,
        )
        try:
            user = user_template.format(**format_kwargs) if user_template else json.dumps(
                {
                    "path": serialized,
                    "parent_id": node.node_id,
                    "parent_title": node.title,
                    "blocked_sites": blocked_sites,
                    "errors": errors,
                    "memories": memories_text,
                    "event_log": event_log,
                },
                ensure_ascii=True,
            )
        except KeyError as fmt_err:
            self._logger.error(f"[EXPANSION] User prompt format error (missing key: {fmt_err}) - using manual substitution")
            user = user_template or ""
            for k, v in format_kwargs.items():
                user = user.replace("{" + k + "}", str(v))
        from agent.app.idea_policies.action_constants import PromptBuilder
        messages = PromptBuilder.build_messages(system_content=system, user_content=user)
        
        total_prompt_size = sum(len(msg.get("content", "")) for msg in messages)
        self._logger.debug(f"[EXPANSION] Prompt size: system={len(system)} chars, user={len(user)} chars, total={total_prompt_size} chars")
        if total_prompt_size > 50000:
            self._logger.warning(f"[EXPANSION] Large prompt detected ({total_prompt_size} chars) - may cause slow expansion. Consider reducing expansion_max_context_nodes or expansion_max_detail_chars")
        
        return messages

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        
        import re
        link_pattern = r'\[link:\s*(https?://[^\]]+)\]'
        match = re.search(link_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        url_pattern = r'https?://[^\s\)\]\>\"\']+'
        match = re.search(url_pattern, text)
        if match:
            url = match.group(0).rstrip('.,;:!?)')
            if url.startswith(("http://", "https://")):
                return url
        
        return None
    
    def _root_mandate(self, graph: Optional["IdeaDag"]) -> str:
        """The root node's mandate text (the task statement), or "" if unavailable."""
        if graph is None:
            return ""
        try:
            root = graph.get_node(graph.root_id())
        except Exception:
            return ""
        if root and isinstance(root.details, dict):
            return str(root.details.get("mandate") or "")
        return ""

    def _mandate_urls(self, graph: Optional["IdeaDag"]) -> List[str]:
        """URLs named in the root mandate (the task statement), in order.

        Used to recover a visit candidate's URL when the LLM emitted a visit node
        without one — explicit-URL mandates otherwise fail because the planner names
        the page in the title but drops the URL from details.
        """
        if graph is None:
            return []
        import re
        try:
            root = graph.get_node(graph.root_id())
        except Exception:
            return []
        mandate = ""
        if root and isinstance(root.details, dict):
            mandate = str(root.details.get("mandate") or "")
        if not mandate:
            return []
        urls: List[str] = []
        for u in re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', mandate):
            cleaned = u.rstrip('.,;:!?)]')
            if cleaned not in urls:
                urls.append(cleaned)
        return urls

    def _match_mandate_url(self, title: str, urls: List[str]) -> Optional[str]:
        """Best mandate URL for a URL-less visit candidate, matched by title overlap.

        Scores each URL by how many slug tokens of its last path segment (e.g.
        ``Eiffel_Tower`` -> {eiffel, tower}) appear in the candidate title. With a
        single mandate URL and no signal, returns it (the only sensible target).
        """
        if not urls:
            return None
        if len(urls) == 1:
            return urls[0]
        import re
        title_l = (title or "").lower()
        best_url, best_score = None, 0
        for u in urls:
            slug = u.rstrip('/').rsplit('/', 1)[-1]
            tokens = [t for t in re.split(r'[_\-%]+', slug.lower()) if len(t) > 2]
            score = sum(1 for t in tokens if t in title_l)
            if score > best_score:
                best_url, best_score = u, score
        return best_url if best_score > 0 else None

    def _is_url_from_visit(self, graph: IdeaDag, node_id: str) -> bool:
        node = graph.get_node(node_id)
        if not node:
            return False
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        action = NodeDetailsExtractor.get_action(node.details)
        return action == IdeaActionType.VISIT.value
    
    def _extract_url_from_path_context_with_source(self, graph: IdeaDag, node_id: str, candidate_title: str = "") -> tuple[Optional[str], Optional[str]]:
        url = self._extract_url_from_path_context(graph, node_id, candidate_title)
        if not url:
            return None, None
        
        path = graph.path_to_root(node_id)
        candidate_keywords = set()
        if candidate_title:
            import re
            words = re.findall(r'\b\w+\b', candidate_title.lower())
            candidate_keywords = {w for w in words if len(w) > 3}
        
        best_match = None
        best_score = 0
        best_source = None
        first_url = None
        first_source = None
        
        for path_node in reversed(path):
            details = path_node.details or {}
            action_result = details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(action_result, dict):
                continue
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            action_type = action_result.get(ActionResultKey.ACTION.value)
            
            if action_type == IdeaActionType.SEARCH.value:
                results = action_result.get(ActionResultKey.RESULTS.value) or []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            result_url = result.get("url") or result.get("link")
                            if result_url and isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                                if not first_url:
                                    first_url = result_url
                                    first_source = path_node.node_id
                                
                                if candidate_keywords:
                                    result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                                    score = sum(1 for kw in candidate_keywords if kw in result_text)
                                    if score > best_score:
                                        best_score = score
                                        best_match = result_url
                                        best_source = path_node.node_id
            
            elif action_type == IdeaActionType.VISIT.value:
                links_inline = action_result.get("_links_inline")
                if links_inline and isinstance(links_inline, str):
                    for line in links_inline.split('\n'):
                        if '[link:' in line:
                            extracted_url = self._extract_url_from_text(line)
                            if extracted_url:
                                if not first_url:
                                    first_url = extracted_url
                                    first_source = path_node.node_id
                                
                                if candidate_keywords:
                                    line_lower = line.lower()
                                    score = sum(1 for kw in candidate_keywords if kw in line_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = extracted_url
                                        best_source = path_node.node_id
        
        if best_match and best_source:
            return best_match, best_source
        if first_url and first_source:
            return first_url, first_source
        return None, None
    
    def _extract_url_from_path_context(self, graph: IdeaDag, node_id: str, candidate_title: str = "") -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        path = graph.path_to_root(node_id)
        candidate_keywords = set()
        if candidate_title:
            import re
            words = re.findall(r'\b\w+\b', candidate_title.lower())
            candidate_keywords = {w for w in words if len(w) > 3}
        
        best_match = None
        best_score = 0
        first_url = None
        
        for path_node in reversed(path):
            details = path_node.details or {}
            action_result = details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(action_result, dict):
                continue
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            action_type = action_result.get(ActionResultKey.ACTION.value)
            
            if action_type == IdeaActionType.SEARCH.value:
                results = action_result.get(ActionResultKey.RESULTS.value) or []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            result_url = result.get("url") or result.get("link")
                            if result_url and isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                                if not first_url:
                                    first_url = result_url
                                
                                if candidate_keywords:
                                    result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                                    score = sum(1 for kw in candidate_keywords if kw in result_text)
                                    if score > best_score:
                                        best_score = score
                                        best_match = result_url
            
            elif action_type == IdeaActionType.VISIT.value:
                links_inline = action_result.get("_links_inline")
                if links_inline and isinstance(links_inline, str):
                    for line in links_inline.split('\n'):
                        if '[link:' in line:
                            url = self._extract_url_from_text(line)
                            if url:
                                if not first_url:
                                    first_url = url
                                
                                if candidate_keywords:
                                    line_lower = line.lower()
                                    score = sum(1 for kw in candidate_keywords if kw in line_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = url
        
        return best_match if best_match else first_url
    
    def _parse_candidates(self, content: Optional[str], graph: Optional[IdeaDag] = None, parent_node_id: Optional[str] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not content:
            return [], {}
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self._logger.error(f"[EXPANSION] JSON PARSE ERROR: {e}")
            self._logger.error(f"[EXPANSION] Content preview (first 500 chars): {content[:500] if content else 'None'}")
            # Salvage a fence-wrapped / prose-wrapped / TRUNCATED plan instead of silently
            # planning nothing (see ``_repair_json_object``); still fails safe when unrepairable.
            data = _repair_json_object(content, required_key="candidates")
            if data is None:
                self._logger.error("[EXPANSION] JSON repair failed - degrading to an empty plan")
                return [], {}
            self._logger.info(
                f"[EXPANSION] Repaired malformed JSON (recovered {len(data.get('candidates') or [])} candidates)"
            )
        except Exception as e:
            self._logger.error(f"[EXPANSION] PARSE EXCEPTION: {e}", exc_info=True)
            self._logger.error(f"[EXPANSION] Content preview (first 500 chars): {content[:500] if content else 'None'}")
            return [], {}

        if not isinstance(data, dict):
            # A bare top-level array is the one non-object shape worth accepting as the plan.
            if isinstance(data, list):
                data = {"candidates": data}
            else:
                self._logger.error(f"[EXPANSION] Response is not a JSON object ({type(data).__name__})")
                return [], {}

        candidates = data.get("candidates", [])
        if not candidates:
            self._logger.error(f"[EXPANSION] NO CANDIDATES IN RESPONSE!")
            self._logger.error(f"[EXPANSION] Response data keys: {list(data.keys())}")
            self._logger.error(f"[EXPANSION] Full response data: {json.dumps(data, indent=2, ensure_ascii=True)[:1000]}")
        meta = data.get("meta") or {}
        cleaned: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            action = candidate.get(DetailKey.ACTION.value)
            title = candidate.get("title") or ""
            details = candidate.get("details") or {}
            if action:
                details = dict(details)
                details[DetailKey.ACTION.value] = action
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            justification = NodeDetailsExtractor.get_justification(candidate)
            if justification:
                details[DetailKey.JUSTIFICATION.value] = str(justification)

            # Opt-in: capture a leaf's measurable output contract. The model may place
            # ``expect`` at the candidate's top level or inside ``details``; both are
            # accepted. No-op when the flag is off or the field is absent (optional).
            if self._cfg.expansion.expect_contract_enabled:
                expect = candidate.get("expect")
                if expect is None:
                    expect = details.get(DetailKey.EXPECT.value)
                if isinstance(expect, str) and expect.strip():
                    details[DetailKey.EXPECT.value] = expect.strip()
                elif DetailKey.EXPECT.value in details:
                    # Drop a non-string/empty expect so it never reaches execution.
                    details.pop(DetailKey.EXPECT.value, None)

            candidate_goal = candidate.get("goal")
            local_goal: Optional[str] = None
            if isinstance(candidate_goal, str) and candidate_goal.strip():
                local_goal = candidate_goal.strip()
            else:
                existing_goal = details.get(DetailKey.GOAL.value) or details.get(DetailKey.ORIGINAL_GOAL.value)
                if isinstance(existing_goal, str) and existing_goal.strip():
                    local_goal = existing_goal.strip()
                elif isinstance(title, str) and title.strip():
                    local_goal = title.strip()

            if local_goal:
                details[DetailKey.GOAL.value] = details.get(DetailKey.GOAL.value) or local_goal
                if not details.get(DetailKey.ORIGINAL_GOAL.value):
                    details[DetailKey.ORIGINAL_GOAL.value] = local_goal
            
            if action == IdeaActionType.VISIT.value:
                url = (
                    details.get(DetailKey.URL.value)
                    or details.get(DetailKey.LINK.value)
                    or details.get("url")
                    or details.get("link")
                    or details.get("optional_url")
                )
                if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    extracted_url = None
                    source_node_id = None
                    
                    if title:
                        extracted_url = self._extract_url_from_text(title)
                    if not extracted_url and justification:
                        extracted_url = self._extract_url_from_text(str(justification))
                    if not extracted_url and graph and parent_node_id:
                        extracted_url, source_node_id = self._extract_url_from_path_context_with_source(graph, parent_node_id, candidate_title=title)
                    
                    if extracted_url:
                        details[DetailKey.URL.value] = extracted_url
                        if source_node_id:
                            source_node = graph.get_node(source_node_id)
                            if source_node:
                                from agent.app.idea_policies.action_constants import NodeDetailsExtractor
                                source_action = NodeDetailsExtractor.get_action(source_node.details)
                                if source_action == IdeaActionType.THINK.value:
                                    details[DetailKey.REQUIRES_DATA.value] = {
                                        "type": "url_from_think",
                                        "source_node_id": source_node_id
                                    }
                                else:
                                    details[DetailKey.REQUIRES_DATA.value] = {
                                        "type": "urls_from_visit" if self._is_url_from_visit(graph, source_node_id) else "urls_from_search",
                                        "source_node_id": source_node_id
                                    }
                            self._logger.info(f"[EXPANSION] Visit candidate requires data from node {source_node_id}: {extracted_url[:60]}...")
                        self._logger.info(f"[EXPANSION] Proactively extracted URL for visit candidate '{title[:50]}...': {extracted_url[:60]}...")
                    else:
                        # Last resort: recover from a URL named in the mandate (explicit-URL
                        # tasks otherwise fail — the planner names the page but drops the URL).
                        recovered = self._match_mandate_url(title, self._mandate_urls(graph))
                        if recovered:
                            details[DetailKey.URL.value] = recovered
                            self._logger.info(f"[EXPANSION] Recovered visit URL from mandate for '{title[:50]}...': {recovered[:60]}...")
                        else:
                            # No URL yet — KEEP the node. In a search-driven task the visit's
                            # URL is resolved at execution time from a sibling search's results
                            # (VisitLeafAction._extract_urls_from_parent_search_results); dropping
                            # it here would break the search->visit pipeline (visits=0).
                            self._logger.warning(f"[EXPANSION] Visit candidate has no URL yet (search-driven?); keeping for runtime resolution: title='{title[:60]}...'")

            if action == IdeaActionType.SEARCH.value:
                details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_search"}
            
            cleaned.append(
                {
                    "title": str(title),
                    "details": details,
                    "score": candidate.get("score"),
                }
            )
        return cleaned, dict(meta)
    
    def _create_fallback_candidate(self, node: "IdeaNode", graph: Optional["IdeaDag"] = None) -> Optional[Dict[str, Any]]:
        import re
        title = node.title or ""
        mandate = node.details.get("mandate") or ""
        text_to_search = f"{title} {mandate}"
        
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text_to_search)
        
        if urls:
            url = urls[0]
            self._logger.info(f"[EXPANSION] Fallback: Creating visit candidate for URL found in mandate: {url[:60]}...")
            return {
                "title": f"Visit {url}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.URL.value: url,
                    "optional_url": url,
                    DetailKey.JUSTIFICATION.value: "Fallback candidate: URL extracted from mandate",
                },
                "score": None,
            }
        
        text_lower = text_to_search.lower()
        if any(keyword in text_lower for keyword in ["visit", "go to", "fetch", "open", "navigate"]):
            if urls:
                url = urls[0]
                return {
                    "title": f"Visit {url}",
                    "details": {
                        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                        DetailKey.URL.value: url,
                        "optional_url": url,
                        DetailKey.JUSTIFICATION.value: "Fallback candidate: Visit action inferred from mandate",
                    },
                    "score": None,
                }
        
        if any(keyword in text_lower for keyword in ["search", "find", "look for", "query"]):
            query = title[:100] if title else "Search"
            return {
                "title": f"Search: {query}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.QUERY.value: query,
                    DetailKey.JUSTIFICATION.value: "Fallback candidate: Search action inferred from mandate",
                },
                "score": None,
            }
        
        self._logger.warning(f"[EXPANSION] Fallback: Creating generic think node (no URL or search query found)")
        return {
            "title": "Analyze and plan next steps",
            "details": {
                DetailKey.ACTION.value: IdeaActionType.THINK.value,
                DetailKey.JUSTIFICATION.value: "Fallback candidate: Generic think node",
            },
            "score": None,
        }