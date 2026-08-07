"""
Authoring — turn N solved task instances of one archetype into ONE generalized strategy note.

Methodology (i) of the two the plan considers: a dedicated, curated authoring pass over task
*statements*. Methodology (ii) — distilling notes from adaptive-engine run traces
(``testing/contract_log.py``) — is DELIBERATELY NOT BUILT: a raw trace contains every fetched
page and every intermediate answer, i.e. the whole literal ledger a leak gate must defend
against is inside the distiller's own input by construction. Prove the smaller leak surface
first.

The pipeline mirrors ``testing/scaffold_compiler.py``'s shape on purpose — an isolated
single-LLM-call function, a disk cache keyed by a content hash, parse-and-validate separated
from the call so everything downstream is unit-testable with a mocked author:

    seed task ids
      -> public statements ONLY                (never a validator, never a docstring)
      -> [pre-flight] the author prompt itself must pass the leak gate
      -> one strong-model JSON call            (_author_note_llm, cached)
      -> StrategyNote (schema-validated)
      -> [write gate] all four leak-gate layers against EVERY seed ledger
      -> notes/<note_id>.json                  (status: candidate)

Two properties are load-bearing and both are enforced here rather than promised:

* **The author model never sees ground truth.** Its input is the seed tasks'
  ``get_task_statement()`` — the same text the solver gets — and the prompt is itself run
  through the ledger check before it is sent. A note cannot generalize a fact its author was
  never shown.
* **A note is born unproven.** It is written with ``status="candidate"`` and zero held-out
  metrics, so :func:`~agent.app.strategy_library.schema.is_active` keeps it out of retrieval
  until ``scripts/eval_strategy_library_generalization.py`` has measured it. Authoring cannot
  promote.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.app.strategy_library import leak_gate
from agent.app.strategy_library.leak_gate import LeakGateError, TaskLedger
from agent.app.strategy_library.retrieval import NOTES_DIR
from agent.app.strategy_library.schema import (
    SOURCE_HAND_AUTHORED,
    STATUS_CANDIDATE,
    HeldOutMetrics,
    NoteValidationError,
    StrategyNote,
    note_to_dict,
    validate_note,
)

logger = logging.getLogger(__name__)

#: Same default as the scaffold compiler: the reference (strong) model authors offline, once.
#: Override with ``IDEA_TEST_STRATEGY_AUTHOR_MODEL``.
DEFAULT_AUTHOR_MODEL = "google/gemini-3.1-pro-preview"

#: Bumped whenever ``_NOTE_META_PROMPT`` changes, so the cache key changes with it — a cached
#: note authored under an older prompt is a different artifact, not a hit.
PROMPT_VERSION = 1

MAX_STATEMENT_CHARS = 6000


class AuthoringError(RuntimeError):
    """Raised when a note cannot be authored (no seeds, no author, unparseable output)."""


_NOTE_META_PROMPT = (
    "You are writing ONE short piece of GENERALIZED STRATEGY ADVICE for a research agent.\n\n"
    "You will be shown several task STATEMENTS that share a structure (an 'archetype'). Your "
    "job is to describe, in prose, the DISCIPLINE that reliably solves tasks OF THAT SHAPE — "
    "the failure mode a weak model falls into and the habit that avoids it. The advice will be "
    "pasted into the prompt of a DIFFERENT, UNSEEN task of the same shape, whose entities, "
    "numbers and answer you do not know and must not assume.\n\n"
    "Return ONLY JSON of this exact shape:\n"
    "{\n"
    '  "title": "a few words naming the discipline",\n'
    '  "advice": "the advice itself, 2-5 sentences of plain prose",\n'
    '  "embedding_text": "how a task of this shape is typically phrased, in the statements\' '
    'own vocabulary, so this note can be retrieved for a task like it"\n'
    "}\n\n"
    "RULES:\n"
    "1. GENERALIZED, NOT MEMORIZED. Do not name any entity, place, person, number, unit value, "
    "date or answer from the statements. Someone reading your advice must not be able to tell "
    "WHICH tasks it came from.\n"
    "2. NO WORKED EXAMPLES. Do not write 'e.g. X -> Y', do not show a sample input and its "
    "result, do not invent illustrative values. State the method; the solver builds its own "
    "examples.\n"
    "3. Write it as an INSTRUCTION to the solver ('list every candidate's value before "
    "concluding'), not as a description of the tasks.\n"
    "4. Name the trap. The most useful sentence is usually the one that says what NOT to "
    "shortcut to.\n"
    "5. Keep 'advice' under 900 characters. It is a prompt addendum, not a document.\n"
    "6. 'embedding_text' is for retrieval only: paraphrase the SHAPE of the request using the "
    "kind of wording these statements use, still naming no specific entity.\n"
    "Output JSON only — no prose, no markdown fences."
)


# --------------------------------------------------------------------------------------
# ids and cache keys
# --------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    words = re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower()).split()
    return "_".join(words[:6]).strip("_")


def default_note_id(archetype: str, seed_task_ids: Sequence[str]) -> str:
    """``<archetype>_from_<ids>`` — readable, and unique per (archetype, seed set)."""
    seeds = "_".join(sorted(str(t).strip() for t in seed_task_ids if str(t).strip()))
    return f"{_slug(archetype) or 'note'}_from_{seeds or 'unknown'}"


def author_key(archetype: str, seed_task_ids: Sequence[str], author_model: str) -> str:
    """Stable cache key: (prompt version, archetype, seed set, author model)."""
    payload = json.dumps(
        {
            "v": PROMPT_VERSION,
            "archetype": str(archetype or "").strip().lower(),
            "seeds": sorted(str(t).strip() for t in seed_task_ids or ()),
            "model": str(author_model or "").strip(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_cache_dir() -> Path:
    """Where raw author outputs are cached. Override with ``IDEA_TEST_STRATEGY_NOTES_DIR``."""
    override = os.environ.get("IDEA_TEST_STRATEGY_NOTES_CACHE", "").strip()
    if override:
        return Path(override)
    # services/agent/strategy_notes_cache (sibling of compiled_plans/)
    return Path(__file__).resolve().parent.parent.parent / "strategy_notes_cache"


# --------------------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------------------


def seed_statements(
    seed_task_ids: Sequence[str], **ledger_kwargs: Any
) -> Tuple[Dict[str, str], List[TaskLedger]]:
    """``({task_id: public statement}, [ledger, ...])`` for the seed tasks.

    Both come from the SAME module load, which is the point: the statements are what the author
    model is allowed to see, and the ledgers are everything it is not — built from the same
    file, so nothing can be forgotten by hand.
    """
    statements: Dict[str, str] = {}
    ledgers: List[TaskLedger] = []
    for raw_id in seed_task_ids or ():
        task_id = str(raw_id).strip()
        if not task_id:
            continue
        path = leak_gate.find_task_file(task_id, **ledger_kwargs)
        if path is None:
            raise AuthoringError(f"no task module found for seed task '{task_id}'")
        module = leak_gate.load_task_module(path)
        ledger = leak_gate.build_ledger(module, task_id=task_id)
        ledgers.append(ledger)
        try:
            statements[task_id] = module.get_task_statement()
        except Exception as exc:  # noqa: BLE001
            raise AuthoringError(f"seed task '{task_id}' has no usable statement: {exc}") from exc
    if not statements:
        raise AuthoringError("no seed tasks supplied")
    return statements, ledgers


def build_author_prompt(archetype: str, statements: Dict[str, str]) -> str:
    """The user message: the archetype plus each seed's PUBLIC statement, verbatim.

    Seeds are numbered, not labelled with their task ids: the id tells the author model nothing
    it can use, and stamping it into the prompt would put a benchmark identifier into the one
    text that must stay purely about task SHAPE (``author_note``'s pre-flight ledger check reads
    this string, and the seeds are recorded in the note's provenance regardless).
    """
    lines = [f"ARCHETYPE: {archetype}", ""]
    for index, (_task_id, statement) in enumerate(sorted(statements.items()), 1):
        lines.append(f"--- TASK {index} ---")
        lines.append(_clip(statement, MAX_STATEMENT_CHARS))
        lines.append("")
    lines.append(
        "Write the generalized strategy note for tasks of this shape. Return the JSON object."
    )
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    out = str(text or "")
    if len(out) <= limit:
        return out
    return out[:limit].rsplit(" ", 1)[0] + " ... [truncated]"


# --------------------------------------------------------------------------------------
# the LLM call (isolated, mockable)
# --------------------------------------------------------------------------------------


async def _author_note_llm(agent_io: Any, prompt: str, author_model: str, max_tokens: int) -> str:
    """Single LLM call to the strong author model; returns the raw JSON string.

    Isolated exactly like ``scaffold_compiler._author_plan_llm`` so parse / validate / gate /
    cache are all testable offline with a mocked author.
    """
    messages = [
        {"role": "system", "content": _NOTE_META_PROMPT},
        {"role": "user", "content": prompt},
    ]
    payload = agent_io.build_llm_payload(
        messages=messages, json_mode=True, model_name=author_model,
        temperature=0.1, max_tokens=max_tokens,
    )
    return (await agent_io.query_llm(payload, model_name=author_model)) or ""


def parse_note_payload(raw: str) -> Dict[str, Any]:
    """Parse the author's raw output into ``{title, advice, embedding_text}``.

    Tolerates markdown fences and surrounding prose exactly like ``scaffold_compiler.parse_plan``
    — a strong model still fences its JSON now and then, and re-authoring costs real money.
    """
    text = (raw or "").strip()
    if not text:
        raise AuthoringError("author returned empty output")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise AuthoringError(f"no JSON object in author output: {text[:200]!r}")
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuthoringError(f"author output is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise AuthoringError(f"author output is not a JSON object: {type(obj).__name__}")
    if not str(obj.get("advice") or "").strip():
        raise AuthoringError("author output has no 'advice'")
    return obj


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------


async def author_note(
    archetype: str,
    seed_task_ids: Sequence[str],
    *,
    agent_io: Any = None,
    author_model: Optional[str] = None,
    note_id: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    notes_dir: Optional[Path] = None,
    max_tokens: int = 1024,
    force: bool = False,
    persist: bool = True,
    require_llm_audit: bool = True,
    auditor_model: Optional[str] = None,
    **ledger_kwargs: Any,
) -> Tuple[StrategyNote, Dict[str, Any]]:
    """Author (or re-load) one :class:`StrategyNote` for ``archetype`` from ``seed_task_ids``.

    :returns: ``(note, info)`` where ``info`` carries the cache verdict, the author model and
        the write-gate verdict (which is also stamped onto the note itself).
    :raises AuthoringError: no seeds, cache miss with no ``agent_io``, unparseable output.
    :raises LeakGateError: the authored advice did not clear the four-layer gate. Deliberately
        an exception rather than a returned flag: there is no sensible partial success, and a
        caller that ignores a return value must not be able to ship a leaking note.
    """
    model = (author_model or os.environ.get("IDEA_TEST_STRATEGY_AUTHOR_MODEL", "").strip()
             or DEFAULT_AUTHOR_MODEL)
    statements, ledgers = seed_statements(seed_task_ids, **ledger_kwargs)
    seeds = sorted(statements)
    prompt = build_author_prompt(archetype, statements)

    # Pre-flight: the prompt we are about to SEND must itself be ground-truth-free. It is built
    # only from public statements, so this should never fire — which is exactly why it is worth
    # asserting: if a future edit starts feeding docstrings or validators into the author, this
    # is what stops it, rather than a reviewer noticing months later.
    prompt_verdict = leak_gate.check_text(prompt, ledgers)
    if not prompt_verdict.passed:
        raise LeakGateError(
            f"the author prompt for archetype '{archetype}' carries seed ground truth: "
            f"{prompt_verdict.summary()}"
        )

    key = author_key(archetype, seeds, model)
    cache_path = (Path(cache_dir) if cache_dir is not None else default_cache_dir()) / f"{key}.json"
    payload: Optional[Dict[str, Any]] = None
    cache_verdict = "miss"
    if not force and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_verdict = "hit"
        except (OSError, ValueError) as exc:
            logger.warning("Strategy library: cached author output %s is unusable (%s)", cache_path, exc)
            payload = None

    if payload is None:
        if agent_io is None:
            raise AuthoringError(
                f"cache miss for archetype '{archetype}' seeds {seeds} and no agent_io to author it"
            )
        logger.info("Strategy library: authoring a note for '%s' from %s with %s",
                    archetype, seeds, model)
        payload = parse_note_payload(await _author_note_llm(agent_io, prompt, model, max_tokens))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    note = _note_from_payload(
        payload, archetype=archetype, seeds=seeds, note_id=note_id, author_model=model,
        prompt=prompt,
    )

    verdict = await leak_gate.audit_text(
        note.advice, ledgers, agent_io,
        require_llm_audit=require_llm_audit,
        model_name=auditor_model,
    )
    note.leak_gate = verdict.as_dict()
    if not verdict.passed:
        raise LeakGateError(
            f"note '{note.note_id}' failed the leak gate and was NOT written: {verdict.summary()}"
        )

    path: Optional[Path] = None
    if persist:
        path = write_note(note, notes_dir=notes_dir)
    return note, {
        "cache": cache_verdict,
        "key": key,
        "cache_path": str(cache_path),
        "author_model": model,
        "seeds": seeds,
        "note_path": str(path) if path else None,
        "leak_gate": verdict.as_dict(),
    }


def _note_from_payload(
    payload: Dict[str, Any],
    *,
    archetype: str,
    seeds: Sequence[str],
    note_id: Optional[str],
    author_model: str,
    prompt: str,
) -> StrategyNote:
    advice = " ".join(str(payload.get("advice") or "").split())
    raw = {
        "note_id": note_id or default_note_id(archetype, seeds),
        "archetype": str(archetype or "").strip(),
        "title": str(payload.get("title") or "").strip(),
        "advice": advice,
        "embedding_text": " ".join(str(payload.get("embedding_text") or "").split()),
        "provenance": {
            # "hand_authored" per ``plan_library``'s vocabulary: a curated authoring pass over
            # task statements, as opposed to "mined" (methodology ii, distilled from run
            # traces). ``authored_by`` records that a strong model drafted the prose.
            "source": SOURCE_HAND_AUTHORED,
            "based_on_tasks": list(seeds),
            "mined_from_trace_ids": [],
            "authored_by": author_model,
            "author_prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt_version": PROMPT_VERSION,
        },
        "evaluation": HeldOutMetrics().as_dict(),
        # Authoring cannot promote: the eval script is the only thing that fills ``evaluation``.
        "status": STATUS_CANDIDATE,
    }
    try:
        return validate_note(raw)
    except NoteValidationError as exc:
        raise AuthoringError(f"authored note is invalid: {exc}") from exc


def write_note(note: StrategyNote, *, notes_dir: Optional[Path] = None) -> Path:
    """Persist a note to the corpus. Callers must gate it FIRST — see :func:`author_note`."""
    directory = Path(notes_dir) if notes_dir is not None else NOTES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{note.note_id}.json"
    path.write_text(
        json.dumps(note_to_dict(note), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def record_evaluation(
    note: StrategyNote,
    metrics: HeldOutMetrics,
    *,
    notes_dir: Optional[Path] = None,
    persist: bool = True,
) -> StrategyNote:
    """Stamp measured held-out metrics onto a note and re-validate it.

    Separate from authoring on purpose: measuring is a live, paid step the eval script owns, and
    keeping it out of ``author_note`` is what makes "authoring cannot promote" structural rather
    than a convention.
    """
    updated = replace(note, evaluation=metrics)
    validated = validate_note(note_to_dict(updated))
    validated.leak_gate = dict(note.leak_gate or {})
    if persist:
        write_note(validated, notes_dir=notes_dir)
    return validated
