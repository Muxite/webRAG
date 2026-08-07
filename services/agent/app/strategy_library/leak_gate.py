"""
Leak gate — the automated, pre-registered check every strategy note must pass.

This repo has already shipped one real leak: commit ``c0dbc720`` fixed two hand-authored
codebench compiled plans that embedded the exact inputs and verdicts of HIDDEN keystone tests
into the instructions handed to the executing model. It was caught by a human re-read, not by
any check — and the one automated check that existed for that domain (``"def test_" not in
plan_text``) could not have caught it, because the leak was natural-language paraphrase, not
source code. A library of *generalized* advice is the same failure mode with a much larger
blast radius: one leaked entry corrupts every future task that retrieves it.

So the gate is four independent layers, each of which catches something the others cannot:

  (a) **Literal ledger** (:func:`build_ledger`) — the ground-truth-bearing literals of ONE task,
      harvested PROGRAMMATICALLY from the task module (hidden test content and keystone ids for
      codebench tasks; answer/decoy constants, entity tables and validator regexes for
      wiki-reasoning tasks), minus everything the task's own public statement already gives
      away. Never hand-copied per note, so the ledger itself is auditable and cannot silently
      omit the one literal someone forgot.

  (b) **Normalized substring overlap** (:func:`overlap_findings`) — every ledger entry looked up
      in the candidate text by plain substring match over a normalized form. Deliberately NOT a
      word-boundary regex: task 146's own test comment records that ``\\bmead\\b`` does not match
      inside ``res_mead``, which is exactly how a leak hides from a ``\\b``-anchored blocklist.
      Normalization folds case and accents and deletes separators, so ``res_mead`` still
      contains ``mead``, ``6,527`` still matches ``6527``, and ``Atatürk`` still matches
      ``ataturk``.

  (c) **LLM auditor** (:func:`llm_audit`) — one cheap, temperature-0 JSON-mode call (reusing
      ``plan_library/slot_fill.py``'s batched idiom) asking a single binary question about the
      ledger's FACTS: does this text state, imply, or make directly derivable any of them, as
      opposed to describing a method that would apply to different values? This is the layer
      that catches a *paraphrased* leak — c30's "the common lines are a, c, d -- 3 'equal' ops"
      restates a hidden assertion without reusing a single one of its long literals.

  (d) **Worked-example lint** (:func:`worked_example_findings`) — a structural rule codifying
      the c0dbc720 fix's own wording as a positive requirement: example-shaped spans (an arrow,
      "returns", "e.g.", a quoted literal) are a leak vector by construction, so a span carrying
      a ledger token is rejected outright, and a span carrying any literal at all is rejected
      unless the text frames it as "construct your own ... rather than reusing the worked
      examples".

Layer (c) costs a call, so the split is: **write time** runs all four (an entry enters the
corpus only once an auditor has seen it); **read time** runs the deterministic three against
the CURRENT task's ledger and, on any finding, drops the note and proceeds with no advice —
fail toward silence, the same philosophy ``plan_library/retrieval.py`` and
``contract_satisfaction.py`` already state.

Pure except for layer (c): the ledger builder imports task modules (which is only module-level
constant evaluation) and nothing here touches the network.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# app/strategy_library/ -> app/
_APP_DIR = Path(__file__).resolve().parent.parent
WIKI_TESTS_DIR = _APP_DIR / "idea_tests"
CODE_TESTS_DIR = _APP_DIR / "idea_code_tests"

KIND_WIKI = "wiki"
KIND_CODE = "code"

# --- entry kinds and their match modes ---------------------------------------------------
#: A single number ("836", "6527"). Matched TIGHT.
ENTRY_NUMBER = "number"
#: A single name/string value ("Lake Mead", "mead", "test_tld_too_short"). Matched TIGHT.
ENTRY_NAME = "name"
#: A whole sequence literal, reduced to its ordered word runs (``["a","b","c","d","e"]`` ->
#: ``a b c d e``). Matched as CONSECUTIVE WHOLE RUNS, because the punctuation between a list's
#: elements is exactly what a paraphrase rewrites (``old=['a','b','c','d','e']`` vs
#: ``["a", "b", "c", "d", "e"]``) while the element ORDER is what it cannot change.
ENTRY_SEQUENCE = "sequence"

#: Minimum normalized length per kind. Below these a "hit" is noise, and a noisy ledger is
#: worse than a small one: it makes the read-time gate drop every note, silently disabling the
#: whole mechanism. Small numbers (counts, indices, 0/1 scores) and short words are therefore
#: left to layers (c) and (d) instead.
MIN_LEN = {ENTRY_NUMBER: 3, ENTRY_NAME: 4, ENTRY_SEQUENCE: 5}

#: Upper bound on a normalized entry. A whole docstring or a whole entity table concatenated
#: can never be a substring hit; keeping it would only bloat the ledger a reviewer has to read.
MAX_ENTRY_CHARS = 400

#: Harvested from a REGEX value, these come from URL/slug scaffolding rather than from ground
#: truth, and would otherwise put "wiki" into the ledger of every wikipedia task — which would
#: reject every note that says the word "Wikipedia". Small, corpus-level (never per-note), and
#: listed here so it is reviewable.
_REGEX_RUN_STOPWORDS = frozenset({
    "wiki", "wikipedia", "http", "https", "www", "html", "index", "php", "org", "com", "net",
    "true", "false", "none", "null", "name", "page", "text", "value", "http s",
})

#: Module-level constant names whose VALUE is ground truth by convention across this corpus.
_GROUND_TRUTH_NAME_RE = re.compile(
    r"(?:^|_)(WINNER|ANSWER|KEYSTONE|EXPECTED|CORRECT|TRUTH|GOLD|DECOY|TARGET|SURVIVOR)"
)
#: ...minus the ones that only hold a URL/slug (see :data:`_REGEX_RUN_STOPWORDS`).
_SLUG_NAME_RE = re.compile(r"(?:^|_)(SLUG|URL|LINK)(?:_|$)")
#: Mapping keys inside an entity table that hold a URL/slug rather than a fact.
_SLUG_KEY_RE = re.compile(r"slug|url|link", re.I)
#: Mapping keys / constant names whose string value is a REGEX, not prose. Their literal word
#: runs are still harvested (that is where ``\bmead\b`` -> ``mead`` comes from), but the raw
#: pattern is not stored as a name entry.
_REGEX_KEY_RE = re.compile(r"(?:^|_)(rx|re|regex|pattern)(?:_|$)", re.I)

#: Truthy keys in an entity table that mark the row as THE answer.
_WINNER_KEYS = ("winner", "answer", "is_answer", "correct", "keystone", "survivor")

_NUMBER_IN_TEXT_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RUN_RE = re.compile(r"[A-Za-zÀ-ɏ]{3,}")
_REGEX_META = set(r"\|()[]{}?*+^$")

# --- layer (d) patterns ------------------------------------------------------------------
#: Cues that a span is presenting a worked example rather than describing a method.
_EXAMPLE_CUE_RE = re.compile(
    r"->|=>|→|\breturns\b|\byields\b|\bevaluates to\b|\bgives\b|\be\.?g\.?\b|"
    r"\bfor example\b|\bworked example\b|\bfor instance\b|\bsuch as\b",
    re.I,
)
#: A concrete literal inside that span: a quoted/backticked string, a bracketed list, or a
#: multi-digit number.
_EXAMPLE_LITERAL_RE = re.compile(r"'[^']{1,80}'|\"[^\"]{1,80}\"|`[^`]{1,80}`|\[[^\]]{1,120}\]|\d{2,}")
#: The c0dbc720 fix's own framing, promoted to a positive rule.
_OWN_CONSTRUCTION_RE = re.compile(
    r"\b(?:of|your|their) own\b|\bconstruct (?:your|their) own\b|\binvent(?:ing)? (?:your|their) own\b|"
    r"\brather than reusing\b|\bdo not reuse\b|\bmake up your own\b",
    re.I,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+|\n+")

# --- severities ---------------------------------------------------------------------------
SEVERITY_REJECT = "reject"
SEVERITY_WARN = "warn"

LAYER_OVERLAP = "overlap"
LAYER_WORKED_EXAMPLE = "worked_example"
LAYER_LLM_AUDIT = "llm_audit"


class LeakGateError(RuntimeError):
    """Raised when a note is written (or authored) despite failing the gate."""


# --------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------


def _fold(text: str) -> str:
    """Lowercase + strip accents. ``Atatürk`` -> ``ataturk``, ``Martín`` -> ``martin``."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def normalize_tight(text: str) -> str:
    """Fold, then delete SEPARATORS only (whitespace, ``_``, ``-``, ``,``).

    Keeps every other character, so ``user@example..com`` stays distinguishable from
    ``user@examplecom`` — the two are different hidden test cases, and an aggressive
    punctuation strip would collapse them into one and silently un-protect both. Deleting
    separators is what makes ``res_mead`` contain ``mead`` and ``6,527`` equal ``6527``.
    """
    return re.sub(r"[\s_\-,]+", "", _fold(text))


def normalize_runs(text: str) -> str:
    """Fold, then reduce to the ordered alphanumeric RUNS, space-separated and space-padded.

    This is the sequence-literal form. Collapsing punctuation entirely (``["delete","a"]`` ->
    ``deletea``) is too lossy in the other direction: it also matches the prose "emitting
    'delete' (advance old only)", which is a false positive on a text that leaks nothing.
    Keeping runs keeps the one property a re-punctuated copy cannot lose — that the elements
    appear, whole and in order, one after another.
    """
    return " " + " ".join(re.findall(r"[a-z0-9]+", _fold(text))) + " "


def _normalized(text: str, mode: str) -> str:
    if mode != ENTRY_SEQUENCE:
        return normalize_tight(text)
    runs = normalize_runs(text).strip()
    return runs


def _entry_length(value: str, kind: str) -> int:
    """Comparable length: a sequence's separators are structure, not content."""
    return len(value.replace(" ", "")) if kind == ENTRY_SEQUENCE else len(value)


def _contains(tight_haystack: str, runs_haystack: str, value: str, kind: str) -> bool:
    """Is ``value`` present? Plain substring — never ``\\b``-anchored (the ``res_mead`` case)."""
    if not value:
        return False
    if kind == ENTRY_SEQUENCE:
        return f" {value} " in runs_haystack
    return value in tight_haystack


# --------------------------------------------------------------------------------------
# (a) the ledger
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    """One ground-truth-bearing literal, with the provenance that makes it auditable."""

    value: str          #: normalized comparison form
    raw: str            #: as it appears in the task module
    kind: str           #: ENTRY_NUMBER | ENTRY_NAME | ENTRY_SEQUENCE
    source: str         #: where it was harvested from, e.g. "ENTITIES[0].depth"

    def as_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "raw": self.raw, "kind": self.kind, "source": self.source}


@dataclass(frozen=True)
class TaskLedger:
    """Everything one task considers secret, plus the public text that defines "secret"."""

    task_id: str
    kind: str
    #: What the solver legitimately receives (statement + fixture files). An entry that appears
    #: here is NOT a secret and is dropped: c32's task statement publishes three worked email
    #: examples on purpose, and flagging a note for repeating a *given* would be noise.
    public_text: str
    entries: Tuple[LedgerEntry, ...] = ()
    #: One-line ground-truth statements for the LLM auditor. Unlike ``entries`` these may
    #: contain public names: the auditor is asked about FACTS ("the answer is X"), which is a
    #: relation the deterministic layer structurally cannot see when every name is public.
    facts: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "entries": [e.as_dict() for e in self.entries],
            "facts": list(self.facts),
        }


class _Harvester:
    """Accumulates entries/facts while walking a task module, de-duping as it goes."""

    def __init__(self, public_text: str, exclude: Iterable[str] = ()) -> None:
        self.public_tight = normalize_tight(public_text)
        self.public_runs = normalize_runs(public_text)
        #: Normalized values that are LABELS rather than ground truth — the task's own id, most
        #: importantly, which a docstring repeats ("Test 084: ...") and which would otherwise
        #: make a ledger reject any text that mentions the task by number.
        self.exclude = {normalize_tight(e) for e in exclude if str(e or "").strip()}
        self._entries: Dict[Tuple[str, str], LedgerEntry] = {}
        self.facts: List[str] = []

    # -- adding ------------------------------------------------------------------------

    def add(self, raw: Any, kind: str, source: str) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        value = _normalized(text, kind)
        if not (MIN_LEN.get(kind, 4) <= _entry_length(value, kind) <= MAX_ENTRY_CHARS):
            return
        if value in self.exclude:
            return
        if _contains(self.public_tight, self.public_runs, value, kind):
            return  # already given away by the task statement -> not a secret
        self._entries.setdefault((kind, value), LedgerEntry(value, text, kind, source))

    def add_fact(self, text: str) -> None:
        cleaned = " ".join(str(text or "").split())
        if cleaned and cleaned not in self.facts:
            self.facts.append(cleaned)

    def entries(self) -> Tuple[LedgerEntry, ...]:
        return tuple(sorted(self._entries.values(), key=lambda e: (e.kind, e.value)))

    # -- walking runtime values ---------------------------------------------------------

    def walk_value(self, value: Any, source: str, *, key: str = "") -> str:
        """Recursively harvest a runtime object; returns its concatenated scalar leaves.

        The return value is what makes a *sequence* entry possible: a list's leaves joined with
        no separator is exactly the form a re-punctuated paraphrase of that list collapses to.
        """
        if value is None or isinstance(value, bool):
            return ""
        if isinstance(value, (int, float)):
            self.add(_render_number(value), ENTRY_NUMBER, source)
            return str(value)
        if isinstance(value, str):
            return self._walk_string(value, source, key=key)
        if hasattr(value, "pattern"):  # a compiled regex
            return self._walk_string(str(value.pattern), source, key=key or "rx")
        if isinstance(value, Mapping):
            parts = []
            for k, v in value.items():
                parts.append(self.walk_value(v, f"{source}.{k}", key=str(k)))
            # No composite entry for a mapping: key order is an implementation detail and the
            # concatenation of unrelated fields is not a literal anyone could leak.
            return " ".join(p for p in parts if p)
        if isinstance(value, (list, tuple, set, frozenset)):
            parts = [
                self.walk_value(item, f"{source}[{i}]", key=key)
                for i, item in enumerate(value)
            ]
            joined = " ".join(p for p in parts if p)
            self.add(joined, ENTRY_SEQUENCE, source)
            return joined
        return ""

    def _walk_string(self, value: str, source: str, *, key: str = "") -> str:
        if _SLUG_KEY_RE.search(key or "") or _SLUG_NAME_RE.search((key or "").upper()):
            return ""  # a URL/slug is scaffolding, not a fact
        is_regex = bool(_REGEX_KEY_RE.search(key or "")) or _looks_like_regex(value)
        for number in _NUMBER_IN_TEXT_RE.findall(value):
            self.add(number, ENTRY_NUMBER, source)
        if is_regex:
            # A validator regex is where a wiki task keeps its answer tokens (``\bmead\b``).
            # Harvest the literal word runs, never the pattern itself.
            for run in _regex_literal_runs(value):
                if _fold(run) not in _REGEX_RUN_STOPWORDS:
                    self.add(run, ENTRY_NAME, f"{source} (regex literal)")
            return ""
        self.add(value, ENTRY_NAME, source)
        return value


def _render_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _looks_like_regex(text: str) -> bool:
    """Cheap heuristic: a string carrying escapes AND metacharacters is a pattern, not prose."""
    if text.startswith("(?"):
        return True
    return "\\" in text and sum(1 for ch in text if ch in _REGEX_META) >= 2


def _regex_literal_runs(pattern: str) -> List[str]:
    """The literal word runs inside a regex — the answer tokens a validator is built around.

    ``\\bmead\\b`` must yield ``mead``, NOT ``bmeadb``: the ``b`` of an escape class is syntax,
    so escape classes are dropped whole before anything is read as content. Escaped literals
    (``\\.``) are unescaped, and groups/char-classes become separators.
    """
    cleaned = re.sub(r"\\[A-Za-z]", " ", pattern)      # \b \s \d \w ... — syntax, not content
    cleaned = re.sub(r"\\(.)", r"\1", cleaned)          # \. \+ ... — a literal character
    cleaned = re.sub(r"\(\?[^)]*\)?", " ", cleaned)     # (?i) (?<!...) (?: openers
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)       # character classes
    return _WORD_RUN_RE.findall(cleaned)


# -- public entry points -------------------------------------------------------------------


def load_task_module(path: Path) -> Any:
    """Import a task file as a throwaway module (module-level constants only, no I/O)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"_leak_gate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise LeakGateError(f"could not load task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_task_file(
    task_id: str,
    *,
    tests_dir: Optional[Path] = None,
    code_tests_dir: Optional[Path] = None,
) -> Optional[Path]:
    """The task file whose id prefix is ``task_id`` (``"084"``, ``"c30"``), or ``None``."""
    wanted = str(task_id or "").strip().lower()
    if not wanted:
        return None
    for directory in (tests_dir or WIKI_TESTS_DIR, code_tests_dir or CODE_TESTS_DIR):
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("test_*.py")):
            if path.stem[len("test_"):].split("_")[0].lower() == wanted:
                return path
    return None


def build_ledger(source: Any, *, task_id: str = "") -> TaskLedger:
    """The literal ledger of one task, from a module / a path / a task id.

    Dispatches on the module's own shape rather than on which directory it came from: a module
    exposing ``get_grading_payload`` is a codebench task (its secret is a hidden pytest file), a
    module exposing ``get_validation_functions`` is a wiki-reasoning task (its secret is an
    answer/decoy table plus validator regexes).
    """
    module = source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            found = find_task_file(str(source))
            if found is None:
                raise LeakGateError(f"no task module for '{source}'")
            path = found
        module = load_task_module(path)

    resolved_id = task_id or _module_task_id(module)
    if hasattr(module, "get_grading_payload"):
        return _ledger_for_code_task(module, resolved_id)
    return _ledger_for_wiki_task(module, resolved_id)


def build_ledgers(task_ids: Sequence[str], **kwargs: Any) -> List[TaskLedger]:
    """Ledgers for several tasks; a task that cannot be located is skipped with a warning."""
    out: List[TaskLedger] = []
    for task_id in task_ids or ():
        try:
            out.append(build_ledger(str(task_id), **kwargs))
        except (LeakGateError, OSError, SyntaxError, ValueError) as exc:  # noqa: PERF203
            logger.warning("Strategy library: could not build a ledger for task %s: %s", task_id, exc)
    return out


def _module_task_id(module: Any) -> str:
    try:
        return str((module.get_test_metadata() or {}).get("test_id") or "")
    except Exception:  # noqa: BLE001 — a missing id only costs a label
        return ""


def _public_text(module: Any) -> str:
    """Everything the solver legitimately receives, concatenated."""
    parts: List[str] = []
    for name in ("get_task_statement", "get_required_deliverables"):
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except Exception:  # noqa: BLE001
            continue
        parts.append(value if isinstance(value, str) else " ".join(str(v) for v in value or ()))
    fixture = getattr(module, "get_sandbox_fixture", None)
    if callable(fixture):
        try:
            files = fixture() or {}
        except Exception:  # noqa: BLE001
            files = {}
        if isinstance(files, Mapping):
            # Fixture files are written into the agent's workdir — the agent reads them, so
            # anything in them is a given, not a secret.
            parts.extend(f"{k}\n{v}" for k, v in files.items())
    return "\n\n".join(p for p in parts if p)


# -- wiki-reasoning tasks -------------------------------------------------------------------


def _ledger_for_wiki_task(module: Any, task_id: str) -> TaskLedger:
    """Answer/decoy constants + entity tables + validator regexes + docstring figures."""
    public = _public_text(module)
    harvester = _Harvester(public, exclude=[task_id])

    for name, value in sorted(vars(module).items()):
        if name.startswith("__") or _SLUG_NAME_RE.search(name.upper()):
            continue
        if _GROUND_TRUTH_NAME_RE.search(name.upper()):
            harvester.walk_value(value, name, key=name)
            continue
        # The corpus-wide convention for the answer/decoy table is a module-level list of
        # mappings (``ENTITIES`` / ``BRANCHES`` / ``CANDIDATES`` / ...). Matched structurally
        # rather than by name so a new task naming it something else is still covered.
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, Mapping) for v in value):
            harvester.walk_value(value, name, key=name)
            for index, row in enumerate(value):
                _add_row_facts(harvester, row, f"{name}[{index}]")

    # Ground truth is often written out ONLY in the module docstring (chain tasks have no
    # entity table at all), so its figures are harvested too. Numbers only: prose spans from a
    # docstring are far too noisy to be ledger entries, and go to the auditor via ``facts``.
    doc = str(getattr(module, "__doc__", "") or "")
    for number in _NUMBER_IN_TEXT_RE.findall(doc):
        harvester.add(number, ENTRY_NUMBER, "module docstring")

    criteria = getattr(module, "get_success_criteria", None)
    if callable(criteria):
        try:
            for line in criteria() or ():
                harvester.add_fact(f"success criterion: {line}")
        except Exception:  # noqa: BLE001
            pass
    if doc.strip():
        harvester.add_fact(f"task {task_id} ground-truth notes: {_clip(doc, 2000)}")

    return TaskLedger(
        task_id=task_id, kind=KIND_WIKI, public_text=public,
        entries=harvester.entries(), facts=tuple(harvester.facts),
    )


def _add_row_facts(harvester: "_Harvester", row: Mapping[str, Any], source: str) -> None:
    """One entity row -> a readable fact line, and an explicit "this one is the answer"."""
    name = str(row.get("name") or row.get("project") or row.get("label") or row.get("key") or "").strip()
    if not name:
        return
    fields = ", ".join(
        f"{k}={v}" for k, v in row.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    if fields:
        harvester.add_fact(f"{name}: {fields}")
    if any(bool(row.get(k)) for k in _WINNER_KEYS):
        harvester.add_fact(f"the correct answer / winning entity of this task is: {name} ({source})")


# -- codebench tasks --------------------------------------------------------------------------


def _ledger_for_code_task(module: Any, task_id: str) -> TaskLedger:
    """The HIDDEN pytest file's literals + the keystone assertions that grade the task."""
    public = _public_text(module)
    harvester = _Harvester(public, exclude=[task_id])

    try:
        payload = module.get_grading_payload() or {}
    except Exception as exc:  # noqa: BLE001
        raise LeakGateError(f"task {task_id}: get_grading_payload() failed: {exc}") from exc

    tests = payload.get("tests") if isinstance(payload, Mapping) else None
    tests = tests if isinstance(tests, Mapping) else {}
    keystones = [str(t) for t in (payload.get("keystone_test_ids") or ())]

    for rel_path, content in tests.items():
        harvester.add(str(rel_path), ENTRY_NAME, "grading_payload.tests key")
        _harvest_python_source(harvester, str(content or ""), f"hidden test {rel_path}")
        for name in keystones:
            fn_name = name.split("::")[-1].strip()
            snippet = _function_source(str(content or ""), fn_name)
            if snippet:
                harvester.add_fact(f"hidden keystone test {fn_name} asserts: {_clip(snippet, 600)}")
    for name in keystones:
        harvester.add(name.split("::")[-1].strip(), ENTRY_NAME, "keystone_test_ids")

    entry = payload.get("entrypoint") if isinstance(payload, Mapping) else None
    if isinstance(entry, Mapping):
        # The module/function names are stated in the task itself; the public subtraction drops
        # them. Harvested anyway so a task that DOESN'T state them stays protected.
        harvester.walk_value(entry, "grading_payload.entrypoint")

    return TaskLedger(
        task_id=task_id, kind=KIND_CODE, public_text=public,
        entries=harvester.entries(), facts=tuple(harvester.facts),
    )


def _harvest_python_source(harvester: "_Harvester", source: str, origin: str) -> None:
    """Every constant and every sequence literal of a Python source string, via ``ast``.

    Source-level rather than value-level because the hidden tests arrive as TEXT, and because a
    test's input vectors are literally its ground truth: ``["a","b","c","d","e"]`` is the exact
    artifact c30's pre-fix plan handed away.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("Strategy library: could not parse %s: %s", origin, exc)
        return

    def _walk(node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                return ""
            if isinstance(value, (int, float)):
                harvester.add(_render_number(value), ENTRY_NUMBER, origin)
                return str(value)
            if isinstance(value, str):
                harvester.add(value, ENTRY_NAME, origin)
                return value
            return ""
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            joined = " ".join(p for p in (_walk(el) for el in node.elts) if p)
            harvester.add(joined, ENTRY_SEQUENCE, origin)
            return joined
        return ""

    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Set)):
            _walk(node)
        elif isinstance(node, ast.FunctionDef):
            harvester.add(node.name, ENTRY_NAME, f"{origin} (test name)")


def _function_source(source: str, name: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""


def _clip(text: str, limit: int) -> str:
    out = " ".join(str(text or "").split())
    return out if len(out) <= limit else out[:limit] + " ...[clipped]"


# --------------------------------------------------------------------------------------
# findings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One reason the gate is unhappy."""

    layer: str
    severity: str
    task_id: str
    detail: str
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer, "severity": self.severity, "task_id": self.task_id,
            "detail": self.detail, "evidence": self.evidence,
        }


@dataclass(frozen=True)
class GateVerdict:
    """The gate's answer. ``passed`` is False as soon as ANY finding is a rejection."""

    findings: Tuple[Finding, ...] = ()
    #: Whether layer (c) actually ran. A verdict with ``audited=False`` passed three layers,
    #: not four — recorded so a corpus can never claim an audit it never had.
    audited: bool = False

    @property
    def rejections(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == SEVERITY_REJECT)

    @property
    def warnings(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == SEVERITY_WARN)

    @property
    def passed(self) -> bool:
        return not self.rejections

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return f"clean ({len(self.warnings)} warning(s), audited={self.audited})"
        return "; ".join(f"[{f.layer}/{f.task_id}] {f.detail}" for f in self.rejections)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "audited": self.audited,
            "findings": [f.as_dict() for f in self.findings],
        }


def _merge(*verdicts: GateVerdict) -> GateVerdict:
    findings: List[Finding] = []
    audited = False
    for verdict in verdicts:
        findings.extend(verdict.findings)
        audited = audited or verdict.audited
    return GateVerdict(findings=tuple(findings), audited=audited)


# --------------------------------------------------------------------------------------
# (b) normalized substring overlap
# --------------------------------------------------------------------------------------


def overlap_findings(text: str, ledger: TaskLedger) -> List[Finding]:
    """Every ledger entry that appears in ``text``.

    PLAIN SUBSTRING over a normalized form, never ``\\b``-anchored: task 146's own leak test
    documents that ``\\bmead\\b`` misses ``res_mead``, so a word-boundary blocklist certifies a
    text as clean precisely when the leak has been glued to an identifier.
    """
    tight = normalize_tight(text)
    runs = normalize_runs(text)
    out: List[Finding] = []
    for entry in ledger.entries:
        if _contains(tight, runs, entry.value, entry.kind):
            out.append(Finding(
                layer=LAYER_OVERLAP,
                severity=SEVERITY_REJECT,
                task_id=ledger.task_id,
                detail=f"contains the ground-truth {entry.kind} {entry.raw!r} (from {entry.source})",
                evidence=entry.value,
            ))
    return out


# --------------------------------------------------------------------------------------
# (d) worked-example lint
# --------------------------------------------------------------------------------------


def worked_example_findings(text: str, ledgers: Sequence[TaskLedger] = ()) -> List[Finding]:
    """Flag example-shaped spans, codifying the c0dbc720 fix's own wording as a rule.

    A span counts as a worked example when one sentence carries BOTH an example cue (``->``,
    "returns", "e.g.", "worked example") and a concrete literal (a quoted string, a bracketed
    list, a multi-digit number). Then:

      * a span that also carries a ledger token is rejected, framing or not — "construct your
        own" does not un-leak a secret that is already printed next to it;
      * any other example span is rejected too, UNLESS the text frames its examples as
        "construct your own ... rather than reusing the worked examples" (the exact remedy
        c0dbc720 applied), in which case it is a warning.

    This is the weakest of the four layers and is documented as such: the framing exemption is
    trivially satisfiable by an author, which is why (b) and (c) are independent of it.
    """
    framed = bool(_OWN_CONSTRUCTION_RE.search(text or ""))
    out: List[Finding] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        if not _EXAMPLE_CUE_RE.search(sentence):
            continue
        literals = _EXAMPLE_LITERAL_RE.findall(sentence)
        if not literals:
            continue
        hit = _first_ledger_hit(sentence, ledgers)
        if hit is not None:
            entry, task_id = hit
            out.append(Finding(
                layer=LAYER_WORKED_EXAMPLE,
                severity=SEVERITY_REJECT,
                task_id=task_id,
                detail=(
                    f"worked example carrying the ground-truth {entry.kind} {entry.raw!r} "
                    f"(from {entry.source})"
                ),
                evidence=_clip(sentence, 240),
            ))
            continue
        out.append(Finding(
            layer=LAYER_WORKED_EXAMPLE,
            severity=SEVERITY_WARN if framed else SEVERITY_REJECT,
            task_id="",
            detail=(
                "worked example with concrete literals "
                + ("framed as 'construct your own'" if framed
                   else "and no 'construct your own' framing — state the METHOD and have the "
                        "solver build its own examples (see commit c0dbc720)")
            ),
            evidence=_clip(sentence, 240),
        ))
    return out


def _first_ledger_hit(
    span: str, ledgers: Sequence[TaskLedger]
) -> Optional[Tuple[LedgerEntry, str]]:
    tight = normalize_tight(span)
    runs = normalize_runs(span)
    for ledger in ledgers or ():
        for entry in ledger.entries:
            if _contains(tight, runs, entry.value, entry.kind):
                return entry, ledger.task_id
    return None


# --------------------------------------------------------------------------------------
# the deterministic gate (layers b + d)
# --------------------------------------------------------------------------------------


def check_text(text: str, ledgers: Sequence[TaskLedger] = ()) -> GateVerdict:
    """Layers (b) and (d) — no LLM, no I/O. This is what the READ-time gate runs.

    Read time deliberately stops here: a note already cleared all four layers when it was
    written, and spending an extra LLM call per run to re-audit it against the current task
    would tax every run for a check that has already happened once. What read time DOES add is
    the thing write time cannot know — this task's own ledger.
    """
    findings: List[Finding] = []
    for ledger in ledgers or ():
        findings.extend(overlap_findings(text, ledger))
    findings.extend(worked_example_findings(text, ledgers))
    return GateVerdict(findings=tuple(findings), audited=False)


# --------------------------------------------------------------------------------------
# (c) the LLM auditor
# --------------------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 700
DEFAULT_TIMEOUT_SECONDS = 90.0
MAX_FACTS = 40
MAX_FACT_CHARS = 700

_AUDIT_SCHEMA_NAME = "leak_audit"

_AUDIT_SYSTEM = (
    "You are a LEAK AUDITOR for a benchmark. You are given a numbered list of PROTECTED FACTS "
    "(the hidden ground truth of one task) and a CANDIDATE TEXT that is about to be shown to a "
    "model that must solve that task WITHOUT being told the answer.\n"
    "Answer ONE question: does the candidate text state, imply, or make directly derivable any "
    "of the protected facts?\n"
    "Rules:\n"
    "- A text that describes a METHOD, a procedure, or a discipline that would apply equally to "
    "DIFFERENT values is NOT a leak, even if it talks about the same kind of quantity.\n"
    "- A paraphrase IS a leak. Restating a hidden value in words ('the common lines are a, c and "
    "d -- three of them'), naming the winner, or giving a worked example whose inputs and result "
    "match a hidden case all count, even with no literal copied.\n"
    "- Being merely ON TOPIC is not a leak. Do not flag generic advice.\n"
    "- When unsure, flag it: a false alarm costs one rejected note, a miss corrupts the "
    "benchmark.\n"
    "Respond with a single JSON object and nothing else."
)


def _audit_schema() -> Dict[str, Any]:
    return {
        "name": _AUDIT_SCHEMA_NAME,
        "schema": {
            "type": "object",
            "properties": {
                "leaks": {
                    "type": "boolean",
                    "description": "true if the candidate text states, implies or makes "
                                   "derivable ANY protected fact.",
                },
                "fact_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "The numbers of the protected facts that leak (empty if none).",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence: which phrase leaks which fact, or why the "
                                   "text is only describing a method.",
                },
            },
            "required": ["leaks"],
            "additionalProperties": False,
        },
    }


def _audit_prompt(text: str, facts: Sequence[str]) -> str:
    lines = ["PROTECTED FACTS (the hidden ground truth — never to be revealed):"]
    lines.extend(f"{i}. {_clip(fact, MAX_FACT_CHARS)}" for i, fact in enumerate(facts, 1))
    lines.append("")
    lines.append("CANDIDATE TEXT:")
    lines.append(_clip(text, 4000))
    lines.append("")
    lines.append(
        "Does the candidate text state, imply, or make directly derivable any protected fact?"
    )
    return "\n".join(lines)


async def llm_audit(
    text: str,
    ledgers: Sequence[TaskLedger],
    io: Any = None,
    *,
    model_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
) -> GateVerdict:
    """Layer (c): ONE batched, temperature-0 JSON-mode call over every ledger's facts.

    Batched exactly like ``plan_library.slot_fill.llm_slot_values``: all facts of all ledgers
    ride in one request, numbered, so the answer can point at which one leaked.

    Returns ``GateVerdict(audited=False)`` when there is no LLM to ask or the call fails — the
    CALLER decides what that means. :func:`audit_text` (write time) treats an unavailable
    auditor as a hard failure, because an unaudited entry must not enter the corpus; read time
    never calls this at all.
    """
    facts: List[Tuple[str, str]] = []  # (task_id, fact)
    for ledger in ledgers or ():
        for fact in ledger.facts:
            facts.append((ledger.task_id, fact))
    facts = facts[:MAX_FACTS]
    if not facts:
        return GateVerdict(audited=False)
    if io is None or not hasattr(io, "query_llm_with_fallback"):
        logger.debug("Strategy library: no LLM available for the leak auditor")
        return GateVerdict(audited=False)

    # Imported here (not at module scope) so the ledger builder and the deterministic layers
    # stay importable with no edge into the LLM stack — the same reason ``retrieval.py`` defers
    # its ``slot_fill`` import.
    from agent.app.idea_policies.action_constants import PromptBuilder
    from agent.app.llm_backends import json_instruction_from_response_format

    system = _AUDIT_SYSTEM
    hint = json_instruction_from_response_format(
        {"type": "json_schema", "json_schema": _audit_schema()}
    )
    if hint:
        system = f"{system}\n\n{hint}"
    messages = PromptBuilder.build_messages(
        system_content=system,
        user_content=_audit_prompt(text, [f for _, f in facts]),
    )
    try:
        payload = io.build_llm_payload(
            messages=messages, json_mode=True, model_name=model_name,
            temperature=temperature, max_tokens=max_tokens, json_schema=None,
        )
        content = await io.query_llm_with_fallback(
            payload, model_name=model_name, fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 — an unreachable auditor is "not audited"
        logger.warning("Strategy library: leak auditor call failed: %s", exc)
        return GateVerdict(audited=False)

    data = _parse_json_object(content)
    if data is None:
        logger.info("Strategy library: leak auditor returned unusable JSON (%d chars)",
                    len(content or ""))
        return GateVerdict(audited=False)

    if not bool(data.get("leaks")):
        return GateVerdict(audited=True)
    reason = str(data.get("reason") or "").strip() or "auditor flagged a leak"
    numbers = [n for n in (data.get("fact_numbers") or []) if isinstance(n, int)]
    task_ids = sorted({facts[n - 1][0] for n in numbers if 1 <= n <= len(facts)})
    return GateVerdict(
        findings=(Finding(
            layer=LAYER_LLM_AUDIT,
            severity=SEVERITY_REJECT,
            task_id=",".join(task_ids),
            detail=reason,
            evidence="; ".join(_clip(facts[n - 1][1], 160) for n in numbers if 1 <= n <= len(facts)),
        ),),
        audited=True,
    )


def _parse_json_object(content: Any) -> Optional[Dict[str, Any]]:
    """``json.loads`` with the same repair fallback ``slot_fill`` uses for weak-model JSON."""
    import json

    if not isinstance(content, str) or not content.strip():
        return None
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if isinstance(data, dict):
        return data
    try:
        from agent.app.idea_policies.expansion import _repair_json_object
    except Exception:  # noqa: BLE001 - pragma: no cover
        return None
    repaired = _repair_json_object(content)
    return repaired if isinstance(repaired, dict) else None


# --------------------------------------------------------------------------------------
# the write-time gate (all four layers)
# --------------------------------------------------------------------------------------


async def audit_text(
    text: str,
    ledgers: Sequence[TaskLedger],
    io: Any = None,
    *,
    require_llm_audit: bool = True,
    **audit_kwargs: Any,
) -> GateVerdict:
    """All four layers. This is what runs before anything is written to the corpus.

    ``require_llm_audit`` (default True) turns "the auditor could not be reached" into a
    rejection rather than a silent three-layer pass: a corpus whose entries claim an audit they
    never had is exactly the "we promise it's generalized" position this gate exists to
    replace. Pass ``False`` only for an offline dry run, and expect ``audited=False`` on the
    verdict it returns.
    """
    verdict = check_text(text, ledgers)
    audit = await llm_audit(text, ledgers, io, **audit_kwargs)
    merged = _merge(verdict, audit)
    if require_llm_audit and not merged.audited:
        merged = _merge(merged, GateVerdict(findings=(Finding(
            layer=LAYER_LLM_AUDIT,
            severity=SEVERITY_REJECT,
            task_id="",
            detail="the LLM leak auditor could not be run (no LLM, no facts, or the call "
                   "failed) and require_llm_audit is on",
        ),)))
    return merged
