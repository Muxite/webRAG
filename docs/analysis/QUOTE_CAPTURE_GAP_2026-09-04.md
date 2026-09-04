# Root cause: `quote_verified` null on ~100% of ladder03 SOURCE nodes (2026-09-04)

Diagnosis only — no code changed. $0, read-only, offline. Triggered by
`ladder03_risk_coverage_final.json` (scratchpad): the full certify signal is
starved to 0% coverage solely by the quote clause; minus-quote it certifies
at 14.3% vs a 61% base. Confirmed against a live cell
(`ladder03_qwen2_5_14b_derive_sq_210_..._r1.json`): every SOURCE node has
`"quote": "", "quote_verified": None, "quote_fail_reason": "empty"`,
`start`/`end` populated (the VALUE's span is captured; the quote is not).

## (1) Why is the field null: host never passes it, model never asked, or verification fails?

**The host never passes it.** This is a pure wiring gap, not a verification
failure and not (functionally) a model-elicitation failure.

Trace:

- `agent/app/testing/evidence_graph.py::EvidenceGraph.add_source()` (line
  1294) takes an optional `quote: str = ""` param. When given, it is
  independently re-checked against the frozen page via
  `verify_against_stored_page()` (imported from
  `execution_evidence_loop.py:579`), which calls `verify_quote()` (line 517)
  — an exact-substring check (verbatim, then whitespace-collapsed, then
  quote-wrapper-stripped). An empty/absent quote hits
  `strip_quote_wrapper(quote)` being falsy and returns
  `QuoteMatch(None, -1, -1, QUOTE_FAIL_EMPTY)` immediately — this is exactly
  the `"quote_fail_reason": "empty"` seen on every node. **The verification
  logic itself never runs and never fails on real input; it never receives
  any input.**
- `agent/app/ledger_tools.py::LedgerToolkit._locate()` is what actually
  mints SOURCE nodes for the `sequential_react` + `derive` arm (confirmed:
  `ladder03`'s `*_derive_sq_*` cells go through `LedgerToolkit`, imported at
  `execution_sequential.py:22`). It calls `self._graph.add_source(...)` at
  **two** call sites — line 307 (the `q<N>`-id-resolved path) and line 322
  (the page-scan literal path) — and **neither passes `quote=`**. Both rely
  on the `quote: str = ""` default. This is the entire mechanism: `add_source`
  supports quotes, `LedgerToolkit` simply never supplies one.
- The `derive` tool's own signature
  (`LedgerToolkit.derive(operation, operands, proposed_value=None)`,
  `ledger_tools.py:203`) has **no quote parameter at all** — the model isn't
  even offered a slot to supply one through this tool call.
- The host prompt in `execution_sequential.py` (`_forced_synthesis`-style
  block around line 388-390) *does* tell the model to "quote the exact value
  from the page and cite the source URL" — but that instruction is scoped
  to the free-text `final_deliverable` answer, not to any ledger tool
  argument. It has no path back into `EvidenceGraph`/`LedgerToolkit` at all.
  So even the one place a quote is "asked for" is structurally disconnected
  from node minting.
- Contrast: `agent/app/testing/execution_evidence_loop.py` (the separate
  `evidence_loop` host variant, not what `ladder03` sequential_react uses)
  *does* pass `quote=quote` into `graph.add_source(...)` at line 1339 —
  proof the plumbing works end-to-end when a caller supplies it. The gap is
  specific to `LedgerToolkit`/`ledger_tools.py`, not to `add_source` or the
  verification function.

Net: not a model failure (model is barely even asked, through this tool),
not a verifier failure (verifier never gets input to check) — a **host
wiring omission** in exactly two call sites of one file.

## (2) Minimal change to populate verified quotes at row-minting time

**Mechanical span capture, not asking the model — strongly preferred and
directly available**, because `LedgerToolkit._locate()` already computes
everything needed:

- The page-scan path (`ledger_tools.py:315-325`) already has `page_text`
  and `match.start`/`match.end` (the exact located span of the *value*) in
  scope from `verify_value(page_text, candidate, unit=candidate_unit)`.
  Passing `quote=page_text[match.start:match.end]` into the `add_source(...)`
  call at line 322 is a same-string slice of text already in hand — zero
  new page reads, zero new model interaction, and it is **guaranteed to
  verify**: `verify_quote()`'s first check is `page_text.find(quote) >= 0`
  on the verbatim string, and a slice of `page_text` trivially satisfies
  that (`start` a real offset into that same text).
- The id-resolved path (`ledger_tools.py:304-309`) uses a
  `QuantityRef` from `quantity_index.py`, which **already carries
  `start`/`end`** (`quantity_index.py:85-86`, "start offset of value in the
  RAW page text"). The page text itself is retrievable via
  `self._graph.page(page_id)["text"]` (already used elsewhere in this
  class). So `quote=page_text[entry.start:entry.end]` is equally available
  with one extra dict lookup.
- `EvidenceGraph.add_source()` needs **no changes** — it already accepts
  `quote` and already independently re-verifies it; this is exactly the
  "attachment surface, host-neutral" contract the module's docstring
  describes working as designed. The gap is purely on the caller side.

**One design nuance worth flagging for the next phase, not blocking**: if
the quote is set to the *identical* span already used to verify the value
(`page_text[start:end]`), `quote_verified` becomes tautologically `True`
for every node that clears value-location at all — it stops being an
independent citation-fidelity check (its original purpose per the
`add_source` docstring: "a weak model's bad citation, not a bad fact") and
becomes a restatement of "the value was found." That still satisfies the
literal KPI ask (verified quotes populated, `quote_fail_reason` no longer
universally `"empty"`), and it's still deterministic/host-side per the
project's stated preference. A slightly richer mechanical option — expand
the span to the containing **line** (page text already has line-boundary
scanning available via helpers in `quantity_index.py` around line
174, `_line_spans`-style logic used to build `QuantityRef`s) rather than
the bare value span — would give the quote field genuine context (a
checkable sentence, not just the digits) while remaining 100% mechanical
and still guaranteed to verify (still an exact substring of the stored
page text). Recommend the next phase pick the line-window variant over the
bare-span variant for this reason, but either is a legitimate "minimal
change."

## (3) Blast radius

Small and tightly scoped:

- **`agent/app/ledger_tools.py`** — the only file needing an actual code
  change. Two call sites inside `LedgerToolkit._locate()`:
  - line 307: `self._graph.add_source(page_id, entry.value, unit=entry.unit or None)`
    → add `quote=` sourced from the resolved `QuantityRef`'s
    `start`/`end` plus that page's stored text.
  - line 322: `self._graph.add_source(page["page_id"], candidate, unit=span_unit or None)`
    → add `quote=` sourced from `match.start`/`match.end` and the
    already-in-scope `page_text`.
  - If the line-window variant is chosen instead of the bare-span variant,
    a small helper (either reused from `quantity_index.py`'s line-splitting
    logic or a short local function) is needed to expand an offset pair to
    its containing line/sentence boundary within `page_text`.
- **`agent/app/testing/evidence_graph.py`** — no change needed;
  `add_source()`'s `quote` parameter and `verify_against_stored_page()` are
  already correct and already exercised by `execution_evidence_loop.py`.
- **`agent/app/quantity_index.py`** — no change needed if the bare-span
  variant is used (`QuantityRef.start`/`.end` already exist). Only touched
  if the line-window variant is chosen and no existing line-boundary helper
  is reused directly.
- **`agent/app/testing/execution_sequential.py`** — no change needed. It
  never touches quotes directly; it only imports and calls
  `LedgerToolkit.register_page()` / `.derive()`, both of which are
  untouched by this fix (the quote capture happens entirely inside
  `_locate()`, one layer below what the host calls).
- **Host prompt text** (the "quote the exact value... and cite the source
  URL" instruction in `execution_sequential.py` around line 388-390) is
  **not** part of the required fix and can be left as-is; it's a
  freestanding instruction for the free-text answer, orthogonal to node
  minting. Could optionally be reworded later to stop implying the model's
  own quote is what gets checked, but that's cleanup, not part of the
  blast radius for closing the KPI gap.
- **Cross-check**: `agent/app/testing/execution_langgraph.py` also imports
  and binds `LedgerToolkit` (not just `execution_sequential.py`), so this
  same defect and the same one-file fix apply to the `langgraph_react`
  derive arm as well, not only `sequential_react` — the fix in
  `ledger_tools.py` covers both hosts at once since they share the same
  `LedgerToolkit._locate()` code path. Also checked
  `agent/tests/ledger_tools_test.py`,
  `agent/tests/sequential_ledger_module_test.py`, and
  `agent/tests/langgraph_ledger_module_test.py` for existing assertions on
  `quote`/`quote_verified`/`quote_fail_reason` — none exist, so no test
  updates are anticipated as part of this fix, only new test coverage to
  add.
- **Estimated size**: ~1 file materially changed (`ledger_tools.py`), ~2
  call sites, roughly 5-15 new lines total for the bare-span variant (a
  few more for the line-window variant plus its helper). No schema changes
  to `EvidenceNode`/`Extraction`/the JSON artifact shape — `quote`,
  `quote_verified`, `quote_fail_reason` are all pre-existing fields that
  simply start getting non-default values. Existing offline tests for
  `ledger_tools.py`/`evidence_graph.py` should mostly continue to pass
  unchanged since they assert on `value`/`unit`/`derivation_valid`
  behavior, not on `quote` being empty — but any test that explicitly
  asserts `quote_verified is None` / `quote == ""` on a `LedgerToolkit`-minted
  node would need updating, and is worth a targeted grep before
  implementing.
