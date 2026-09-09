# Ledger run visualization

Three surfaces over one projection. Everything here reads a stored per-cell result JSON from
`agent/idea_test_results/` and invents nothing: if a fact is on a slide, the run recorded it.

```
stored cell .json
      │
      ├─ ledger_trace.project_cell()   typed, parented, timed control plane
      ├─ output.evidence_graph         pages + text, source spans, derivations, refusals
      ├─ output.host_derive            per-slot operand attribution + ranker scores
      ├─ output.answer_audit           every number in the answer, backed / derived / unbacked
      └─ output.confidence(_basis)     the verdict, as arithmetic
                    │
      agent/app/testing/ledger_story.build()  ->  [Beat, Beat, Beat, ...]
                    │
     ┌──────────────┼──────────────────┐
  HTML deck     PNG frames          4 charts
```

## The storyboard — `agent/app/testing/ledger_story.py`

The only module with interpretation logic. A `Beat` answers the same four questions every time,
so the renderers hold layout and nothing else:

| field | question | example |
|---|---|---|
| `title` | what is being done | `LOCATE` |
| `subject` | the text being handled | `508.2` |
| `origin` | where it came from | `p3 · en.wikipedia.org/Taipei_101 · chars 1074-1081` |
| `target` | where it goes | `E1` |

plus `ledger_after`, a cumulative snapshot of the evidence rail, and `trace_ref` back to the
`TraceNode` it came from. Kinds, in narrative order: `question search visit locate mint rank
derive refuse audit verdict`. Statuses are `ledger_trace`'s, not a second vocabulary.

Three rules are enforced in the projection rather than left to the renderers:

- **The tri-state survives.** `quote_verified is None` ("never checked") is not folded into
  `False` ("checked, and absent"). Same for `derivation_valid` and `operand_supported`.
- **Absent is never zero.** A beat with no interval carries no duration key at all.
- **Refusals are beats.** The Ledger declining to assert something is the behaviour the whole
  subsystem exists to produce, and it is the one thing a demo must not quietly drop.

Because it is a projection of what a run already recorded, it works on every cell already on
disk, including cells written before it existed. `agent/tests/ledger_story_test.py` exercises it
against 120 real stored cells as well as fixtures.

## 1. The deck — `scripts/render_ledger_deck.py`

```
PYTHONPATH=.:services:agent python3 scripts/render_ledger_deck.py <cell>.json -o deck.html
```

One self-contained HTML file: no CDN, no build step, openable from disk. Arrow keys or click to
step, space to play/pause. Autoplay dwell varies by beat kind, so a twelve-span mint run moves
quickly and the refusal and the verdict hold.

Every frame uses the same grammar — the act on the left, the accumulating ledger on the right,
the whole run as a clickable ribbon along the bottom — so the eye learns the layout once and
then only reads what changed. Colours come from `plot_style.dark_tokens()`.

`window.gotoBeat(i)` is exposed for headless screenshotting.

## 2. The frames — `scripts/render_ledger_frames.py`

```
PYTHONPATH=.:services:agent python3 scripts/render_ledger_frames.py <cell>.json -o frames/
```

Numbered PNGs, one per beat, plus `contact_sheet.png` and `storyboard.json`.

These are screenshots of the deck itself, driven headlessly. Reimplementing the frame grammar a
second time in matplotlib would be a second layout engine to keep in step with the first, and
every later change to the deck would silently stop being true of the exported frames.

## 3. The charts — `scripts/render_ledger_charts.py`

```
PYTHONPATH=.:services:agent python3 scripts/render_ledger_charts.py \
    --cell <cell>.json --prefix mint04 -o gallery/
```

Square PNGs on the frozen light paper surface, for docs, a README or a post.

| file | what it shows | scope |
|---|---|---|
| `risk_coverage.png` | risk against coverage as the certify threshold moves | campaign (`--prefix`) |
| `provenance_flow.png` | printed figure → checkable figure, loss drawn as width | one run |
| `run_timeline.png` | every step on a time axis, evidence plane banded off the clock | one run |
| `answer_audit.png` | every number in the answer, ranked by how uniquely it is backed | one run |

Three decisions in these figures are load-bearing and should not be "tidied" away:

- **The accuracy anchor is printed on the risk-coverage figure**, and cells with no signal are
  reported as UNKNOWN rather than imputed as zero. Both are frozen KPI reading rules
  (`docs/LEDGER_KPI_SPEC.md`), not stylistic choices.
- **The provenance funnel counts one thing at every stage** — a number printed in the final
  answer. An earlier version began with "pages fetched", which made the chart *widen* from 5 to
  26 and invited a comparison between a page and a span. The evidence base is stated in the
  caption instead, where it is context rather than a stage.
- **The timeline's bottom band has no time meaning.** Derivations and refusals cost no
  measurable time, so pinning them to `t=0` would place the run's most consequential steps at
  the origin and imply they happened first.

## Palette

`agent/app/testing/plot_style.py` holds both grounds: the frozen light paper surface for the
charts, and `dark_tokens()` for the deck. Both draw accents from the same magma family, so the
two surfaces read as one system. A new surface reads its colours from that module or it is not a
house surface.

Note that `dag_visualizer.py` still carries its own `_WAVE_FILLS` list, predating this. That is
known drift, left alone deliberately rather than repointed as a drive-by change.

## Regenerating the demo

Any stored cell works; `--cell` is the whole interface. The reference run used throughout this
document is task 221 on `qwen2.5:7b` (10 pages, 26 located spans, 11 derivations, one typed
refusal, PARTIAL verdict), which exercises every narrative stage including the ones that fail.
