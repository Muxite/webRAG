#!/usr/bin/env python3
"""render_ledger_deck.py: draw one run of the Euglena Ledger as a step-through deck.

Takes a stored per-cell result JSON (anything in ``agent/idea_test_results/*.json``), projects it
with :mod:`agent.app.testing.ledger_story`, and writes ONE self-contained HTML file: no network,
no CDN, no build step, openable from disk and publishable as-is.

    PYTHONPATH=.:services:agent python3 scripts/render_ledger_deck.py <cell>.json -o deck.html

Each frame answers the same four questions in the same four places -- what is being done, the
text being handled, where it came from, where it goes -- so the eye learns the layout once and
then only reads what changed. The right-hand rail accumulates, which is what makes it a story
rather than a slideshow: the viewer watches the evidence get built.

Colours come from :mod:`agent.app.testing.plot_style` (the dark tokens), so this deck and the
static gallery are recognisably one system rather than two products.

Controls: arrow keys or click to step, space to play/pause, Home/End to jump. ``window.gotoBeat(i)``
is exposed for headless screenshotting.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_ROOT, os.path.join(_ROOT, "services"), os.path.join(_ROOT, "agent")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.app.testing import ledger_story  # noqa: E402
from agent.app.testing import plot_style  # noqa: E402

#: Autoplay dwell per beat kind, in milliseconds. Repetitive beats (a page yielding twelve spans)
#: move fast; the ones a viewer has to actually read hold. A single global interval would either
#: rush the verdict or make the mint run tedious.
DWELL_MS = {
    "question": 5200, "search": 1500, "visit": 2200, "locate": 2400, "mint": 1100,
    "rank": 2200, "derive": 2200, "refuse": 6000, "audit": 6000, "verdict": 8000,
}

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
__TOKENS__
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
html,body{height:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
     font-variant-numeric:tabular-nums;
     font-size:15px;line-height:1.5;overflow:hidden;-webkit-font-smoothing:antialiased}
#deck{display:flex;flex-direction:column;height:100vh;padding:18px 20px 12px;gap:12px}

/* ---- header: identity + position. Constant across every frame. ---- */
header{display:flex;align-items:baseline;gap:16px;flex:0 0 auto;
       border-bottom:1px solid var(--hairline);padding-bottom:11px}
.counter{font-family:var(--mono);font-size:13px;color:var(--ink-muted);letter-spacing:.08em}
.kind{font-family:var(--mono);font-size:20px;font-weight:600;letter-spacing:.16em;
      color:var(--accent-hot)}
.kind[data-status="refused"],.kind[data-status="invalid"]{color:var(--serious)}
.kind[data-status="error"]{color:var(--critical)}
.kind[data-status="unknown"]{color:var(--warning)}
.ident{margin-left:auto;font-family:var(--mono);font-size:12.5px;color:var(--ink-muted);
       text-align:right;letter-spacing:.03em}
.ident b{color:var(--ink-secondary);font-weight:500}

/* ---- body: the act (left) and the accumulating ledger (right) ---- */
main{display:grid;grid-template-columns:minmax(0,1fr) 384px;gap:16px;flex:1 1 auto;min-height:0}
.stage{background:var(--panel);border:1px solid var(--hairline);border-radius:10px;
       padding:22px 24px;min-height:0;overflow:auto;display:flex;flex-direction:column;gap:14px}
.stage::-webkit-scrollbar,.rail-list::-webkit-scrollbar{width:8px}
.stage::-webkit-scrollbar-thumb,.rail-list::-webkit-scrollbar-thumb{background:var(--hairline);border-radius:4px}

.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-family:var(--mono);
      font-size:12.5px;color:var(--ink-muted);flex:0 0 auto}
.flow .from{color:var(--ink-secondary);overflow-wrap:anywhere}
.flow .arrow{color:var(--accent-cool);letter-spacing:.1em}
.flow .to{color:var(--accent-hot);font-weight:600}
.flow .to.neutral{color:var(--ink-secondary);font-weight:500}

.headline{font-size:29px;line-height:1.28;font-weight:600;color:var(--ink);word-break:break-word}
.headline.mono{font-family:var(--mono);font-size:33px;letter-spacing:-.01em}
.mandate{font-size:19px;line-height:1.6;color:var(--ink-secondary);white-space:pre-wrap;
         max-height:100%;overflow:auto}

/* ---- the locate window: the one place page text is shown, because it IS the beat ---- */
.window{font-family:var(--mono);font-size:16px;line-height:2.0;background:var(--ground);
        border:1px solid var(--hairline);border-radius:8px;padding:18px 20px;
        color:var(--ink-muted);word-break:break-word}
.window .nl{color:var(--accent-cool);opacity:.75;padding:0 4px}
.window .span{background:var(--accent-hot);color:#140b00;font-weight:700;padding:2px 5px;
              border-radius:3px;box-shadow:0 0 0 4px rgba(252,165,10,.16)}
.window .ell{color:var(--hairline)}
.offsets{font-family:var(--mono);font-size:12px;color:var(--ink-muted);letter-spacing:.06em}

/* ---- chips: badges, evidence numbers, meta ---- */
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{font-family:var(--mono);font-size:12px;padding:3px 9px;border-radius:20px;
      border:1px solid var(--hairline);color:var(--ink-secondary);background:var(--panel-hi);
      white-space:nowrap}
.chip.ok{border-color:var(--ok);color:var(--ok)}
.chip.warn{border-color:var(--warning);color:var(--warning)}
.chip.bad{border-color:var(--critical);color:var(--critical)}
.chip.hot{border-color:var(--accent-hot);color:var(--accent-hot)}
.chip.mid{border-color:var(--accent-mid);color:#e688b8}
.chip.dash{border-style:dashed;border-color:var(--warning);color:var(--warning)}

/* ---- formula: a derivation drawn as what it is ---- */
.formula{display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-family:var(--mono);
         font-size:26px;color:var(--ink)}
.formula .op{color:var(--accent-mid);font-size:19px;letter-spacing:.1em}
.formula .operand{background:var(--panel-hi);border:1px solid var(--accent-mid);
                  border-radius:6px;padding:5px 12px;font-size:22px;color:var(--ink);
                  display:inline-flex;flex-direction:column;align-items:center;line-height:1.25}
.formula .operand em{font-style:normal;font-size:13px;color:var(--accent-mid);letter-spacing:.02em}
.formula .eq{color:var(--ink-muted)}
.formula .result{background:var(--accent-hot);color:#140b00;border-radius:6px;
                 padding:5px 13px;font-weight:700}

/* ---- score bar: the ranker's decision against its floor ---- */
.scorebar{position:relative;height:26px;background:var(--ground);border:1px solid var(--hairline);
          border-radius:5px;overflow:hidden}
.scorebar .fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent-mid)}
.scorebar .fill.pass{background:var(--ok)}
.scorebar .floor{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--accent-hot)}
.scorelabels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11.5px;
             color:var(--ink-muted)}

/* ---- audit grid ---- */
.numgrid{display:flex;flex-wrap:wrap;gap:6px}
.num{font-family:var(--mono);font-size:14px;padding:5px 10px;border-radius:5px;
     border:1px solid var(--hairline);background:var(--panel-hi)}
.num.backed{border-color:var(--ok);color:var(--ok)}
.num.derived{border-color:var(--accent-mid);color:#e688b8}
.num.unbacked{border-color:var(--critical);color:var(--critical)}
.num.ambig{border-style:dashed}
.num.ambig sub{color:var(--warning);font-weight:700}
.num sub{font-size:10px;opacity:.8;margin-left:4px}

.note{font-size:14.5px;color:var(--ink-secondary);line-height:1.6}
.note.crit{color:var(--serious)}
.answer{background:var(--ground);border:1px solid var(--hairline);border-radius:8px;
        padding:14px 18px;font-size:14px;line-height:1.65;color:var(--ink-secondary);
        white-space:pre-wrap;max-height:38vh;overflow:auto}
.answer h3{margin:0 0 8px;font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;
           text-transform:uppercase;color:var(--ink-muted);font-weight:600}
.verdictbig{font-family:var(--mono);font-size:64px;font-weight:700;letter-spacing:.06em}
.verdictbig.ANSWER{color:var(--ok)} .verdictbig.PARTIAL{color:var(--warning)}
.verdictbig.ABSTAIN{color:var(--serious)}

/* ---- right rail: the ledger, accumulating ---- */
.rail{background:var(--panel);border:1px solid var(--hairline);border-radius:10px;
      display:flex;flex-direction:column;min-height:0}
.rail h2{margin:0;padding:13px 16px 10px;font-family:var(--mono);font-size:11.5px;
         font-weight:600;letter-spacing:.18em;color:var(--ink-muted);
         border-bottom:1px solid var(--hairline)}
.rail-counts{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--hairline);
             border-bottom:1px solid var(--hairline)}
.rail-counts div{background:var(--panel);padding:9px 6px;text-align:center}
.rail-counts .n{font-family:var(--mono);font-size:19px;color:var(--ink);display:block;line-height:1.15}
.rail-counts .l{font-size:9.5px;letter-spacing:.1em;color:var(--ink-muted);text-transform:uppercase}
.rail-counts .n.hot{color:var(--accent-hot)}
.rail-list{flex:1 1 auto;overflow:auto;padding:8px 8px 28px}
.h-row{display:grid;grid-template-columns:34px 1fr;gap:9px;padding:6px 8px;border-radius:6px;
       align-items:baseline;border:1px solid transparent}
.h-row.active{background:var(--panel-hi);border-color:var(--accent-hot)}
.h-row.operand{background:var(--panel-hi);border-color:var(--accent-mid)}
.h-row .ref{font-family:var(--mono);font-size:12.5px;font-weight:700;color:var(--accent-mid)}
.h-row.derived .ref{color:var(--accent-hot)}
.h-row .val{font-family:var(--mono);font-size:13px;color:var(--ink);word-break:break-word}
.h-row .org{font-size:11.5px;color:var(--ink-secondary);overflow-wrap:anywhere;display:block;margin-top:3px;line-height:1.45}
.h-row .tick{font-size:11px}
.rail-empty{padding:22px 14px;font-size:13px;color:var(--ink-muted);text-align:center}

/* ---- bottom ribbon: the whole run, current beat lit ---- */
.ribbon{flex:0 0 auto;display:flex;gap:2px;height:30px;align-items:stretch}
.rib{flex:1 1 0;min-width:2px;background:var(--hairline);border-radius:2px;cursor:pointer;
     position:relative;transition:transform .12s ease}
.rib:hover{transform:scaleY(1.25)}
.rib.on{background:var(--accent-hot);box-shadow:0 0 12px rgba(252,165,10,.55)}
.rib.past{opacity:.95}
.rib.future{opacity:.30}
.legend{flex:0 0 auto;display:flex;gap:14px;font-family:var(--mono);font-size:10.5px;
        color:var(--ink-muted);letter-spacing:.06em;align-items:center;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
          vertical-align:-1px;font-style:normal}
.hint{margin-left:auto;color:var(--ink-muted)}
:focus-visible{outline:2px solid var(--accent-hot);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
kbd{font-family:var(--mono);background:var(--panel-hi);border:1px solid var(--hairline);
    border-radius:3px;padding:1px 5px;font-size:10px;color:var(--ink-secondary)}
"""

JS = r"""
const S = window.__STORY__;
const B = S.beats;
let i = 0, playing = false, timer = null;

const el = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// One frame grammar: every beat kind fills the same three slots, and only the STAGE body
// changes. That constancy is what lets a viewer read the deck instead of re-orienting.
function stageFor(b) {
  const d = b.detail || {};
  const parts = [];
  switch (b.kind) {
    case 'question':
      parts.push(`<div class="mandate">${esc(b.subject)}</div>`);
      break;
    case 'search':
      parts.push(`<div class="headline mono">${esc(b.subject)}</div>`);
      parts.push(chips([[b.target, ''], ['result titles are not kept in the trace', '']]));
      break;
    case 'visit':
      parts.push(`<div class="headline mono">${esc(b.subject)}</div>`);
      parts.push(chips([
        d.http_status ? ['HTTP ' + d.http_status, d.http_status === 200 ? 'ok' : 'bad'] : null,
        d.chars ? [d.chars.toLocaleString() + ' chars stored', ''] : null,
        d.truncated ? ['truncated', 'warn'] : null,
        d.prefetched ? ['host prefetch — not a model visit', 'warn'] : null,
      ]));
      break;
    case 'locate': {
      const w = d.window || {};
      // Stored page text is a flattened infobox: long runs of one-token lines. Rendered with
      // pre-wrap it becomes forty lines of vertical noise and pushes the highlighted span off
      // screen entirely. Line breaks are shown as a marker instead, so the structure is still
      // visible but the sentence reads across. The offsets printed below are the raw ones.
      const flat = t => esc(t).replace(/\s*\n\s*/g, '<span class="nl">\u23ce</span>')
                              .replace(/[ \t]{2,}/g, ' ');
      parts.push(`<div class="window">${w.truncated_left ? '<span class="ell">… </span>' : ''}${
        flat(w.before)}<span class="span">${esc(w.span)}</span>${flat(w.after)}${
        w.truncated_right ? '<span class="ell"> …</span>' : ''}</div>`);
      parts.push(`<div class="offsets">characters ${w.start}–${w.end} of the stored page` +
        ` &nbsp;·&nbsp; <span class="nl">⏎</span> marks a line break in the stored text</div>`);
      parts.push(chips([
        d.unit_bearing ? ['carries its unit', 'ok'] : ['bare number', 'warn'],
        d.occurrences > 1 ? [d.occurrences + ' occurrences on this page', 'warn'] : null,
      ]));
      break;
    }
    case 'mint':
      parts.push(`<div class="headline mono">${esc(b.subject)}</div>`);
      parts.push(chips([
        d.verified === true ? ['value located on the page', 'ok'] :
          d.verified === false ? ['value NOT on the page', 'bad'] : ['never checked', 'warn'],
        d.quote_verified === true ? ['quote verified', 'ok'] :
          d.quote_verified === false ? ['quote not found: ' + (d.quote_fail_reason || 'absent'), 'bad'] :
          ['quote unchecked', 'warn'],
        d.minted_by ? ['minted by ' + d.minted_by, ''] : null,
      ]));
      break;
    case 'rank': {
      const s = d.score, floor = d.min_score;
      parts.push(`<div class="note">${esc(d.entity)} — <b>${esc(d.field_phrase)}</b></div>`);
      parts.push(`<div class="headline mono">${esc(b.subject)}</div>`);
      if (typeof s === 'number') {
        const pass = s >= floor;
        parts.push(`<div class="scorebar"><div class="fill ${pass ? 'pass' : ''}" style="width:${
          Math.max(0, Math.min(1, s)) * 100}%"></div><div class="floor" style="left:${floor * 100}%"></div></div>`);
        parts.push(`<div class="scorelabels"><span>attribution score ${s.toFixed(3)}</span>` +
          `<span>floor ${floor}</span></div>`);
      } else {
        parts.push(chips([['no candidate on any fetched page', 'bad']]));
      }
      break;
    }
    case 'derive': {
      // An operand shown only as "E16" makes the arithmetic unverifiable by eye. The handle is
      // the provenance; the value is the thing being divided. Show both, or the frame asks the
      // viewer to trust it -- which is the opposite of the point.
      const byRef = {};
      ((b.ledger_after || {}).handles || []).forEach(h => { byRef[h.ref] = h; });
      const ins = (d.inputs || []).map(x => {
        const h = byRef[x] || {};
        const v = h.value ? `${h.value}${h.unit ? ' ' + h.unit : ''}` : '';
        return `<span class="operand">${esc(x)}${v ? `<em>${esc(v)}</em>` : ''}</span>`;
      }).join(`<span class="op">·</span>`);
      parts.push(`<div class="formula"><span class="op">${esc(d.operation)}</span>${ins}` +
        `<span class="eq">=</span><span class="result">${esc(b.subject)}</span></div>`);
      parts.push(chips([
        d.derivation_valid === true ? ['arithmetic recomputes', 'ok'] :
          d.derivation_valid === false ? ['arithmetic FAILS to recompute', 'bad'] : ['never assessed', 'warn'],
        d.operand_supported === true ? ['both operands traced to a page', 'ok'] :
          d.operand_supported === false ? ['an operand is not grounded', 'bad'] : null,
        d.message ? [d.message, ''] : null,
      ]));
      // The checks are mechanical and LOCAL. They confirm the sum was computed correctly from
      // operands that exist on a page; they say nothing about whether this was a sensible thing
      // to compute. Leaving that unsaid lets two green chips imply the model reasoned well --
      // which, on this run, it did not: a cold reader spotted a height/height ratio wearing them.
      parts.push(`<div class="note" style="color:var(--ink-muted)">Checked: the value recomputes` +
        ` from these operands, and each operand traces to a page. Not checked: whether this` +
        ` calculation answers the question — the model chose the operands.</div>`);
      break;
    }
    case 'refuse':
      parts.push(`<div class="headline mono" style="color:var(--serious)">${esc(d.code)}</div>`);
      parts.push(`<div class="note crit">${esc(d.message)}</div>`);
      parts.push(`<div class="note">Nothing was asserted. The gap is reported instead of filled.</div>`);
      break;
    case 'audit': {
      // The old headline ("28 of 28 backed") sat in big type above a small grey "blocked"
      // line, and a cold reader took the headline as the outcome. The outcome leads now.
      const blocked = !d.answer_supported;
      parts.push(`<div class="headline"${blocked ? ' style="color:var(--warning)"' : ''}>${
        blocked ? esc(b.target) : 'every figure is uniquely backed'}</div>`);
      parts.push(`<div class="note">${esc(b.subject)} — but backed is not the same as backed by` +
        ` exactly one value.</div>`);
      const nums = (d.numbers || []).map(n => {
        const cls = [n.status, (n.ambiguity > 1 ? 'ambig' : '')].join(' ');
        const amb = n.ambiguity > 1 ? `<sub>×${n.ambiguity}</sub>` : '';
        return `<span class="num ${cls}" title="${esc(n.status)}${n.ref ? ' · ' + esc(n.ref) : ''}">${
          esc(n.text)}${n.unit ? ' ' + esc(n.unit) : ''}${amb}</span>`;
      }).join('');
      parts.push(`<div class="numgrid">${nums}</div>`);
      parts.push(chips([['backed by a span', 'ok'], ['derived in code', 'mid'],
                        ['dashed = matches several values, so backs none of them uniquely', 'dash']]));
      break;
    }
    case 'verdict': {
      parts.push(`<div class="verdictbig ${esc(b.subject)}">${esc(b.subject)}</div>`);
      parts.push(`<div class="note">${esc(d.basis || '')}</div>`);
      const c = d.counts || {};
      parts.push(chips([
        [`${c.pages || 0} pages read`, ''],
        [`${c.sources || 0} spans located`, ''],
        [`${c.verified || 0} verified on the page`, c.verified ? 'ok' : ''],
        [`${c.unchecked || 0} never checked`, c.unchecked ? 'warn' : ''],
        [`${c.derived || 0} values derived in code`, 'mid'],
        [`${c.refusals || 0} refused`, c.refusals ? 'bad' : ''],
      ]));
      parts.push(`<div class="note">Every figure above traces to a character range on a stored page. Nothing here is asserted from model memory.</div>`);
      // A deck that never restates the answer leaves the viewer unable to say what the run
      // concluded -- which is the question slide 1 asked.
      if (d.answer) {
        // The model emits markdown; the panel shows prose. Strip only the markers, never words.
        const prose = String(d.answer).replace(/^#{1,6}\s*/gm, '').replace(/\*\*/g, '');
        parts.push(`<div class="answer"><h3>what the run answered</h3>${esc(prose).trim()}</div>`);
      }
      // Two different subsystems count two different things, and a reader who assumes one
      // denominator will find the numbers irreconcilable. Say which is which.
      parts.push(`<div class="note" style="color:var(--ink-muted)">The audit counts numbers` +
        ` printed in the answer; the verdict counts checkable <em>claims</em>. Different` +
        ` denominators, different subsystems — they are not meant to match.</div>`);
      break;
    }
    default:
      parts.push(`<div class="headline">${esc(b.subject)}</div>`);
  }
  return parts.join('');
}

function chips(rows) {
  const out = rows.filter(Boolean).map(([t, c]) => `<span class="chip ${c || ''}">${esc(t)}</span>`).join('');
  return out ? `<div class="chips">${out}</div>` : '';
}

function render() {
  const b = B[i];
  el('counter').textContent = String(i + 1).padStart(2, '0') + ' / ' + B.length;
  const k = el('kind'); k.textContent = b.title; k.dataset.status = b.status;
  el('stage').innerHTML =
    `<div class="flow"><span class="from">${esc(b.origin || '—')}</span>` +
    `<span class="arrow">──▶</span><span class="to ${b.target && /^[ED]\d+$/.test(b.target) ? '' : 'neutral'}">${
      esc(b.target || '—')}</span></div>` + stageFor(b);
  el('stage').scrollTop = 0;

  const snap = b.ledger_after || {}, c = snap.counts || {};
  el('counts').innerHTML =
    `<div><span class="n">${c.pages || 0}</span><span class="l">pages</span></div>` +
    `<div><span class="n">${c.sources || 0}</span><span class="l">spans</span></div>` +
    `<div><span class="n">${c.derived || 0}</span><span class="l">derived</span></div>` +
    `<div><span class="n" style="color:var(--ok)">${c.verified || 0}</span><span class="l">verified</span></div>` +
    `<div><span class="n" style="color:var(--warning)">${c.unchecked || 0}</span><span class="l">unchecked</span></div>` +
    `<div><span class="n" style="color:var(--serious)">${c.refusals || 0}</span><span class="l">refused</span></div>`;

  let hs = (snap.handles || []).slice().reverse();
  // Newest-first means E1 is permanently scrolled out of sight -- and E1 is exactly the operand a
  // DERIVE frame is asking you to check. Float this beat's operands to the top with it.
  const ops = new Set((b.detail && b.detail.inputs) || []);
  if (ops.size) hs = hs.filter(h => ops.has(h.ref)).concat(hs.filter(h => !ops.has(h.ref)));
  el('rail').innerHTML = hs.length ? hs.map(h => {
    const on = h.ref === b.target;
    const tick = h.verified === true ? '<span class="tick" style="color:var(--ok)">✓</span>'
      : h.verified === false ? '<span class="tick" style="color:var(--critical)">✕</span>'
      : '<span class="tick" style="color:var(--warning)">?</span>';
    return `<div class="h-row ${h.kind} ${on ? 'active' : ops.has(h.ref) ? 'operand' : ''}"><span class="ref">${esc(h.ref)}</span>` +
      `<span class="val">${esc(h.value)}${h.unit ? ' ' + esc(h.unit) : ''} ${tick}` +
      `<span class="org">${esc(h.origin || '')}</span></span></div>`;
  }).join('') : '<div class="rail-empty">no evidence admitted yet</div>';

  document.querySelectorAll('.rib').forEach((r, n) => {
    r.className = 'rib ' + (n === i ? 'on' : n < i ? 'past' : 'future');
  });
}

function goto(n) { i = Math.max(0, Math.min(B.length - 1, n)); render(); }
window.gotoBeat = goto;                       // headless screenshotting hooks in here
window.beatCount = B.length;

function step(d) { if (i + d >= B.length) { stop(); return; } goto(i + d); }
function tick() { if (!playing) return; if (i >= B.length - 1) { stop(); return; }
  goto(i + 1); timer = setTimeout(tick, B[i].dwell); }
function stop() { playing = false; clearTimeout(timer); el('play').textContent = '▶'; }
function play() { if (i >= B.length - 1) goto(0); playing = true; el('play').textContent = '❚❚';
  timer = setTimeout(tick, B[i].dwell); }

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'PageDown') { stop(); step(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { stop(); step(-1); e.preventDefault(); }
  else if (e.key === ' ') { playing ? stop() : play(); e.preventDefault(); }
  else if (e.key === 'Home') { stop(); goto(0); }
  else if (e.key === 'End') { stop(); goto(B.length - 1); }
});
document.getElementById('play').addEventListener('click', () => playing ? stop() : play());

const rib = document.getElementById('ribbon');
rib.innerHTML = B.map((b, n) =>
  `<div class="rib" data-n="${n}" tabindex="0" role="button" aria-label="step ${n + 1}: ${
    esc(b.title)}" title="${esc(b.title)} — ${esc(b.subject).slice(0, 60)}" style="${
    b.status === 'refused' || b.status === 'error' ? 'background:var(--serious)' : ''}"></div>`).join('');
rib.addEventListener('click', e => { const t = e.target.closest('.rib'); if (t) { stop(); goto(+t.dataset.n); } });
rib.addEventListener('keydown', e => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const t = e.target.closest('.rib'); if (t) { stop(); goto(+t.dataset.n); e.preventDefault(); }
});
document.getElementById('stage').addEventListener('click', () => { stop(); step(1); });
render();
"""

PAGE = """<meta charset="utf-8">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>__CSS__</style>
<div id="deck">
  <header>
    <span class="counter" id="counter"></span>
    <span class="kind" id="kind"></span>
    <button id="play" style="background:var(--panel-hi);color:var(--ink-secondary);
      border:1px solid var(--hairline);border-radius:5px;width:30px;height:26px;cursor:pointer;
      font-size:11px">&#9654;</button>
    <span class="ident">__IDENT__</span>
  </header>
  <main>
    <section class="stage" id="stage"></section>
    <aside class="rail">
      <h2>THE LEDGER</h2>
      <div class="rail-counts" id="counts"></div>
      <div class="rail-list" id="rail"></div>
    </aside>
  </main>
  <div class="ribbon" id="ribbon"></div>
  <div class="legend">
    <span><i style="background:var(--accent-hot)"></i>current step</span>
    <span><i style="background:var(--serious)"></i>refused / failed</span>
    <span><i style="background:var(--hairline)"></i>run timeline — click any step</span>
    <span class="hint"><kbd>&larr;</kbd><kbd>&rarr;</kbd> step &nbsp; <kbd>space</kbd> play</span>
  </div>
</div>
<script>window.__STORY__ = __DATA__;</script>
<script>__JS__</script>
"""


def build_html(story: ledger_story.Storyboard, title: str) -> str:
    """The whole deck as one string. No external references of any kind."""
    payload: Dict[str, Any] = story.as_dict()
    for row in payload["beats"]:
        row["dwell"] = DWELL_MS.get(row["kind"], 2200)
    tokens = "\n".join(f"  --{name}:{value};" for name, value in plot_style.dark_tokens().items())
    ident = " · ".join(html.escape(x) for x in [
        f"task <b>{story.task_id}</b>" if story.task_id else "",
        html.escape(story.model) if story.model else "",
        html.escape(story.variant) if story.variant else "",
    ] if x).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    return (PAGE
            .replace("__TITLE__", html.escape(title))
            .replace("__CSS__", CSS.replace("__TOKENS__", tokens))
            .replace("__IDENT__", ident)
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__JS__", JS))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cell", help="a stored per-cell result JSON")
    parser.add_argument("-o", "--out", default="ledger_deck.html", help="output HTML path")
    parser.add_argument("--title", default="", help="deck title (default: derived from the run)")
    parser.add_argument("--json", default="", help="also write the storyboard JSON here")
    args = parser.parse_args(argv)

    story = ledger_story.build_file(args.cell)
    title = args.title or f"Ledger Run {story.task_id or ''}".strip()
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(build_html(story, title))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(story.as_dict(), handle, indent=1, ensure_ascii=False)
    print(f"{len(story)} beats -> {args.out}  ({os.path.getsize(args.out) / 1024:.0f} KB)")
    print("  " + "  ".join(f"{k}:{v}" for k, v in story.counts().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
