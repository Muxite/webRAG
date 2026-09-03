#!/usr/bin/env python3
"""ledger_run.py: answer ONE question with the Euglena Ledger and print a pinned result.

This is the component's command line. It needs no task module, no graders, no benchmark harness
and no results directory -- a question, and optionally the sources it may be answered from::

    python scripts/ledger_run.py "How much taller is Denali than Mount Whitney?" \\
        --source https://en.wikipedia.org/wiki/Denali \\
        --source https://en.wikipedia.org/wiki/Mount_Whitney

With no ``--source`` the configured live search backend is used, exactly as a benchmark cell
does. With one or more, retrieval is restricted to them: search ranks only those documents and a
visit outside the set is refused (see :mod:`agent.app.ledger_api`).

``--source-file`` reads a JSONL file of ``{"url", "title", "text"}`` rows -- the same shape a
frozen corpus uses (``agent/idea_test_results/corpus/*/documents.jsonl``) -- so a question can be
answered fully offline, at $0, from documents already on disk.

The human output shows what the product claims: the answer, the verdict, and per claim the value
with its source URL and whether its quote was mechanically found on that page. An ABSTAIN, an
unverified quote and a refused derivation are all printed as they are. ``--json`` emits
``LedgerResult.to_dict()``.

Exit status is 0 for ANSWER or PARTIAL and 2 for ABSTAIN, so a script can tell the difference
between "answered" and "declined" without parsing prose. ``--strict`` also exits 2 when any
surfaced claim is not backed by a verified quote.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(1, str(_ROOT / "services"))

from agent.app import ledger_api  # noqa: E402
from agent.app.ledger_api import (QUOTE_FAILED, QUOTE_UNKNOWN, QUOTE_VERIFIED,  # noqa: E402
                                  VERDICT_ABSTAIN, LedgerResult, Source)

#: Exit code for a run that declined to answer. Distinct from 1 (the script itself failed) so a
#: caller never has to read the prose to tell an honest abstention from a crash.
EXIT_ABSTAIN = 2

#: How each verification state reads to a human. UNKNOWN is worded as "not checked", never as a
#: failure -- absent is not zero.
_MARK = {QUOTE_VERIFIED: "verified quote", QUOTE_FAILED: "QUOTE NOT FOUND ON PAGE",
         QUOTE_UNKNOWN: "quote not checked"}


def load_source_file(path: str) -> List[Source]:
    """Read a JSONL source file into :class:`Source` rows.

    :param path: file of one JSON object per line, carrying ``url`` and optionally ``title`` /
        ``text``. A row with no ``url`` is skipped; a malformed line raises, because a silently
        dropped source changes which question was actually answered.
    :returns: the sources, in file order.
    """
    sources: List[Source] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON ({exc})") from exc
        if not isinstance(row, dict) or not row.get("url"):
            continue
        sources.append(Source(url=str(row["url"]), text=str(row.get("text") or ""),
                              title=str(row.get("title") or "")))
    return sources


def select(sources: List[Source], urls: List[str]) -> List[Source]:
    """The subset of ``sources`` whose URL is listed in ``urls``, in the order given.

    :raises ValueError: naming a URL the file does not hold, which would otherwise answer the
        question from fewer sources than the caller asked for without saying so.
    """
    index = {source.url: source for source in sources}
    chosen = []
    for url in urls:
        if url not in index:
            raise ValueError(f"--source {url} is not in the source file")
        chosen.append(index[url])
    return chosen


def render(result: LedgerResult) -> str:
    """The human report: answer, verdict, then one pinned line per claim."""
    lines = [f"QUESTION: {result.question}", "", "ANSWER:", result.answer.strip() or "(none)", ""]
    counts = result.resolution_counts
    lines.append(f"VERDICT: {result.verdict}   "
                 f"({counts.get('resolved', 0)}/{counts.get('rows', 0)} claims resolved, "
                 f"{counts.get('resolved_verified', 0)} backed by a verified quote)")
    lines.append("")
    lines.append(f"CLAIMS ({len(result.claims)}):")
    for number, claim in enumerate(result.claims, 1):
        value = claim.value or "(not established)"
        lines.append(f"{number}. [{claim.status}] {claim.entity} = {value}")
        lines.append(f"     source: {claim.source_url or '(none)'}")
        pin = (f" @{claim.page_id}[{claim.quote_start}:{claim.quote_end}]"
               if claim.page_id and claim.quote_start >= 0 else "")
        lines.append(f"     quote:  {_MARK[claim.quote_verification]}{pin}")
        if claim.quote:
            lines.append(f"             \"{claim.quote[:160]}\"")
    if result.derivations:
        lines.append("")
        lines.append(f"DERIVATIONS ({len(result.derivations)}):")
        for derivation in result.derivations:
            state = {True: "recomputed and agrees", False: "POSTCONDITION FAILED",
                     None: "not assessed"}[derivation.valid]
            lines.append(f"  {derivation.operation} = {derivation.value} "
                         f"{derivation.unit}".rstrip() + f"  [{state}]")
            if derivation.detail:
                lines.append(f"    {derivation.detail}")
    if result.derivation_refusals:
        lines.append("")
        lines.append("REFUSED DERIVATIONS: " + ", ".join(
            f"{code} x{count}" for code, count in sorted(result.derivation_refusals.items())))
    lines.append("")
    lines.append(f"PAGES READ: {len(result.pages)}"
                 + (f"   SEARCHES: {len(result.search_provenance)} "
                    f"({', '.join(result.search_provenance)})" if result.search_provenance else ""))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledger_run.py", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--source", action="append", default=[], metavar="URL",
                        help="a source URL the answer may be built from; repeatable. Omit "
                             "entirely to use the configured live search backend.")
    parser.add_argument("--source-file", metavar="PATH",
                        help="JSONL file of {url,title,text} rows to answer from (offline). "
                             "With --source, only the named URLs from it are used.")
    parser.add_argument("--model", help="executor model; defaults to $MODEL_NAME")
    parser.add_argument("--max-steps", type=int, default=ledger_api.DEFAULT_MAX_STEPS)
    parser.add_argument("--max-tokens", type=int, default=ledger_api.DEFAULT_MAX_TOKENS)
    parser.add_argument("--json", action="store_true", help="emit LedgerResult.to_dict()")
    parser.add_argument("--no-page-text", action="store_true",
                        help="with --json, drop each page's stored text (hashes are kept)")
    parser.add_argument("--scratchpad", action="store_true",
                        help="with --json, include the loop's per-step trace")
    parser.add_argument("--strict", action="store_true",
                        help="exit %d unless every claim is backed by a verified quote"
                             % EXIT_ABSTAIN)
    return parser


def resolve_sources(args: argparse.Namespace):
    """The source set for this run, or None for the configured live backend."""
    if args.source_file:
        available = load_source_file(args.source_file)
        return select(available, args.source) if args.source else available
    return list(args.source) or None


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = resolve_sources(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = asyncio.run(ledger_api.run(args.question, sources, model=args.model,
                                        max_steps=args.max_steps, max_tokens=args.max_tokens))
    if args.json:
        payload: Dict[str, Any] = result.to_dict(include_scratchpad=args.scratchpad,
                                                 include_page_text=not args.no_page_text)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render(result))
    if result.verdict == VERDICT_ABSTAIN:
        return EXIT_ABSTAIN
    if args.strict and any(claim.quote_verification != QUOTE_VERIFIED
                           for claim in result.claims):
        return EXIT_ABSTAIN
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
