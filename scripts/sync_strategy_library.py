#!/usr/bin/env python3
"""
Sync the PROMOTED strategy notes into their Chroma index — idempotent, hash-gated.

The strategy-library twin of ``scripts/sync_plan_library.py``, with one extra rule that matters:
**only notes that clear the promotion gate are indexed**. A note whose held-out uplift has not
been measured (or no longer clears the bar) is not merely un-retrievable in code — it is not in
the index at all, so there is no path by which an unproven note reaches a run.

Chroma is an INDEX, never the source of truth: ``strategy_library/notes/*.json`` is. The
manifest (``notes/_manifest.json``) records ``{note_id: content_hash}`` so an unchanged note
costs zero re-embedding, and — because the manifest is committed while the collection is
per-environment — every "unchanged" verdict is verified against the target collection before it
is believed.

An explicit AUTHORING step: author/measure a note, run this, commit the note and the manifest
together. Deliberately NOT run at engine startup (re-embedding from every worker subprocess is
the Chroma write contention this codebase already had to fix once).

Usage::

    CHROMA_MODE=embedded CHROMA_EMBEDDED_PATH=./.chroma_strategy_library \\
      PYTHONPATH=services:services/agent ./.venv/bin/python scripts/sync_strategy_library.py

    CHROMA_URL=http://localhost:8001 \\
      PYTHONPATH=services:services/agent ./.venv/bin/python scripts/sync_strategy_library.py

    ... scripts/sync_strategy_library.py --dry-run      # reads the index, writes nothing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "services", _ROOT / "services" / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.connector_config import ConnectorConfig  # noqa: E402
from agent.app.connector_chroma import ConnectorChroma  # noqa: E402
from agent.app.strategy_library.retrieval import (  # noqa: E402
    COLLECTION_NAME,
    StrategyLibrary,
    document_metadata,
    document_text,
    note_hash,
)


async def _plan(library: StrategyLibrary, chroma: ConnectorChroma, force: bool) -> Dict[str, List[str]]:
    """``{added, changed, unchanged, removed, unverified}`` note ids.

    ``library.notes`` is already the promoted-only view, so a note that LOST its promotion shows
    up here as ``removed`` and gets deleted from the index — which is the whole reason the
    manifest is compared against that view rather than against everything on disk.
    """
    manifest = library.read_manifest()
    added, changed, unchanged = [], [], []
    for note_id, note in sorted(library.notes.items()):
        recorded = manifest.get(note_id)
        if recorded is None:
            added.append(note_id)
        elif force or recorded != note_hash(note):
            changed.append(note_id)
        else:
            unchanged.append(note_id)

    unverified: List[str] = []
    if unchanged:
        present = await library.indexed_ids(chroma, unchanged)
        if present is None:
            unverified = list(unchanged)
            print(f"  WARNING: could not read '{library.collection}' — the 'unchanged' verdict "
                  "below is the manifest's word alone", file=sys.stderr)
        else:
            missing = [nid for nid in unchanged if nid not in present]
            if missing:
                print(f"  MISSING from '{library.collection}' despite the manifest: "
                      f"{', '.join(missing)}")
                added = sorted(added + missing)
                unchanged = [nid for nid in unchanged if nid in present]

    removed = sorted(nid for nid in manifest if nid not in library.notes)
    return {"added": added, "changed": changed, "unchanged": unchanged,
            "removed": removed, "unverified": unverified}


async def _sync(library: StrategyLibrary, chroma: ConnectorChroma, work: Dict[str, List[str]]) -> bool:
    collection = await library.ensure_collection(chroma)
    if collection is None:
        print(f"ERROR: could not open collection '{library.collection}' (is chroma reachable?)",
              file=sys.stderr)
        return False

    ok = True
    stale = work["removed"] + work["changed"]
    if stale:
        # Chroma's ``add`` keeps the existing row, so a changed note is delete-then-add.
        ok = await chroma.delete_from_chroma(library.collection, stale) and ok
        print(f"  deleted {len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'}: "
              f"{', '.join(stale)}")

    write = work["added"] + work["changed"]
    if write:
        notes = [library.notes[nid] for nid in write]
        ok = await chroma.add_to_chroma(
            collection=library.collection,
            ids=list(write),
            metadatas=[document_metadata(n) for n in notes],
            documents=[document_text(n) for n in notes],
        ) and ok
        print(f"  embedded {len(write)} note(s): {', '.join(write)}")
    return ok


def _write_manifest(library: StrategyLibrary) -> Path:
    path = library.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {nid: note_hash(n) for nid, n in sorted(library.notes.items())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


async def main_async(args: argparse.Namespace) -> int:
    library = StrategyLibrary(
        notes_dir=Path(args.notes_dir) if args.notes_dir else None,
        collection=args.collection,
        warn_on_drift=False,  # this script IS the drift resolution
        include_inactive=bool(args.include_unpromoted),
    )
    if args.include_unpromoted:
        print("WARNING: --include-unpromoted — indexing notes that have NOT cleared the "
              "promotion gate. Only the held-out measurement run wants this; re-sync without "
              "it afterwards.", file=sys.stderr)
    print(f"strategy library: {len(library.all_notes)} note(s) in {library.notes_dir}; "
          f"{len(library.notes)} promoted")
    for name, error in sorted(library.load_errors.items()):
        print(f"  SKIPPED {name}: {error}", file=sys.stderr)
    for note_id, reason in library.promotion_report().items():
        print(f"  {note_id}: {reason}")

    chroma = ConnectorChroma(ConnectorConfig())
    work = await _plan(library, chroma, force=args.force)
    for bucket in ("added", "changed", "unchanged", "removed"):
        if work[bucket]:
            print(f"  {bucket}: {', '.join(work[bucket])}")

    if args.dry_run:
        print("dry run — nothing was embedded and the manifest was not touched")
        return 0

    if not (work["added"] or work["changed"] or work["removed"]):
        if work["unverified"]:
            print(f"ERROR: could not read '{library.collection}', so its contents are unknown — "
                  "refusing to report it up to date (is chroma reachable?)", file=sys.stderr)
            return 1
        print(f"index already up to date ({library.collection}); nothing to embed")
        _write_manifest(library)
        return 0

    if not await _sync(library, chroma, work):
        print("ERROR: sync failed; manifest NOT updated", file=sys.stderr)
        return 1

    print(f"manifest updated: {_write_manifest(library)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--notes-dir", default=None, help="Override the notes directory.")
    parser.add_argument("--force", action="store_true", help="Re-embed every promoted note.")
    parser.add_argument(
        "--include-unpromoted", action="store_true",
        help="Also index notes that have not cleared the promotion gate — the held-out "
             "measurement bootstrap only (pair with "
             "IDEA_TEST_STRATEGY_LIBRARY_INCLUDE_UNPROMOTED=1), never a production sync.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the plan only — reads the collection, writes nothing.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
