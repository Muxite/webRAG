#!/usr/bin/env python3
"""
Sync the hand-authored plan-library corpus into its Chroma index — idempotent, hash-gated.

Chroma is an INDEX, never the source of truth: ``plan_library/templates/*.json`` is. This
script embeds each template's ``embedding_text`` into the ``plan_library_templates_v1``
collection so retrieval can rank ``template_id``s, and records ``{template_id: content_hash}``
in ``templates/_manifest.json`` so an unchanged template costs ZERO re-embedding on the next
run (the same lockfile pattern as ``scaffold_compiler``'s mandate hash).

The manifest is a necessary but NOT sufficient condition for skipping: it is committed to the
repo while the collection is per-environment, so "unchanged" is only believed once the target
collection is confirmed to actually hold that id (see :func:`_plan`). Without that check a
fresh checkout pointed at an empty collection reports "nothing to embed" and stays empty
forever.

It is an explicit AUTHORING step — edit a template, run this, commit the template and the
manifest together. It is deliberately NOT run at engine startup: re-embedding from every
worker subprocess is exactly the multi-process Chroma write contention this codebase already
had to fix once. ``PlanLibrary`` only *warns* about manifest drift at construction.

Chroma's ``add`` silently keeps the OLD row when an id already exists, so a changed template
is delete-then-add, and a template whose file is gone has its index entry deleted outright.

Usage::

    # embedded mode: a private per-process SQLite store, no chroma service needed
    CHROMA_MODE=embedded CHROMA_EMBEDDED_PATH=./.chroma_plan_library \\
      PYTHONPATH=.:services:agent ./.venv/bin/python scripts/sync_plan_library.py

    # against the running chroma service
    CHROMA_URL=http://localhost:8001 \\
      PYTHONPATH=.:services:agent ./.venv/bin/python scripts/sync_plan_library.py

    # what would change: reads the collection, embeds nothing, manifest untouched
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/sync_plan_library.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Mirror the runner's import roots so this works from a plain checkout.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "services", _ROOT / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.connector_config import ConnectorConfig  # noqa: E402
from agent.app.connector_chroma import ConnectorChroma  # noqa: E402
from agent.app.plan_library.retrieval import (  # noqa: E402
    COLLECTION_NAME,
    PlanLibrary,
    document_metadata,
    document_text,
    template_hash,
)


async def _plan(
    library: PlanLibrary, chroma: ConnectorChroma, force: bool
) -> Dict[str, List[str]]:
    """What has to change: ``{added, changed, unchanged, removed, unverified}`` template ids.

    The manifest ALONE cannot answer this. It is committed to the repo — a record of "these
    templates were embedded", with no idea WHERE — while the target collection is
    per-environment (an embedded SQLite path, or wherever ``CHROMA_URL`` points). Gating on it
    alone means a fresh checkout aimed at an empty collection reports "nothing to embed" and
    exits 0, leaving an index that every later retrieval silently misses on, forever.

    So every manifest-says-unchanged id is verified against the TARGET collection and folded
    into ``added`` when it is not actually there — "added" rather than "changed" because
    nothing is indexed under it, so there is nothing to delete first. One extra round-trip per
    sync, which is free at this cadence: sync is an explicit, infrequent authoring step.

    ``unverified`` carries the ids the check could NOT reach (an unreachable collection), so
    the caller can refuse to call an unreadable index "up to date" — the same silent-success
    trap, one layer out.
    """
    manifest = library.read_manifest()
    added, changed, unchanged = [], [], []
    for template_id, template in sorted(library.templates.items()):
        recorded = manifest.get(template_id)
        if recorded is None:
            added.append(template_id)
        elif force or recorded != template_hash(template):
            changed.append(template_id)
        else:
            unchanged.append(template_id)

    unverified: List[str] = []
    if unchanged:
        present = await library.indexed_ids(chroma, unchanged)
        if present is None:
            unverified = list(unchanged)
            print(f"  WARNING: could not read '{library.collection}' — the 'unchanged' verdict "
                  "below is the manifest's word alone", file=sys.stderr)
        else:
            missing = [tid for tid in unchanged if tid not in present]
            if missing:
                print(f"  MISSING from '{library.collection}' despite the manifest: "
                      f"{', '.join(missing)}")
                added = sorted(added + missing)
                unchanged = [tid for tid in unchanged if tid in present]

    removed = sorted(tid for tid in manifest if tid not in library.templates)
    return {
        "added": added, "changed": changed, "unchanged": unchanged,
        "removed": removed, "unverified": unverified,
    }


async def _sync(library: PlanLibrary, chroma: ConnectorChroma, work: Dict[str, List[str]]) -> bool:
    """Apply ``work`` to the collection. Returns True when every op succeeded."""
    collection = await library.ensure_collection(chroma)
    if collection is None:
        print("ERROR: could not open collection "
              f"'{library.collection}' (is chroma reachable?)", file=sys.stderr)
        return False

    ok = True
    stale = work["removed"] + work["changed"]
    if stale:
        # Changed templates are deleted first: chroma's ``add`` keeps the existing row.
        ok = await chroma.delete_from_chroma(library.collection, stale) and ok
        print(f"  deleted {len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'}: "
              f"{', '.join(stale)}")

    write = work["added"] + work["changed"]
    if write:
        templates = [library.templates[tid] for tid in write]
        ok = await chroma.add_to_chroma(
            collection=library.collection,
            ids=list(write),
            metadatas=[document_metadata(t) for t in templates],
            documents=[document_text(t) for t in templates],
        ) and ok
        print(f"  embedded {len(write)} template(s): {', '.join(write)}")
    return ok


def _write_manifest(library: PlanLibrary) -> Path:
    path = library.manifest_path()
    payload = {tid: template_hash(t) for tid, t in sorted(library.templates.items())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


async def main_async(args: argparse.Namespace) -> int:
    library = PlanLibrary(
        templates_dir=Path(args.templates_dir) if args.templates_dir else None,
        collection=args.collection,
        warn_on_drift=False,  # this script IS the drift resolution
    )
    print(f"plan library: {len(library)} template(s) in {library.templates_dir}")
    for name, error in sorted(library.load_errors.items()):
        print(f"  SKIPPED {name}: {error}", file=sys.stderr)
    if not library.templates:
        print("nothing to sync", file=sys.stderr)
        return 1

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
            # "Nothing to do" is only trustworthy when the index was actually read.
            print(f"ERROR: could not read '{library.collection}', so its contents are unknown "
                  "— refusing to report it up to date (is chroma reachable?)", file=sys.stderr)
            return 1
        print(f"index already up to date ({library.collection}); nothing to embed")
        return 0

    ok = await _sync(library, chroma, work)
    if not ok:
        print("ERROR: sync failed; manifest NOT updated", file=sys.stderr)
        return 1

    path = _write_manifest(library)
    print(f"manifest updated: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--templates-dir", default=None, help="Override the template directory.")
    parser.add_argument("--force", action="store_true", help="Re-embed every template.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the plan only — reads the collection to verify it, writes nothing.",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
