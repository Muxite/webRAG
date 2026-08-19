"""Run the (model x family x variant x item) matrix and append one JSONL row per cell.

Resumable: a completed cell key is never re-run, so a crash or an interrupt costs
wall clock but never re-spends money. Cells are ordered smallest-model-first
because OLLAMA_MAX_LOADED_MODELS=1 makes every model swap expensive, and a swap
per cell would dominate the run.

Grading happens here, and it is the one place the Label is legitimately opened --
via ``.expose(reason)``, which returns an ORACLE-tainted Signal. The prompt builder
never sees it.

INAPPLICABLE CELLS ARE NEVER PLACED
-----------------------------------
``factors.is_applicable`` decides whether a (family, variant) pair asks a coherent
question. v1 ran SHIPPED x select -- a four-way truth verdict asked to name one of
five candidate marathons -- and discarded every one of those cells in analysis.
Here the calls are not made at all, so the waste is avoided rather than cleaned up.

The base URL comes from ``PROMPTBENCH_BASE_URL`` when set. On the host that stays
badmodel-ollama's published port; inside the compose profile it is the service
name on the shared network, and nothing else about the run differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from agent.app.promptbench.availability import Item, PromptContext
from agent.app.promptbench.factors import (
    ALL_VARIANTS,
    CALIBRATION_VARIANTS,
    PRIMARY_VARIANTS,
    build_prompt,
    is_applicable,
)
from agent.app.promptbench.families import ALL_FAMILIES, REGISTRY, build as build_families
from agent.app.promptbench.grade import grade_confidence
from agent.app.promptbench.http_llm import HttpLLM
from agent.app.promptbench.items import census as candidate_census, load_all_specs, load_specs
from agent.app.promptbench.items_engine import census as engine_census
from agent.app.promptbench.items_keystone import census as keystone_census

DEFAULT_BASE_URL = os.environ.get("PROMPTBENCH_BASE_URL", "http://127.0.0.1:11435/v1")
ABSTAIN = ("INSUFFICIENT", "UNKNOWN", "UNCLEAR", "UNVERIFIABLE")


def _cell_key(model: str, family: str, variant: str, item_id: str, rep: int) -> str:
    return f"{model}|{family}|{variant}|{item_id}|{rep}"


def _load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["cell"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def full_census() -> Dict[str, object]:
    """Everything the matrix would be built from, before any call is placed."""
    specs = load_all_specs()
    families = build_families(list(ALL_FAMILIES), specs)
    return {
        "candidate_families": candidate_census(load_specs()),
        "keystone_family": keystone_census(specs),
        "engine_families": engine_census(specs),
        "matrix": {
            name: {
                "items": len(items),
                "clusters": len({i.cluster for i in items}),
                "has_shipped": REGISTRY[name].has_shipped,
                "applicable_variants": [v for v in ALL_VARIANTS if is_applicable(name, v)],
            }
            for name, items in families.items()
        },
        "total_items": sum(len(v) for v in families.values()),
        "union_clusters": len({i.cluster for v in families.values() for i in v}),
    }


def run(models: List[str], variants: List[str], families: List[str], reps: int,
        out_path: Path, base_url: str, api_key: str, max_cells: int) -> int:
    fam_items = build_families(families)
    done = _load_done(out_path)
    llm = HttpLLM(base_url, api_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    planned = sum(
        len(items) * sum(1 for v in variants if is_applicable(family, v))
        for family, items in fam_items.items()
    ) * len(models) * reps
    skipped = sum(
        len(items) * sum(1 for v in variants if not is_applicable(family, v))
        for family, items in fam_items.items()
    ) * len(models) * reps
    print(f"planned cells: {planned} | already done: {len(done)} | cap: {max_cells}", flush=True)
    if skipped:
        # Named rather than silently dropped: a coverage gap that prints is a
        # decision, one that does not is an accident waiting to be misread.
        print(f"skipping {skipped} inapplicable (family, variant) cells", flush=True)

    ran = 0
    t_start = time.time()
    with out_path.open("a") as fh:
        for model in models:
            for family, items in fam_items.items():
                for variant in variants:
                    if not is_applicable(family, variant):
                        continue
                    for rep in range(reps):
                        for item in items:
                            key = _cell_key(model, family, variant, item.item_id, rep)
                            if key in done:
                                continue
                            if ran >= max_cells:
                                print(f"CELL CAP {max_cells} reached; stopping cleanly", flush=True)
                                return ran
                            ctx = PromptContext(family=family, variant=variant, model=model)
                            prompt = build_prompt(item.runtime, ctx)
                            try:
                                c = llm.complete(prompt, model=model, temperature=0.0)
                            except Exception as exc:                     # noqa: BLE001
                                fh.write(json.dumps({
                                    "cell": key, "model": model, "family": family,
                                    "variant": variant, "item_id": item.item_id,
                                    "cluster": item.cluster, "rep": rep,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }) + "\n")
                                fh.flush()
                                ran += 1
                                continue
                            truth = item.label.expose("grading a promptbench prediction")
                            verdict, confidence = grade_confidence(
                                c.text, truth.value, item.runtime["choices"],
                                abstain_choices=list(ABSTAIN))
                            fh.write(json.dumps({
                                "cell": key, "model": model, "family": family,
                                "variant": variant, "item_id": item.item_id,
                                "cluster": item.cluster, "rep": rep,
                                "polarity": item.posthoc.get("polarity", ""),
                                "correct": bool(verdict.correct),
                                "parse_failed": bool(verdict.parse_failed),
                                "abstained": bool(verdict.abstained),
                                "parsed": verdict.parsed,
                                # None when the model stated none. Never defaulted to
                                # 0.5, which would fabricate an average judge out of
                                # a model that said nothing.
                                "confidence": confidence,
                                "prompt_tokens": c.prompt_tokens,
                                "completion_tokens": c.completion_tokens,
                                "cached_prompt_tokens": c.cached_prompt_tokens,
                                "latency_s": round(c.latency_s, 3),
                                # 1500 chars allows offline re-grading. 200 chars was insufficient:
                                # it captures the answer for answer-first shapes but truncates it for
                                # reasoning-first ones, creating bias in any regrade.
                                "raw": c.text[:1500],
                                "raw_head": c.text[:200].replace("\n", " "),
                            }) + "\n")
                            fh.flush()
                            ran += 1
                            if ran % 50 == 0:
                                rate = ran / max(1e-9, time.time() - t_start)
                                print(f"  {ran} cells | {rate:.1f}/s | {model} {family} {variant}",
                                      flush=True)
    return ran


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    # Not required: --census must run without naming a model. The reproduce block in
    # PROMPTBENCH_RESULTS_2026-08-19.md prints `runner --census` and it exited 2.
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS))
    p.add_argument("--families", nargs="+", default=["verify", "select"])
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--out", default="agent/idea_test_results/promptbench_runs.jsonl")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=os.environ.get("PROMPTBENCH_API_KEY", "ollama"))
    p.add_argument("--max-cells", type=int, default=100000)
    p.add_argument("--census", action="store_true")
    a = p.parse_args(argv)

    if a.census:
        print(json.dumps(full_census(), indent=2, default=str))
        return 0

    if not a.models:
        print("--models is required unless --census is given", file=sys.stderr)
        return 2

    unknown = set(a.variants) - set(ALL_VARIANTS)
    if unknown:
        print(f"unknown variants: {sorted(unknown)}", file=sys.stderr)
        return 2
    unknown_families = set(a.families) - set(ALL_FAMILIES)
    if unknown_families:
        print(f"unknown families: {sorted(unknown_families)}", file=sys.stderr)
        return 2

    # A calibration family run without a C arm produces zero cells and no error.
    if "calibration" in a.families and not set(a.variants) & set(CALIBRATION_VARIANTS + ("SHIPPED",)):
        print("family 'calibration' needs at least one of "
              f"{list(CALIBRATION_VARIANTS)} or SHIPPED", file=sys.stderr)
        return 2

    n = run(a.models, a.variants, a.families, a.reps, Path(a.out),
            a.base_url, a.api_key, a.max_cells)
    print(f"ran {n} cells -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
