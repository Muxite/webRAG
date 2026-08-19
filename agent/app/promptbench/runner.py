"""Run the (model x variant x item) matrix and append one JSONL row per cell.

Resumable: a completed cell key is never re-run, so a crash or an interrupt
costs wall clock but never re-spends money. Cells are ordered
smallest-model-first because OLLAMA_MAX_LOADED_MODELS=1 makes every model swap
expensive, and a swap per cell would dominate the run.

Grading happens here, and it is the one place the Label is legitimately opened
-- via ``.expose(reason)``, which returns an ORACLE-tainted Signal. The prompt
builder never sees it.
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
from agent.app.promptbench.factors import ALL_VARIANTS, PRIMARY_VARIANTS, build_prompt
from agent.app.promptbench.grade import grade_enum
from agent.app.promptbench.http_llm import HttpLLM
from agent.app.promptbench.items import build_select_items, build_verify_items, census, load_specs

OLLAMA_URL = "http://127.0.0.1:11435/v1"
ABSTAIN = ("INSUFFICIENT", "UNKNOWN", "UNCLEAR")


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


def _families(names: Iterable[str]) -> Dict[str, List[Item]]:
    specs = load_specs()
    all_families = {"verify": build_verify_items(specs), "select": build_select_items(specs)}
    return {k: v for k, v in all_families.items() if k in set(names)}


def run(models: List[str], variants: List[str], families: List[str], reps: int,
        out_path: Path, base_url: str, api_key: str, max_cells: int) -> int:
    fam_items = _families(families)
    done = _load_done(out_path)
    llm = HttpLLM(base_url, api_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    planned = sum(len(items) for items in fam_items.values()) * len(variants) * len(models) * reps
    print(f"planned cells: {planned} | already done: {len(done)} | cap: {max_cells}", flush=True)

    ran = 0
    t_start = time.time()
    with out_path.open("a") as fh:
        for model in models:
            for family, items in fam_items.items():
                for variant in variants:
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
                            verdict = grade_enum(c.text, truth.value,
                                                 item.runtime["choices"], abstain_choices=list(ABSTAIN))
                            fh.write(json.dumps({
                                "cell": key, "model": model, "family": family,
                                "variant": variant, "item_id": item.item_id,
                                "cluster": item.cluster, "rep": rep,
                                "polarity": item.posthoc.get("polarity", ""),
                                "correct": bool(verdict.correct),
                                "parse_failed": bool(verdict.parse_failed),
                                "abstained": bool(verdict.abstained),
                                "parsed": verdict.parsed,
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
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS))
    p.add_argument("--families", nargs="+", default=["verify", "select"])
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--out", default="agent/idea_test_results/promptbench_runs.jsonl")
    p.add_argument("--base-url", default=OLLAMA_URL)
    p.add_argument("--api-key", default=os.environ.get("PROMPTBENCH_API_KEY", "ollama"))
    p.add_argument("--max-cells", type=int, default=100000)
    p.add_argument("--census", action="store_true")
    a = p.parse_args(argv)

    if a.census:
        print(json.dumps(census(load_specs()), indent=2))
        return 0

    unknown = set(a.variants) - set(ALL_VARIANTS)
    if unknown:
        print(f"unknown variants: {sorted(unknown)}", file=sys.stderr)
        return 2

    n = run(a.models, a.variants, a.families, a.reps, Path(a.out),
            a.base_url, a.api_key, a.max_cells)
    print(f"ran {n} cells -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
