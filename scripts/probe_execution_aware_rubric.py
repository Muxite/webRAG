#!/usr/bin/env python3
"""Does an execution-aware batch rubric differentiate ALREADY-EXECUTED siblings?

ASSUMPTION_AUDIT.md T1-3a/b left one live wire: with `evaluate_parallel_siblings` on, the
code cap stops binding and the judge still returns 0.2 for every executed sibling, because
`evaluation_batch_system_prompt` says *"Nodes with actions but no action_result score <=0.2"*
and its planning addendum repeats the order as MANDATORY. The sentence was written for work
that has not run; nothing in it tells the judge what to do with work that HAS run.

This script tests the prompt in isolation, off the engine, on recorded batches:

* **Corpus.** Sibling batches from `agent/idea_test_results` with >=2 executed non-merge
  children. Prompts are rebuilt the way `LlmEvaluationPolicy._build_messages` builds them
  (same path serialization, same `evaluation_max_detail_chars` truncation, same user
  template), with the recorded `evaluation` key stripped from every node's details -- it is
  written *after* scoring, so leaving it in would show the judge the answer.
* **Arms.** OLD is the shipped pair of texts. NEW_A rewrites both to condition on execution
  state. NEW_B adds an explicit anti-tie clause on top of NEW_A. Splitting the two keeps
  "the rubric line was the binding constraint" separable from "the judge only spreads when
  ordered to".
* **Natural condition.** Differentiation on the batch as recorded: flat rate, spread,
  distinct values, how much of the mass still lands in the `<=0.2` band.
* **Spiked condition.** One candidate per batch (seeded choice) has its `action_result`
  replaced by a failure payload. Ground truth then exists: that candidate should be the
  uniquely lowest-scored one. Chance is 1/k, so this separates real discrimination from
  spread-for-its-own-sake, which the natural condition alone cannot do.

Analysis and probe only. Writes JSONL to `--out` and prints; nothing the engine reads.
Live LLM calls, so it takes `--max-usd` and stops before crossing it.

Usage::

    set -a; source services/keys.env; set +a
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/probe_execution_aware_rubric.py \\
      --batches 36 --model gpt-4.1-nano --max-usd 0.60 --out /tmp/rubric_probe.jsonl
    ./.venv/bin/python scripts/probe_execution_aware_rubric.py --analyze /tmp/rubric_probe.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_DEFAULT = REPO_ROOT / "agent" / "idea_test_results"
SETTINGS_PATH = REPO_ROOT / "agent" / "app" / "idea_dag_settings.json"

DEFAULT_SEED = 20260820
#: gpt-4.1-nano list price, USD per token. Only used for the spend guard.
USD_PER_PROMPT_TOKEN = 0.10 / 1_000_000
USD_PER_COMPLETION_TOKEN = 0.40 / 1_000_000

#: What replaces a spiked candidate's result. Shaped like a real failed search/visit so the
#: judge sees a plausible failure rather than a malformed node.
SPIKE_RESULT = {
    "success": False,
    "error": "request failed: no usable content returned",
    "results": [],
    "count": 0,
    "content": "",
}

# --- the three rubric arms -------------------------------------------------------------------

OLD_SYSTEM = (
    "You are the Score operation in a Graph-of-Thought system. Rate all candidates 0-1. "
    "Reward: independent subproblems, evidence-gathering (search+visit). Penalize: redundancy, "
    "tiny steps, skipping visit after search. Nodes with actions but no action_result score "
    '<=0.2. Return JSON: {"scores": [{"id": string, "score": float}, ...]}. '
    "Use candidate 'id' (1,2,3...) not node_id."
)
OLD_ADDENDUM = (
    "Prefer candidates that: (1) gather verifiable evidence, (2) define explicit output fields, "
    "(3) include source validation or cross-checking, and (4) reduce hallucination risk. For "
    "fact-checking or claim-verification mandates, reward a 'verify' candidate that cross-checks "
    "a claim against gathered evidence and cites a contradicting authoritative source. Penalize "
    "vague steps without concrete evidence collection plans. MANDATORY: If a node has an action "
    "(search/visit) but no action_result, score it ≤0.2 - it represents incomplete/unexecuted "
    "work."
)

#: Execution-aware rewrite. Same two sentences, split by execution state.
NEW_SYSTEM = (
    "You are the Score operation in a Graph-of-Thought system. Rate all candidates 0-1. "
    "Reward: independent subproblems, evidence-gathering (search+visit). Penalize: redundancy, "
    "tiny steps, skipping visit after search. A candidate with an action and NO action_result "
    "has not run yet: score it <=0.2. A candidate that HAS an action_result has already run: "
    "score it on what that result contains - usable on-goal evidence, specific facts, working "
    "URLs, relevant page text score high; empty, failed, off-goal or duplicate results score "
    'low. Return JSON: {"scores": [{"id": string, "score": float}, ...]}. '
    "Use candidate 'id' (1,2,3...) not node_id."
)
NEW_ADDENDUM = (
    "Prefer candidates that: (1) gather verifiable evidence, (2) define explicit output fields, "
    "(3) include source validation or cross-checking, and (4) reduce hallucination risk. For "
    "fact-checking or claim-verification mandates, reward a 'verify' candidate that cross-checks "
    "a claim against gathered evidence and cites a contradicting authoritative source. Penalize "
    "vague steps without concrete evidence collection plans. MANDATORY: If a node has an action "
    "(search/visit) but no action_result, score it ≤0.2 - it represents unexecuted work. If a "
    "node HAS an action_result, that work is already done and the ≤0.2 rule does not apply to "
    "it: score it across the full 0-1 range on how much usable, on-goal evidence the result "
    "actually contains."
)
#: NEW_A plus one anti-tie clause, to separate "the rule was binding" from "spread on command".
ANTI_TIE = (
    " Executed candidates whose results differ in usefulness must receive different scores; "
    "give two executed candidates the same score only when their results are genuinely "
    "equivalent."
)

ARMS: Dict[str, Tuple[str, str]] = {
    "OLD": (OLD_SYSTEM, OLD_ADDENDUM),
    "NEW_A": (NEW_SYSTEM, NEW_ADDENDUM),
    "NEW_B": (NEW_SYSTEM, NEW_ADDENDUM + ANTI_TIE),
}
CONDITIONS = ("natural", "spiked")


def parity_check() -> List[str]:
    """OLD must still be byte-identical to what the engine ships, or the A/B is against a ghost."""
    problems: List[str] = []
    try:
        settings = json.loads(SETTINGS_PATH.read_text())
    except OSError as exc:
        return [f"cannot read {SETTINGS_PATH}: {exc}"]
    shipped_system = str(settings.get("evaluation_batch_system_prompt") or "").replace("{{", "{").replace("}}", "}")
    shipped_addendum = str(settings.get("evaluation_planning_addendum") or "")
    if shipped_system != OLD_SYSTEM:
        problems.append("evaluation_batch_system_prompt drifted from OLD_SYSTEM")
    if shipped_addendum != OLD_ADDENDUM:
        problems.append("evaluation_planning_addendum drifted from OLD_ADDENDUM")
    return problems


# --- corpus ----------------------------------------------------------------------------------


@dataclass
class ProbeBatch:
    source: str
    parent_id: str
    parent_goal: str
    path: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]          # {node_id, title, details, action, degenerate}
    recorded_scores: List[Optional[float]] = field(default_factory=list)

    @property
    def key(self) -> str:
        blob = json.dumps([self.parent_goal, [c["title"] for c in self.candidates]], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _strip_evaluation(details: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the score the engine wrote after judging; at scoring time it does not exist."""
    return {k: v for k, v in details.items() if k != "evaluation"}


def _is_degenerate(action_result: Any) -> bool:
    """Did this executed action come back with nothing usable? The offline quality proxy."""
    if not isinstance(action_result, dict):
        return True
    if action_result.get("success") is False:
        return True
    if action_result.get("error"):
        return True
    results = action_result.get("results")
    if isinstance(results, list) and not results and action_result.get("count") in (0, None):
        content = action_result.get("content") or action_result.get("content_full") or ""
        if len(str(content)) < 500:
            return True
    content = action_result.get("content")
    if content is not None and len(str(content)) < 200 and not results:
        return True
    return False


def batches_from_result(payload: Dict[str, Any], source: str, max_detail_chars: int,
                        max_context_nodes: int, min_executed: int,
                        max_candidates: int) -> List[ProbeBatch]:
    execution = payload.get("execution") if isinstance(payload, dict) else None
    graph = execution.get("graph") if isinstance(execution, dict) else None
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict):
        return []

    def ancestry(node_id: str) -> List[str]:
        chain, seen, cur = [], set(), node_id
        while cur and cur in nodes and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = (nodes[cur] or {}).get("parent_id")
        return chain

    out: List[ProbeBatch] = []
    for parent_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        children = [c for c in (node.get("children") or []) if isinstance(nodes.get(c), dict)]
        executed = [
            c for c in children
            if (nodes[c].get("details") or {}).get("action") not in (None, "merge")
            and (nodes[c].get("details") or {}).get("action_result") is not None
        ]
        if len(executed) < min_executed:
            continue
        executed = executed[:max_candidates]
        path = []
        for anc_id in ancestry(parent_id)[:max_context_nodes]:
            anc = nodes[anc_id]
            details = json.dumps(_strip_evaluation(anc.get("details") or {}),
                                 ensure_ascii=True, default=str)[:max_detail_chars]
            path.append({
                "node_id": anc_id,
                "title": anc.get("title") or "",
                "status": anc.get("status") or "",
                "score": anc.get("score"),
                "details": details,
            })
        candidates = []
        for child_id in executed:
            child = nodes[child_id]
            details = _strip_evaluation(child.get("details") or {})
            candidates.append({
                "node_id": child_id,
                "title": child.get("title") or "",
                "action": details.get("action"),
                "details_obj": details,
                "degenerate": _is_degenerate(details.get("action_result")),
            })
        out.append(ProbeBatch(
            source=source,
            parent_id=parent_id,
            parent_goal=(node.get("details") or {}).get("parent_goal") or node.get("title") or "",
            path=path,
            candidates=candidates,
            recorded_scores=[nodes[c].get("score") for c in executed],
        ))
    return out


def load_batches(results_dir: Path, pattern: str, max_detail_chars: int, max_context_nodes: int,
                 min_executed: int, max_candidates: int, limit_files: Optional[int] = None
                 ) -> List[ProbeBatch]:
    out: List[ProbeBatch] = []
    paths = sorted(results_dir.glob(pattern))
    if limit_files:
        paths = paths[:limit_files]
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001. A corrupt or non-result JSON is simply skipped
            continue
        out.extend(batches_from_result(payload, path.name, max_detail_chars, max_context_nodes,
                                       min_executed, max_candidates))
    return out


def sample_batches(batches: Sequence[ProbeBatch], n: int, seed: int) -> List[ProbeBatch]:
    """One batch per (source-prefix, shape) key first, so one noisy run cannot own the sample."""
    unique: Dict[str, ProbeBatch] = {}
    for batch in batches:
        unique.setdefault(batch.key, batch)
    pool = list(unique.values())
    rng = random.Random(seed)
    rng.shuffle(pool)
    by_prefix: Dict[str, List[ProbeBatch]] = defaultdict(list)
    for batch in pool:
        by_prefix[batch.source.split("_")[0]].append(batch)
    picked: List[ProbeBatch] = []
    while len(picked) < n and any(by_prefix.values()):
        for prefix in sorted(by_prefix):
            if not by_prefix[prefix]:
                continue
            picked.append(by_prefix[prefix].pop())
            if len(picked) >= n:
                break
    return picked


# --- prompt construction ---------------------------------------------------------------------


def build_messages(batch: ProbeBatch, arm: str, condition: str, spike_index: int,
                   max_detail_chars: int, user_template: str) -> Tuple[str, str]:
    system, addendum = ARMS[arm]
    system = f"{system}\n\n{addendum}"
    candidates = []
    for idx, cand in enumerate(batch.candidates, start=1):
        details = dict(cand["details_obj"])
        if condition == "spiked" and idx - 1 == spike_index:
            spiked = dict(SPIKE_RESULT)
            spiked["action"] = details.get("action")
            details["action_result"] = spiked
        text = json.dumps(details, ensure_ascii=True, default=str)[:max_detail_chars]
        candidates.append({"id": str(idx), "title": cand["title"], "details": text})
    user = user_template.format(
        path_json=json.dumps(batch.path, ensure_ascii=True),
        parent_id=batch.parent_id,
        parent_goal=batch.parent_goal,
        candidates_json=json.dumps(candidates, ensure_ascii=True),
    )
    return system, user


def parse_scores(text: str, n: int) -> Dict[str, float]:
    if not text:
        return {}
    blob = text.strip()
    start, end = blob.find("{"), blob.rfind("}")
    if start >= 0 and end > start:
        blob = blob[start:end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    out: Dict[str, float] = {}
    for item in (data.get("scores") or []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or item.get("node_id") or "")
        try:
            value = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if key in {str(i) for i in range(1, n + 1)}:
            out[key] = value
    return out


# --- run ---------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    from agent.app.promptbench.http_llm import HttpLLM

    settings = json.loads(SETTINGS_PATH.read_text())
    max_detail_chars = int(settings.get("evaluation_max_detail_chars", 5000))
    max_context_nodes = int(settings.get("evaluation_max_context_nodes", 5))
    max_candidates = int(settings.get("evaluation_batch_max_candidates", 5))
    user_template = settings.get("evaluation_batch_user_prompt") or (
        '{{"path": {path_json}, "parent_id": "{parent_id}", "candidates": {candidates_json}}}'
    )

    drift = parity_check()
    if drift and not args.allow_drift:
        print("PARITY FAILURE (pass --allow-drift to override):", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        return 2

    batches = load_batches(Path(args.results_dir), args.pattern, max_detail_chars,
                           max_context_nodes, args.min_executed, max_candidates, args.limit_files)
    picked = sample_batches(batches, args.batches, args.seed)
    print(f"corpus: {len(batches)} executed sibling batches | probing {len(picked)}", flush=True)
    if not picked:
        return 1

    # Stripped: services/keys.env is CRLF, so a sourced value carries a trailing \r that
    # makes the Authorization header invalid.
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("OPENAI_API_KEY is unset; source services/keys.env first", file=sys.stderr)
        return 2
    llm = HttpLLM(args.base_url, api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A cell that only ever errored is NOT done: a 429 must be retried on the next
            # pass, or a rate-limited run silently becomes a smaller sample.
            if "cell" in row and "error" not in row:
                done.add(row["cell"])

    rng = random.Random(args.seed)
    spike_for = {b.key: rng.randrange(len(b.candidates)) for b in picked}
    spent = 0.0
    calls = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for condition in CONDITIONS:
            if args.conditions and condition not in args.conditions:
                continue
            for batch in picked:
                for arm, rep in [(a, r) for a in ARMS for r in range(args.reps)]:
                    if args.arms and arm not in args.arms:
                        continue
                    # rep 0 keeps the original cell key, so a resumed run reuses the cells
                    # already paid for instead of re-placing them under a renamed key.
                    cell = (f"{args.model}|{arm}|{condition}|{batch.key}"
                            + (f"|r{rep}" if rep else ""))
                    if cell in done:
                        continue
                    if spent >= args.max_usd:
                        print(f"USD CEILING {args.max_usd} reached after {calls} calls "
                              f"(${spent:.4f}); stopping cleanly", flush=True)
                        return 0
                    system, user = build_messages(batch, arm, condition, spike_for[batch.key],
                                                  max_detail_chars, user_template)
                    completion = None
                    last_exc: Optional[Exception] = None
                    # These prompts are ~6k tokens each, so a serial run still trips the
                    # per-minute token limit; back off rather than dropping the cell, which
                    # would shrink the sample in a way correlated with prompt size.
                    for attempt in range(args.retries + 1):
                        try:
                            completion = llm.complete(user, model=args.model,
                                                      temperature=args.temperature,
                                                      max_tokens=args.max_tokens, system=system)
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_exc = exc
                            if attempt < args.retries:
                                time.sleep(args.backoff * (2 ** attempt))
                    if completion is None:
                        fh.write(json.dumps({"cell": cell,
                                             "error": f"{type(last_exc).__name__}: {last_exc}",
                                             "arm": arm, "condition": condition,
                                             "batch": batch.key}) + "\n")
                        fh.flush()
                        continue
                    spent += (completion.prompt_tokens * USD_PER_PROMPT_TOKEN
                              + completion.completion_tokens * USD_PER_COMPLETION_TOKEN)
                    calls += 1
                    scores = parse_scores(completion.text, len(batch.candidates))
                    fh.write(json.dumps({
                        "cell": cell,
                        "arm": arm,
                        "rep": rep,
                        "temperature": args.temperature,
                        "condition": condition,
                        "batch": batch.key,
                        "source": batch.source,
                        "parent_id": batch.parent_id,
                        "width": len(batch.candidates),
                        "actions": [c["action"] for c in batch.candidates],
                        "degenerate": [bool(c["degenerate"]) for c in batch.candidates],
                        "spike_index": spike_for[batch.key],
                        "recorded_scores": batch.recorded_scores,
                        "scores": [scores.get(str(i + 1)) for i in range(len(batch.candidates))],
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "usd_running": round(spent, 5),
                        "raw": completion.text[:600],
                    }) + "\n")
                    fh.flush()
                    if calls % 10 == 0:
                        print(f"  {calls} calls | ${spent:.4f} | {time.time() - t0:.0f}s",
                              flush=True)
    print(f"ran {calls} calls, ${spent:.4f} -> {out_path}")
    return 0


# --- analysis -------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _binom_p(successes: int, n: int, p0: float) -> float:
    """Two-sided exact binomial tail. Used for the spiked-detection rate against 1/k chance."""
    if n == 0:
        return float("nan")

    def pmf(k: int) -> float:
        return math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))

    observed = pmf(successes)
    return min(1.0, sum(pmf(k) for k in range(n + 1) if pmf(k) <= observed * (1 + 1e-9)))


def _mcnemar(pairs: Sequence[Tuple[bool, bool]]) -> Tuple[int, int, float]:
    """Exact McNemar on paired booleans; returns (b, c, two-sided p)."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def analyze(paths: Sequence[Path]) -> int:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in row:
                rows.append(row)
    if not rows:
        print("no usable rows")
        return 1

    # The arm tables read rep 0 only; repeats are a noise measurement, not extra sample.
    by_cell: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        if int(row.get("rep", 0)) == 0:
            by_cell[(row["arm"], row["condition"], row["batch"])] = row

    print(f"cells: {len(by_cell)} | batches: {len({k[2] for k in by_cell})}")
    print()
    print("=== differentiation (natural condition) ===")
    print(f"{'arm':7} {'n':>4} {'flat':>6} {'rate':>6} {'spread':>7} {'distinct':>9} "
          f"{'mean':>6} {'<=0.2':>7}")
    natural: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (arm, condition, _), row in by_cell.items():
        if condition == "natural":
            natural[arm].append(row)
    for arm in ARMS:
        rows_a = [r for r in natural.get(arm, []) if all(s is not None for s in r["scores"])]
        if not rows_a:
            continue
        flats = [len(set(r["scores"])) == 1 for r in rows_a]
        spreads = [max(r["scores"]) - min(r["scores"]) for r in rows_a]
        distinct = [len(set(r["scores"])) for r in rows_a]
        allscores = [s for r in rows_a for s in r["scores"]]
        print(f"{arm:7} {len(rows_a):>4} {sum(flats):>6} {_mean([float(f) for f in flats]):>6.3f} "
              f"{_mean(spreads):>7.3f} {_mean([float(d) for d in distinct]):>9.2f} "
              f"{_mean(allscores):>6.3f} "
              f"{_mean([float(s <= 0.2) for s in allscores]):>7.3f}")

    print()
    print("=== paired vs OLD (natural, same batch) ===")
    for arm in ("NEW_A", "NEW_B"):
        pairs, spread_delta = [], []
        for (a, condition, batch), row in by_cell.items():
            if a != arm or condition != "natural":
                continue
            old = by_cell.get(("OLD", "natural", batch))
            if not old or any(s is None for s in row["scores"] + old["scores"]):
                continue
            pairs.append((len(set(old["scores"])) == 1, len(set(row["scores"])) == 1))
            spread_delta.append((max(row["scores"]) - min(row["scores"]))
                                - (max(old["scores"]) - min(old["scores"])))
        if not pairs:
            continue
        b, c, p = _mcnemar(pairs)
        print(f"{arm}: n={len(pairs)} flat-only-OLD={b} flat-only-{arm}={c} McNemar p={p:.4f} "
              f"| mean spread delta {_mean(spread_delta):+.3f}")

    print()
    print("=== degraded-sibling detection (spiked condition) ===")
    print(f"{'arm':7} {'n':>4} {'hit':>5} {'rate':>6} {'chance':>7} {'p':>8} {'flat':>6}")
    for arm in ARMS:
        hits, chance, flats = [], [], []
        for (a, condition, _), row in by_cell.items():
            if a != arm or condition != "spiked":
                continue
            scores = row["scores"]
            if any(s is None for s in scores):
                continue
            idx = row["spike_index"]
            lowest = min(scores)
            unique_low = sum(1 for s in scores if s == lowest) == 1
            hits.append(bool(unique_low and scores[idx] == lowest))
            chance.append(1.0 / len(scores))
            flats.append(len(set(scores)) == 1)
        if not hits:
            continue
        p0 = _mean(chance)
        p = _binom_p(sum(hits), len(hits), p0)
        print(f"{arm:7} {len(hits):>4} {sum(hits):>5} {_mean([float(h) for h in hits]):>6.3f} "
              f"{p0:>7.3f} {p:>8.4f} {_mean([float(f) for f in flats]):>6.3f}")

    print()
    print("=== recorded-degenerate contrast (natural, batches with both kinds) ===")
    for arm in ARMS:
        good, bad = [], []
        for (a, condition, _), row in by_cell.items():
            if a != arm or condition != "natural":
                continue
            flags = row["degenerate"]
            if any(s is None for s in row["scores"]) or len(set(flags)) < 2:
                continue
            for score, is_bad in zip(row["scores"], flags):
                (bad if is_bad else good).append(score)
        if good and bad:
            print(f"{arm:7} substantive {_mean(good):.3f} (n={len(good)}) | "
                  f"degenerate {_mean(bad):.3f} (n={len(bad)}) | "
                  f"delta {_mean(good) - _mean(bad):+.3f}")

    # Test-retest: the same arm, the same batch, a second call. Whatever this says is the
    # floor a prompt-arm difference has to clear before it means anything.
    repeats: Dict[Tuple[str, str, str], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        repeats[(row["arm"], row["condition"], row["batch"])][int(row.get("rep", 0))] = row
    retest = {k: v for k, v in repeats.items() if len(v) > 1}
    if retest:
        print()
        print("=== test-retest (same arm, same batch, repeated call) ===")
        for arm in ARMS:
            deltas, same_flat, identical = [], [], []
            for (a, _, _), reps in retest.items():
                if a != arm:
                    continue
                ordered = [reps[i] for i in sorted(reps)][:2]
                s0, s1 = ordered[0]["scores"], ordered[1]["scores"]
                if any(s is None for s in s0 + s1):
                    continue
                deltas.extend(abs(x - y) for x, y in zip(s0, s1))
                same_flat.append((len(set(s0)) == 1) == (len(set(s1)) == 1))
                identical.append(s0 == s1)
            if deltas:
                print(f"{arm:7} batches={len(identical)} mean |score delta| {_mean(deltas):.3f} "
                      f"| identical score vectors {_mean([float(i) for i in identical]):.3f} "
                      f"| flat-verdict agreement {_mean([float(s) for s in same_flat]):.3f}")

    parse_fail = sum(1 for r in rows if any(s is None for s in r["scores"]))
    tokens = sum(r.get("prompt_tokens", 0) for r in rows)
    ctokens = sum(r.get("completion_tokens", 0) for r in rows)
    usd = tokens * USD_PER_PROMPT_TOKEN + ctokens * USD_PER_COMPLETION_TOKEN
    print()
    print(f"parse failures / partial score sets: {parse_fail}/{len(rows)} | "
          f"tokens {tokens} in / {ctokens} out | ${usd:.4f} at nano list price")
    print("arms by parse failure:", dict(Counter(
        r["arm"] for r in rows if any(s is None for s in r["scores"]))))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", nargs="+", default=None, help="JSONL run files to summarise")
    p.add_argument("--results-dir", default=str(RESULTS_DIR_DEFAULT))
    p.add_argument("--pattern", default="**/*.json")
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--batches", type=int, default=36)
    p.add_argument("--min-executed", type=int, default=2)
    p.add_argument("--conditions", nargs="+", default=None, choices=list(CONDITIONS))
    p.add_argument("--model", default="gpt-4.1-nano")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--max-usd", type=float, default=0.50)
    p.add_argument("--reps", type=int, default=1,
                   help="repeat each cell; measures judge test-retest noise, which bounds "
                        "how large a prompt-arm difference has to be to matter")
    p.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--backoff", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--allow-drift", action="store_true")
    p.add_argument("--out", default="/tmp/rubric_probe.jsonl")
    a = p.parse_args(argv)

    if a.analyze:
        return analyze([Path(x) for x in a.analyze])
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
