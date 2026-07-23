#!/usr/bin/env python3
"""Parallel compute-ladder A/B driver for the NATIVE adaptive engine (LIVE $).

Question: can we get materially better reasoning out of ONE cheap "bad" model by burning more of
its (cheap) tokens/searches — i.e. does an adaptive agent that re-expands, re-grounds on low
confidence, and self-consistency-votes beat the same model run bare? And how close does it get to a
premium model at a fraction of the cost?

Design (all decided with the caller):
  * ONE cheap agent model (default openai/gpt-5-mini), execution_variant=graph, in three arms that
    spend progressively more compute:
        baseline      — adaptive OFF (bare cheap model)
        good_adaptive — re-expansion + confidence-driven re-grounding + corrective context
        full          — good_adaptive + k-vote x3 + backtrack + expect-contract (max searches/tokens)
  * A premium REFERENCE bar (default google/gemini-3.1-pro-preview + sequential_react) — the agent
    never uses it; it is only the quality ceiling we compare the cheap ladder against.
  * 8 tasks spanning all 4 adaptive archetypes incl. the D re-expansion flagship; R=5 for the cheap
    ladder, R=3 for the reference.

Why a bespoke driver rather than native_ab_run.sh: that driver is serial (one cell at a time,
~14h for this matrix). Here every cell is an ISOLATED process at IDEA_TEST_CONCURRENCY=1 (clean
per-run cost/score attribution — no shared-connector cross-talk), but up to JOBS run at once for
wall-clock. Fairness rules are preserved: connector-retry ON in every arm, fixtures OFF, and arms
of the same (task,rep) are enqueued adjacently so they share the same network window; the analyzer
pairs on (task,rep). A hard global BUDGET stops enqueueing, and a per-run USD ceiling guards runaways.

Usage (from repo root):
  PYTHONPATH=services:services/agent ./.venv/bin/python scripts/adaptive_ladder_run.py \
      --run-id ladder --jobs 6 --budget 22 --reps 5 --ref-reps 3
Resume-safe: cells whose result JSON already exists are skipped.
"""
import argparse, glob, json, os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

REPO = "/home/muk/projects/webRAG"
RESULTS_DIR = f"{REPO}/services/agent/idea_test_results"
LADDER_ARMS = ["baseline", "good_adaptive", "full"]

# Chroma isolation/embedding config for this run, populated in main() from CLI args.
# embedded mode gives each cell subprocess its OWN SQLite file (no cross-subprocess
# write-lock contention) and runs embedding off the loop via to_thread; embed_device
# 'cuda'/'auto' pushes embedding to the GPU. Defaults chosen for the barrage relaunch.
RUN_CFG = {"chroma_mode": "embedded", "embed_device": "auto", "embedded_root": None}


def cell_db_path(cell):
    """Unique per-cell embedded-chroma dir (fresh memory per run_id+task)."""
    return os.path.join(RUN_CFG["embedded_root"], f"{cell['run_id']}_{cell['task']}")

# Named task sets (see services/agent/app/BENCHMARK_SUITE_50.md). All ids are validated + deduped.
TASK_SETS = {
    # the original 8-task smoke across 4 archetypes
    "smoke8": ["122", "125", "128", "130", "134", "138", "140", "144"],
    # Tier A: the 24 adaptive-targeted core (4 archetypes x 6), all grounding-gated
    "core24": [f"{n:03d}" for n in range(122, 146)],
    # The full validated + deduped 50-task suite (Tier A core 24 + B coverage 15 + C depth 11)
    "suite50": (
        [f"{n:03d}" for n in range(122, 146)]                                   # Tier A (24)
        + ["049", "055", "059", "060", "067", "072", "075", "041", "062",       # Tier B (15)
           "024", "044", "093", "046", "047", "073"]
        + ["068", "095", "108", "040", "054", "065", "042", "056", "061",       # Tier C (11)
           "070", "090"]
    ),
}
TASKS = TASK_SETS["smoke8"]  # default; overridden by --task-set / --tasks


def keyval(name):
    with open(f"{REPO}/services/keys.env") as fh:
        for line in fh:
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('\r\n').strip('"')
    return ""


def base_env():
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": "services:services/agent",
        "OPENROUTER_API_KEY": keyval("OPENROUTER_API_KEY"),
        "SEARCH_API_KEY": keyval("SEARCH_API_KEY"),
        "LLM_PROVIDER": "openrouter",
        "MODEL_API_URL": "https://openrouter.ai/api/v1",
        "CHROMA_URL": "http://localhost:8001",
        "DEFAULT_TIMEOUT": "45", "DEFAULT_DELAY": "2", "JITTER_SECONDS": "0.5",
        # fixed harness controls (identical to native_ab_run.sh) ---------------------------------
        "IDEA_TEST_CONCURRENCY": "1",            # isolate: one run per process
        "IDEA_TEST_PARALLEL_ACTION_LIMIT": "1",  # mandatory for wide-breadth tasks
        "IDEA_TEST_FIXTURES": "off",             # native engine explores variably
        "IDEA_TEST_CONNECTOR_RETRY": "1",        # infra-fairness in EVERY arm
        "IDEA_TEST_EFFORT_TIERS": "0",
        "IDEA_TEST_PREFLIGHT_JSON_TOKENS": "4096",  # reasoning models pass the JSON gate
        "IDEA_TEST_RUNS": "1",
        "IDEA_TEST_USD_CEILING": "0.60",         # per-RUN runaway guard (global cap is in this driver)
    })
    return env


def cell_env(cell):
    env = base_env()
    env["IDEA_TEST_IDS"] = cell["task"]
    env["IDEA_TEST_MODELS"] = cell["model"]
    env["IDEA_TEST_EXECUTION_VARIANTS"] = cell["variant"]
    env["IDEA_TEST_RUN_ID"] = cell["run_id"]
    # Grading is deterministic grep (no LLM judge), so the validation model is only preflighted.
    # Point it at the cell's own model so no SEPARATE (e.g. expensive gpt-5-mini) model is probed —
    # a mismatched validation model's preflight can 402/abort the whole run (learned the hard way).
    env["IDEA_TEST_VALIDATION_MODEL"] = cell["model"]
    if cell["arm"]:
        env["IDEA_TEST_ARM"] = cell["arm"]
    else:
        env.pop("IDEA_TEST_ARM", None)
    # Chroma isolation + embedding device. In embedded mode each cell gets its own
    # SQLite file (no shared-server contention) at a unique path; embedding runs on the
    # chosen device (GPU under 'cuda'/'auto').
    env["CHROMA_MODE"] = RUN_CFG["chroma_mode"]
    env["CHROMA_EMBED_DEVICE"] = RUN_CFG["embed_device"]
    if RUN_CFG["chroma_mode"] == "embedded":
        env["CHROMA_EMBEDDED_PATH"] = cell_db_path(cell)
    # per-cell "burn" env (e.g. deepseek's full arm: extra k-vote + deeper re-expansion)
    for k, v in (cell.get("burn") or {}).items():
        env[k] = str(v)
    return env


def result_files(run_id, task):
    fs = [f for f in glob.glob(f"{RESULTS_DIR}/{run_id}_{task}_*.json")
          if not f.endswith("_summary.json")]
    return fs


def cell_usd_and_score(run_id, task):
    """Best-effort (usd, score) for a completed cell — mirrors adaptive_ab_analyze._obs."""
    usd = score = None
    for f in result_files(run_id, task):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ex = d.get("execution", {})
        ob = ex.get("output", {}).get("observability") or ex.get("observability") or {}
        usd = (ob.get("cost", {}) or {}).get("usd")
        score = d.get("validation", {}).get("overall_score")
    return usd, score


# Multi-model AXIS (decided 2026-07-22): two cheap agent LADDERS + one strong REACT-only reference,
# all in ONE driver process so the per-task lock prevents same-task chroma-memory collisions ACROSS
# models. run_id encodes a model tag so the analyzer pairs within a model:
#   {RUN_ID}_{tag}_{arm}_rep{r}  (ladder)   /   {RUN_ID}_{tag}_ref_rep{r}  (reference)
AXES = {
    "final3": {
        "ladders": [
            {"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 5, "burn": None},
            {"model": "deepseek/deepseek-v4-flash", "tag": "ds", "reps": 5,
             # "ridiculous burn" on deepseek's full arm only (output is $0.196/1M): k-vote 3→5, reexpand 2→3
             "burn": {"IDEA_TEST_NATIVE_VOTE_K": 5, "IDEA_TEST_GOT_REEXPAND_MAX_ITER": 3}},
        ],
        "reference": {"model": "anthropic/claude-sonnet-5", "tag": "sonnet",
                      "variant": "sequential_react", "reps": 3},
    },
}


def build_cells(run_id, reps, ref_reps, model, ref_model, ref_variant, include_ref, tasks):
    """Single-model ladder + optional reference (legacy/simple mode)."""
    cells = []
    for task in tasks:
        for rep in range(1, reps + 1):
            for arm in LADDER_ARMS:
                cells.append({"task": task, "rep": rep, "arm": arm, "model": model,
                              "variant": "graph", "burn": None, "run_id": f"{run_id}_{arm}_rep{rep}"})
    if include_ref:
        for task in tasks:
            for rep in range(1, ref_reps + 1):
                cells.append({"task": task, "rep": rep, "arm": None, "model": ref_model,
                              "variant": ref_variant, "burn": None,
                              "run_id": f"{run_id}_reference_rep{rep}"})
    return cells


def build_axis_cells(run_id, axis, tasks):
    """Full multi-model axis. Ordered task-major so same-task cells are adjacent (per-task lock +
    shared window). burn applies only to the 'full' arm of a ladder that declares it."""
    cells = []
    for task in tasks:
        for lad in axis["ladders"]:
            for rep in range(1, lad["reps"] + 1):
                for arm in LADDER_ARMS:
                    cells.append({"task": task, "rep": rep, "arm": arm, "model": lad["model"],
                                  "variant": "graph",
                                  "burn": lad.get("burn") if arm == "full" else None,
                                  "run_id": f"{run_id}_{lad['tag']}_{arm}_rep{rep}"})
        ref = axis.get("reference")
        if ref:
            for rep in range(1, ref["reps"] + 1):
                cells.append({"task": task, "rep": rep, "arm": None, "model": ref["model"],
                              "variant": ref["variant"], "burn": None,
                              "run_id": f"{run_id}_{ref['tag']}_ref_rep{rep}"})
    return cells


def run_cell(cell):
    if result_files(cell["run_id"], cell["task"]):
        return cell, "skip", 0.0, None, 0.0  # already have a result → resume-safe
    t0 = time.time()
    try:
        # 1800s (was 1200): the 'full' arm's longest runs are its highest-scorers; a too-tight cap
        # dropped them and biased 'full' downward (survivorship). Give heavy runs room to finish.
        proc = subprocess.run([f"{REPO}/.venv/bin/python", "-m", "agent.app.idea_test_runner"],
                              cwd=REPO, env=cell_env(cell), stdout=subprocess.DEVNULL,
                              stderr=subprocess.STDOUT, timeout=1800)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    dt = time.time() - t0
    usd, score = cell_usd_and_score(cell["run_id"], cell["task"])
    status = "ok" if rc == 0 else f"{rc}"
    # Reclaim the per-cell embedded DB (fresh memory is per-run; results already scraped).
    if RUN_CFG["chroma_mode"] == "embedded" and RUN_CFG["embedded_root"]:
        shutil.rmtree(cell_db_path(cell), ignore_errors=True)
    return cell, status, (usd or 0.0), score, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="ladder")
    ap.add_argument("--jobs", type=int, default=8, help="max parallel cells; effective cap = #distinct tasks (per-task serialized)")
    ap.add_argument("--budget", type=float, default=22.0, help="hard global USD stop (stops enqueueing)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ref-reps", type=int, default=3)
    ap.add_argument("--model", default="openai/gpt-5-mini")
    ap.add_argument("--ref-model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--ref-variant", default="sequential_react")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--task-set", default="smoke8", choices=list(TASK_SETS),
                    help="named validated task set (see BENCHMARK_SUITE_50.md)")
    ap.add_argument("--tasks", default="", help="explicit space/comma task ids; overrides --task-set")
    ap.add_argument("--axis", default="", choices=[""] + list(AXES),
                    help="multi-model axis (e.g. 'final3'); overrides --model/--ref-* single-model mode")
    ap.add_argument("--chroma-mode", default="embedded", choices=["http", "embedded"],
                    help="embedded = per-cell isolated SQLite (no shared-server contention); http = shared server")
    ap.add_argument("--embed-device", default="auto", choices=["cpu", "cuda", "auto"],
                    help="chroma embedding device; auto = GPU if available else CPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RUN_CFG["chroma_mode"] = args.chroma_mode
    RUN_CFG["embed_device"] = args.embed_device

    tasks = ([t for t in args.tasks.replace(",", " ").split()] if args.tasks
             else TASK_SETS[args.task_set])
    if args.axis:
        cells = build_axis_cells(args.run_id, AXES[args.axis], tasks)
    else:
        cells = build_cells(args.run_id, args.reps, args.ref_reps, args.model,
                            args.ref_model, args.ref_variant, not args.no_ref, tasks)
    out_dir = f"{RESULTS_DIR}/_{args.run_id}"
    os.makedirs(out_dir, exist_ok=True)
    RUN_CFG["embedded_root"] = f"{out_dir}/_chroma"
    if RUN_CFG["chroma_mode"] == "embedded":
        os.makedirs(RUN_CFG["embedded_root"], exist_ok=True)
    logpath = f"{out_dir}/driver.log"
    log = open(logpath, "a")

    def emit(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    n_done = sum(1 for c in cells if result_files(c["run_id"], c["task"]))
    emit(f"run_id={args.run_id} cells={len(cells)} (already-done={n_done}) jobs={args.jobs} "
         f"budget=${args.budget:.2f} tasks={len(tasks)} chroma={RUN_CFG['chroma_mode']}/{RUN_CFG['embed_device']} "
         f"{'axis='+args.axis if args.axis else 'model='+args.model}")
    from collections import Counter
    by_mc = Counter((c["model"].split("/")[-1], c["arm"] or "ref") for c in cells)
    for (m, arm), n in sorted(by_mc.items()):
        emit(f"  {m:<24} {arm:<14} {n} cells")
    if args.dry_run:
        emit("dry-run — not executing"); return

    # Per-task serialization: at most ONE in-flight cell per task, because the engine's working
    # memory lives in a chroma collection keyed by mandate hash (idea_engine._memo_namespace →
    # mem_<sha256(mandate)>), which is shared across arms/reps of the SAME task. Running two same-task
    # cells at once would cross-contaminate that store. Different tasks are fully independent, so we
    # still parallelize across the 8 tasks — and arms of a given (task,rep) run back-to-back, which
    # preserves the shared-network-window pairing the analyzer relies on.
    spent = 0.0
    done = 0
    stopped = False
    pending = list(cells)
    busy_tasks = set()
    futs = {}  # future -> cell

    def fill(ex):
        for c in list(pending):
            if len(futs) >= args.jobs:
                break
            if c["task"] in busy_tasks:
                continue
            pending.remove(c)
            busy_tasks.add(c["task"])
            futs[ex.submit(run_cell, c)] = c

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        fill(ex)
        while futs:
            finished, _ = wait(set(futs.keys()), return_when=FIRST_COMPLETED)
            for fut in finished:
                cell = futs.pop(fut)
                busy_tasks.discard(cell["task"])
                _, status, usd, score, dt = fut.result()
                if status != "skip":
                    spent += usd
                    done += 1
                sc = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
                tag = cell["arm"] or "reference"
                emit(f"  task={cell['task']} rep={cell['rep']} {tag:<13} "
                     f"{status:<6} score={sc:<5} ${usd:.3f} {dt:>5.0f}s  | done={done}/{len(cells)} "
                     f"inflight={len(futs)} spent≈${spent:.2f}")
                if spent >= args.budget and not stopped:
                    stopped = True
                    emit(f"!! BUDGET ${args.budget:.2f} reached (spent≈${spent:.2f}) — no new cells enqueued")
            if not stopped:
                fill(ex)
    emit(f"DONE — executed {done} cells, spent≈${spent:.2f}, {len(pending)} unstarted. "
         f"Analyze with adaptive_ab_analyze.py")


if __name__ == "__main__":
    main()
