# Agent Readiness Checklist

Scaffolding that must be in place (and verified) before the main cost-recovery experiment.
Built in the "Grounded link-following + decision-trace observability" pass (uncommitted).

## Interfaces / classes (the contracts)

| Interface | Where | Purpose |
|---|---|---|
| `MandateRequirements` + `parse_mandate_requirements()` | `idea_policies/mandate_requirements.py` | Single source of truth for what a mandate requires (named URLs, must-visit/search, **navigation**, **grounding**, `nav_targets`). Both the hooks and the grounding gate consume it. |
| `PostExpansionHook` (Protocol) | `idea_policies/post_expansion_hooks.py` | Enforcement extension point. `apply(graph, node_id, step_index, mandate, logger, telemetry=None)`. Hooks must be idempotent. |
| `GroundingResult` + `evaluate_grounding()` | `idea_policies/grounding.py` | Decides whether the answer is backed by actually-visited pages. |
| `DecisionStage` + `TelemetrySession.record_decision()` | `telemetry.py` | Structured thought-process trace (7 stages). Compact always; full rationale at `IDEA_TEST_REPORT_VERBOSITY>=2`. |
| `LeafActionRegistry` (incl. `verify`) | `idea_policies/actions.py` | Action dispatch; `verify` registered + reachable + in `allowed_actions`. |

## Capability — grounded follow-through (soft enforcement)
- [x] `MandateNavigationHook` in the default hook bundle (`default_post_expansion_hooks()`) —
      injects a link-follow visit (`link_idea` = nav target) once a source page is visited.
- [x] `VisitLeafAction._select_links_with_llm` uses the execution model (model_name=None) so a
      descriptive `link_idea` ("rocket that launched the mission") resolves to the right link.
- [x] Soft grounding gate in `idea_engine._grounding_replan` — re-plan up to
      `grounding_max_replans` (default 2), then finalize-but-flag. Hard-capped; cannot hang.
- [x] `final_payload` carries `grounded` + `missing_requirements` + `grounding_replans`.

## Observability — decision trace + grounding
- [x] `record_decision` wired at all 7 stages: EXPANSION + SELECTION + ACTION + FINALIZE
      (`idea_engine.py`), EVALUATION (`evaluation.py`), ENFORCE (`post_expansion_hooks.py`),
      GROUNDING (`idea_engine._grounding_replan`).
- [x] `summarize_observability` exposes a `grounding` block `{grounded, missing, replans}` and a
      `decisions` block `{count, by_stage, trace}` (`testing/utils.py`).
- [x] `scripts/level_ladder.py` reports the **real** grounded verdict (falls back to the
      validation proxy only for legacy results).

## Settings / env
- `grounding_max_replans` (default 2) in `idea_dag_settings.json`.
- `IDEA_NAV_LINK_COUNT` (default 3) — candidate links the follow-up visit considers.
- Benchmark still requires `IDEA_TEST_CONCURRENCY=1`, `IDEA_TEST_PARALLEL_ACTION_LIMIT=1`.

## Verification status
- [x] Offline: `parse_mandate_requirements` + `evaluate_grounding` unit-checked on 040/045/046/047.
- [x] Offline: `record_decision` API + `summary()` include decisions.
- [x] Live: the grounding gate FIRES in the benchmark path (`testing/execution.py`), re-plans,
      and injects link-follow nodes. On 046 the agent went 1 visit -> 7 visits and `grounded`
      flipped False -> True; the full 7-stage decision trace is captured per cell.
> Note: the gate runs in BOTH entry points — `idea_engine.run()` and `testing/execution.py`'s
> loop. The benchmark uses the latter; that is where the gate must live.

## Known limitation → next workstream (NOT a scaffolding gap)
The infrastructure works, but the navigation *score* on 046/047 is still 0: the agent follows
links but **link retrieval cannot resolve a vague descriptive `link_idea`** ("rocket that
launched the mission") **to the specific target** (Saturn V). Across runs it picked
`carrier_rocket`, `Main_Page`, then `donate.wikimedia.org` — generic/chrome links ranked above
the right one by Chroma semantic similarity. Added a Wikipedia chrome filter
(`VisitLeafAction._is_wiki_chrome`) but the core issue is **link-discovery quality**, a separate
workstream: e.g. rank/select over anchor text with the LLM seeing the page's actual link list
(not Chroma top-k), prefer same-host `/wiki/` content links, or have the planner name the target
page once it reads the source page. The decision trace now makes each bad pick fully visible —
that is the point of the observability. 047 wiki-race additionally needs intermediate-chain
planning. **This is what "all interfaces in place before experimentation" enables: the gap is now
measurable and isolated, not hidden.**
