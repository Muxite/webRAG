# Strategy notes — the corpus

One `<note_id>.json` per note (`_`-prefixed files are ignored by the loader; `_manifest.json` is
the sync lockfile). The schema is `strategy_library/schema.py`.

**This directory is intentionally empty of notes.** Nothing has cleared the promotion gate yet,
and shipping a note that has not been measured would be exactly the overclaim the gate exists to
prevent. The machinery — authoring, the leak gate, retrieval, the eval script — is what landed;
promoting the first note is a separate, live-evaluation step.

## Adding a note

```bash
# 1. author it (one strong-model call, cached; runs the four-layer leak gate before writing)
PYTHONPATH=.:services:agent ./.venv/bin/python - <<'PY'
import asyncio
from agent.app.strategy_library.authoring import author_note
# ... build an AgentIO, then:
# note, info = asyncio.run(author_note("argmax", ["062", "077"], agent_io=io))
PY

# 2. measure it on HELD-OUT instances (this is the step that costs money)
PYTHONPATH=.:services:agent ./.venv/bin/python \
  scripts/eval_strategy_library_generalization.py plan \
  --note argmax_from_062_077 --held-out 084 091 --seed 062 --model openai/gpt-5-nano --repeats 4
#   ... run the printed commands ...
PYTHONPATH=.:services:agent ./.venv/bin/python \
  scripts/eval_strategy_library_generalization.py score \
  --note argmax_from_062_077 --held-out 084 091 --seed 062 \
  --on-run-id <run> --off-run-id <run> --write

# 3. index it (only promoted notes are embedded)
CHROMA_URL=http://localhost:8001 PYTHONPATH=.:services:agent \
  ./.venv/bin/python scripts/sync_strategy_library.py
```

## The bar

`schema.is_active` — `held_out_n >= 2` **and** `held_out_uplift >= 0.05`, both pre-registered in
`schema.py` before any note existed. `status: "active"` in a JSON file does not promote anything
on its own; the measured metrics are the gate.
