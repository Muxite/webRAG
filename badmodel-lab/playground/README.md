# badmodel playground

Chat with a genuinely weak, small local LLM made agentic (web search + read + reasoning)
by the same engine that powers the full webRAG/Euglena agent — entirely on your own
machine, on your own GPU. No API keys, no signups, no cloud calls: the model runs
locally via [Ollama](https://ollama.com), and web search runs through a bundled,
self-hosted [SearXNG](https://docs.searxng.org/) instance.

This is a research toy, not a polished product — these are small models being pushed to
do agentic work they're bad at, on purpose. Expect it to fail in interesting ways. That's
the point: see [What to send back](#what-to-send-back).

## Quickstart

```sh
git clone <repo-url>
cd webRAG/badmodel-lab/playground

# check your GPU's VRAM first (see "Picking a tier" below)
docker compose up badmodel-8gb --build -d      # pick the tier matching your GPU

./chat.sh 8gb                                   # wait for "models ready", then chat
```

In a second terminal, any time:
```sh
./shell.sh 8gb    # diagnostics — tail logs, poke at files, alongside an active chat
./logs.sh 8gb      # follow container logs (useful during the first-boot model download)
```

When you're done:
```sh
./down.sh 8gb    # stops everything; type :exit in chat first if you can (see below)
```
Model weights and the badmodel's long-term memory persist across `down.sh`/`up` — only
`docker compose down -v` would wipe them, and nothing here does that for you.

**Use `./down.sh <tier>`, not a bare `docker compose down`.** Every service in this
stack is tier-profile-gated so `docker compose up badmodel-8gb --build -d` only starts
that one tier — but that also means a bare `docker compose down` silently does
**nothing** (Compose resolves zero "active" services with no profile selected, so it
looks like it worked and doesn't). `./down.sh 8gb` runs the correct
`docker compose --profile 8gb down` for you.

## Picking a tier

Four flavors, one per VRAM tier — pick the one at or below your GPU's actual VRAM:

| Tier | Command | Default model | Also includes | First-boot download |
|---|---|---|---|---|
| `<4GB` | `badmodel-4gb` | qwen2.5:0.5b | tinyllama, qwen2.5:1.5b, llama3.2:1b | ~3.3GB |
| `<6GB` | `badmodel-6gb` | llama3.2:3b | gemma2:2b, phi3:mini | ~5.9GB |
| `<8GB` | `badmodel-8gb` | qwen2.5:7b | llama3.1:8b | ~9.6GB |
| `<12GB` | `badmodel-12gb` | qwen2.5:14b | qwen2.5:7b | ~13.7GB |

Check your VRAM: `nvidia-smi --query-gpu=memory.total --format=csv`

Every model in a tier is pulled on first boot and available mid-chat via basic_cli's
`:model` command — comparing a genuinely-broken small model against a getting-decent
bigger one in the same tier is worth trying. See `tier_roster.yaml` for exact sizes and
the (honest, partly-estimated) VRAM methodology — the `<12GB` tier in particular is tight
(~1.4GB headroom measured), so don't run anything else GPU-heavy alongside it.

There's no `>12GB` tier — untested, no hardware to validate it against.

## Prerequisites

- Docker + Docker Compose.
- **Linux**: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured for Docker. This is required — every tier needs GPU passthrough to the shared `ollama` container.
- **Windows**: Docker Desktop with WSL2, and WSL2 GPU passthrough enabled (a different setup path than nvidia-container-toolkit — see [Docker's WSL2 GPU guide](https://docs.docker.com/desktop/gpu/)).
- No API keys, no accounts, no signup. Nothing in this stack calls out to the internet except the model's own web searches (via the bundled SearXNG) and the initial Docker image builds.

## Overriding the model or mitigation profile

- **Model**: type `:model` in chat to pick any model in your tier's roster.
- **Mitigation profile** (how the engine is coached to plan/re-check its own work): set
  `BADMODEL_PROFILE` before bringing the tier up. Only 5 of `badmodel-lab/profiles/`'s 13
  profiles apply to interactive chat — see [MITIGATION_BRIDGE.md](MITIGATION_BRIDGE.md)
  for which, and why the rest are silent no-ops here.
  ```sh
  BADMODEL_PROFILE=a3_native_expect_contract docker compose up badmodel-8gb --build -d
  ```

## What to send back

Everything a session produces lands in `badmodel-lab/playground/logs/` inside your local
clone (bind-mounted from the container, so it's just files on disk — nothing hidden):

- `logs/transcripts/<session>.jsonl` — the full conversation + tool-call trace for one
  `chat.sh` session (every search, every page visited, the literal prompts sent to the
  model).
- `logs/session_summary.jsonl` — one row per completed mandate: which model/tier/profile,
  how long it took, whether it succeeded, how many sources it grounded its answer in.

If you hit something interesting — a weird failure, a surprisingly good answer, a
mitigation that clearly helps or hurts — zip up the `logs/` directory and send it back.
That's exactly what drives the next round of work on making small models viable for
agentic tasks.

## A note on `./down.sh` mid-chat

If you kill the stack while a mandate is actively running, that chat process is
terminated abruptly with no chance to finish cleanly. The transcript log flushes on every
single event, so it's loss-safe up to whatever was last completed — but the one component
that could theoretically get corrupted by an abrupt kill mid-write is the badmodel's own
long-term memory (an embedded, per-tier SQLite-backed store). Model weights are immutable
once pulled and are never at risk. Type `:exit` in chat before running `./down.sh` when
you can; if you forget, the practical worst case is deleting one tier's memory volume,
never lost logs or a broken model.

## Background

This playground packages `badmodel-lab`'s research — making genuinely weak local models
viable for agentic (search/read/reason) tasks — for other people to run and experiment
with. See [`../README.md`](../README.md) for the full research context, and
[`../HANDOFF.md`](../HANDOFF.md) for the current state of that work.
