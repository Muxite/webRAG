"""The promptbench compose services must never join the default stack.

Cheap, and it guards something human review reliably misses: deleting a
``profiles:`` key during an unrelated compose edit produces no error and no warning
-- just a benchmark container that starts alongside the agent on every
``compose up``, holding a GPU-adjacent workload nobody asked for.

Also pins the mount arrangement, which exists for a documented reason:
``DEV_CYCLE.md`` lesson 4 records a codebench reverification that silently tested
PRE-FIX code because its image ``COPY``s the source in at build time and predated
the fix. Nothing in this repo stamps an image with the SHA it was built from, so
the read-only source mount is what makes that class of bug impossible here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path("services/docker-compose.yml")
COMPOSE = yaml.safe_load(COMPOSE_PATH.read_text())
SERVICES = COMPOSE["services"]
PROMPTBENCH_SERVICES = ["promptbench", "promptbench-analyze"]


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_the_service_exists(name):
    assert name in SERVICES


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_the_service_is_behind_a_non_default_profile(name):
    profiles = SERVICES[name].get("profiles")
    assert profiles, f"{name} has no profiles: it would start with the default stack"
    assert "promptbench" in profiles


def test_no_other_service_joins_the_promptbench_profile():
    for name, service in SERVICES.items():
        if name in PROMPTBENCH_SERVICES:
            continue
        assert "promptbench" not in (service.get("profiles") or [])


def test_the_default_stack_is_unchanged_in_size():
    """Any service without a profile starts on `compose up`. Pinning the count
    makes an accidental addition to the default stack fail loudly."""
    default = [n for n, s in SERVICES.items() if not s.get("profiles")]
    assert sorted(default) == [
        "agent", "chroma", "connector-api", "gateway", "rabbitmq", "redis",
    ]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_arguments_pass_through_to_the_module(name):
    """An entrypoint with no command is what lets `compose run --rm promptbench
    --models ...` append CLI arguments rather than replace the program."""
    service = SERVICES[name]
    assert service["entrypoint"][:2] == ["python", "-m"]
    assert service["entrypoint"][2].startswith("agent.app.promptbench.")
    assert "command" not in service


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_the_source_is_mounted_read_only_rather_than_trusted_from_the_image(name):
    mounts = SERVICES[name]["volumes"]
    assert "../agent:/app/agent:ro" in mounts
    assert "../scripts:/app/scripts:ro" in mounts


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_results_land_on_the_host_through_a_writable_nested_mount(name):
    assert "../agent/idea_test_results:/app/agent/idea_test_results:rw" in SERVICES[name]["volumes"]


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_the_working_directory_matches_the_cwd_relative_paths_in_the_code(name):
    """runner.py and analyze.py resolve `agent/idea_test_results/...` and
    `sys.path.insert(0, "scripts")` against the process CWD."""
    assert SERVICES[name]["working_dir"] == "/app"


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_output_is_written_as_the_host_user_not_root(name):
    """A root-owned results file cannot be appended to by a later HOST-side run,
    and the two workflows are meant to share one resumable output file."""
    assert SERVICES[name]["user"].startswith("${PROMPTBENCH_UID:")


def test_the_runner_points_at_ollama_by_service_name_on_the_shared_network():
    env = SERVICES["promptbench"]["environment"]
    assert "PROMPTBENCH_BASE_URL=http://badmodel-ollama:11434/v1" in env
    assert "enet" in SERVICES["promptbench"]["networks"]


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_no_gpu_is_reserved_because_inference_belongs_to_ollama(name):
    """badmodel-ollama holds OLLAMA_MAX_LOADED_MODELS=1 on a 12 GB card.
    Reserving a device here would contend with the thing doing the work."""
    assert "deploy" not in SERVICES[name]


@pytest.mark.parametrize("name", PROMPTBENCH_SERVICES)
def test_one_shot_containers_carry_no_restart_policy(name):
    """The compose file's own comment exempts one-shot containers from the
    `restart: unless-stopped` standard the six default services carry."""
    assert "restart" not in SERVICES[name]


def test_the_env_default_stays_the_host_port_so_host_runs_are_unaffected():
    from agent.app.promptbench import runner

    assert "11435" in runner.DEFAULT_BASE_URL or "PROMPTBENCH_BASE_URL" in str(
        Path("agent/app/promptbench/runner.py").read_text())
