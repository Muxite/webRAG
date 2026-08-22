"""The ladder-benchmark compose service must stay a one-shot, host-owned, local-model run.

Same guard as ``promptbench_compose_profile_test``, for the other benchmark service: a
deleted ``profiles:`` key produces no error and no warning, just a benchmark container
that starts alongside the live agent stack on every ``compose up``. The rest pins the
wiring that makes the container run reproduce the host run -- the driver hard-codes a
host repo path and a ``.venv`` interpreter, and the axis tables hard-code the HOST
publication of the Ollama endpoint, so all three need an override to work in-container.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path("services/docker-compose.yml")
COMPOSE = yaml.safe_load(COMPOSE_PATH.read_text())
SERVICES = COMPOSE["services"]
NAME = "ladder-benchmark"


def test_the_service_exists():
    assert NAME in SERVICES


def test_it_is_behind_its_own_non_default_profile():
    assert SERVICES[NAME].get("profiles") == ["ladder-benchmark"]
    for name, service in SERVICES.items():
        if name != NAME:
            assert "ladder-benchmark" not in (service.get("profiles") or [])


def test_one_shot_containers_carry_no_restart_policy():
    assert "restart" not in SERVICES[NAME]


def test_no_gpu_is_reserved_because_inference_belongs_to_ollama():
    assert "deploy" not in SERVICES[NAME]


def test_the_source_is_mounted_read_only_rather_than_trusted_from_the_image():
    mounts = SERVICES[NAME]["volumes"]
    assert "../agent:/app/agent:ro" in mounts
    assert "../scripts:/app/scripts:ro" in mounts


def test_results_land_on_the_host_through_a_writable_nested_mount():
    """Result JSON, cell logs, the attempt ledger and the embedded per-cell chroma dirs
    all live under the run's output dir; a resumed HOST run must be able to extend them."""
    assert "../agent/idea_test_results:/app/agent/idea_test_results:rw" in SERVICES[NAME]["volumes"]


def test_output_is_written_as_the_host_user_not_root():
    assert SERVICES[NAME]["user"].startswith("${LADDER_UID:")


def test_the_working_directory_matches_the_drivers_repo_relative_paths():
    assert SERVICES[NAME]["working_dir"] == "/app"


@pytest.mark.parametrize("var", ["WEBRAG_REPO=/app", "WEBRAG_PYTHON=python"])
def test_the_host_repo_path_and_venv_interpreter_are_overridden(var):
    assert var in SERVICES[NAME]["environment"]


def test_local_inference_is_addressed_by_service_name_on_the_shared_network():
    assert "LOCAL_API_URL=http://badmodel-ollama:11434/v1" in SERVICES[NAME]["environment"]
    assert "enet" in SERVICES[NAME]["networks"]


def test_the_default_command_is_a_free_dry_run_smoke_test():
    """`compose run --rm ladder-benchmark` with no arguments must not spend money or
    touch a model: it prints the cell plan and exits."""
    command = SERVICES[NAME]["command"]
    assert "--dry-run" in command
    assert SERVICES[NAME]["entrypoint"] == ["python", "scripts/adaptive_ladder_run.py"]
