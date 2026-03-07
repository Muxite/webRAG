import importlib.util
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(8):
        if (cur / "scripts" / "build_task_definition.py").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate repo root containing scripts/build_task_definition.py")


def _load_build_task_definition_module():
    repo_root = _find_repo_root(Path(__file__).resolve())
    path = repo_root / "scripts" / "build_task_definition.py"
    spec = importlib.util.spec_from_file_location("build_task_definition", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _env_as_dict(env_list):
    return {item["name"]: item["value"] for item in (env_list or []) if isinstance(item, dict) and "name" in item and "value" in item}


def test_agent_task_definition_includes_startup_preflight_env_defaults():
    mod = _load_build_task_definition_module()

    task_def = mod.build_agent_task_definition(
        account_id="123456789012",
        region="us-east-1",
        secret_name="s",
        secret_arn_suffix="x",
        secret_keys=[],
        env_vars={},
        aws_env={},
        gateway_host="euglena-gateway.euglena.local",
    )
    container = task_def["containerDefinitions"][0]
    env = _env_as_dict(container.get("environment"))

    assert env["AGENT_START_PREFLIGHT_ENABLED"] == "1"
    assert env["AGENT_START_PREFLIGHT_URL"].startswith("https://en.wikipedia.org/wiki/")
    assert int(env["AGENT_START_PREFLIGHT_MIN_CHARS"]) >= 20000
    assert int(env["AGENT_START_PREFLIGHT_TIMEOUT_SECONDS"]) >= 5
    assert env["AGENT_START_PREFLIGHT_BROWSER"] in ("0", "1")
    assert env["AGENT_START_PREFLIGHT_FAIL_HARD"] in ("0", "1")


def test_single_service_task_definition_includes_startup_preflight_env_defaults():
    mod = _load_build_task_definition_module()

    task_def = mod.build_euglena_task_definition(
        account_id="123456789012",
        region="us-east-1",
        secret_name="s",
        secret_arn_suffix="x",
        secret_keys=[],
        env_vars={},
        aws_env={},
    )
    containers = task_def["containerDefinitions"]
    agent = next(c for c in containers if c.get("name") == "agent")
    env = _env_as_dict(agent.get("environment"))

    assert env["AGENT_START_PREFLIGHT_ENABLED"] == "1"
    assert int(env["AGENT_START_PREFLIGHT_MIN_CHARS"]) >= 20000

