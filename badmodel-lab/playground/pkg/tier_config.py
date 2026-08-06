"""Resolve BADMODEL_TIER/BADMODEL_PROFILE against the bind-mounted tier_roster.yaml /
tier_profiles.yaml, and parse a badmodel-lab mitigation profile .env file.

Config directory layout (see docker-compose.yml for the bind mounts):
    /app/playground_config/tier_roster.yaml
    /app/playground_config/tier_profiles.yaml
    /app/playground_config/mitigation_profiles/<profile>.env   (badmodel-lab/profiles/)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional

import yaml

CONFIG_ROOT = Path("/app/playground_config")
TIER_ROSTER_PATH = CONFIG_ROOT / "tier_roster.yaml"
TIER_PROFILES_PATH = CONFIG_ROOT / "tier_profiles.yaml"
MITIGATION_PROFILES_DIR = CONFIG_ROOT / "mitigation_profiles"

DEFAULT_PROFILE = "a0_native_baseline"


class TierConfig(NamedTuple):
    tier: str
    default_model: str
    models: List[str]
    profile: str
    profile_env: Dict[str, str]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Expected config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Minimal KEY=VALUE .env parser matching badmodel-lab/profiles/*.env's format:
    no quoting, no interpolation, `#`-prefixed comment lines, blank lines ignored."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def resolve_tier_config(environ: Optional[Mapping[str, str]] = None) -> TierConfig:
    """Resolve the active tier's model roster and mitigation profile from env + the two
    bind-mounted YAML files. Raises loudly (not silently) on missing/unknown config —
    this only ever runs inside a container started by the playground docker-compose.yml,
    where BADMODEL_TIER is always set by the matching tier service.
    """
    env = os.environ if environ is None else environ
    tier = (env.get("BADMODEL_TIER") or "").strip()
    if not tier:
        raise RuntimeError(
            "BADMODEL_TIER is not set — this container wasn't started via one of the "
            "playground's tier services (badmodel-4gb/6gb/8gb/12gb)."
        )

    roster = _load_yaml(TIER_ROSTER_PATH)
    tier_entry = (roster.get("tiers") or {}).get(tier)
    if not tier_entry:
        known = sorted((roster.get("tiers") or {}).keys())
        raise RuntimeError(f"Unknown BADMODEL_TIER={tier!r}; tier_roster.yaml only defines {known}.")
    default_model = tier_entry["default"]
    models = list(tier_entry.get("models") or [default_model])
    if default_model not in models:
        models.insert(0, default_model)

    requested_profile = (env.get("BADMODEL_PROFILE") or "").strip()
    if requested_profile:
        profile = requested_profile
    else:
        profiles_doc = _load_yaml(TIER_PROFILES_PATH)
        profile = (profiles_doc.get("defaults") or {}).get(tier, DEFAULT_PROFILE)

    profile_env = _parse_env_file(MITIGATION_PROFILES_DIR / f"{profile}.env")
    if requested_profile and not profile_env:
        raise RuntimeError(
            f"BADMODEL_PROFILE={requested_profile!r} has no matching "
            f"{MITIGATION_PROFILES_DIR / (requested_profile + '.env')} — check the name "
            f"against badmodel-lab/profiles/ (only a0-a4 apply to interactive chat, "
            f"see MITIGATION_BRIDGE.md)."
        )

    return TierConfig(
        tier=tier,
        default_model=default_model,
        models=models,
        profile=profile,
        profile_env=profile_env,
    )


def merged_environ(tier_config: TierConfig, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Overlay the resolved profile's env vars under the real process environment, so an
    explicit ambient env var always wins over the profile file — matching the precedence
    `agent.app.idea_test_runner._apply_got_experiment_overrides` already documents for
    `IDEA_TEST_ARM` profiles.
    """
    env = os.environ if environ is None else environ
    return {**tier_config.profile_env, **env}
