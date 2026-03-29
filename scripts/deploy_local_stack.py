"""
Local Docker Compose stack. Same Supabase config as production; point the frontend at this API with VITE_GATEWAY_URL.

Commands: up, up-spa, build-frontend, down, ps, funnel-help

:returns: None (process exit code 0 on success).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    """
    Resolve repository root (parent of scripts/).

    :returns: Absolute path to repo root.
    """
    return Path(__file__).resolve().parent.parent


def _services_dir() -> Path:
    """
    Resolve services directory containing compose files.

    :returns: Absolute path to services/.
    """
    return _repo_root() / "services"


def _compose_files() -> list[str]:
    """
    Compose file arguments for local stack (base + local override).

    :returns: Flat list of -f path pairs for subprocess.
    """
    base = _services_dir() / "docker-compose.yml"
    override = _services_dir() / "docker-compose.local.yml"
    return ["-f", str(base), "-f", str(override)]


def _compose_cmd(*, profile_local_spa: bool = False) -> list[str]:
    """
    Build docker compose command with local compose files and optional local-spa profile.

    :param profile_local_spa: When True, enable nginx + static SPA service.
    :returns: Command prefix list for subprocess.
    """
    cmd = ["docker", "compose", *_compose_files()]
    if profile_local_spa:
        cmd.extend(["--profile", "local-spa"])
    return cmd


def _run(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> int:
    """
    Run a command and stream output to stdout/stderr.

    :param cmd: Argument list (executable first).
    :param cwd: Working directory, or None for inherit.
    :param env: Full environment dict, or None to inherit.
    :returns: Process exit code.
    """
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return int(result.returncode)


def cmd_build_frontend(_: argparse.Namespace) -> int:
    """
    Build the Vite bundle for nginx same-origin (VITE_GATEWAY_RELATIVE) or local dev.

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    root = _repo_root()
    frontend = root / "frontend"
    pkg = frontend / "package.json"
    if not pkg.is_file():
        print(f"Missing {pkg}", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env["VITE_GATEWAY_RELATIVE"] = "true"
    env["VITE_USE_LOCAL"] = "false"
    code = _run(
        ["npm", "run", "build"],
        cwd=frontend,
        env=env,
    )
    if code == 0:
        dist = frontend / "dist"
        print(f"Built {dist}")
    return code


def cmd_up(_: argparse.Namespace) -> int:
    """
    Start local backend only (rabbitmq, redis, chroma, agents, gateway). No nginx.

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    return _run(
        _compose_cmd() + ["up", "-d", "--build", "--scale", "agent=3"],
        cwd=_services_dir(),
    )


def cmd_up_spa(_: argparse.Namespace) -> int:
    """
    Start local backend plus nginx serving frontend/dist on port 80.

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    dist = _repo_root() / "frontend" / "dist" / "index.html"
    if not dist.is_file():
        print("No frontend/dist. Run: python scripts/deploy_local_stack.py build-frontend", file=sys.stderr)
        return 1
    return _run(
        _compose_cmd(profile_local_spa=True) + ["up", "-d", "--build", "--scale", "agent=3"],
        cwd=_services_dir(),
    )


def cmd_down(_: argparse.Namespace) -> int:
    """
    Stop and remove containers for this stack (includes optional nginx when it was used).

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    return _run(_compose_cmd(profile_local_spa=True) + ["down"], cwd=_services_dir())


def cmd_ps(_: argparse.Namespace) -> int:
    """
    Show compose service status.

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    return _run(_compose_cmd() + ["ps"], cwd=_services_dir())


def cmd_funnel_help(_: argparse.Namespace) -> int:
    """
    Print notes for Tailscale and env vars.

    :param _: Parsed arguments (unused).
    :returns: Always 0.
    """
    print(
        "AWS: deploy with your usual ECS flow.\n"
        "Local: run up, then tailscale funnel --bg --yes 18080. Set VITE_GATEWAY_URL on Vercel to the "
        "HTTPS URL Tailscale prints. Redeploy the frontend.\n"
        "up-spa with nginx on port 80: tailscale funnel --bg --yes 80\n"
        "Supabase Site URL: keep your Vercel hostname unless users hit the funnel URL.\n"
        "Gateway host port: 127.0.0.1:18080 (see docker-compose.local.yml). up uses --scale agent=3."
    )
    return 0


def main() -> int:
    """
    Parse CLI and dispatch subcommands.

    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description="Local docker compose helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("up", help="Backend services only (no nginx)")
    sub.add_parser("up-spa", help="Backend plus nginx on port 80 (run build-frontend first)")
    sub.add_parser("build-frontend", help="Production build with VITE_GATEWAY_RELATIVE=true")
    sub.add_parser("down", help="Stop stack")
    sub.add_parser("ps", help="Compose ps")
    sub.add_parser("funnel-help", help="Notes for Tailscale and env vars")

    args = parser.parse_args()
    handlers = {
        "up": cmd_up,
        "up-spa": cmd_up_spa,
        "build-frontend": cmd_build_frontend,
        "down": cmd_down,
        "ps": cmd_ps,
        "funnel-help": cmd_funnel_help,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
