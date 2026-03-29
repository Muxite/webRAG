"""
Deploy Euglena on a single machine with Docker Compose (fixed replicas, no ECS autoscale).

Typical host: Ubuntu Server with Docker; expose HTTPS via Tailscale funnel to nginx on port 80.

Commands:
  python scripts/deploy_local_stack.py build-frontend
  python scripts/deploy_local_stack.py up
  python scripts/deploy_local_stack.py down
  python scripts/deploy_local_stack.py funnel-help

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


def _compose_cmd() -> list[str]:
    """
    Build docker compose base command with both compose files.

    :returns: Command prefix list for subprocess.
    """
    base = _services_dir() / "docker-compose.yml"
    override = _services_dir() / "docker-compose.local.yml"
    return [
        "docker",
        "compose",
        "-f",
        str(base),
        "-f",
        str(override),
    ]


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
    Build the Vite frontend with same-origin gateway URLs for nginx.

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
        print(f"Frontend build OK: {dist}")
    return code


def cmd_up(_: argparse.Namespace) -> int:
    """
    Start the stack (build images if needed).

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    dist = _repo_root() / "frontend" / "dist" / "index.html"
    if not dist.is_file():
        print("frontend/dist missing. Run: python scripts/deploy_local_stack.py build-frontend", file=sys.stderr)
        return 1
    return _run(_compose_cmd() + ["up", "-d", "--build"], cwd=_services_dir())


def cmd_down(_: argparse.Namespace) -> int:
    """
    Stop and remove containers for this stack.

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    return _run(_compose_cmd() + ["down"], cwd=_services_dir())


def cmd_ps(_: argparse.Namespace) -> int:
    """
    Show compose service status.

    :param _: Parsed arguments (unused).
    :returns: Exit code (0 on success).
    """
    return _run(_compose_cmd() + ["ps"], cwd=_services_dir())


def cmd_funnel_help(_: argparse.Namespace) -> int:
    """
    Print Tailscale funnel and Supabase checklist (no network operations).

    :param _: Parsed arguments (unused).
    :returns: Always 0.
    """
    print(
        """
Tailscale funnel (run on the Ubuntu host, not inside Docker):
  1) Install Tailscale and log in: https://tailscale.com/download/linux
  2) Enable Funnel in your tailnet ACL (see Tailscale docs: tailscale funnel).
  3) Start the stack first so nginx listens on port 80, then:
       sudo tailscale funnel 80
     This exposes http://127.0.0.1:80 (nginx + SPA + proxied API) at https://<device>.<tailnet>.ts.net
     Run `tailscale funnel --help` for your CLI version (e.g. background: tailscale funnel --bg 80).

Supabase (dashboard -> Authentication -> URL configuration):
  - Add your funnel URL to "Site URL" and "Redirect URLs" (e.g. https://YOURHOST.tailXXXX.ts.net).

Stack notes:
  - Gateway is only on 127.0.0.1:8080; public traffic should go to nginx on :80 via funnel.
  - Fixed worker count: default is one agent container. To run two agents:
       docker compose -f services/docker-compose.yml -f services/docker-compose.local.yml up -d --scale agent=2

AWS/ECS deployments are unchanged; this compose stack is separate.
"""
    )
    return 0


def main() -> int:
    """
    Parse CLI and dispatch subcommands.

    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description="Local Docker deployment helper (Compose + optional Tailscale funnel).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-frontend", help="npm run build with VITE_GATEWAY_RELATIVE=true")
    sub.add_parser("up", help="docker compose up -d --build (requires frontend/dist)")
    sub.add_parser("down", help="docker compose down")
    sub.add_parser("ps", help="docker compose ps")
    sub.add_parser("funnel-help", help="Print Tailscale funnel and Supabase steps")

    args = parser.parse_args()
    handlers = {
        "build-frontend": cmd_build_frontend,
        "up": cmd_up,
        "down": cmd_down,
        "ps": cmd_ps,
        "funnel-help": cmd_funnel_help,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
