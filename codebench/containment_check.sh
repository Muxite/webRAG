#!/usr/bin/env bash
# The containment invariant for codebench sandboxes — run this INSIDE a codebench-badmodel
# or codebench-aider container (e.g. via `docker exec`) to verify hardening before trusting
# any live task run. Adapted from badmodel-lab/localagent/docker/containment_check.sh:
# same checks 1/2/4/5 verbatim, check 3 flipped for codebench's network topology (a
# writable, execution-capable sandbox on codebench_sandbox, with exactly TWO intentional
# holes — SearXNG and codebench-ollama-proxy, both reached by container name, NOT
# host.docker.internal (--internal removes the bridge's host-gateway route entirely,
# verified live) — instead of localagent's fully-isolated segment).
#
# badmodel-ollama itself is now on the BLOCKED list (§3c), not the allowed list: an
# earlier topology joined it directly, giving the sandbox unauthenticated access to its
# entire admin API (verified live: GET /api/tags, DELETE /api/delete, POST /api/pull all
# worked with zero auth) — see setup_network.sh's SECURITY comment. codebench-ollama-proxy
# is the fix, and §3d below verifies the proxy's OWN allowlist actually holds, not just
# that it's reachable.
set +e
fail=0

echo "[1] workdir writable + confined:"
if touch /work/_probe 2>/dev/null; then echo "  OK (/work writable)"; rm -f /work/_probe; else echo "  LEAK: /work not writable"; fail=1; fi
if : > /rootprobe 2>/dev/null; then echo "  LEAK: root fs writable"; rm -f /rootprobe; fail=1; else echo "  OK (rootfs read-only)"; fi

echo "[2] host filesystem not mounted (probe host-specific paths, NOT the container's own /home/agent):"
leaks=""
for d in /home/muk /mnt/storage /mnt/archive /host /hostfs; do
  [ -e "$d" ] && leaks="$leaks $d"
done
if [ -n "$leaks" ]; then echo "  LEAK: host paths visible:$leaks"; fail=1; else echo "  OK (no host paths visible)"; fi

echo "[3a] SearXNG + codebench-ollama-proxy reachable — the TWO intentional holes:"
for hp in badmodel-playground-searxng:8080 codebench-ollama-proxy:8090; do
  host="${hp%%:*}"; port="${hp##*:}"
  if timeout 3 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
    echo "  OK ($hp reachable — expected)"
  else
    echo "  LEAK: $hp NOT reachable (investigate setup_network.sh)"; fail=1
  fi
done

echo "[3b] host gateway route removed by --internal (belt-and-suspenders, not the boundary):"
if timeout 2 bash -c "exec 3<>/dev/tcp/host.docker.internal/11435" 2>/dev/null; then
  echo "  LEAK: host.docker.internal reached (--internal should block this route entirely)"; fail=1
else
  echo "  OK (host.docker.internal unreachable, as --internal guarantees)"
fi

echo "[3c] internal stack + the REAL ollama (direct) + a DIFFERENT ollama + real internet unreachable:"
for hp in euglena-redis-1:6379 euglena-chroma:8000 euglena-rabbitmq-1:5672 \
          badmodel-ollama:11434 badmodel-playground-ollama:11434 8.8.8.8:53 1.1.1.1:443; do
  host="${hp%%:*}"; port="${hp##*:}"
  if timeout 2 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then echo "  LEAK: reached $hp"; fail=1; else echo "  OK ($hp blocked)"; fi
done

echo "[3d] the proxy's OWN allowlist actually holds (not just reachability):"
if command -v python3 >/dev/null 2>&1; then
  code=$(python3 - <<'PYEOF' 2>/dev/null
import urllib.request
try:
    urllib.request.urlopen("http://codebench-ollama-proxy:8090/api/tags", timeout=3)
    print(200)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:
    print("ERR")
PYEOF
)
  if [ "$code" = "403" ]; then
    echo "  OK (GET /api/tags via the proxy correctly returns 403)"
  else
    echo "  LEAK: GET /api/tags via the proxy returned '$code', expected 403"; fail=1
  fi
else
  echo "  SKIPPED (no python3 in this image to probe with)"
fi

echo "[4] cloud metadata blocked:"
if timeout 2 bash -c "exec 3<>/dev/tcp/169.254.169.254/80" 2>/dev/null; then echo "  LEAK: metadata reachable"; fail=1; else echo "  OK (metadata blocked)"; fi

echo "[5] non-root:"
[ "$(id -u)" != "0" ] && echo "  OK (uid $(id -u))" || { echo "  LEAK: running as root"; fail=1; }

echo "---"
[ "$fail" = "0" ] && echo "CONTAINMENT OK" || echo "CONTAINMENT FAILED ($fail leaks)"
exit "$fail"
