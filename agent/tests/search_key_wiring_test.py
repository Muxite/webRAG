"""Regression guard for the Serper key-wiring gap (HANDOFF.md item 5, 2026-08-10).

``ConnectorConfig.search_provider`` defaults to ``"serper"``, and when it's serper the key comes
from ``SERPER_KEY`` (falling back to the Brave ``SEARCH_API_KEY`` only if ``SERPER_KEY`` is unset
-- see ``services/shared/connector_config.py``). A driver that sources ``SEARCH_API_KEY`` from
``services/keys.env`` but never sources ``SERPER_KEY`` silently hands the Serper backend a Brave
key and gets a 403 on every search that looks identical to "Serper is down". This bit every
``scripts/*.sh`` driver except ``badmodel-lab/run_cell.sh``/``run_adaptive_cell.sh`` until fixed.

This isn't exhaustive shell parsing -- it's a plain substring check pinning the one property that
actually matters here: any driver that sources ``SEARCH_API_KEY`` from ``keys.env`` must also
source ``SERPER_KEY`` from it, so a future driver can't reintroduce this exact gap silently.
"""
import glob
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")


def _drivers_sourcing_search_api_key():
    """``scripts/*.sh`` files that read ``SEARCH_API_KEY`` out of ``services/keys.env``."""
    hits = []
    for path in sorted(glob.glob(os.path.join(_SCRIPTS_DIR, "*.sh"))):
        text = open(path, encoding="utf-8").read()
        if "SEARCH_API_KEY" in text and "services/keys.env" in text:
            hits.append(path)
    return hits


def test_every_driver_sourcing_search_api_key_also_sources_serper_key():
    drivers = _drivers_sourcing_search_api_key()
    assert drivers, "expected at least one scripts/*.sh driver sourcing SEARCH_API_KEY"
    missing = [
        os.path.basename(path) for path in drivers
        if "SERPER_KEY" not in open(path, encoding="utf-8").read()
    ]
    assert not missing, (
        f"these drivers source SEARCH_API_KEY (Brave) from keys.env but never SERPER_KEY, so "
        f"they'll silently hand the default serper search_provider a Brave key and 403: {missing}"
    )
