"""Unit tests for `data_contracts.py` — no external deps required.

We deliberately load `data_contracts` via importlib rather than through
`agent.app.idea_policies` because the package `__init__.py` pulls in the
full action stack (including `bs4`), which we don't want as a test-time
dependency for what is otherwise a pure-Python module.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


def _load_data_contracts():
    here = Path(__file__).resolve().parent
    # tests/ -> agent/ -> services/ -> repo root
    target = here.parent / "ideaengine" / "idea_policies" / "data_contracts.py"
    spec = importlib.util.spec_from_file_location(
        "_data_contracts_under_test",
        target,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_data_contracts_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_dc = _load_data_contracts()
CHUNK_FROM_VISIT = _dc.CHUNK_FROM_VISIT
ContractRegistry = _dc.ContractRegistry
DataContract = _dc.DataContract
URL_FROM_THINK = _dc.URL_FROM_THINK
URLS_FROM_SEARCH = _dc.URLS_FROM_SEARCH
URLS_FROM_VISIT = _dc.URLS_FROM_VISIT
default_contract_registry = _dc.default_contract_registry


@dataclass
class _FakeNode:
    details: Dict[str, Any] = field(default_factory=dict)


def test_default_registry_has_four_builtins():
    r = default_contract_registry()
    assert set(r.names()) == {
        "urls_from_search",
        "urls_from_visit",
        "url_from_think",
        "chunk_from_visit",
    }


def test_contract_for_action_maps_default_producers():
    r = default_contract_registry()
    assert r.contract_for_action("search") is URLS_FROM_SEARCH
    assert r.contract_for_action("visit") is URLS_FROM_VISIT
    assert r.contract_for_action("think") is None
    assert r.contract_for_action(None) is None


def test_get_returns_none_for_unknown():
    r = default_contract_registry()
    assert r.get("bogus") is None
    assert r.get(None) is None


def test_each_builtin_contract_has_callable_is_ready():
    """`is_ready` is a callable; behavioral tests live in the integration
    suite where the deferred `agent.app.idea_policies.action_constants`
    imports inside the predicates are resolvable."""
    for c in (URLS_FROM_SEARCH, URLS_FROM_VISIT, URL_FROM_THINK, CHUNK_FROM_VISIT):
        assert callable(c.is_ready)
        assert isinstance(c.name, str) and c.name


def test_builtin_default_for_action_is_consistent():
    assert URLS_FROM_SEARCH.default_for_action == "search"
    assert URLS_FROM_VISIT.default_for_action == "visit"
    assert URL_FROM_THINK.default_for_action is None
    assert CHUNK_FROM_VISIT.default_for_action is None


def test_register_overrides_default_for_action():
    """A later-registered contract claiming the same default action wins."""

    def _always_ready(_result, _node) -> bool:
        return True

    custom = DataContract(
        name="custom_search_evidence",
        is_ready=_always_ready,
        default_for_action="search",
    )
    r = default_contract_registry()
    r.register(custom)
    assert r.contract_for_action("search") is custom
    # original is still queryable by name
    assert r.get("urls_from_search") is URLS_FROM_SEARCH


def test_custom_contract_registration_lookup_by_name():
    def _ready(_r, _n) -> bool:
        return True

    custom = DataContract(name="rows_from_db", is_ready=_ready)
    r = ContractRegistry()
    r.register(custom)
    assert r.get("rows_from_db") is custom
    # No default_for_action set → no action mapping.
    assert r.contract_for_action("db_query") is None
