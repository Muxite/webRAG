"""Shared test fixtures and fake connectors for the connector API tests.

The real connectors touch the network; tests inject these fakes via
``create_app(...)`` so nothing leaves the process.
"""
from __future__ import annotations

from typing import List, Optional

import pytest

from shared.request_result import RequestResult


class FakeSearch:
    """Stand-in for ConnectorSearch with a scripted query_search."""

    def __init__(self, results: Optional[List[dict]] = None, error: Optional[Exception] = None):
        self._results = results
        self._error = error
        self.reset_called = False

    async def query_search(self, query: str, count: int = 10):
        if self._error is not None:
            raise self._error
        return self._results

    async def _reset_session(self) -> None:
        self.reset_called = True


class FakeHttp:
    """Stand-in for ConnectorHttp with a scripted request()."""

    def __init__(self, result: Optional[RequestResult] = None, exc: Optional[Exception] = None):
        self._result = result
        self._exc = exc
        self.reset_called = False

    async def request(self, method: str, url: str, retries: int = 2, **kwargs) -> RequestResult:
        if self._exc is not None:
            raise self._exc
        return self._result

    async def _reset_session(self) -> None:
        self.reset_called = True


class FakeBrowser:
    """Stand-in for ConnectorBrowser with a scripted fetch_page()."""

    def __init__(self, result: Optional[RequestResult] = None, exc: Optional[Exception] = None):
        self._result = result
        self._exc = exc
        self.closed = False

    async def fetch_page(self, url: str, timeout: Optional[float] = None) -> RequestResult:
        if self._exc is not None:
            raise self._exc
        return self._result

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def make_client():
    """
    Factory that builds a FastAPI TestClient with injected fake connectors.

    :returns: Callable(**connectors) -> starlette.testclient.TestClient.
    """
    from fastapi.testclient import TestClient
    from app.main import create_app

    def _factory(search=None, http=None, browser=None):
        app = create_app(search=search, http=http, browser=browser)
        return TestClient(app)

    return _factory
