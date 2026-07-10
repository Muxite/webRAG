"""
Pydantic request/response models for the connector reachability API.

These models double as the OpenAPI documentation surface (visible at
``/docs`` and ``/redoc``). Field descriptions are intentionally written for a
task-author audience: someone deciding whether a page/query is reachable
*before* writing a benchmark task around it.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Parameters for a Brave Search API query, matching ``ConnectorSearch.query_search``."""

    query: str = Field(
        ...,
        min_length=1,
        description="The search query string sent verbatim to the Brave Search API.",
        examples=["site:sec.gov Apple 10-K 2023"],
    )
    count: int = Field(
        default=10,
        ge=1,
        le=20,
        description=(
            "Maximum number of web results to request (Brave caps this at 20). "
            "The API may return fewer than requested."
        ),
    )


class SearchResultItem(BaseModel):
    """A single normalized search hit, mirroring the dicts ``query_search`` returns."""

    title: str = Field(description="Result title as reported by the search provider (may be empty).")
    url: str = Field(description="The canonical result URL. Feed this straight into POST /visit.")
    description: str = Field(
        description="Snippet/description for the result (may be empty)."
    )


class SearchResponse(BaseModel):
    """Result of a search query. An empty ``results`` list means the query returned zero hits."""

    query: str = Field(description="The query that was executed (echoed back for correlation).")
    count: int = Field(description="The ``count`` that was requested.")
    result_count: int = Field(description="Number of results actually returned.")
    results: List[SearchResultItem] = Field(
        description="Normalized search hits (title/url/description). Empty when nothing matched."
    )


class VisitRequest(BaseModel):
    """Parameters for a reachability probe of a single URL."""

    url: str = Field(
        ...,
        min_length=1,
        description="The absolute URL to fetch (include the scheme, e.g. https://).",
        examples=["https://en.wikipedia.org/wiki/Euglena"],
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Optional per-attempt timeout in seconds. When omitted the connector's "
            "own defaults apply. Applied to both the HTTP attempt and the headless "
            "browser fallback."
        ),
    )
    max_content_chars: int = Field(
        default=20000,
        ge=0,
        le=5_000_000,
        description=(
            "Cap on how many characters of fetched content to return in the response "
            "body. Set to 0 to omit content entirely and only get the reachability "
            "verdict. ``content_chars`` always reports the true, un-truncated length."
        ),
    )


class VisitResponse(BaseModel):
    """
    Reachability verdict for a URL.

    This is the load-bearing shape: ``reachable`` is the yes/no a task author
    needs, and ``reason``/``status``/``via`` explain *why* a fetch failed
    (timeout vs. 404 vs. bot-blocked vs. connection refused) so the connector
    can be fixed before a task depends on the page.
    """

    url: str = Field(description="The URL that was probed (echoed back).")
    reachable: bool = Field(
        description=(
            "True only if content was successfully fetched (HTTP 2xx/3xx with a body). "
            "False for any failure, including bot-blocks the browser fallback could not clear."
        )
    )
    status: Optional[int] = Field(
        default=None,
        description=(
            "The HTTP status code observed on the deciding attempt, or null when no "
            "response was received at all (DNS failure, connection refused, timeout)."
        ),
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable explanation of the outcome. On failure this distinguishes "
            "timeout vs. 404 vs. forbidden/blocked vs. connection error. Null on success."
        ),
    )
    via: Optional[str] = Field(
        default=None,
        description=(
            "Which transport produced the result: 'http' for the plain aiohttp fetch, "
            "'browser' when the headless-Chromium fallback (used on 401/403 bot-blocks) "
            "succeeded, or null when nothing succeeded."
        ),
    )
    browser_fallback_used: bool = Field(
        default=False,
        description="True if the headless-browser fallback was attempted (regardless of outcome).",
    )
    content_chars: int = Field(
        default=0,
        description="True length in characters of the fetched content, before any truncation.",
    )
    content: Optional[str] = Field(
        default=None,
        description=(
            "The raw fetched content (HTML/text/JSON), truncated to ``max_content_chars``. "
            "Null when the fetch failed or ``max_content_chars`` was 0."
        ),
    )
    content_truncated: bool = Field(
        default=False,
        description="True when ``content`` was truncated because it exceeded ``max_content_chars``.",
    )
