"""
Diagnostic tests for visit URL extraction issues.

Tests focus on diagnosing why visit nodes fail with "no URLs to visit"
when they should extract URLs from search results.

Key scenarios tested:
1. String "None" in optional_url field
2. URL extraction from parent search results
3. Multiple URL extraction (link_count > 1)
4. Empty or missing search results
5. Invalid URL formats in search results
6. Deep parent hierarchies
7. Production error scenario reproduction

Run with: pytest tests/visit_url_extraction_test.py -v
"""

import pytest
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.actions import VisitLeafAction
from agent.app.idea_policies.action_constants import ActionResultKey


class FakeChroma:
    """Mock Chroma connector for testing."""
    
    async def add_to_chroma(self, collection, ids, metadatas, documents):
        return True
    
    async def list_collections(self):
        return []
    
    async def query_chroma(self, collection, query_texts, n_results=10):
        return {"metadatas": [[]], "distances": [[]]}


class FakeIO:
    """Mock IO for testing URL extraction logic."""
    
    def __init__(self):
        self.last_visit = None
        self.last_fetch = None
        self.telemetry = None
        self.connector_chroma = FakeChroma()
    
    async def fetch_url(self, url: str, retries: int = 3, timeout_seconds=None) -> str:
        self.last_fetch = {"url": url, "retries": retries}
        return "<html><body><h1>Test Page</h1><p>Content</p><a href='https://link.example'>Link</a></body></html>"
    
    async def visit(self, url: str, timeout_seconds=None) -> str:
        self.last_visit = {"url": url}
        return "Test Page Content"
    
    def build_llm_payload(self, **kwargs):
        return {}
    
    async def query_llm_with_fallback(self, payload, **kwargs):
        return None


@pytest.mark.asyncio
async def test_visit_with_string_none_optional_url():
    """Test that string 'None' in optional_url is properly filtered out."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "visit",
        details={"optional_url": "None", "link_idea": "test page"},
    )
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, node.node_id, io)
    
    # Should fail gracefully, not crash
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    # Should attempt to extract URLs from other sources
    assert ActionResultKey.ERROR.value in payload or payload[ActionResultKey.SUCCESS.value] is False


@pytest.mark.asyncio
async def test_visit_extracts_urls_from_parent_search_results():
    """Test that visit nodes extract URLs from parent search results."""
    graph = IdeaDag(root_title="root")
    
    # Create a search node with results
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        },
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            # Keep exactly 1 URL so VisitLeafAction does not invoke LLM link-selection.
            {"title": "Python Wikipedia", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "snippet": "Python programming language"},
        ],
    }
    
    # Create a visit node with only link_idea (no optional_url)
    visit_node = graph.add_child(
        search_node.node_id,
        "visit python page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "Python programming language Wikipedia page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should successfully extract URL from parent search results
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.URL.value] is not None
    assert "wikipedia.org" in payload[ActionResultKey.URL.value]
    assert io.last_visit is not None


@pytest.mark.asyncio
async def test_visit_extracts_multiple_urls_from_search_results():
    """Test that visit nodes can extract multiple URLs when link_count > 1."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "Page 1", "url": "https://example.com/page1", "snippet": "Page 1 content"},
            {"title": "Page 2", "url": "https://example.com/page2", "snippet": "Page 2 content"},
            # Keep exactly 2 URLs so VisitLeafAction does not invoke LLM link-selection.
        ],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit pages",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "example pages",
            "link_count": 2,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    assert payload[ActionResultKey.SUCCESS.value] is True
    urls_visited = payload.get("urls_visited", [])
    assert len(urls_visited) >= 1


@pytest.mark.asyncio
async def test_visit_fails_when_no_urls_available():
    """Test that visit fails gracefully when no URLs can be extracted."""
    graph = IdeaDag(root_title="root")
    
    # Visit node with link_idea but no parent search results
    visit_node = graph.add_child(
        graph.root_id(),
        "visit page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "some page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.SUCCESS.value] is False
    assert ActionResultKey.ERROR.value in payload
    err = payload[ActionResultKey.ERROR.value].lower()
    assert ("no urls to visit" in err) or ("missing valid url" in err) or ("missing valid url or link_idea" in err)


@pytest.mark.asyncio
async def test_visit_handles_empty_search_results():
    """Test that visit handles empty search results gracefully."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "some page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.SUCCESS.value] is False
    assert ActionResultKey.ERROR.value in payload


@pytest.mark.asyncio
async def test_visit_handles_invalid_urls_in_search_results():
    """Test that visit filters out invalid URLs from search results."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "Valid", "url": "https://example.com", "snippet": "Valid URL"},
            {"title": "Invalid", "url": "not-a-url", "snippet": "Invalid URL"},
            {"title": "None", "url": None, "snippet": "None URL"},
            {"title": "Empty", "url": "", "snippet": "Empty URL"},
        ],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "example page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should extract the valid URL and ignore invalid ones
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.URL.value] == "https://example.com"


@pytest.mark.asyncio
async def test_visit_extracts_from_grandparent_search():
    """Test that visit extracts URLs from grandparent search nodes."""
    graph = IdeaDag(root_title="root")
    
    # Create grandparent search
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "Target Page", "url": "https://target.example.com", "snippet": "Target content"},
        ],
    }
    
    # Create intermediate think node
    think_node = graph.add_child(
        search_node.node_id,
        "think",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
    )
    think_node.status = IdeaNodeStatus.DONE
    
    # Create visit node under think node
    visit_node = graph.add_child(
        think_node.node_id,
        "visit target",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "target page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should extract URL from grandparent search
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.URL.value] == "https://target.example.com"


@pytest.mark.asyncio
async def test_visit_handles_missing_url_fields_in_search_results():
    """Test that visit handles search results with missing 'url' field."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "Page 1", "link": "https://example.com/page1", "snippet": "Page 1"},
            {"title": "Page 2", "href": "https://example.com/page2", "snippet": "Page 2"},
        ],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "example page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should extract URLs from 'link' or 'href' fields
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.URL.value] is not None
    assert "example.com" in payload[ActionResultKey.URL.value]


@pytest.mark.asyncio
async def test_visit_prefers_explicit_url_over_extraction():
    """Test that explicit optional_url is preferred over extraction."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "Search Result", "url": "https://search-result.example.com", "snippet": "Result"},
        ],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit explicit",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "https://explicit.example.com",
            "link_idea": "some page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should use explicit URL, not extracted one
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.URL.value] == "https://explicit.example.com"


@pytest.mark.asyncio
async def test_visit_string_none_with_search_results():
    """Test the specific error case: optional_url='None' string with link_idea and search results."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "Spotify API Docs", "url": "https://developer.spotify.com/documentation/web-api", "snippet": "Official Spotify Developer documentation"},
            {"title": "Spotify API Reference", "url": "https://developer.spotify.com/documentation/web-api/reference", "snippet": "API reference"},
            {"title": "Spotify Auth", "url": "https://developer.spotify.com/documentation/web-api/concepts/authorization", "snippet": "Authorization"},
        ],
    }
    
    # This simulates the exact error case from the logs
    visit_node = graph.add_child(
        search_node.node_id,
        "Visit the official Spotify Developer documentation pages",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "None",  # String "None", not None value
            "link_idea": "Official Spotify Developer Web API documentation",
            "link_count": 3,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should extract URLs from parent search results despite string "None"
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    # Should succeed by extracting from search results
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.URL.value] is not None
    assert "spotify.com" in payload[ActionResultKey.URL.value]


@pytest.mark.asyncio
async def test_extract_urls_from_parent_search_results_method():
    """Test the _extract_urls_from_parent_search_results method directly."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "A", "url": "https://a.example.com", "snippet": "A"},
            {"title": "B", "url": "https://b.example.com", "snippet": "B"},
            {"title": "C", "url": "https://c.example.com", "snippet": "C"},
        ],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    
    action = VisitLeafAction()
    urls = action._extract_urls_from_parent_search_results(graph, visit_node)
    
    assert len(urls) == 3
    assert "https://a.example.com" in urls
    assert "https://b.example.com" in urls
    assert "https://c.example.com" in urls


@pytest.mark.asyncio
async def test_extract_urls_handles_different_result_formats():
    """Test that extraction handles different search result formats."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [
            {"title": "URL field", "url": "https://url.example.com"},
            {"title": "Link field", "link": "https://link.example.com"},
            {"title": "Href field", "href": "https://href.example.com"},
        ],
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    
    action = VisitLeafAction()
    urls = action._extract_urls_from_parent_search_results(graph, visit_node)
    
    assert len(urls) == 3
    assert "https://url.example.com" in urls
    assert "https://link.example.com" in urls
    assert "https://href.example.com" in urls


@pytest.mark.asyncio
async def test_visit_diagnostic_production_error_scenario():
    """
    Diagnostic test for the exact production error scenario:
    - Visit node with optional_url='None' (string)
    - link_idea provided
    - link_count > 1
    - Parent search node exists but may not have accessible results
    """
    graph = IdeaDag(root_title="root")
    
    # Simulate search node that completed but results might not be accessible
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    
    # Case 1: Search results exist but in wrong format
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": None,  # Results might be None instead of empty list
    }
    
    visit_node = graph.add_child(
        search_node.node_id,
        "Visit the official Spotify Developer documentation pages",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "None",
            "link_idea": "Official Spotify Developer Web API documentation",
            "link_count": 3,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should handle gracefully - either extract URLs or fail with clear error
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    # Check error message format matches production logs
    if not payload[ActionResultKey.SUCCESS.value]:
        error = payload[ActionResultKey.ERROR.value]
        assert isinstance(error, str)
        # Error should mention link_count and link_idea
        assert "link_count" in error.lower() or "link_idea" in error.lower() or "url" in error.lower()


@pytest.mark.asyncio
async def test_visit_with_pending_parent_search():
    """Test visit node when parent search is still pending (not DONE)."""
    graph = IdeaDag(root_title="root")
    
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.PENDING  # Not done yet
    # No ACTION_RESULT yet
    
    visit_node = graph.add_child(
        search_node.node_id,
        "visit page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "some page",
            "link_count": 1,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # Should fail because parent search hasn't completed
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.SUCCESS.value] is False


@pytest.mark.asyncio
async def test_visit_extraction_depth_limit():
    """Test that URL extraction respects max_depth limit."""
    graph = IdeaDag(root_title="root")

    # Build ancestor search node near the root
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": "Deep", "url": "https://deep.example.com"}],
    }

    # Create a chain so the visit node is multiple levels below the search node
    n1 = graph.add_child(search_node.node_id, "think1", details={DetailKey.ACTION.value: IdeaActionType.THINK.value}, status=IdeaNodeStatus.DONE)
    n2 = graph.add_child(n1.node_id, "think2", details={DetailKey.ACTION.value: IdeaActionType.THINK.value}, status=IdeaNodeStatus.DONE)
    n3 = graph.add_child(n2.node_id, "think3", details={DetailKey.ACTION.value: IdeaActionType.THINK.value}, status=IdeaNodeStatus.DONE)
    visit_node = graph.add_child(n3.node_id, "visit", details={DetailKey.ACTION.value: IdeaActionType.VISIT.value})
    
    action = VisitLeafAction()
    # From visit_node -> think3 (1) -> think2 (2) -> think1 (3) -> search (4)
    # max_depth=2 should not reach the search node.
    urls = action._extract_urls_from_parent_search_results(graph, visit_node, max_depth=2)
    
    # Should not find URLs beyond depth limit
    assert len(urls) == 0
    
    # But with max_depth=5 should find it
    urls = action._extract_urls_from_parent_search_results(graph, visit_node, max_depth=5)
    assert len(urls) == 1
    assert "https://deep.example.com" in urls


@pytest.mark.asyncio
async def test_visit_handles_search_result_structure_variations():
    """Test that extraction handles various search result data structures."""
    graph = IdeaDag(root_title="root")
    
    # Test with results as list of strings (edge case)
    search_node1 = graph.add_child(
        graph.root_id(),
        "search1",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node1.status = IdeaNodeStatus.DONE
    search_node1.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": "not-a-list",  # Wrong type
    }
    
    visit_node1 = graph.add_child(
        search_node1.node_id,
        "visit1",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    
    action = VisitLeafAction()
    urls1 = action._extract_urls_from_parent_search_results(graph, visit_node1)
    assert len(urls1) == 0  # Should handle gracefully
    
    # Test with results as dict instead of list
    search_node2 = graph.add_child(
        graph.root_id(),
        "search2",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node2.status = IdeaNodeStatus.DONE
    search_node2.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": {"url": "https://dict.example.com"},  # Dict instead of list
    }
    
    visit_node2 = graph.add_child(
        search_node2.node_id,
        "visit2",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    
    urls2 = action._extract_urls_from_parent_search_results(graph, visit_node2)
    assert len(urls2) == 0  # Should handle gracefully
    
    # Test with proper list structure
    search_node3 = graph.add_child(
        graph.root_id(),
        "search3",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node3.status = IdeaNodeStatus.DONE
    search_node3.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": "Valid", "url": "https://valid.example.com"}],
    }
    
    visit_node3 = graph.add_child(
        search_node3.node_id,
        "visit3",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    
    urls3 = action._extract_urls_from_parent_search_results(graph, visit_node3)
    assert len(urls3) == 1
    assert "https://valid.example.com" in urls3


@pytest.mark.asyncio
async def test_visit_production_error_exact_scenario():
    """
    Exact reproduction of production error:
    - optional_url='None' (string)
    - link_idea provided
    - link_count=3
    - Parent search exists but visit still fails
    """
    graph = IdeaDag(root_title="root")
    
    # Create search node (may or may not have results accessible)
    search_node = graph.add_child(
        graph.root_id(),
        "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    search_node.status = IdeaNodeStatus.DONE
    
    # Simulate search that completed but results might be empty or inaccessible
    # This could happen if search failed or returned no results
    search_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [],  # Empty results - this is likely the issue
    }
    
    # Exact node structure from production logs
    visit_node = graph.add_child(
        search_node.node_id,
        "Visit the official Spotify Developer documentation pages discovered by search",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "None",  # String "None"
            "link_idea": "Official Spotify Developer Web API documentation — API overview, reference for endpoints (Tracks, Albums, Playlists, Search, Player, Users), authentication (Authorization Code, Client Credentials, PKCE), rate limits, SDKs and the Console",
            "link_count": 3,
        },
    )
    
    io = FakeIO()
    action = VisitLeafAction()
    
    payload = await action.execute(graph, visit_node.node_id, io)
    
    # This should fail with the exact error message from production
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.SUCCESS.value] is False
    error = payload[ActionResultKey.ERROR.value]
    
    # Verify we fail with a visit-related error message (exact text may vary by path)
    err_l = str(error).lower()
    assert ("no urls" in err_l) or ("missing valid url" in err_l)


@pytest.mark.asyncio
async def test_visit_with_search_results_but_wrong_action_type():
    """Test when parent has results but wrong action type."""
    graph = IdeaDag(root_title="root")
    
    # Create a think node (not search) with results-like data
    think_node = graph.add_child(
        graph.root_id(),
        "think",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
    )
    think_node.status = IdeaNodeStatus.DONE
    think_node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.THINK.value,  # Wrong action type
        "success": True,
        "results": [{"title": "A", "url": "https://a.example.com"}],
    }
    
    visit_node = graph.add_child(
        think_node.node_id,
        "visit",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": "some page",
            "link_count": 1,
        },
    )
    
    action = VisitLeafAction()
    urls = action._extract_urls_from_parent_search_results(graph, visit_node)
    
    # Should not extract from think node
    assert len(urls) == 0
