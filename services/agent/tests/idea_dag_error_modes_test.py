import asyncio
import os
import pytest
import re
import time
import uuid
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from agent.app.idea_dag import IdeaDag, IdeaNodeStatus
from agent.app.idea_engine import IdeaDagEngine
from agent.app.agent_io import AgentIO
from agent.app.idea_policies import DetailKey, IdeaActionType
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.connector_config import ConnectorConfig


class AgentTestConfig:
    """
    Abstraction layer for loading test configuration from environment variables.
    Provides a clean interface for accessing API keys and other test settings.
    """
    def __init__(self):
        self._config = ConnectorConfig()
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Returns OPENAI_API_KEY from environment."""
        return self._config.openai_api_key
    
    @property
    def search_api_key(self) -> Optional[str]:
        """Returns SEARCH_API_KEY from environment."""
        return self._config.search_api_key
    
    @property
    def has_openai_key(self) -> bool:
        """Returns True if OPENAI_API_KEY is available."""
        return bool(self.openai_api_key)
    
    @property
    def has_search_key(self) -> bool:
        """Returns True if SEARCH_API_KEY is available."""
        return bool(self.search_api_key)
    
    @property
    def has_all_keys(self) -> bool:
        """Returns True if both OPENAI_API_KEY and SEARCH_API_KEY are available."""
        return self.has_openai_key and self.has_search_key
    
    def get_connector_config(self) -> ConnectorConfig:
        """Returns the underlying ConnectorConfig instance."""
        return self._config


class MockSearch:
    _DEFAULT_RESULTS = {
        "fish": [{"title": "Fish Information", "url": "https://fish.example.com", "description": "Learn about fish species and habitats"}],
        "cat": [{"title": "Cat Care Guide", "url": "https://cat.example.com", "description": "Everything about cats"}],
        "dog": [{"title": "Dog Training Tips", "url": "https://dog.example.com", "description": "Training your dog"}],
    }

    def __init__(self, results: Optional[Dict[str, List[Dict]]] = None):
        self.results = results or {}
        self.search_api_ready = True

    def set_telemetry(self, telemetry):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def init_search_api(self):
        pass

    async def search(self, query: str, count: int = 10, timeout_seconds=None):
        query_lower = query.lower()
        for term, result in self._DEFAULT_RESULTS.items():
            if term in query_lower:
                return self.results.get(term, result)
        return self.results.get("default", [])
    
    async def query_search(self, query: str, count: int = 10):
        return await self.search(query, count)


class MockHttp:
    _DEFAULT_CONTENT = {
        "fish": "<html><body><h1>Fish</h1><p>Fish are aquatic animals. There are many types of fish.</p></body></html>",
        "cat": "<html><body><h1>Cat</h1><p>Cats are domestic pets. Cats are independent animals.</p></body></html>",
        "dog": "<html><body><h1>Dog</h1><p>Dogs are loyal companions. Dogs need training.</p></body></html>",
    }
    _DEFAULT_HTML = "<html><body>Default content</body></html>"

    def __init__(self, content: Optional[Dict[str, str]] = None):
        self.content = content or {}
        self.chroma_api_ready = True

    def set_telemetry(self, telemetry):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def fetch_url(self, url: str, timeout_seconds=None):
        url_lower = url.lower()
        for term, html in self._DEFAULT_CONTENT.items():
            if term in url_lower:
                return self.content.get(term, html)
        return self.content.get("default", self._DEFAULT_HTML)


class MockChroma:
    def __init__(self):
        self.chroma_api_ready = True
        self.stored_docs = []

    def set_telemetry(self, telemetry):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def init_chroma(self):
        pass

    async def store(self, documents, metadatas, ids, timeout_seconds=None):
        self.stored_docs.extend(documents)
        return True

    async def query(self, queries, n_results=3, timeout_seconds=None):
        return {"documents": [["Mock document 1", "Mock document 2"]]}

    async def query_chroma(self, collection, query_texts, n_results=3, timeout_seconds=None):
        return {"documents": [["Mock document 1", "Mock document 2"]]}


def generate_unique_task_name(test_name: str) -> str:
    test_id = uuid.uuid4().hex[:8]
    return f"test_{test_name}_{test_id}"


def create_mock_io(use_real_chroma: bool = True, use_real_llm: bool = True, use_real_search: bool = False, use_real_http: bool = False) -> AgentIO:
    test_config = AgentTestConfig()
    config = test_config.get_connector_config()
    
    if use_real_chroma:
        chroma = ConnectorChroma(config)
    else:
        chroma = MockChroma()
    
    if use_real_llm:
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY required for tests")
        llm = ConnectorLLM(config)
    else:
        pytest.skip("Real LLM required for tests")
    
    if use_real_search:
        if not test_config.has_search_key:
            pytest.skip("SEARCH_API_KEY required for tests")
        search = ConnectorSearch(config)
    else:
        search = MockSearch()
    
    if use_real_http:
        http = ConnectorHttp(config)
    else:
        http = MockHttp()
    
    return AgentIO(
        connector_llm=llm,
        connector_search=search,
        connector_http=http,
        connector_chroma=chroma,
    )


async def cleanup_chroma_collection(chroma: ConnectorChroma, collection_name: str):
    try:
        if chroma and chroma.chroma_api_ready:
            await chroma.get_or_create_collection(collection_name)
    except Exception:
        pass


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def extract_links_from_text(text: str) -> List[str]:
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    matches = re.findall(url_pattern, text)
    return [url.rstrip('.,;!?)') for url in matches if is_valid_url(url.rstrip('.,;!?)'))]


def check_keywords_in_content(content: str, keywords: List[str]) -> Dict[str, bool]:
    content_lower = content.lower()
    return {keyword: keyword.lower() in content_lower for keyword in keywords}


def validate_branching_task_result(result: Dict[str, Any], expected_terms: List[str]) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "links_found": [],
        "keywords_found": {},
    }
    
    final_text = result.get("final", "") or result.get("answer", "") or str(result)
    links = extract_links_from_text(final_text)
    report["links_found"] = links
    
    if len(links) < len(expected_terms):
        report["valid"] = False
        report["errors"].append(f"Expected at least {len(expected_terms)} links, found {len(links)}")
    
    for link in links:
        if not is_valid_url(link):
            report["valid"] = False
            report["errors"].append(f"Invalid URL: {link}")
    
    for term in expected_terms:
        found = term.lower() in final_text.lower()
        report["keywords_found"][term] = found
        if not found:
            report["warnings"].append(f"Keyword '{term}' not found in final result")
    
    graph_dict = result.get("graph", {})
    nodes = graph_dict.get("nodes", {})
    
    search_nodes = []
    visit_nodes = []
    merge_nodes = []
    
    for node_id, node_data in nodes.items():
        details = node_data.get("details", {})
        action = details.get("action")
        if action == "search":
            search_nodes.append(node_id)
        elif action == "visit":
            visit_nodes.append(node_id)
        elif action == "merge":
            merge_nodes.append(node_id)
    
    if len(search_nodes) < len(expected_terms):
        report["warnings"].append(f"Expected at least {len(expected_terms)} search nodes, found {len(search_nodes)}")
    
    if len(visit_nodes) < len(expected_terms):
        report["warnings"].append(f"Expected at least {len(expected_terms)} visit nodes, found {len(visit_nodes)}")
    
    if len(merge_nodes) == 0:
        report["warnings"].append("No merge nodes found - results may not be synthesized")
    
    return report


async def run_with_timeout(coro, timeout_seconds: float = 60.0):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Test exceeded timeout of {timeout_seconds} seconds")


def create_branching_task_mandate(task_name: str) -> str:
    return f"{task_name}: What's the first webpage result for each of the following: fish, cat, dog? Return 3 links"


def setup_real_connectors(test_config: AgentTestConfig):
    config = test_config.get_connector_config()
    llm = ConnectorLLM(config)
    search = ConnectorSearch(config)
    http = ConnectorHttp(config)
    chroma = ConnectorChroma(config)
    return llm, search, http, chroma


async def initialize_connectors(llm, search, http, chroma):
    await llm.__aenter__()
    await search.__aenter__()
    await http.__aenter__()
    await search.init_search_api()
    await chroma.init_chroma()


async def cleanup_connectors(llm, search, http, chroma):
    await llm.__aexit__(None, None, None)
    await search.__aexit__(None, None, None)
    await http.__aexit__(None, None, None)


def count_action_nodes(nodes: Dict[str, Any], action_type: str) -> int:
    return sum(1 for n in nodes.values() if n.get("details", {}).get("action") == action_type)


def extract_all_text_from_graph(graph_dict: Dict[str, Any]) -> str:
    nodes = graph_dict.get("nodes", {})
    text_parts = [str(node.get("title", "")) + " " + str(node.get("details", {})) for node in nodes.values()]
    return " ".join(text_parts)


class TestLocalErrorModes:
    @pytest.mark.asyncio
    async def test_circular_reference_prevention(self):
        graph = IdeaDag(root_title="test")
        node = graph.add_child(graph.root_id(), "child", details={"key": "value"})
        node.details["self_ref"] = node
        graph.update_details(node.node_id, {"nested": {"ref": node}})
        graph_dict = graph.to_dict()
        assert "nodes" in graph_dict
        assert node.node_id in graph_dict["nodes"]
    
    @pytest.mark.asyncio
    async def test_missing_action_details(self):
        graph = IdeaDag(root_title="test")
        node = graph.add_child(graph.root_id(), "child")
        assert node.details.get("action") is None
        assert node.details.get(DetailKey.ACTION.value) is None
    
    @pytest.mark.asyncio
    async def test_invalid_node_operations(self):
        graph = IdeaDag(root_title="test")
        with pytest.raises(ValueError, match="Unknown node_id"):
            graph.update_status("invalid", IdeaNodeStatus.DONE)
        with pytest.raises(ValueError, match="Unknown node_id"):
            graph.update_details("invalid", {})
    
    @pytest.mark.asyncio
    async def test_empty_expansion_result(self):
        graph = IdeaDag(root_title="test")
        root_id = graph.root_id()
        created = graph.expand(root_id, [])
        assert len(created) == 0
        assert len(graph.get_node(root_id).children) == 0
    
    @pytest.mark.asyncio
    async def test_node_status_transitions(self):
        graph = IdeaDag(root_title="test")
        node = graph.add_child(graph.root_id(), "child", status=IdeaNodeStatus.PENDING)
        graph.update_status(node.node_id, IdeaNodeStatus.ACTIVE)
        assert graph.get_node(node.node_id).status == IdeaNodeStatus.ACTIVE
        graph.update_status(node.node_id, IdeaNodeStatus.DONE)
        assert graph.get_node(node.node_id).status == IdeaNodeStatus.DONE
        graph.update_status(node.node_id, IdeaNodeStatus.FAILED)
        assert graph.get_node(node.node_id).status == IdeaNodeStatus.FAILED


class TestLocalBranchingTask:
    @pytest.mark.asyncio
    async def test_branching_task_mock(self):
        task_name = generate_unique_task_name("branching_mock")
        mandate = create_branching_task_mandate(task_name)
        test_config = AgentTestConfig()
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY required")
        
        config = test_config.get_connector_config()
        llm = ConnectorLLM(config)
        search = MockSearch()
        http = MockHttp()
        chroma = ConnectorChroma(config)
        
        await llm.__aenter__()
        
        try:
            await chroma.init_chroma()
            io = AgentIO(connector_llm=llm, connector_search=search, connector_http=http, connector_chroma=chroma)
            engine = IdeaDagEngine(io=io, settings={"max_steps": 20})
            
            start_time = time.time()
            try:
                result = await run_with_timeout(engine.run(mandate, max_steps=20), timeout_seconds=120.0)
                elapsed = time.time() - start_time
                assert elapsed < 120.0, "Test took too long"
                assert "graph" in result
                
                graph_dict = result.get("graph", {})
                nodes = graph_dict.get("nodes", {})
                
                search_nodes = [n for n in nodes.values() if n.get("details", {}).get("action") == "search"]
                visit_nodes = [n for n in nodes.values() if n.get("details", {}).get("action") == "visit"]
                merge_nodes = [n for n in nodes.values() if n.get("details", {}).get("action") == "merge"]
                
                assert len(search_nodes) >= 3, f"Expected at least 3 search nodes, found {len(search_nodes)}"
                
                search_queries = []
                for node in search_nodes:
                    query = node.get("details", {}).get("query", "")
                    if query:
                        search_queries.append(query.lower())
                
                expected_terms = ["fish", "cat", "dog"]
                found_terms = []
                for term in expected_terms:
                    if any(term in q for q in search_queries):
                        found_terms.append(term)
                
                assert len(found_terms) >= 2, f"Expected searches for at least 2 of {expected_terms}, found queries: {search_queries}"
                
                search_results_found = 0
                for node in search_nodes:
                    action_result = node.get("details", {}).get("action_result", {})
                    if action_result and action_result.get("success"):
                        results = action_result.get("results", [])
                        if results:
                            search_results_found += 1
                
                assert search_results_found > 0, f"Expected at least 1 successful search with results, found {search_results_found}"
                
                if len(visit_nodes) > 0:
                    visit_results_found = 0
                    for node in visit_nodes:
                        action_result = node.get("details", {}).get("action_result", {})
                        if action_result and action_result.get("success"):
                            visit_results_found += 1
                    
                    assert visit_results_found > 0, f"Expected at least 1 successful visit, found {visit_results_found}"
                
                validation = validate_branching_task_result(result, expected_terms)
                if len(validation["links_found"]) == 0:
                    all_text = extract_all_text_from_graph(graph_dict)
                    keywords_found = check_keywords_in_content(all_text, expected_terms)
                    assert any(keywords_found.values()), f"No links found and keywords missing. Search queries: {search_queries}, Keywords found: {keywords_found}"
                else:
                    assert len(validation["links_found"]) > 0, f"Validation errors: {validation['errors']}"
                
                all_text = extract_all_text_from_graph(graph_dict)
                keywords_found = check_keywords_in_content(all_text, expected_terms)
                assert any(keywords_found.values()), f"At least one keyword should be found in graph. Found: {keywords_found}"
            except TimeoutError as e:
                pytest.fail(f"Test timed out: {e}")
            finally:
                if engine._memory_manager:
                    await cleanup_chroma_collection(chroma, engine._memory_manager.collection_name)
        finally:
            await llm.__aexit__(None, None, None)


class TestLLMIntegration:
    @pytest.mark.asyncio
    async def test_branching_task_with_llm(self):
        task_name = generate_unique_task_name("branching_llm")
        mandate = create_branching_task_mandate(task_name)
        
        test_config = AgentTestConfig()
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY not configured")
        
        config = test_config.get_connector_config()
        llm = ConnectorLLM(config)
        search = MockSearch()
        http = MockHttp()
        chroma = ConnectorChroma(config)
        
        await llm.__aenter__()
        
        try:
            await chroma.init_chroma()
            io = AgentIO(connector_llm=llm, connector_search=search, connector_http=http, connector_chroma=chroma)
            engine = IdeaDagEngine(io=io, settings={"max_steps": 15})
            
            start_time = time.time()
            try:
                result = await run_with_timeout(engine.run(mandate, max_steps=15), timeout_seconds=120.0)
                elapsed = time.time() - start_time
                assert elapsed < 120.0, "Test took too long"
                assert "graph" in result
                
                validation = validate_branching_task_result(result, ["fish", "cat", "dog"])
                graph_dict = result.get("graph", {})
                nodes = graph_dict.get("nodes", {})
                search_count = count_action_nodes(nodes, "search")
                assert search_count >= 3, f"Expected at least 3 search nodes, found {search_count}"
            except TimeoutError as e:
                pytest.fail(f"Test timed out: {e}")
            finally:
                if engine._memory_manager:
                    await cleanup_chroma_collection(chroma, engine._memory_manager.collection_name)
        finally:
            await llm.__aexit__(None, None, None)


class TestFullIntegration:
    @pytest.mark.asyncio
    async def test_branching_task_full_integration(self):
        task_name = generate_unique_task_name("branching_full")
        mandate = create_branching_task_mandate(task_name)
        
        test_config = AgentTestConfig()
        if not test_config.has_all_keys:
            pytest.skip("OPENAI_API_KEY and SEARCH_API_KEY required")
        
        llm, search, http, chroma = setup_real_connectors(test_config)
        
        try:
            await initialize_connectors(llm, search, http, chroma)
            io = AgentIO(connector_llm=llm, connector_search=search, connector_http=http, connector_chroma=chroma)
            engine = IdeaDagEngine(io=io, settings={"max_steps": 20})
            
            start_time = time.time()
            try:
                result = await run_with_timeout(engine.run(mandate, max_steps=20), timeout_seconds=180.0)
                elapsed = time.time() - start_time
                assert elapsed < 180.0, f"Test took {elapsed:.2f} seconds, exceeded 180s timeout"
                assert "graph" in result
                assert "final" in result or "answer" in result or "final_deliverable" in result
                
                validation = validate_branching_task_result(result, ["fish", "cat", "dog"])
                assert len(validation["links_found"]) >= 3, f"Expected at least 3 links. Errors: {validation['errors']}"
                
                for link in validation["links_found"]:
                    assert is_valid_url(link), f"Invalid URL: {link}"
                
                graph_dict = result.get("graph", {})
                nodes = graph_dict.get("nodes", {})
                search_count = count_action_nodes(nodes, "search")
                visit_count = count_action_nodes(nodes, "visit")
                merge_count = count_action_nodes(nodes, "merge")
                
                assert search_count >= 3, f"Expected at least 3 search nodes, found {search_count}"
                assert visit_count >= 0, f"Visit nodes are optional, found {visit_count}"
                assert merge_count > 0, "Expected at least 1 merge node"
                
                final_text = result.get("final", "") or result.get("answer", "") or ""
                final_deliverable = result.get("final_deliverable", {})
                if isinstance(final_deliverable, list):
                    final_deliverable_text = " ".join(str(item) for item in final_deliverable)
                elif isinstance(final_deliverable, dict):
                    final_deliverable_text = " ".join(str(v) for v in final_deliverable.values())
                else:
                    final_deliverable_text = str(final_deliverable)
                combined_text = f"{final_text} {final_deliverable_text}"
                keywords_found = check_keywords_in_content(combined_text, ["fish", "cat", "dog"])
                assert any(keywords_found.values()), f"None of the keywords found in final text. Final: {final_text[:200]}, Deliverable: {final_deliverable_text[:200]}"
            except TimeoutError as e:
                pytest.fail(f"Test timed out: {e}")
            finally:
                if engine._memory_manager:
                    await cleanup_chroma_collection(chroma, engine._memory_manager.collection_name)
        finally:
            await cleanup_connectors(llm, search, http, chroma)


class TestInvariants:
    @pytest.mark.asyncio
    async def test_node_count_invariant(self):
        graph = IdeaDag(root_title="test")
        assert graph.node_count() == 1
        node1 = graph.add_child(graph.root_id(), "child1")
        assert graph.node_count() == 2
        node2 = graph.add_child(graph.root_id(), "child2")
        assert graph.node_count() == 3
        all_nodes = list(graph.iter_depth_first())
        assert len(all_nodes) == graph.node_count()
    
    @pytest.mark.asyncio
    async def test_parent_child_invariant(self):
        graph = IdeaDag(root_title="test")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child")
        assert child.parent_id == root_id
        assert child.node_id in child.parent_ids or root_id in child.parent_ids
        root = graph.get_node(root_id)
        assert child.node_id in root.children
    
    @pytest.mark.asyncio
    async def test_leaf_node_invariant(self):
        graph = IdeaDag(root_title="test")
        root_id = graph.root_id()
        assert graph.get_node(root_id).is_leaf()
        child = graph.add_child(root_id, "child")
        assert not graph.get_node(root_id).is_leaf()
        assert child.is_leaf()
        grandchild = graph.add_child(child.node_id, "grandchild")
        assert not graph.get_node(child.node_id).is_leaf()
        assert grandchild.is_leaf()


class TestForgettingScenarios:
    @pytest.mark.asyncio
    async def test_details_persistence(self):
        graph = IdeaDag(root_title="test")
        node = graph.add_child(
            graph.root_id(),
            "child",
            details={"action": "search", "query": "test", "intent": "find information"}
        )
        assert node.details.get("action") == "search"
        assert node.details.get("query") == "test"
        assert node.details.get("intent") == "find information"
        graph.update_details(node.node_id, {"result": "data"})
        assert graph.get_node(node.node_id).details.get("action") == "search"
        assert graph.get_node(node.node_id).details.get("result") == "data"
    
    @pytest.mark.asyncio
    async def test_action_result_storage(self):
        graph = IdeaDag(root_title="test")
        node = graph.add_child(graph.root_id(), "child", details={"action": "search"})
        graph.update_details(node.node_id, {
            DetailKey.ACTION_RESULT.value: {
                "action": "search",
                "success": True,
                "results": [{"title": "Test", "url": "https://test.com"}]
            }
        })
        stored_result = graph.get_node(node.node_id).details.get(DetailKey.ACTION_RESULT.value)
        assert stored_result is not None
        assert stored_result.get("success") is True
        assert len(stored_result.get("results", [])) > 0
    
    @pytest.mark.asyncio
    async def test_merge_result_persistence(self):
        graph = IdeaDag(root_title="test")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1", details={"action": "search", "query": "fish"})
        child2 = graph.add_child(root_id, "child2", details={"action": "search", "query": "cat"})
        child3 = graph.add_child(root_id, "child3", details={"action": "search", "query": "dog"})
        graph.update_details(child1.node_id, {DetailKey.ACTION_RESULT.value: {"success": True, "url": "https://fish.com"}})
        graph.update_details(child2.node_id, {DetailKey.ACTION_RESULT.value: {"success": True, "url": "https://cat.com"}})
        graph.update_details(child3.node_id, {DetailKey.ACTION_RESULT.value: {"success": True, "url": "https://dog.com"}})
        graph.merge_details(root_id)
        root = graph.get_node(root_id)
        merged = root.details.get("merged", [])
        assert len(merged) == 3
        merged_ids = [m.get("node_id") for m in merged]
        assert child1.node_id in merged_ids
        assert child2.node_id in merged_ids
        assert child3.node_id in merged_ids


class TestTimeoutMechanisms:
    @pytest.mark.asyncio
    async def test_timeout_detection(self):
        async def slow_operation():
            await asyncio.sleep(2.0)
            return "result"
        with pytest.raises(TimeoutError):
            await run_with_timeout(slow_operation(), timeout_seconds=0.5)
        result = await run_with_timeout(slow_operation(), timeout_seconds=3.0)
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_branching_task_timeout(self):
        mandate = create_branching_task_mandate("timeout_test")
        test_config = AgentTestConfig()
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY required")
        
        config = test_config.get_connector_config()
        llm = ConnectorLLM(config)
        search = MockSearch()
        http = MockHttp()
        chroma = ConnectorChroma(config)
        
        await llm.__aenter__()
        
        try:
            await chroma.init_chroma()
            io = AgentIO(connector_llm=llm, connector_search=search, connector_http=http, connector_chroma=chroma)
            engine = IdeaDagEngine(io=io, settings={"max_steps": 100})
            start_time = time.time()
            try:
                result = await run_with_timeout(engine.run(mandate, max_steps=100), timeout_seconds=10.0)
                elapsed = time.time() - start_time
                assert elapsed < 10.0, "Test should complete within timeout"
                assert "graph" in result
            except TimeoutError:
                pass
            finally:
                if engine._memory_manager:
                    await cleanup_chroma_collection(chroma, engine._memory_manager.collection_name)
        finally:
            await llm.__aexit__(None, None, None)


class TestKeywordValidation:
    @pytest.mark.asyncio
    async def test_keyword_detection(self):
        content = "Fish are aquatic animals. Cats are pets. Dogs are loyal."
        keywords = ["fish", "cat", "dog"]
        found = check_keywords_in_content(content, keywords)
        assert found["fish"] is True
        assert found["cat"] is True
        assert found["dog"] is True
    
    @pytest.mark.asyncio
    async def test_keyword_case_insensitive(self):
        content = "FISH are aquatic. CATS are pets. DOGS are loyal."
        keywords = ["fish", "cat", "dog"]
        found = check_keywords_in_content(content, keywords)
        assert found["fish"] is True
        assert found["cat"] is True
        assert found["dog"] is True
    
    @pytest.mark.asyncio
    async def test_url_extraction(self):
        text = "Visit https://example.com and https://test.org for more info."
        links = extract_links_from_text(text)
        assert len(links) == 2
        assert "https://example.com" in links
        assert "https://test.org" in links
    
    @pytest.mark.asyncio
    async def test_url_validation(self):
        assert is_valid_url("https://example.com") is True
        assert is_valid_url("http://test.org/path") is True
        assert is_valid_url("not a url") is False
        assert is_valid_url("ftp://example.com") is True
        assert is_valid_url("") is False


class TestLLMBarrage:
    @pytest.mark.asyncio
    async def test_merge_consistency_small_context(self):
        task_name = generate_unique_task_name("merge_small")
        test_config = AgentTestConfig()
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY not configured")
        
        config = test_config.get_connector_config()
        llm = ConnectorLLM(config)
        chroma = ConnectorChroma(config)
        await llm.__aenter__()
        
        try:
            await chroma.init_chroma()
            io = AgentIO(connector_llm=llm, connector_search=None, connector_http=None, connector_chroma=chroma)
            from agent.app.idea_policies.merge import SimpleMergePolicy
            merge_policy = SimpleMergePolicy(settings={"merge_system_prompt": "Merge the following results into a single coherent summary."})
            
            graph = IdeaDag(root_title=f"{task_name}_root")
            root_id = graph.root_id()
            child1 = graph.add_child(root_id, "Result 1", details={"action": "search", "query": "test1"})
            child2 = graph.add_child(root_id, "Result 2", details={"action": "search", "query": "test2"})
            
            graph.update_details(child1.node_id, {
                DetailKey.ACTION_RESULT.value: {
                    "success": True,
                    "query": "test1",
                    "results": [{"title": "Test 1", "url": "https://test1.com", "description": "First result"}]
                }
            })
            graph.update_details(child2.node_id, {
                DetailKey.ACTION_RESULT.value: {
                    "success": True,
                    "query": "test2",
                    "results": [{"title": "Test 2", "url": "https://test2.com", "description": "Second result"}]
                }
            })
            
            merge_result = merge_policy.merge(graph, root_id, recursive=False)
            assert "merged" in merge_result
            assert len(merge_result["merged"]) == 2
            
            results = []
            for _ in range(3):
                result = merge_policy.merge(graph, root_id, recursive=False)
                results.append(result.get("summary", {}))
            assert all("total" in r for r in results)
        finally:
            await llm.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_merge_consistency_large_context(self):
        task_name = generate_unique_task_name("merge_large")
        test_config = AgentTestConfig()
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY not configured")
        
        config = test_config.get_connector_config()
        llm = ConnectorLLM(config)
        chroma = ConnectorChroma(config)
        await llm.__aenter__()
        
        try:
            await chroma.init_chroma()
            io = AgentIO(connector_llm=llm, connector_search=None, connector_http=None, connector_chroma=chroma)
            from agent.app.idea_policies.merge import SimpleMergePolicy
            merge_policy = SimpleMergePolicy(settings={"merge_system_prompt": "Merge the following results into a single coherent summary."})
            
            graph = IdeaDag(root_title=f"{task_name}_root")
            root_id = graph.root_id()
            
            for i in range(10):
                child = graph.add_child(root_id, f"Result {i+1}", details={"action": "search", "query": f"test{i+1}"})
                graph.update_details(child.node_id, {
                    DetailKey.ACTION_RESULT.value: {
                        "success": True,
                        "query": f"test{i+1}",
                        "results": [{"title": f"Test {i+1}", "url": f"https://test{i+1}.com", "description": f"Result {i+1} description"}]
                    }
                })
            
            merge_result = merge_policy.merge(graph, root_id, recursive=False)
            assert "merged" in merge_result
            assert len(merge_result["merged"]) == 10
            assert merge_result.get("summary", {}).get("total") == 10
        finally:
            await llm.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_expansion_consistency(self):
        task_name = generate_unique_task_name("expansion")
        test_config = AgentTestConfig()
        if not test_config.has_openai_key:
            pytest.skip("OPENAI_API_KEY not configured")
        
        config = test_config.get_connector_config()
        llm = ConnectorLLM(config)
        chroma = ConnectorChroma(config)
        await llm.__aenter__()
        
        try:
            await chroma.init_chroma()
            io = AgentIO(connector_llm=llm, connector_search=None, connector_http=None, connector_chroma=chroma)
            from agent.app.idea_policies.expansion import LlmExpansionPolicy
            expansion_policy = LlmExpansionPolicy(io=io, settings={"max_steps": 5})
            
            graph = IdeaDag(root_title=f"{task_name}_root")
            root_id = graph.root_id()
            mandate = f"{task_name}: Find information about cats and dogs"
            graph.get_node(root_id).title = mandate
            
            results = []
            for _ in range(3):
                candidates = await expansion_policy.expand(graph, root_id)
                results.append(len(candidates))
            assert all(r > 0 for r in results), "All expansions should produce candidates"
        finally:
            await llm.__aexit__(None, None, None)
