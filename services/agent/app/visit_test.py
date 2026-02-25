"""
Comprehensive Visit Test Suite
Tests multiple visit scenarios with increased complexity and validation.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.connector_config import ConnectorConfig
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_dag import IdeaDag
from agent.app.idea_finalize import build_final_payload
from agent.app.agent_io import AgentIO
from agent.app.telemetry import TelemetrySession
from agent.app.idea_policies.base import DetailKey

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
_logger = logging.getLogger(__name__)


class VisitTestCase:
    """
    Single visit test case definition.
    """
    
    def __init__(
        self,
        name: str,
        mandate: str,
        required_urls: List[str],
        min_visits: int,
        min_links_per_visit: int = 2,
        required_content_checks: Optional[List[str]] = None,
        required_mentions: Optional[List[str]] = None,
    ):
        """
        Initialize test case.
        
        :param name: Test case name
        :param mandate: Task mandate
        :param required_urls: URLs that must be visited
        :param min_visits: Minimum number of visits required
        :param min_links_per_visit: Minimum links per visit
        :param required_content_checks: Content patterns to check for
        :param required_mentions: Terms that must appear in final output
        """
        self.name = name
        self.mandate = mandate
        self.required_urls = required_urls
        self.min_visits = min_visits
        self.min_links_per_visit = min_links_per_visit
        self.required_content_checks = required_content_checks or []
        self.required_mentions = required_mentions or []


TEST_CASES = [
    VisitTestCase(
        name="Basic Multi-URL Visit",
        mandate=(
            "You MUST visit these 3 websites directly using the visit action: "
            "(1) https://www.python.org/about/ "
            "(2) https://www.rust-lang.org/learn "
            "(3) https://go.dev/doc/ "
            "Do NOT use search - the URLs are already provided. "
            "For each website you visit: "
            "- Extract the main heading (H1 or first prominent heading) "
            "- Extract at least 2 links found on that page "
            "- Extract a brief summary (1-2 sentences) of what the page is about "
            "After visiting all 3 websites, provide a summary showing: "
            "- The heading from each page "
            "- At least 2 links from each page (showing the links you found) "
            "- A brief description of each page "
            "You must visit each URL to get the full page content - search results only show brief snippets. "
            "Visiting allows you to read the complete webpage content and extract links, while searches only provide brief overviews. "
            "The visit action returns both the page content AND a list of links found on the page - use those links in your report."
        ),
        required_urls=[
            "https://www.python.org/about/",
            "https://www.rust-lang.org/learn",
            "https://go.dev/doc/",
        ],
        min_visits=3,
        min_links_per_visit=2,
        required_mentions=["python.org", "rust-lang.org", "go.dev"],
    ),
    VisitTestCase(
        name="Wikipedia Deep Visit",
        mandate=(
            "Visit the Wikipedia page about 'Python (programming language)' at "
            "https://en.wikipedia.org/wiki/Python_(programming_language) "
            "and extract the following information directly from the page content: "
            "(1) The year Python was first released, "
            "(2) The name of Python's creator, "
            "(3) The current stable version number, "
            "(4) At least 5 links to related Wikipedia pages found on that page. "
            "You MUST visit the URL - do not rely on search results. "
            "Provide the information with citations from the actual page content. "
            "After visiting, also visit at least 2 of the related Wikipedia pages you found "
            "and extract one key fact from each."
        ),
        required_urls=[
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        ],
        min_visits=3,
        min_links_per_visit=5,
        required_content_checks=["1991", "Guido", "version"],
        required_mentions=["wikipedia", "python"],
    ),
    VisitTestCase(
        name="Multi-Domain Link Following",
        mandate=(
            "Visit https://www.python.org/ and extract at least 5 links from the main page. "
            "Then visit at least 2 of those links (choose interesting ones like documentation, downloads, or community pages) "
            "and extract: "
            "(1) The page title/heading, "
            "(2) At least 3 links found on that page, "
            "(3) A brief summary of what that page is about. "
            "Provide a comprehensive report showing the main page links you found and details from the 2+ pages you visited."
        ),
        required_urls=[
            "https://www.python.org/",
        ],
        min_visits=3,
        min_links_per_visit=3,
        required_mentions=["python.org"],
    ),
]


async def run_test_case(
    test_case: VisitTestCase,
    model_name: str,
    config: ConnectorConfig,
    idea_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single visit test case.
    
    :param test_case: Test case definition
    :param model_name: Model name to use
    :param config: Connector config
    :param idea_settings: Idea DAG settings
    :returns: Test results dictionary
    """
    _logger.info(f"\n{'='*70}")
    _logger.info(f"Running Test Case: {test_case.name}")
    _logger.info(f"{'='*70}")
    
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)
    
    async with connector_search, connector_http, connector_llm:
        await connector_search.init_search_api()
        await connector_chroma.init_chroma()
        
        telemetry = TelemetrySession(
            enabled=True,
            mandate=test_case.mandate,
            correlation_id=f"visit_test_{test_case.name.lower().replace(' ', '_')}",
        )
        
        agent_io = AgentIO(
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            telemetry=telemetry,
            collection_name="visit_test",
        )
        
        engine = IdeaDagEngine(
            io=agent_io,
            settings=idea_settings,
            model_name=model_name,
        )
        
        graph = IdeaDag(root_title=test_case.mandate, root_details={"mandate": test_case.mandate})
        current_id = graph.root_id()
        
        max_steps = 30
        for step_num in range(max_steps):
            try:
                result_id = await engine.step(graph, current_id, step_num)
                if result_id is None:
                    break
                current_id = result_id
                node = graph.get_node(current_id)
                if node and node.status.value == "done":
                    break
            except Exception as exc:
                _logger.error(f"Step {step_num} failed: {exc}", exc_info=True)
                break
        
        final_node = graph.get_node(current_id)
        if final_node:
            output = await build_final_payload(
                io=engine.io,
                settings=idea_settings,
                graph=graph,
                mandate=test_case.mandate,
                model_name=model_name,
            )
        else:
            output = {}
        
        telemetry.finish(success=output.get("success", False))
        
        visit_count = 0
        visit_urls = []
        visit_data = []
        all_links_found = []
        failed_visits = []
        visit_attempts = []
        required_urls_visited = set()
        
        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node:
                continue
            
            action_result = node.details.get(DetailKey.ACTION_RESULT.value)
            if not action_result or not isinstance(action_result, dict):
                continue
            
            action_type = action_result.get("action")
            if action_type == "visit":
                url = action_result.get("url", "") or node.title
                success = action_result.get("success", False)
                visit_attempts.append({
                    "node_id": node_id,
                    "url": url,
                    "success": success,
                    "action_result": action_result,
                })
                
                if success:
                    visit_count += 1
                    content = action_result.get("content", "")
                    content_full = action_result.get("content_full", "")
                    links = action_result.get("links", [])
                    
                    if content_full:
                        content_chars = len(content_full)
                    else:
                        content_chars = action_result.get("content_total_chars", 0)
                        if not content_chars or content_chars < 50:
                            if action_result.get("content_with_links"):
                                content_chars = len(action_result.get("content_with_links"))
                            elif content:
                                content_chars = len(content)
                            else:
                                content_chars = 0
                    
                    content_is_truncated = action_result.get("content_is_truncated", False)
                    
                    if url:
                        visit_urls.append(url)
                        for required_url in test_case.required_urls:
                            if required_url in url or url in required_url:
                                required_urls_visited.add(required_url)
                        
                        links_count = len(links) if isinstance(links, list) else 0
                        content_viable = bool(
                            (content_chars and content_chars > 50) or
                            (success and links_count >= test_case.min_links_per_visit)
                        )
                        visit_data.append({
                            "url": url,
                            "content_length": content_chars,
                            "content_viable": content_viable,
                            "content_is_truncated": content_is_truncated,
                            "content_preview": content[:200] if content else "",
                            "links_count": links_count,
                            "links": links[:10] if isinstance(links, list) else [],
                        })
                        if isinstance(links, list):
                            all_links_found.extend(links)
                else:
                    error = action_result.get("error", "Unknown error")
                    error_type = action_result.get("error_type", "Unknown")
                    http_status = action_result.get("http_status")
                    retryable = action_result.get("retryable", False)
                    root_cause = action_result.get("root_cause", error)
                    
                    failed_visits.append({
                        "node_id": node_id,
                        "url": url,
                        "error": error,
                        "error_type": error_type,
                        "http_status": http_status,
                        "retryable": retryable,
                        "root_cause": root_cause,
                    })
        
        root = graph.get_node(graph.root_id())
        if root:
            merged = root.details.get(DetailKey.MERGED_RESULTS.value, [])
            if isinstance(merged, list):
                for item in merged:
                    if not isinstance(item, dict):
                        continue
                    result = item.get("result")
                    if isinstance(result, dict):
                        if result.get("action") == "visit":
                            url = result.get("url", "")
                            success = result.get("success", False)
                            if success and url and url not in visit_urls:
                                visit_count += 1
                                visit_urls.append(url)
                                for required_url in test_case.required_urls:
                                    if required_url in url or url in required_url:
                                        required_urls_visited.add(required_url)
                                links = result.get("links", [])
                                if isinstance(links, list):
                                    all_links_found.extend(links)
                            elif not success:
                                error = result.get("error", "Unknown error")
                                error_type = result.get("error_type", "Unknown")
                                http_status = result.get("http_status")
                                retryable = result.get("retryable", False)
                                root_cause = result.get("root_cause", error)
                                
                                if not any(fv["url"] == url for fv in failed_visits):
                                    failed_visits.append({
                                        "url": url,
                                        "error": error,
                                        "error_type": error_type,
                                        "http_status": http_status,
                                        "retryable": retryable,
                                        "root_cause": root_cause,
                                    })
        
        final_deliverable = output.get('final_deliverable', '') or ''
        final_output_lower = str(final_deliverable).lower()
        
        viable_content_count = sum(1 for d in visit_data if d.get('content_viable', False))
        
        has_min_visits = visit_count >= test_case.min_visits
        has_required_urls = len(required_urls_visited) >= len(test_case.required_urls)
        has_min_links = all(d.get('links_count', 0) >= test_case.min_links_per_visit for d in visit_data)
        has_viable_content = viable_content_count >= test_case.min_visits
        
        has_required_mentions = True
        missing_mentions = []
        for mention in test_case.required_mentions:
            if mention.lower() not in final_output_lower:
                has_required_mentions = False
                missing_mentions.append(mention)
        
        has_content_checks = True
        missing_checks = []
        for check in test_case.required_content_checks:
            if check.lower() not in final_output_lower:
                has_content_checks = False
                missing_checks.append(check)
        
        critical_errors = []
        for failure in failed_visits:
            error_type = failure.get('error_type', '').lower()
            http_status = failure.get('http_status')
            if http_status and http_status >= 500:
                critical_errors.append(f"Server error (HTTP {http_status}) for {failure['url']}")
            elif error_type in ['timeouterror', 'timeout']:
                critical_errors.append(f"Timeout error for {failure['url']}")
            elif error_type == 'invalidurl':
                critical_errors.append(f"Invalid URL: {failure['url']}")
            elif not failure.get('retryable', False) and http_status not in [403, 401]:
                critical_errors.append(f"Non-retryable error ({error_type}) for {failure['url']}")
        
        has_critical_errors = len(critical_errors) > 0
        
        overall_pass = (
            has_min_visits and
            has_required_urls and
            has_viable_content and
            has_required_mentions and
            (has_content_checks if test_case.required_content_checks else True) and
            not has_critical_errors
        )
        
        print(f"\n{'='*70}")
        print(f"Test Case: {test_case.name}")
        print(f"{'='*70}")
        print(f"Total visit attempts: {len(visit_attempts)}")
        print(f"Successful visits: {visit_count}")
        print(f"Failed visits: {len(failed_visits)}")
        unique_visit_urls = list(set(visit_urls))
        print(f"Unique URLs visited: {len(unique_visit_urls)}")
        print(f"Required URLs visited: {len(required_urls_visited)}/{len(test_case.required_urls)}")
        if visit_urls:
            print(f"Visit URLs: {', '.join(unique_visit_urls[:5])}...")
        
        if failed_visits:
            print(f"\n{'='*70}")
            print(f"FAILED VISITS:")
            print(f"{'='*70}")
            for i, failure in enumerate(failed_visits[:3], 1):
                print(f"  {i}. {failure['url']}: {failure['error_type']}")
                if failure.get('error'):
                    print(f"     Error: {failure['error'][:100]}")
        
        print(f"\n{'='*70}")
        print(f"Visit Details:")
        print(f"{'='*70}")
        for i, data in enumerate(visit_data[:5], 1):
            print(f"  {i}. {data['url']}")
            print(f"     Content: {data['content_length']} chars, Links: {data['links_count']}")
            if data.get('content_preview'):
                print(f"     Preview: {data['content_preview'][:100]}...")
        
        print(f"\nValidation Checks:")
        print(f"  - Minimum visits ({test_case.min_visits}): {has_min_visits} ({visit_count})")
        print(f"  - Required URLs visited: {has_required_urls} ({len(required_urls_visited)}/{len(test_case.required_urls)})")
        print(f"  - Minimum links per visit ({test_case.min_links_per_visit}): {has_min_links}")
        print(f"  - Viable content: {has_viable_content} ({viable_content_count} viable)")
        if test_case.required_mentions:
            print(f"  - Required mentions: {has_required_mentions}")
            if missing_mentions:
                print(f"    Missing: {', '.join(missing_mentions)}")
        if test_case.required_content_checks:
            print(f"  - Content checks: {has_content_checks}")
            if missing_checks:
                print(f"    Missing: {', '.join(missing_checks)}")
        if has_critical_errors:
            print(f"  - Critical errors: {len(critical_errors)}")
            for error in critical_errors[:3]:
                print(f"    ⚠ {error}")
        
        try:
            from agent.app.idea_graph_visualizer import idea_graph_to_ascii
            print(f"\n{'='*70}")
            print(f"IdeaDAG Graph Visualization")
            print(f"{'='*70}")
            graph_ascii = idea_graph_to_ascii(graph)
            print(graph_ascii)
            print(f"{'='*70}")
        except Exception as exc:
            _logger.warning(f"Failed to render graph: {exc}")
        
        print(f"\nOverall result: {'PASSED' if overall_pass else 'FAILED'}")
        print(f"{'='*70}\n")
        
        return {
            "test_name": test_case.name,
            "passed": overall_pass,
            "visit_count": visit_count,
            "required_urls_visited": len(required_urls_visited),
            "total_links": len(set(all_links_found)),
            "viable_content_count": viable_content_count,
            "has_critical_errors": has_critical_errors,
            "validation": {
                "min_visits": has_min_visits,
                "required_urls": has_required_urls,
                "min_links": has_min_links,
                "viable_content": has_viable_content,
                "required_mentions": has_required_mentions,
                "content_checks": has_content_checks,
            },
        }


async def main():
    """
    Run all visit test cases.
    """
    model_name = os.environ.get("MODEL_NAME", "gpt-5-mini")
    
    config = ConnectorConfig()
    idea_settings = load_idea_dag_settings()
    idea_settings["log_dag_ascii"] = True
    idea_settings["log_dag_step_interval"] = 1
    idea_settings["allowed_actions"] = ["search", "visit", "save", "think"]
    idea_settings["expansion_max_context_nodes"] = 10
    idea_settings["evaluation_max_context_nodes"] = 10
    idea_settings["max_observation_chars"] = 10000
    
    results = []
    for test_case in TEST_CASES:
        try:
            result = await run_test_case(test_case, model_name, config, idea_settings)
            results.append(result)
        except Exception as exc:
            _logger.error(f"Test case {test_case.name} failed with exception: {exc}", exc_info=True)
            results.append({
                "test_name": test_case.name,
                "passed": False,
                "error": str(exc),
            })
    
    print(f"\n{'='*70}")
    print(f"Test Suite Summary")
    print(f"{'='*70}")
    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)
    print(f"Tests passed: {passed_count}/{total_count}")
    
    for result in results:
        status = "PASSED" if result.get("passed", False) else "FAILED"
        print(f"  {result['test_name']}: {status}")
        if result.get("visit_count"):
            print(f"    Visits: {result['visit_count']}, Links: {result.get('total_links', 0)}")
    
    print(f"{'='*70}\n")
    
    overall_success = passed_count == total_count
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
