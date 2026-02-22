
import asyncio
import logging
import os

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
_logger = logging.getLogger(__name__)


async def main():
    mandate = (
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
    )
    
    model_name = os.environ.get("MODEL_NAME", "gpt-5-mini")
    
    config = ConnectorConfig()
    idea_settings = load_idea_dag_settings()
    idea_settings["log_dag_ascii"] = True
    idea_settings["log_dag_step_interval"] = 1
    idea_settings["allowed_actions"] = ["search", "visit", "save"]
    
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)
    
    async with connector_search, connector_http, connector_llm:
        await connector_search.init_search_api()
        await connector_chroma.init_chroma()
        
        telemetry = TelemetrySession(
            enabled=True,
            mandate=mandate,
            correlation_id="visit_test_001",
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
        
        graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
        current_id = graph.root_id()
        
        max_steps = 20
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
                mandate=mandate,
                model_name=model_name,
            )
        else:
            output = {}
        
        telemetry.finish(success=output.get("success", False))
        
        from agent.app.idea_policies.base import DetailKey
        
        visit_count = 0
        visit_urls = []
        visit_data = []
        all_links_found = []
        
        for node_id in graph._nodes.keys():
            node = graph.get_node(node_id)
            if not node:
                continue
            
            action_result = node.details.get(DetailKey.ACTION_RESULT.value)
            if not action_result or not isinstance(action_result, dict):
                continue
            
            action_type = action_result.get("action")
            if action_type == "visit":
                success = action_result.get("success", False)
                if success:
                    visit_count += 1
                    url = action_result.get("url", "")
                    content = action_result.get("content", "")
                    links = action_result.get("links", [])
                    content_chars = action_result.get("content_total_chars", 0)
                    content_is_truncated = action_result.get("content_is_truncated", False)
                    
                    if url:
                        visit_urls.append(url)
                        content_viable = bool(content and len(content) > 50)
                        visit_data.append({
                            "url": url,
                            "content_length": content_chars,
                            "content_viable": content_viable,
                            "content_is_truncated": content_is_truncated,
                            "content_preview": content[:200] if content else "",
                            "links_count": len(links) if isinstance(links, list) else 0,
                            "links": links[:5] if isinstance(links, list) else [],
                        })
                        if isinstance(links, list):
                            all_links_found.extend(links)
        
        root = graph.get_node(graph.root_id())
        if root:
            merged = root.details.get(DetailKey.MERGED_RESULTS.value, [])
            if isinstance(merged, list):
                for item in merged:
                    if not isinstance(item, dict):
                        continue
                    result = item.get("result")
                    if isinstance(result, dict):
                        if result.get("action") == "visit" and result.get("success"):
                            url = result.get("url", "")
                            if url and url not in visit_urls:
                                visit_count += 1
                                visit_urls.append(url)
                                links = result.get("links", [])
                                if isinstance(links, list):
                                    all_links_found.extend(links)
        
        final_deliverable = output.get('final_deliverable', '') or ''
        final_output_lower = str(final_deliverable).lower()
        
        viable_content_count = sum(1 for d in visit_data if d.get('content_viable', False))
        
        has_visit_in_graph = visit_count >= 3
        has_python_org = "python.org" in final_output_lower
        has_rust_lang = "rust-lang.org" in final_output_lower or "rust" in final_output_lower
        has_go_dev = "go.dev" in final_output_lower or "golang" in final_output_lower
        has_multiple_sites = sum([has_python_org, has_rust_lang, has_go_dev]) >= 2
        has_links_mentioned = any(link in final_output_lower for link in ["http://", "https://", "www."]) or "link" in final_output_lower
        
        has_viable_content = viable_content_count >= 2
        has_visit_evidence = has_visit_in_graph and has_multiple_sites and has_viable_content
        
        print(f"\n{'='*70}")
        print(f"Visit Test Results")
        print(f"{'='*70}")
        print(f"Visits found in graph: {visit_count}")
        print(f"Unique URLs visited: {len(set(visit_urls))}")
        if visit_urls:
            print(f"Visit URLs: {', '.join(set(visit_urls))}")
        
        print(f"\nVisit Data Details:")
        for i, data in enumerate(visit_data, 1):
            print(f"  Visit {i}: {data['url']}")
            print(f"    Content length: {data['content_length']} chars")
            print(f"    Content viable (>50 chars): {data['content_viable']}")
            print(f"    Content truncated: {data['content_is_truncated']}")
            print(f"    Links found: {data['links_count']}")
            if data['links']:
                print(f"    Sample links: {', '.join(data['links'][:3])}")
            if data['content_preview']:
                print(f"    Content preview: {data['content_preview'][:100]}...")
        
        print(f"\nTotal links extracted: {len(set(all_links_found))}")
        print(f"\nFinal output length: {len(final_deliverable)} chars")
        print(f"Final output preview: {final_deliverable[:200]}...")
        print(f"\nEvidence checks:")
        print(f"  - At least 3 visits in graph: {has_visit_in_graph}")
        print(f"  - Python.org mentioned: {has_python_org}")
        print(f"  - Rust-lang.org mentioned: {has_rust_lang}")
        print(f"  - Go.dev mentioned: {has_go_dev}")
        print(f"  - Multiple sites (2+) mentioned: {has_multiple_sites}")
        print(f"  - Links mentioned in output: {has_links_mentioned}")
        print(f"  - Viable content from visits (>=2): {has_viable_content} ({viable_content_count} viable)")
        print(f"Overall result: {'PASSED' if has_visit_evidence else 'FAILED'}")
        print(f"{'='*70}\n")
        
        if has_visit_evidence:
            print(f"[PASSED] Agent executed {visit_count} visit action(s) to {len(set(visit_urls))} unique URLs")
            if visit_data:
                print(f"[INFO] Visit data is viable - content lengths: {[d['content_length'] for d in visit_data]}")
                print(f"[INFO] Links extracted: {len(set(all_links_found))} unique links found across all visits")
                print(f"[INFO] Viable content from {viable_content_count} out of {len(visit_data)} visits")
            return True
        else:
            reasons = []
            if visit_count < 3:
                reasons.append(f"not enough visits ({visit_count} < 3)")
            if not has_multiple_sites:
                reasons.append("not enough different sites covered")
            if not has_viable_content:
                reasons.append(f"not enough viable content ({viable_content_count} < 2)")
            
            print(f"[FAILED] Agent did not meet all requirements: {', '.join(reasons)}")
            if visit_count > 0:
                print(f"[INFO] Partial success - {visit_count} visit(s) executed, {viable_content_count} with viable content")
            return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
