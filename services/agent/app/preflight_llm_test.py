"""
Standalone preflight test for LLM connector with DAG expansion.
Tests the same settings as the real expansion call.

Run with: python -m agent.app.preflight_llm_test
Or: cd /app && python -m agent.app.preflight_llm_test
"""
import asyncio
import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from agent.app.agent_io import AgentIO
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_dag_settings import load_idea_dag_settings
from shared.connector_config import ConnectorConfig
import pytest


@pytest.mark.asyncio
async def test_llm_preflight():
    """
    Test LLM with a minimal DAG step using the same settings as expansion.
    """
    print("=== LLM Preflight Test ===\n")
    
    # Load settings
    idea_settings = load_idea_dag_settings()
    model_name = idea_settings.get('expansion_model') or os.environ.get('MODEL_NAME', 'default')
    print(f"Using model: {model_name}")
    print(f"Reasoning effort: {idea_settings.get('reasoning_effort', 'high')}")
    print(f"Text verbosity: {idea_settings.get('text_verbosity', 'medium')}")
    print(f"Expansion max tokens: {idea_settings.get('expansion_max_tokens', 4096)}")
    print(f"Expansion temperature: {idea_settings.get('expansion_temperature', 0.4)}")
    print()
    
    # Initialize connectors
    config = ConnectorConfig()
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)
    
    try:
        async with connector_search, connector_http, connector_llm:
            # Initialize IO
            io = AgentIO(
                connector_llm=connector_llm,
                connector_search=connector_search,
                connector_http=connector_http,
                connector_chroma=connector_chroma,
            )
            
            # Create engine with settings
            engine = IdeaDagEngine(
                io=io,
                settings=idea_settings,
                model_name=connector_llm.model_name,
            )
            
            # Create a simple test graph
            from agent.app.idea_dag import IdeaDag
            test_mandate = "Test: Find one example of a simple acknowledgment word like 'OK'"
            test_graph = IdeaDag(root_title=test_mandate, root_details={"mandate": test_mandate})
            current_id = test_graph.root_id()
            
            print(f"Test mandate: {test_mandate}")
            print(f"Root node ID: {current_id}")
            print("\nRunning DAG step...\n")
            
            # Run one step - this will test expansion with real settings
            result_id = await engine.step(test_graph, current_id, 0)
            
            # Check results
            node = test_graph.get_node(current_id)
            print(f"\n=== Results ===")
            print(f"Node children: {len(node.children) if node else 0}")
            if node:
                print(f"Node status: {node.status.value}")
                print(f"Node details keys: {list(node.details.keys())}")
                if "expansion_error" in node.details:
                    print(f"Expansion error: {node.details['expansion_error']}")
                if node.children:
                    print(f"\nChildren created:")
                    for child_id in node.children:
                        child = test_graph.get_node(child_id)
                        if child:
                            print(f"  - {child.title[:60]}...")
            
            if node and len(node.children) > 0:
                print("\n✓ Preflight test PASSED - Expansion worked!")
                return True
            else:
                print("\n✗ Preflight test FAILED - No children created")
                return False
                
    except Exception as exc:
        print(f"\n✗ Preflight test FAILED with exception: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_llm_preflight())
    sys.exit(0 if success else 1)
