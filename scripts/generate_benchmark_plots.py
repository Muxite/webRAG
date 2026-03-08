"""
Generate benchmark plots and copy to docs/benchmark/ for README display.

This script automatically:
1. Finds the latest test run from JSON files
2. Generates the 3 core visualization images needed for documentation
3. Copies them to docs/benchmark/ with the correct names

Usage:
    python scripts/generate_benchmark_plots.py

The generated images are:
- executive_summary.png: Overall performance summary
- score_heatmap.png: Per-test, per-model performance heatmap
- efficiency_dashboard.png: Cost, time, and token efficiency metrics
"""

import sys
import os
import shutil
from pathlib import Path

# Set up paths before any imports
project_root = Path(__file__).parent.parent
services_dir = project_root / "services"
agent_app_dir = services_dir / "agent" / "app"

# Add services to path for agent.app and shared imports
sys.path.insert(0, str(services_dir))

# Import visualization modules (services/ is in sys.path, so agent.app is available)
try:
    from agent.app.testing.visualization_main import generate_docs_plots
except ImportError as e:
    print(f"Error: Could not import visualization modules: {e}")
    print("Make sure you're running from the project root and all dependencies are installed.")
    print(f"Current directory: {os.getcwd()}")
    print(f"Services dir: {services_dir}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """
    Generate the 3 core plots needed for documentation and copy to docs/benchmark/.
    """
    results_dir = services_dir / "agent" / "idea_test_results"
    benchmark_dir = project_root / "docs" / "benchmark"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    try:
        print("--- Generating Documentation Plots ---")
        mappings = generate_docs_plots(results_dir)
        
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        output_dir = results_dir / "plots_latest"
        
        print("\n--- Copying to docs/benchmark/ ---")
        copied = 0
        for source_name, dest_name in mappings.items():
            source = output_dir / source_name
            dest = benchmark_dir / dest_name
            
            if source.exists():
                shutil.copy2(source, dest)
                print(f"  [OK] {source_name} -> {dest_name}")
                copied += 1
            else:
                print(f"  [SKIP] {source_name} not found (skipping)")
        
        print(f"\n[OK] Copied {copied}/{len(mappings)} files to {benchmark_dir}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
