import sys
from pathlib import Path

# Put badmodel-lab/ (the parent of the `agent` package) on the path so tests import
# `agent.*` as a top-level package and its relative imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
