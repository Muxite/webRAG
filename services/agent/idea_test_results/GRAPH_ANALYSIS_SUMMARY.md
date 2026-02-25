# Graph Visualization and Analysis Summary

## Overview

This document summarizes the graph visualization and analysis functionality added to idea test result JSON files.

## Script: `add_graph_visualization.py`

### Features

1. **Graph Visualization**
   - Generates ASCII art visualization of the execution graph
   - Extracts graph data structure (nodes and edges) for programmatic access
   - Cleans ANSI color codes from output for JSON storage

2. **Graph Analysis**
   - **Action Distribution**: Counts and ratios of different action types (think, search, visit, merge, etc.)
   - **Think Action Analysis**: Identifies excessive use of "think" actions
   - **Title Repetition Detection**: Finds nodes with identical or very similar titles
   - **Action Diversity**: Checks for low diversity in action types

3. **Issue Detection**
   - **Excessive Think Actions**: Flags when >50% of nodes are "think" actions (high severity) or >30% (medium severity)
   - **Repeated Titles**: Identifies nodes with identical titles
   - **Similar Long Titles**: Detects merge nodes that duplicate root node mandates
   - **Low Action Diversity**: Warns when graphs have too few action types

## Usage

### Single File
```bash
python add_graph_visualization.py <json_file_path>
```

### Directory (all graph files)
```bash
python add_graph_visualization.py <directory_path> --pattern "*_graph_r*.json"
```

## Output Structure

The script adds a `graph_visualization` section to the `execution` object:

```json
{
  "execution": {
    "graph_visualization": {
      "ascii": "...ASCII visualization...",
      "graph_data": {
        "nodes": [...],
        "edges": [...]
      },
      "analysis": {
        "total_nodes": 4,
        "action_counts": {...},
        "think_count": 1,
        "think_ratio": 0.25,
        "issues": [...],
        "issue_count": 1
      }
    }
  }
}
```

## Issues Found in Test File: `20260224_005057_014_gpt-5-mini_graph_r1.json`

### Graph Statistics
- **Total Nodes**: 4
- **Think Actions**: 1 (25.0%)
- **Action Distribution**: think (1), search (1), merge (1)
- **Action Diversity**: 3 unique action types

### Issues Detected

1. **Similar Long Titles (Medium Severity)**
   - **Issue**: Found 1 pair of nodes with very similar long titles
   - **Details**: The merge node duplicates the root node's mandate title
   - **Recommendation**: Review merge nodes - they may be duplicating the root node's mandate unnecessarily
   - **Impact**: This indicates redundant planning/merging that could be optimized

### Analysis

The graph shows a relatively simple structure with:
- 1 root node (the main task)
- 1 think action (planning)
- 1 search action (finding news articles)
- 1 merge action (combining results)

**Key Observations:**
1. **Repetition Issue**: The merge node's title is essentially identical to the root node's mandate, suggesting the merge operation may be unnecessary or could be simplified.

2. **Think Action Ratio**: At 25%, the think action ratio is acceptable (below the 30% threshold), but the graph is quite small (only 4 nodes), so this may not be representative of larger graphs.

3. **Missing Actions**: Notably, there are **0 visit actions** despite the task requiring following links from a news article. This suggests the graph execution may have been incomplete or the task was completed without visiting the required links.

## Recommendations

1. **Optimize Merge Nodes**: Review merge operations to ensure they're not simply duplicating root mandates
2. **Complete Execution**: Ensure visit actions are properly executed when following links
3. **Monitor Think Ratio**: For larger graphs, keep think action ratio below 30%
4. **Action Diversity**: Encourage use of diverse action types (search, visit, save) rather than excessive planning
