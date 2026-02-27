# IdeaDAG System Improvement Plan
## Analysis of Test Results: 20260226_231551

### Critical Issues Identified

#### 🔴 **CRITICAL: Graph Never Expands (All Tests)**
**Symptom:**
- All 4 tests show graphs with only 1 node (root)
- Root node status: "active" (never transitions to "expanding" or "done")
- No children created
- No actions executed (visit_count: 0, search_count: 0)
- Only 2 LLM calls total (likely just finalize step)
- Graph visualization shows single root node with no edges

**Impact:**
- **Test 019**: Score 0.633 (should be ~0.95+) - No visits executed despite explicit requirement
- **Test 012**: Score 0.3 (should be ~0.9+) - No visits, cannot collect links
- **Test 013**: Score 0.69 (should be ~0.85+) - No visits, cannot explore path
- **Test 025**: Score 0.333 (should be ~0.9+) - No visits, cannot build chain

**Root Cause Hypothesis:**
1. Expansion policy not being called for root node
2. Engine exiting too early before expansion
3. Expansion failing silently
4. Root node not being recognized as needing expansion
5. Step function returning None immediately

**Evidence:**
- `events_count: 2` suggests only initialization + finalize
- `total_nodes: 1` confirms no expansion occurred
- `action_counts: {}` confirms no actions created
- `status: "active"` suggests node never processed

---

#### 🟡 **HIGH: Final Output Quality Issues**

**Test 019 - Malformed JSON:**
- Output contains duplicate JSON: `{"deliverable":"", "summary":""}\n{"deliverable":"..."}`
- Empty JSON object followed by populated one
- Suggests finalize step has JSON parsing/formatting bug

**Test 012, 013, 025 - Missing Evidence Messages:**
- LLM correctly identifies missing visit evidence
- But system never actually executes visits to gather evidence
- Final output is just "I cannot complete because X is missing" instead of executing X

---

#### 🟡 **MEDIUM: Hallucination in Test 019**
- Agent provided correct answers (1991, Guido van Rossum, 3.11.5) without visiting
- Suggests LLM used pre-trained knowledge instead of actual page content
- Validation caught this (visit_evidence: false) but score still 0.633 due to correct content

---

### Improvement Plans

## Phase 1: Fix Critical Graph Expansion Failure (Priority: CRITICAL)

### 1.1 Root Node Expansion Logic
**Problem:** Root node never expands into child nodes with actions

**Investigation Steps:**
1. Add detailed logging to `idea_engine.py`:
   - Log when `step()` is called
   - Log node status transitions
   - Log expansion policy calls
   - Log why expansion might be skipped

2. Check `_handle_expansion_node()`:
   - Verify it's being called for root node
   - Check if expansion policy returns empty candidates
   - Verify node status transitions (active → expanding → done)

3. Check expansion policy:
   - Verify LLM is being called
   - Check if JSON parsing fails silently
   - Verify candidates are being created

**Fix Implementation:**
```python
# In idea_engine.py step() method:
# Ensure root node always gets expanded if it has no children
if current_id == graph.root_id():
    root = graph.get_node(current_id)
    if root and not root.children and root.status == IdeaNodeStatus.ACTIVE:
        self._logger.info(f"[STEP {step_index}] Root node has no children - forcing expansion")
        return await self._handle_expansion_node(graph, current_id, step_index, None)
```

**Expected Outcome:**
- Root node expands into visit/search nodes
- Actions get executed
- Visit counts > 0
- Scores improve dramatically (0.3-0.7 → 0.8-0.95)

---

### 1.2 Step Function Early Exit Prevention
**Problem:** Engine may exit before processing root

**Fix:**
- Add minimum step guarantee (at least 3 steps before exit)
- Ensure root expansion happens in step 0 or 1
- Add validation: if root has no children after step 1, force expansion

---

### 1.3 Expansion Policy Error Handling
**Problem:** Silent failures in expansion

**Fix:**
- Wrap expansion in try/except with detailed logging
- If expansion returns 0 candidates, retry with different prompt
- Add fallback: if LLM fails, create default visit/search node based on mandate keywords

---

## Phase 2: Fix Final Output Quality (Priority: HIGH)

### 2.1 JSON Formatting in Finalize
**Problem:** Duplicate/malformed JSON in final output

**Fix:**
- In `idea_finalize.py`, ensure single JSON object output
- Parse LLM response, extract deliverable/summary, rebuild clean JSON
- Add validation: if JSON is malformed, fix it before returning

**Implementation:**
```python
# In idea_finalize.py finalize_run():
response = await io.query_llm(...)
# Parse and clean
try:
    data = json.loads(response)
    # If it's a string containing JSON, parse again
    if isinstance(data, str):
        data = json.loads(data)
    # Ensure single object
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
except json.JSONDecodeError:
    # Try to extract JSON from text
    import re
    json_match = re.search(r'\{[^{}]*"deliverable"[^{}]*\}', response)
    if json_match:
        data = json.loads(json_match.group())
    else:
        data = {"deliverable": response, "summary": ""}
```

---

### 2.2 Action Execution Before Finalize
**Problem:** Finalize runs even when no actions executed

**Fix:**
- Add check: if no actions executed, don't call finalize
- Instead, force expansion and retry
- Or return error message explaining why execution failed

---

## Phase 3: Improve Action Execution (Priority: HIGH)

### 3.1 Visit Action Execution
**Problem:** Visit actions created but never executed

**Investigation:**
- Check if visit nodes are being selected for execution
- Verify `_has_required_data()` doesn't block execution incorrectly
- Check if URL extraction fails silently

**Fix:**
- Add explicit URL validation before visit execution
- If URL missing, extract from mandate or parent nodes
- Add fallback: if URL extraction fails, create search node to find URL

---

### 3.2 Multi-Step Visit Chains
**Problem:** Tests 013, 025 require sequential visits but system doesn't chain them

**Fix:**
- Ensure visit nodes can extract URLs from previous visit results
- Use Chroma link storage to enable semantic link discovery
- Create expansion candidates that explicitly chain visits

---

## Phase 4: Improve Expansion Quality (Priority: MEDIUM)

### 4.1 URL Extraction from Mandate
**Problem:** Expansion doesn't extract URLs from mandate text

**Fix:**
- Pre-process mandate to extract URLs using regex
- If URL found in mandate, create visit node with that URL
- Add URL to expansion context so LLM sees it

**Implementation:**
```python
# In LlmExpansionPolicy._parse_candidates():
# Extract URLs from mandate/parent context
import re
url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
urls_in_context = re.findall(url_pattern, context_text)
# If candidate is visit and no URL provided, use extracted URL
```

---

### 4.2 Action Type Detection
**Problem:** Expansion may not create correct action types

**Fix:**
- Analyze mandate keywords to suggest action types
- "visit", "go to", "fetch" → visit action
- "search", "find", "look for" → search action
- "collect links", "extract" → visit + think actions

---

## Phase 5: Validation & Testing (Priority: MEDIUM)

### 5.1 Pre-Execution Validation
**Problem:** System runs to completion but produces invalid results

**Fix:**
- Add validation checkpoints:
  - After step 1: verify root expanded
  - After step 3: verify at least one action executed
  - Before finalize: verify actions produced results

**Implementation:**
```python
# In idea_engine.py run():
for step_index in range(max_steps):
    current_id = await self.step(graph, current_id, step_index)
    
    # Validation checkpoint
    if step_index == 1:
        root = graph.get_node(graph.root_id())
        if not root.children:
            self._logger.error("[VALIDATION] Root has no children after step 1 - forcing expansion")
            current_id = await self._handle_expansion_node(graph, graph.root_id(), step_index, None)
            if not current_id:
                return {"success": False, "error": "Failed to expand root node"}
    
    if step_index == 3:
        action_count = sum(1 for n in graph.nodes.values() if n.details.get(DetailKey.ACTION.value))
        if action_count == 0:
            self._logger.error("[VALIDATION] No actions created after step 3")
            # Force action creation or return error
```

---

### 5.2 Test-Specific Improvements

**Test 019 (Explicit Visit):**
- Ensure URL in mandate triggers visit node creation
- Verify visit executes before finalize
- Add validation: visit must succeed before claiming success

**Test 012 (Link Collection):**
- Create visit node for Main_Page URL
- After visit, create think node to extract and format links
- Ensure links are included in final output

**Test 013 (Exploration):**
- Create initial visit node for start URL
- After each visit, create expansion candidates for next link
- Chain visits until target URL reached
- Document path with adjacency evidence

**Test 025 (Link Chain):**
- Similar to 013 but simpler (just build chain)
- Ensure each visit extracts links for next visit
- Verify adjacency pairs in output

---

## Phase 6: Performance & Reliability (Priority: LOW)

### 6.1 Error Recovery
- If expansion fails, retry with simpler prompt
- If action fails, create alternative action
- If finalize fails, return partial results

### 6.2 Logging & Observability
- Add structured logging for each phase
- Log expansion candidates created
- Log action execution results
- Log merge operations

### 6.3 Memory Management
- Ensure Chroma storage happens after each action
- Verify memory retrieval works for context
- Clean up temporary collections

---

## Success Metrics

### Target Scores (After Fixes):
- **Test 019**: 0.633 → **0.95+** (visit executes, facts extracted correctly)
- **Test 012**: 0.3 → **0.9+** (visit executes, 10 links collected)
- **Test 013**: 0.69 → **0.85+** (path documented, target reached)
- **Test 025**: 0.333 → **0.9+** (chain built, adjacency verified)

### Key Indicators:
- ✅ Root node expands into children (total_nodes > 1)
- ✅ Actions executed (visit_count > 0, search_count > 0)
- ✅ Node status transitions (active → expanding → done)
- ✅ Final output contains actual results, not "missing evidence" messages
- ✅ JSON output is clean and well-formed

---

## Implementation Order

1. **IMMEDIATE**: Fix root expansion (Phase 1.1, 1.2, 1.3)
2. **URGENT**: Fix JSON formatting (Phase 2.1)
3. **HIGH**: Fix action execution (Phase 3.1, 3.2)
4. **MEDIUM**: Improve expansion (Phase 4.1, 4.2)
5. **MEDIUM**: Add validation (Phase 5.1, 5.2)
6. **LOW**: Performance improvements (Phase 6)

---

## Notes

- All tests show same pattern: graph never expands
- This suggests a systemic issue in the engine, not test-specific
- Fixing root expansion should improve all tests significantly
- Secondary issues (JSON formatting, action execution) can be addressed after expansion works
