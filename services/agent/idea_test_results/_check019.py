import json

import glob
matches = sorted(glob.glob(r"C:\Users\mukch\projects\euglena\services\agent\idea_test_results\*019*_graph_r1.json"))
path = matches[-1]
print(f"Using: {path}")
print()
d = json.loads(open(path).read())
nodes = d.get("execution", {}).get("graph", {}).get("nodes", {})

for nid, node in nodes.items():
    details = node.get("details", {})
    action = details.get("action")
    title = node.get("title", "")[:80]
    status = node.get("status", "?")
    score = node.get("score")
    print(f"\n=== {nid[:8]} [{status}] score={score} action={action} ===")
    print(f"  Title: {title}")
    
    result = details.get("action_result", {})
    if result and isinstance(result, dict):
        print(f"  Result success: {result.get('success')}")
        content = result.get("content", "")
        if content:
            print(f"  Content length: {len(content)}")
            print(f"  Content preview: {content[:500]}...")
        links = result.get("_links_inline", "")
        if links:
            print(f"  Links count: {links.count('[link:')}")
        source_url = result.get("source_url")
        if source_url:
            print(f"  Source URL: {source_url}")
        error = result.get("error")
        if error:
            print(f"  ERROR: {error}")

# Check validation
val = d.get("validation", {})
print(f"\n--- VALIDATION ---")
print(f"Passed: {val.get('overall_passed')}, Score: {val.get('overall_score')}")
for check in val.get("grep_validations", []):
    icon = "+" if check.get("passed") else "x"
    print(f"  [{icon}] {check.get('check','?')}: {check.get('score',0):.1f} - {check.get('reason','')[:120]}")
llm_val = val.get("llm_validation", {})
if llm_val:
    for r in (llm_val.get("reasons") or []):
        print(f"  LLM: {r[:200]}")
