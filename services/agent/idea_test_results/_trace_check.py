import json

path = r'C:\Users\mukch\projects\euglena\services\agent\idea_test_results\20260227_060658_025_gpt-5-mini.jsonl'
lines = open(path).readlines()
print(f"Total trace entries: {len(lines)}")

# Show last 15 entries
for i, line in enumerate(lines[-15:]):
    idx = len(lines) - 15 + i
    entry = json.loads(line)
    event = entry.get("event", "?")
    payload = entry.get("payload", {})
    
    if event == "timing":
        name = payload.get("name", "")
        success = payload.get("success", "?")
        dur = payload.get("duration", 0)
        err = payload.get("error", "")
        print(f"  [{idx}] timing: {name} success={success} dur={dur:.1f}s")
        if err:
            print(f"    ERROR: {str(err)[:300]}")
    elif event == "connector_io":
        conn = payload.get("connector", "")
        direction = payload.get("direction", "")
        op = payload.get("operation", "")
        err = payload.get("error", "")
        print(f"  [{idx}] io: {conn} {direction} {op}")
        if err:
            print(f"    ERROR: {str(err)[:300]}")
    elif event == "summary":
        print(f"  [{idx}] SUMMARY: success={payload.get('success', '?')}")
    else:
        print(f"  [{idx}] {event}: {str(payload)[:200]}")
