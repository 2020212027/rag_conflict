import json

lines = []
with open("rewritten_misinfo.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            lines.append(json.loads(line))

print(f"Total completed: {len(lines)}")
if lines:
    print(f"Last pilot_id: {lines[-1]['pilot_id']}")
    empty_variants = sum(1 for r in lines for v in r["variants"] if not v)
    print(f"Empty variants: {empty_variants}")
    ids = [r["pilot_id"] for r in lines]
    print(f"Duplicate IDs: {len(ids) - len(set(ids))}")
    print(f"ID range: {min(ids)} - {max(ids)}")
