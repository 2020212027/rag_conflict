import json

# 读取所有行，去重保留每个 pilot_id 的第一条
records = []
seen_ids = set()
with open("rewritten_misinfo.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r["pilot_id"] not in seen_ids:
                seen_ids.add(r["pilot_id"])
                records.append(r)

print(f"Before dedup: 51 lines, After dedup: {len(records)} unique pilot_ids")
print(f"IDs: {sorted([r['pilot_id'] for r in records])}")

# 检查每条的 variants 完整性
for r in records:
    empty = sum(1 for v in r["variants"] if not v)
    if empty:
        print(f"  WARNING: pilot_id={r['pilot_id']} has {empty} empty variants")

# 覆盖写入去重后的数据
with open("rewritten_misinfo.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nCleaned file written: {len(records)} records")
