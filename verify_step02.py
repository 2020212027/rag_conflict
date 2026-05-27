import json

records = []
with open(r"d:\pythonProject\rewritten_misinfo.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"=== Step 0.2 Verification ===")
print(f"Total records: {len(records)}")
print(f"All have 8 variants: {all(len(r['variants']) == 8 for r in records)}")

total_empty = sum(1 for r in records for v in r["variants"] if not v)
print(f"Empty variants: {total_empty}")

# 检查改写是否保留了错误答案
print(f"\n--- Quality Preview (first 3) ---")
for r in records[:3]:
    print(f"\nPilot #{r['pilot_id']} | Q: {r['question'][:60]}")
    print(f"  Wrong answer: {r['wrong_answer']}")
    print(f"  Seed: {r['seed_text'][:100]}...")
    print(f"  V0:   {r['variants'][0][:100]}...")
    print(f"  V7:   {r['variants'][7][:100]}...")
    # 检查 wrong_answer 是否出现在变体中
    wa = r["wrong_answer"].lower()
    hits = sum(1 for v in r["variants"] if wa in v.lower())
    print(f"  Wrong answer appears in {hits}/8 variants")

# 总体统计
wa_coverage = []
for r in records:
    wa = r["wrong_answer"].lower()
    hits = sum(1 for v in r["variants"] if wa in v.lower())
    wa_coverage.append(hits)

avg_cov = sum(wa_coverage) / len(wa_coverage)
full_cov = sum(1 for c in wa_coverage if c == 8)
zero_cov = sum(1 for c in wa_coverage if c == 0)
print(f"\n--- Wrong Answer Retention ---")
print(f"Avg variants containing wrong_answer: {avg_cov:.1f}/8")
print(f"Samples with 8/8 retention: {full_cov}/{len(records)}")
print(f"Samples with 0/8 retention: {zero_cov}/{len(records)}")
