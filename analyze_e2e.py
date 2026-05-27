import json
from collections import Counter

# 1. Check amp_8 doc structure
with open("dataset_amp_8.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]

print("=== AMP_8 Document Structure ===")
for i, item in enumerate(data[:5]):
    docs = item["documents"]
    types = Counter(d["type"] for d in docs)
    print(f"Sample {i}: {len(docs)} docs, types={dict(types)}, wrong={item['wrong_answer'][:20]}")

# 2. Analyze e2e results
print("\n=== E2E Results Detail ===")
results = []
seen = set()
with open("e2e_checkpoint.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            key = f"{r['condition']}_{r['sample_idx']}"
            if key not in seen:
                seen.add(key)
                results.append(r)

for r in results:
    if r["condition"] == "amp_8":
        flag = "WRONG" if r["naive_wrong"] else ("OK" if r["naive_correct"] else "IDK")
        dedup_flag = "WRONG" if r["dedup_wrong"] else ("OK" if r["dedup_correct"] else "IDK")
        print(f"  amp_{r['sample_idx']}: {r['num_docs']}docs rm={r['num_removed']} "
              f"naive={r['naive_answer'][:20]}({flag}) dedup={r['dedup_answer'][:20]}({dedup_flag})")

# 3. Key insight
print("\n=== KEY INSIGHT ===")
amp_results = [r for r in results if r["condition"] == "amp_8"]
wrong_naive = sum(1 for r in amp_results if r["naive_wrong"])
wrong_dedup = sum(1 for r in amp_results if r["dedup_wrong"])
avg_rm = sum(r["num_removed"] for r in amp_results) / len(amp_results)
print(f"Amp8: naive wrong={wrong_naive}/{len(amp_results)}, dedup wrong={wrong_dedup}/{len(amp_results)}")
print(f"Avg removed: {avg_rm:.1f}")
print(f"Problem: amp_8 has 8 misinfo docs, but we only remove {avg_rm:.1f} on avg")
print("Need to remove MORE dependent docs to shift the majority vote")
