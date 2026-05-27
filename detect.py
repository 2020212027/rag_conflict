import json
from collections import Counter

data = []
with open("RAMDocs_test.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(f"总样本数: {len(data)}")

# ---- 有 misinfo 的样本占比 ----
has_misinfo = [s for s in data if any(d["type"] == "misinfo" for d in s["documents"])]
has_wrong = [s for s in data if s.get("wrong_answers")]
print(f"含 misinfo doc 的样本: {len(has_misinfo)} / {len(data)}")
print(f"有 wrong_answers 的样本: {len(has_wrong)} / {len(data)}")

# ---- 每条样本的文档数和类型分布 ----
for s in has_misinfo[:20]:
    types = Counter(d["type"] for d in s["documents"])
    misinfo_answers = [d["answer"] for d in s["documents"] if d["type"] == "misinfo"]
    unique_wrong = set(misinfo_answers)
    print(f"Q: {s['question'][:60]}")
    print(f"  docs: {len(s['documents'])}, {dict(types)}")
    print(f"  gold: {s['gold_answers']}, wrong: {s['wrong_answers']}")
    print(f"  misinfo answers: {misinfo_answers}, all same: {len(unique_wrong) == 1}")
    print()
