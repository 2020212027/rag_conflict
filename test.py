import json
from collections import Counter

data = []
with open("RAMDocs_test.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

has_misinfo = [s for s in data if any(d["type"] == "misinfo" for d in s["documents"])]

# 1. 文档数分布
doc_counts = [len(s["documents"]) for s in has_misinfo]
print("=== 含 misinfo 样本的文档数分布 ===")
for k, v in sorted(Counter(doc_counts).items()):
    print(f"  {k} docs: {v} samples")
print()

# 2. 类型分布
type_totals = Counter()
for s in has_misinfo:
    for d in s["documents"]:
        type_totals[d["type"]] += 1
print(f"=== 文档类型总分布 (仅含 misinfo 的样本) ===")
for t, c in type_totals.most_common():
    print(f"  {t}: {c}")
print()

# 3. all_same_wrong 统计
same_count = 0
diff_count = 0
single_misinfo = 0
for s in has_misinfo:
    misinfo_answers = [d["answer"] for d in s["documents"] if d["type"] == "misinfo"]
    if len(misinfo_answers) == 1:
        single_misinfo += 1
    elif len(set(misinfo_answers)) == 1:
        same_count += 1
    else:
        diff_count += 1
print(f"=== Misinfo 答案模式 ===")
print(f"  只有 1 篇 misinfo: {single_misinfo}")
print(f"  多篇 misinfo 同一错误: {same_count}")
print(f"  多篇 misinfo 不同错误: {diff_count}")
print()

# 4. 无 correct doc 的样本
no_correct = [s for s in has_misinfo
              if not any(d["type"] == "correct" for d in s["documents"])]
print(f"=== 含 misinfo 但无 correct doc: {len(no_correct)} / {len(has_misinfo)} ===")
print()

# 5. Gap 样本检查
gap = [s for s in data
       if s.get("wrong_answers")
       and not any(d["type"] == "misinfo" for d in s["documents"])]
print(f"=== 有 wrong_answers 但无 misinfo doc: {len(gap)} ===")
noise_with_answer = 0
for s in gap:
    for d in s["documents"]:
        if d["type"] == "noise" and d.get("answer", "unknown") != "unknown":
            noise_with_answer += 1
            break
print(f"  其中 noise doc 含非 unknown answer: {noise_with_answer}")
