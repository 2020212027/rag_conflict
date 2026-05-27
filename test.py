# step_0_1_filter_and_export.py

import json
import random
from collections import Counter

random.seed(42)

# ========================================
# 1. 加载数据
# ========================================
DATA_PATH = "RAMDocs_test.jsonl"  # ← 改成你的实际路径

data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"总样本数: {len(data)}")
print("=" * 60)

# ========================================
# 2. 过滤
# ========================================
pilot_samples = []
excluded_no_misinfo = 0
excluded_no_correct = 0
excluded_diff_wrong = 0
excluded_no_wrong_answers = 0

for s in data:
    docs = s["documents"]
    misinfo_docs = [d for d in docs if d["type"] == "misinfo"]
    correct_docs = [d for d in docs if d["type"] == "correct"]
    noise_docs = [d for d in docs if d["type"] == "noise"]

    # 必须有 misinfo
    if not misinfo_docs:
        excluded_no_misinfo += 1
        continue

    # 必须有 correct
    if not correct_docs:
        excluded_no_correct += 1
        continue

    # 必须有 wrong_answers
    if not s.get("wrong_answers") or len(s["wrong_answers"]) == 0:
        excluded_no_wrong_answers += 1
        continue

    # 所有 misinfo 必须指向同一个错误答案
    misinfo_answers = set(d["answer"] for d in misinfo_docs)
    if len(misinfo_answers) > 1:
        excluded_diff_wrong += 1
        continue

    # 通过所有过滤条件
    pilot_samples.append(s)

print(f"过滤结果:")
print(f"  通过:                         {len(pilot_samples)}")
print(f"  排除 - 无 misinfo doc:        {excluded_no_misinfo}")
print(f"  排除 - 无 correct doc:        {excluded_no_correct}")
print(f"  排除 - 无 wrong_answers:      {excluded_no_wrong_answers}")
print(f"  排除 - misinfo 指向不同错误:  {excluded_diff_wrong}")
print(f"  总计:                         {len(pilot_samples) + excluded_no_misinfo + excluded_no_correct + excluded_no_wrong_answers + excluded_diff_wrong}")
print("=" * 60)

# ========================================
# 3. 统计报告
# ========================================

# 3a. 文档数分布
doc_counts = [len(s["documents"]) for s in pilot_samples]
print(f"\n--- Pilot 样本文档数分布 ---")
for k, v in sorted(Counter(doc_counts).items()):
    print(f"  {k} docs: {v} samples")

# 3b. 类型分布
type_totals = Counter()
for s in pilot_samples:
    for d in s["documents"]:
        type_totals[d["type"]] += 1
print(f"\n--- 文档类型总分布 ---")
for t, c in type_totals.most_common():
    print(f"  {t}: {c}")
avg_correct = type_totals["correct"] / len(pilot_samples)
avg_misinfo = type_totals["misinfo"] / len(pilot_samples)
avg_noise = type_totals["noise"] / len(pilot_samples)
print(f"\n  平均 per sample: correct={avg_correct:.1f}, misinfo={avg_misinfo:.1f}, noise={avg_noise:.1f}")

# 3c. misinfo 数量分布
misinfo_counts = []
for s in pilot_samples:
    n = sum(1 for d in s["documents"] if d["type"] == "misinfo")
    misinfo_counts.append(n)
print(f"\n--- 每条样本的 misinfo 文档数分布 ---")
for k, v in sorted(Counter(misinfo_counts).items()):
    print(f"  {k} misinfo docs: {v} samples")

# 3d. correct 数量分布
correct_counts = []
for s in pilot_samples:
    n = sum(1 for d in s["documents"] if d["type"] == "correct")
    correct_counts.append(n)
print(f"\n--- 每条样本的 correct 文档数分布 ---")
for k, v in sorted(Counter(correct_counts).items()):
    print(f"  {k} correct docs: {v} samples")

print("=" * 60)

# ========================================
# 4. 为每条样本选择 seed misinfo doc
# ========================================
for s in pilot_samples:
    misinfo_docs = [d for d in s["documents"] if d["type"] == "misinfo"]
    # 如果有多篇 misinfo（都指向同一错误），随机选一篇作为 seed
    seed = random.choice(misinfo_docs)
    s["_seed_misinfo_text"] = seed["text"]
    s["_seed_misinfo_answer"] = seed["answer"]
    s["_wrong_answer"] = list(set(d["answer"] for d in misinfo_docs))[0]

# ========================================
# 5. 保存 pilot_samples.jsonl
# ========================================
PILOT_OUTPUT = "pilot_samples.jsonl"

with open(PILOT_OUTPUT, "w", encoding="utf-8") as f:
    for i, s in enumerate(pilot_samples):
        s["_pilot_id"] = i  # 给每条分配一个编号
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\n已保存 pilot 样本: {PILOT_OUTPUT} ({len(pilot_samples)} 条)")

# ========================================
# 6. 导出 seed_misinfo.jsonl (待 LLM 改写)
# ========================================
SEED_OUTPUT = "seed_misinfo.jsonl"

with open(SEED_OUTPUT, "w", encoding="utf-8") as f:
    for i, s in enumerate(pilot_samples):
        record = {
            "pilot_id": i,
            "question": s["question"],
            "gold_answers": s["gold_answers"],
            "wrong_answer": s["_wrong_answer"],
            "seed_text": s["_seed_misinfo_text"],
            "seed_answer": s["_seed_misinfo_answer"],
            "num_original_docs": len(s["documents"]),
            "num_original_correct": sum(1 for d in s["documents"] if d["type"] == "correct"),
            "num_original_misinfo": sum(1 for d in s["documents"] if d["type"] == "misinfo"),
            "num_original_noise": sum(1 for d in s["documents"] if d["type"] == "noise"),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"已保存 seed misinfo: {SEED_OUTPUT} ({len(pilot_samples)} 条)")

# ========================================
# 7. 手动检查：打印前 5 条 seed
# ========================================
print("\n" + "=" * 60)
print("前 5 条 seed misinfo 预览:")
print("=" * 60)

for i, s in enumerate(pilot_samples[:5]):
    print(f"\n--- Pilot #{i} ---")
    print(f"Question:     {s['question']}")
    print(f"Gold answer:  {s['gold_answers']}")
    print(f"Wrong answer: {s['_wrong_answer']}")
    print(f"Docs:         {len(s['documents'])} total "
          f"(C={sum(1 for d in s['documents'] if d['type']=='correct')}, "
          f"M={sum(1 for d in s['documents'] if d['type']=='misinfo')}, "
          f"N={sum(1 for d in s['documents'] if d['type']=='noise')})")
    print(f"Seed text:    {s['_seed_misinfo_text'][:200]}...")
    print()
