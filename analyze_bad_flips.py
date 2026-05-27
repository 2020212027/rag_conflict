"""Analyze the 19 bad flip cases in clean E2E run."""
import json

# Load clean results
with open(r"d:\pythonProject\results_e2e_full_clean.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(l) for l in f if l.strip()]

# Load clean dataset for full document details
with open(r"d:\pythonProject\dataset_clean.jsonl", "r", encoding="utf-8") as f:
    dataset = [json.loads(l) for l in f if l.strip()]

bad_flips = [r for r in results if r["naive_correct"] and not r["dedup_correct"]]
print(f"Total bad flips: {len(bad_flips)}")

output_lines = []

for i, r in enumerate(bad_flips):
    idx = r["idx"]
    sample = dataset[idx]
    question = r["question"]
    gold = r["gold_answers"]
    naive_ans = r["naive_answer"]
    dedup_ans = r["dedup_answer"]
    num_docs = r["num_docs"]
    num_kept = r["num_kept"]
    num_removed = r["num_removed"]

    docs = sample["documents"]

    output_lines.append("=" * 70)
    output_lines.append(f"BAD FLIP #{i+1} (idx={idx})")
    output_lines.append(f"Q: {question}")
    output_lines.append(f"Gold: {gold}")
    output_lines.append(f"Naive answer: {naive_ans}")
    output_lines.append(f"Dedup answer: {dedup_ans}")
    output_lines.append(f"Docs: {num_docs} -> Kept: {num_kept} (removed {num_removed})")
    output_lines.append("")

    # Show all docs with first 150 chars
    output_lines.append("--- ALL DOCUMENTS ---")
    for j, d in enumerate(docs[:10]):
        text_preview = d["text"][:150].replace("\n", " ")
        output_lines.append(f"  Doc {j}: {text_preview}")
    output_lines.append("")

with open(r"d:\pythonProject\bad_flip_analysis.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"Written to bad_flip_analysis.txt ({len(output_lines)} lines)")
print(f"\nQuick stats:")
print(f"  Avg docs removed: {sum(r['num_removed'] for r in bad_flips)/len(bad_flips):.1f}")
print(f"  Cases with 1 removed: {sum(1 for r in bad_flips if r['num_removed']==1)}")
print(f"  Cases with 2 removed: {sum(1 for r in bad_flips if r['num_removed']==2)}")
print(f"  Cases with 3+ removed: {sum(1 for r in bad_flips if r['num_removed']>=3)}")
