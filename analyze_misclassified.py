"""Analyze the 15 misclassified independent pairs (D3/D4)."""
import json
from itertools import combinations

CLEAN_PATH = r"d:\pythonProject\dataset_clean.jsonl"
EVIDENCE_PATH = r"d:\pythonProject\pilot_v2_evidence.jsonl"
PAIRS_PATH = r"d:\pythonProject\pilot_v2_pairs.jsonl"
OUTPUT_PATH = r"d:\pythonProject\misclassified_analysis.txt"

# Load clean data
with open(CLEAN_PATH, "r", encoding="utf-8") as f:
    clean_data = [json.loads(l) for l in f]

# Load evidence
with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
    all_evidence = [json.loads(l) for l in f]

# Filter independent evidence (source=clean)
ind_evidence = [e for e in all_evidence if e["source"] == "clean"]

# Reconstruct the same pairing logic as pilot_v2
# Group by data_idx
from collections import defaultdict
evidence_by_query = defaultdict(list)
for ev in ind_evidence:
    evidence_by_query[ev["data_idx"]].append(ev)

# Rebuild pairs in same order
TARGET = 80
pairs_with_context = []
count = 0
for data_idx in sorted(evidence_by_query.keys()):
    query_ev = evidence_by_query[data_idx]
    question = query_ev[0]["question"]
    for i, j in combinations(range(len(query_ev)), 2):
        if count >= TARGET:
            break
        pairs_with_context.append({
            "pair_idx": count,
            "question": question,
            "data_idx": data_idx,
            "ev_a": query_ev[i],
            "ev_b": query_ev[j],
        })
        count += 1
    if count >= TARGET:
        break

# Load judgments
with open(PAIRS_PATH, "r", encoding="utf-8") as f:
    all_pairs = [json.loads(l) for l in f]

ind_pairs = [p for p in all_pairs if p["category"] == "independent_pair"]

# Match judgments to context
assert len(ind_pairs) == len(pairs_with_context), f"{len(ind_pairs)} != {len(pairs_with_context)}"

# Find misclassified
misclassified = []
for ctx, pair in zip(pairs_with_context, ind_pairs):
    level = pair["judgment"].get("level", "")
    if level in ("D3", "D4"):
        misclassified.append({**ctx, "judgment": pair["judgment"]})

# Now get original doc texts
output_lines = []
output_lines.append(f"{'='*70}")
output_lines.append(f"MISCLASSIFIED INDEPENDENT PAIRS: {len(misclassified)} cases")
output_lines.append(f"These are pairs of CORRECT docs (same answer) judged as D3/D4")
output_lines.append(f"{'='*70}\n")

for case_idx, case in enumerate(misclassified):
    data_idx = case["data_idx"]
    question = case["question"]
    judgment = case["judgment"]
    ev_a = case["ev_a"]["evidence_unit"]
    ev_b = case["ev_b"]["evidence_unit"]
    
    # Get original doc texts from clean_data
    item = clean_data[data_idx]
    correct_docs = [d for d in item["documents"] if d["type"] == "correct"]
    
    # Find which docs correspond to ev_a and ev_b by matching evidence_span
    doc_a_text = "N/A"
    doc_b_text = "N/A"
    for doc in correct_docs:
        span_a = ev_a.get("evidence_span", "")
        span_b = ev_b.get("evidence_span", "")
        if span_a and span_a in doc["text"]:
            doc_a_text = doc["text"]
        if span_b and span_b in doc["text"]:
            doc_b_text = doc["text"]

    output_lines.append(f"{'─'*70}")
    output_lines.append(f"Case {case_idx+1}/{len(misclassified)} | Level: {judgment['level']} | Score: {judgment.get('score','?')}")
    output_lines.append(f"Question: {question}")
    output_lines.append(f"Model's reason: {judgment.get('key_evidence','')}")
    output_lines.append(f"")
    output_lines.append(f"  [Evidence Unit A]")
    output_lines.append(f"    answer: {ev_a.get('answer','')}")
    output_lines.append(f"    facts:  {ev_a.get('supporting_facts','')}")
    output_lines.append(f"    details: {ev_a.get('specific_details','')}")
    output_lines.append(f"    span:   {ev_a.get('evidence_span','')}")
    output_lines.append(f"")
    output_lines.append(f"  [Evidence Unit B]")
    output_lines.append(f"    answer: {ev_b.get('answer','')}")
    output_lines.append(f"    facts:  {ev_b.get('supporting_facts','')}")
    output_lines.append(f"    details: {ev_b.get('specific_details','')}")
    output_lines.append(f"    span:   {ev_b.get('evidence_span','')}")
    output_lines.append(f"")
    output_lines.append(f"  [Original Doc A] ({len(doc_a_text)} chars)")
    output_lines.append(f"    {doc_a_text[:300]}")
    output_lines.append(f"")
    output_lines.append(f"  [Original Doc B] ({len(doc_b_text)} chars)")
    output_lines.append(f"    {doc_b_text[:300]}")
    output_lines.append(f"")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"Saved {len(misclassified)} cases to {OUTPUT_PATH}")
