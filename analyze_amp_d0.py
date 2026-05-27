"""Analyze the 17 amplified pairs that were judged D0 (false negatives)."""
import json
from itertools import combinations
from collections import defaultdict

EVIDENCE_PATH = r"d:\pythonProject\pilot_run_evidence.jsonl"
JUDGMENT_PATH = r"d:\pythonProject\pilot_run_judgments.jsonl"
AMP8_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
OUTPUT = r"d:\pythonProject\amp_d0_analysis.txt"

# Load evidence
evidence = [json.loads(l) for l in open(EVIDENCE_PATH, "r", encoding="utf-8")]
amp_evidence = [e for e in evidence if e["source"] == "amp"]

# Rebuild amp pairs (same logic as step_2_pilot_run.py)
amp_by_query = defaultdict(list)
for rec in amp_evidence:
    ev = rec.get("evidence")
    if not ev or not ev.get("evidence_span") or not ev.get("answer"):
        continue
    amp_by_query[rec["group_key"]].append(rec)

amp_pairs = []
for gk, units in amp_by_query.items():
    by_ans = defaultdict(list)
    for u in units:
        by_ans[u["evidence"]["answer"].lower().strip()].append(u)
    for ans, group in by_ans.items():
        if len(group) < 2:
            continue
        for i, j in combinations(range(len(group)), 2):
            amp_pairs.append(("amplified_pair", group[i], group[j]))
            if len(amp_pairs) >= 80:
                break
        if len(amp_pairs) >= 80:
            break
    if len(amp_pairs) >= 80:
        break

# Load judgments - amp judgments start after ind judgments (78)
judgments = [json.loads(l) for l in open(JUDGMENT_PATH, "r", encoding="utf-8")]
amp_judgments = [j for j in judgments if j["category"] == "amplified_pair"]

# Find D0 cases
lines = []
lines.append("=" * 70)
lines.append(f"AMPLIFIED PAIRS JUDGED D0: {sum(1 for j in amp_judgments if j['judgment'].get('level')=='D0')}")
lines.append("=" * 70)

d0_count = 0
for idx, (pair, judgment) in enumerate(zip(amp_pairs, amp_judgments)):
    if judgment["judgment"].get("level") != "D0":
        continue
    d0_count += 1
    _, ua, ub = pair
    ea = ua["evidence"]
    eb = ub["evidence"]
    
    lines.append(f"\n{'─'*70}")
    lines.append(f"Case {d0_count} | Pair idx={idx}")
    lines.append(f"Question: {ua['question']}")
    lines.append(f"Reason: {judgment['judgment'].get('key_evidence','?')}")
    lines.append(f"")
    lines.append(f"  Unit A answer: {ea.get('answer','')}")
    lines.append(f"  Unit A facts:  {ea.get('supporting_facts','')}")
    lines.append(f"  Unit A details: {ea.get('specific_details','')}")
    lines.append(f"  Unit A span:   {ea.get('evidence_span','')[:150]}")
    lines.append(f"")
    lines.append(f"  Unit B answer: {eb.get('answer','')}")
    lines.append(f"  Unit B facts:  {eb.get('supporting_facts','')}")
    lines.append(f"  Unit B details: {eb.get('specific_details','')}")
    lines.append(f"  Unit B span:   {eb.get('evidence_span','')[:150]}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved {d0_count} cases to {OUTPUT}")
