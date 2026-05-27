import json, collections

with open("RAMDocs/RAMDocs_test.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

doc_counts = collections.Counter()
type_counts = collections.Counter()
multi_gold = 0
has_misinfo = 0
has_wrong = 0

for l in lines:
    d = json.loads(l)
    doc_counts[len(d["documents"])] += 1
    if len(d["gold_answers"]) > 1:
        multi_gold += 1
    if d.get("wrong_answers"):
        has_wrong += 1
    for doc in d["documents"]:
        type_counts[doc["type"]] += 1
        if doc["type"] == "misinfo":
            has_misinfo += 1

print(f"Total samples: {len(lines)}")
print(f"Doc count distribution: {dict(doc_counts)}")
print(f"Multi gold answers: {multi_gold} / {len(lines)}")
print(f"Samples with wrong_answers: {has_wrong}")
print(f"Doc type distribution: {dict(type_counts)}")
print(f"Samples containing misinfo docs: {has_misinfo}")
print()

# Check a few samples with misinfo
count = 0
for l in lines:
    d = json.loads(l)
    types = [doc["type"] for doc in d["documents"]]
    if "misinfo" in types:
        print(f"Q: {d['question'][:60]}")
        print(f"  gold: {d['gold_answers']}, wrong: {d['wrong_answers']}")
        print(f"  types: {types}")
        print(f"  answers: {[doc['answer'] for doc in d['documents']]}")
        print()
        count += 1
        if count >= 3:
            break
