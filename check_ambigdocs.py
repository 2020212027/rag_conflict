import json

with open("d:/pythonProject/AmbigDocs_data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
s = data[0]
print(f"Keys: {list(s.keys())}")
print(f"Question: {s['question']}")
print(f"Entity: {s.get('ambiguous_entity')}")
print(f"Num docs: {len(s['documents'])}")

for i, d in enumerate(s["documents"][:4]):
    print(f"  doc[{i}] title={d.get('title')}, answer={d.get('answer')}")
    print(f"    text[:120]: {d['text'][:120]}")

print("---")
num_docs = [len(s["documents"]) for s in data]
num_answers = [len(set(d["answer"] for d in s["documents"])) for s in data]
print(f"docs/sample: min={min(num_docs)} max={max(num_docs)} avg={sum(num_docs)/len(num_docs):.1f}")
print(f"unique answers/sample: min={min(num_answers)} max={max(num_answers)} avg={sum(num_answers)/len(num_answers):.1f}")

# Check doc keys
print(f"Doc keys: {list(data[0]['documents'][0].keys())}")

# Check if there's type/gold_answers/wrong_answers fields
print(f"Has 'gold_answers'? {'gold_answers' in data[0]}")
print(f"Has 'wrong_answers'? {'wrong_answers' in data[0]}")
print(f"Has doc 'type'? {'type' in data[0]['documents'][0]}")

# Show a few more samples
for idx in [0, 1, 100]:
    s = data[idx]
    answers = [d["answer"] for d in s["documents"]]
    print(f"\n[{idx}] Q={s['question']}")
    print(f"  entity={s['ambiguous_entity']}")
    print(f"  answers={answers}")
    print(f"  titles={[d['title'] for d in s['documents']]}")
