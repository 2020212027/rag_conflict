import json, random

pilots = []
with open(r"d:\pythonProject\pilot_samples.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pilots.append(json.loads(line))

rewrites = {}
with open(r"d:\pythonProject\rewritten_misinfo.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            rewrites[r["pilot_id"]] = r

print(f"Pilots: {len(pilots)}, Rewrites: {len(rewrites)}")

AMP = {"clean": 0, "amp_2": 2, "amp_4": 4, "amp_8": 8}

for name, n_amp in AMP.items():
    path = rf"d:\pythonProject\dataset_{name}.jsonl"
    records = []

    for s in pilots:
        pid = s["_pilot_id"]
        rw = rewrites.get(pid)

        docs = [{"type": d["type"], "text": d["text"], "answer": d.get("answer", "unknown")}
                for d in s["documents"]]

        n_added = 0
        if n_amp > 0 and rw:
            for vi in range(min(n_amp, len(rw["variants"]))):
                if rw["variants"][vi]:
                    docs.append({
                        "type": "misinfo_amplified",
                        "text": rw["variants"][vi],
                        "answer": rw["wrong_answer"],
                    })
                    n_added += 1

        random.seed(42 + pid)
        random.shuffle(docs)

        records.append({
            "pilot_id": pid,
            "question": s["question"],
            "gold_answers": s["gold_answers"],
            "wrong_answer": s.get("_wrong_answer", ""),
            "documents": docs,
            "num_total_docs": len(docs),
            "num_correct": sum(1 for d in docs if d["type"] == "correct"),
            "num_misinfo_original": sum(1 for d in docs if d["type"] == "misinfo"),
            "num_misinfo_amplified": sum(1 for d in docs if d["type"] == "misinfo_amplified"),
            "num_noise": sum(1 for d in docs if d["type"] == "noise"),
            "amplification_level": n_amp,
        })

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_t = sum(r["num_total_docs"] for r in records) / len(records)
    avg_c = sum(r["num_correct"] for r in records) / len(records)
    avg_mo = sum(r["num_misinfo_original"] for r in records) / len(records)
    avg_ma = sum(r["num_misinfo_amplified"] for r in records) / len(records)
    avg_n = sum(r["num_noise"] for r in records) / len(records)
    mn = min(r["num_total_docs"] for r in records)
    mx = max(r["num_total_docs"] for r in records)

    print(f"\n{name} -> {path}: {len(records)} samples")
    print(f"  docs: avg={avg_t:.1f} (min={mn}, max={mx})")
    print(f"  C={avg_c:.1f} | M_orig={avg_mo:.1f} | M_amp={avg_ma:.1f} | N={avg_n:.1f}")
