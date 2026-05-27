import json

for condition in ["clean", "amp_8"]:
    path = f"results_naive_rag_{condition}.jsonl"
    records = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as file:
        original_records = [json.loads(line) for line in file if line.strip()]

    for record in original_records:
        if record["pilot_id"] in seen_ids:
            continue
        seen_ids.add(record["pilot_id"])
        records.append(record)

    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    exact_match = sum(1 for record in records if record["is_correct"]) / len(records) * 100
    wrong_rate = sum(1 for record in records if record["matches_wrong_answer"]) / len(records) * 100
    removed_duplicates = len(original_records) - len(records)
    print(
        f"{condition}: N={len(records)} | EM={exact_match:.1f}% | "
        f"Wrong={wrong_rate:.1f}% | dups_removed={removed_duplicates}"
    )
