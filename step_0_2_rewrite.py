import json, time, os
from openai import OpenAI

SEED_INPUT = r"d:\pythonProject\seed_misinfo.jsonl"
REWRITE_OUTPUT = r"d:\pythonProject\rewritten_misinfo.jsonl"
NUM_VARIANTS = 8
MODEL = "gpt-4o-mini"
RESUME = True

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

STYLE_HINTS = [
    "Use a more formal, encyclopedic tone.",
    "Use a conversational, journalistic tone.",
    "Restructure the passage by starting with the conclusion and then providing details.",
    "Break the content into shorter, punchier sentences.",
    "Use passive voice more frequently and adopt an academic style.",
    "Rewrite as if summarizing the key points for a brief report.",
    "Use a narrative storytelling approach.",
    "Reorganize the information in reverse chronological or logical order.",
]


def call_rewrite(seed_text: str, variant_index: int) -> str:
    prompt = f"""Rewrite the following passage to convey the same information but with different wording, sentence structure, and phrasing. 

IMPORTANT RULES:
1. Keep ALL factual claims exactly identical — do not change any names, numbers, dates, or facts.
2. Do not add any new information that is not in the original.
3. Do not remove any factual claims from the original.
4. Only change the wording, structure, and style.

Style hint: {STYLE_HINTS[variant_index % len(STYLE_HINTS)]}

Original passage:
{seed_text}

Rewritten passage:"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Warning: API error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return ""


# 加载 seed
seeds = []
with open(SEED_INPUT, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            seeds.append(json.loads(line))

# 断点续跑
completed_ids = set()
if RESUME and os.path.exists(REWRITE_OUTPUT):
    with open(REWRITE_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_ids.add(json.loads(line)["pilot_id"])
    print(f"Already completed {len(completed_ids)} samples, skipping")
else:
    open(REWRITE_OUTPUT, "w").close()

remaining = len(seeds) - len(completed_ids)
print(f"To process: {remaining} samples, API calls needed: {remaining * NUM_VARIANTS}")

start = time.time()
api_calls = 0

for idx, seed in enumerate(seeds):
    pid = seed["pilot_id"]
    if pid in completed_ids:
        continue

    print(f"[{idx+1}/{len(seeds)}] Pilot #{pid}: {seed['question'][:50]}...")

    variants = []
    for v in range(NUM_VARIANTS):
        variants.append(call_rewrite(seed["seed_text"], v))
        api_calls += 1
        time.sleep(0.15)

    record = {
        "pilot_id": pid,
        "question": seed["question"],
        "gold_answers": seed["gold_answers"],
        "wrong_answer": seed["wrong_answer"],
        "seed_text": seed["seed_text"],
        "seed_answer": seed["seed_answer"],
        "variants": variants,
        "num_original_docs": seed["num_original_docs"],
        "num_original_correct": seed["num_original_correct"],
        "num_original_misinfo": seed["num_original_misinfo"],
        "num_original_noise": seed["num_original_noise"],
    }

    with open(REWRITE_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    done = idx + 1 - len(completed_ids)
    elapsed = time.time() - start
    eta = elapsed / done * (len(seeds) - len(completed_ids) - done) / 60
    print(f"  Done | calls={api_calls} | ETA={eta:.1f}min")

# === 质量预览 ===
print(f"\n{'='*50}")
print(f"Finished! API calls: {api_calls}, Time: {(time.time()-start)/60:.1f}min\n")

preview = []
with open(REWRITE_OUTPUT, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            preview.append(json.loads(line))
        if len(preview) >= 3:
            break

for r in preview:
    print(f"--- Pilot #{r['pilot_id']} | Wrong: {r['wrong_answer']} ---")
    print(f"Seed:  {r['seed_text'][:120]}...")
    print(f"V0:    {r['variants'][0][:120]}...")
    print(f"V7:    {r['variants'][7][:120]}...")
    empty = sum(1 for v in r["variants"] if not v)
    if empty:
        print(f"  WARNING: {empty} empty variants")
    print()

# 总体统计
all_records = []
with open(REWRITE_OUTPUT, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

total_empty = sum(1 for r in all_records for v in r["variants"] if not v)
print(f"Total records: {len(all_records)}")
print(f"Total empty variants: {total_empty}")
