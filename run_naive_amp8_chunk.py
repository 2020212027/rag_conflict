import json
import os
import time
from openai import OpenAI

MODEL = "gpt-4o-mini"
MAX_NEW = 20
INPUT_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
OUTPUT_PATH = r"d:\pythonProject\results_naive_rag_amp_8.jsonl"

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1",
)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_prompt(question, documents):
    doc_texts = "\n\n".join(f"[Document {index + 1}]\n{doc['text']}" for index, doc in enumerate(documents))
    return f"""Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


def call_llm(prompt):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            print(f"API error attempt {attempt + 1}: {error}", flush=True)
            time.sleep(5 * (attempt + 1))
    return ""


def normalized_exact_match(prediction, gold_answers):
    prediction_normalized = prediction.lower().strip().rstrip(".")
    return any(
        gold.lower().strip() in prediction_normalized or prediction_normalized in gold.lower().strip()
        for gold in gold_answers
    )


samples = load_jsonl(INPUT_PATH)
existing = load_jsonl(OUTPUT_PATH)
completed_ids = {result["pilot_id"] for result in existing}
print(f"Before: {len(completed_ids)}/{len(samples)}", flush=True)

processed = 0
with open(OUTPUT_PATH, "a", encoding="utf-8") as output_file:
    for sample in samples:
        if sample["pilot_id"] in completed_ids:
            continue
        prediction = call_llm(build_prompt(sample["question"], sample["documents"]))
        result = {
            "pilot_id": sample["pilot_id"],
            "question": sample["question"],
            "gold_answers": sample["gold_answers"],
            "wrong_answer": sample["wrong_answer"],
            "prediction": prediction,
            "is_correct": normalized_exact_match(prediction, sample["gold_answers"]),
            "matches_wrong_answer": normalized_exact_match(prediction, [sample["wrong_answer"]]) if sample["wrong_answer"] else False,
            "num_docs": sample["num_total_docs"],
        }
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_file.flush()
        completed_ids.add(sample["pilot_id"])
        processed += 1
        print(f"  processed pilot_id={sample['pilot_id']} total={len(completed_ids)}", flush=True)
        if processed >= MAX_NEW:
            break
        time.sleep(0.1)

print(f"After: {len(completed_ids)}/{len(samples)} | added={processed}", flush=True)
