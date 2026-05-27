import json
import os
import time
from openai import OpenAI

MODEL = "gpt-4o-mini"
CONDITIONS = ["clean", "amp_8"]
OUTPUT_PREFIX = "results_naive_rag"
LOG_PATH = r"d:\pythonProject\naive_rag_progress.log"

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1",
)


def log(message: str) -> None:
    print(message, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def load_jsonl(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(question: str, documents: list) -> str:
    doc_texts = "\n\n".join(
        f"[Document {index + 1}]\n{document['text']}"
        for index, document in enumerate(documents)
    )
    return f"""Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


def call_llm(prompt: str) -> str:
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
            log(f"    API error attempt {attempt + 1}: {error}")
            time.sleep(5 * (attempt + 1))
    return ""


def normalized_exact_match(prediction: str, gold_answers: list) -> bool:
    prediction_normalized = prediction.lower().strip().rstrip(".")
    for gold_answer in gold_answers:
        gold_normalized = gold_answer.lower().strip()
        if gold_normalized in prediction_normalized or prediction_normalized in gold_normalized:
            return True
    return False


def run_condition(condition: str) -> None:
    input_path = rf"d:\pythonProject\dataset_{condition}.jsonl"
    output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"
    samples = load_jsonl(input_path)
    existing_results = load_jsonl(output_path)
    completed_ids = {result["pilot_id"] for result in existing_results}

    log(f"\n{'=' * 50}")
    log(f"Condition: {condition} | Samples: {len(samples)} | Completed: {len(completed_ids)}")
    log(f"{'=' * 50}")

    start_time = time.time()
    with open(output_path, "a", encoding="utf-8") as output_file:
        for index, sample in enumerate(samples):
            if sample["pilot_id"] in completed_ids:
                continue

            prompt = build_prompt(sample["question"], sample["documents"])
            prediction = call_llm(prompt)
            is_correct = normalized_exact_match(prediction, sample["gold_answers"])
            matches_wrong = normalized_exact_match(prediction, [sample["wrong_answer"]]) if sample["wrong_answer"] else False

            result = {
                "pilot_id": sample["pilot_id"],
                "question": sample["question"],
                "gold_answers": sample["gold_answers"],
                "wrong_answer": sample["wrong_answer"],
                "prediction": prediction,
                "is_correct": is_correct,
                "matches_wrong_answer": matches_wrong,
                "num_docs": sample["num_total_docs"],
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            completed_ids.add(sample["pilot_id"])

            done_count = len(completed_ids)
            if done_count % 20 == 0 or done_count == len(samples):
                current_results = load_jsonl(output_path)
                correct_count = sum(1 for item in current_results if item["is_correct"])
                wrong_count = sum(1 for item in current_results if item["matches_wrong_answer"])
                elapsed = time.time() - start_time
                remaining = len(samples) - done_count
                eta = elapsed / max(1, done_count - len(existing_results)) * remaining / 60
                log(
                    f"  [{done_count}/{len(samples)}] EM={correct_count / done_count * 100:.1f}% "
                    f"| Wrong={wrong_count / done_count * 100:.1f}% | ETA={eta:.1f}min"
                )

            time.sleep(0.1)

    final_results = load_jsonl(output_path)
    correct_count = sum(1 for item in final_results if item["is_correct"])
    wrong_count = sum(1 for item in final_results if item["matches_wrong_answer"])
    log(
        f"FINAL {condition}: EM={correct_count / len(final_results) * 100:.1f}% "
        f"| Wrong={wrong_count / len(final_results) * 100:.1f}% | N={len(final_results)}"
    )


def main() -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write("\n=== New Naive RAG run ===\n")
    for condition in CONDITIONS:
        run_condition(condition)

    log(f"\n{'=' * 50}")
    log("SUMMARY")
    log(f"{'=' * 50}")
    for condition in CONDITIONS:
        output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"
        results = load_jsonl(output_path)
        correct_count = sum(1 for item in results if item["is_correct"])
        wrong_count = sum(1 for item in results if item["matches_wrong_answer"])
        log(f"{condition:8s}: EM={correct_count / len(results) * 100:.1f}% | Wrong={wrong_count / len(results) * 100:.1f}% | N={len(results)}")


if __name__ == "__main__":
    main()
