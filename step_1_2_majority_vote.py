"""
Step 1.2 - Majority Vote Baseline
For each document, extract an answer. Then take majority vote.
Run on clean + amp_8.
"""
import json
import os
import time
from collections import Counter
from openai import OpenAI

MODEL = "gpt-4o-mini"
CONDITIONS = ["clean", "amp_8"]
OUTPUT_PREFIX = "results_majority_vote"
MAX_PER_RUN = 9999

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1",
)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def extract_answer_from_doc(question, doc_text):
    prompt = f"""Based ONLY on the following document, answer the question with a short, direct answer (a few words or a short phrase). If the document does not contain relevant information, say "NOT FOUND".

Question: {question}

Document:
{doc_text}

Answer:"""
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
            print(f"    API error attempt {attempt + 1}: {error}", flush=True)
            time.sleep(5 * (attempt + 1))
    return ""


def majority_vote(answers):
    """Take majority vote, ignoring NOT FOUND and empty answers."""
    filtered = [
        answer.lower().strip().rstrip(".")
        for answer in answers
        if answer and "not found" not in answer.lower() and "i don't know" not in answer.lower()
    ]
    if not filtered:
        return ""
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def normalized_exact_match(prediction, gold_answers):
    if not prediction:
        return False
    prediction_normalized = prediction.lower().strip().rstrip(".")
    return any(
        gold.lower().strip() in prediction_normalized or prediction_normalized in gold.lower().strip()
        for gold in gold_answers
    )


def run_condition(condition):
    input_path = rf"d:\pythonProject\dataset_{condition}.jsonl"
    output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"

    samples = load_jsonl(input_path)
    existing = load_jsonl(output_path)
    completed_ids = {result["pilot_id"] for result in existing}

    remaining = len(samples) - len(completed_ids)
    print(f"\n{'=' * 50}", flush=True)
    print(f"Condition: {condition} | Total: {len(samples)} | Done: {len(completed_ids)} | Remaining: {remaining}", flush=True)

    if remaining == 0:
        print("  Already complete!", flush=True)
        return

    processed = 0
    with open(output_path, "a", encoding="utf-8") as output_file:
        for sample in samples:
            if sample["pilot_id"] in completed_ids:
                continue

            # Extract answer from each document
            doc_answers = []
            for doc in sample["documents"]:
                answer = extract_answer_from_doc(sample["question"], doc["text"])
                doc_answers.append(answer)
                time.sleep(0.05)

            # Majority vote
            prediction = majority_vote(doc_answers)
            is_correct = normalized_exact_match(prediction, sample["gold_answers"])
            matches_wrong = normalized_exact_match(prediction, [sample["wrong_answer"]]) if sample["wrong_answer"] else False

            result = {
                "pilot_id": sample["pilot_id"],
                "question": sample["question"],
                "gold_answers": sample["gold_answers"],
                "wrong_answer": sample["wrong_answer"],
                "prediction": prediction,
                "doc_answers": doc_answers,
                "is_correct": is_correct,
                "matches_wrong_answer": matches_wrong,
                "num_docs": sample["num_total_docs"],
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            completed_ids.add(sample["pilot_id"])
            processed += 1

            if processed % 5 == 0:
                all_results = load_jsonl(output_path)
                em = sum(1 for r in all_results if r["is_correct"]) / len(all_results) * 100
                wr = sum(1 for r in all_results if r["matches_wrong_answer"]) / len(all_results) * 100
                print(f"  [{len(completed_ids)}/{len(samples)}] EM={em:.1f}% Wrong={wr:.1f}%", flush=True)

            if processed >= MAX_PER_RUN:
                print(f"  Batch limit reached ({MAX_PER_RUN}). Re-run to continue.", flush=True)
                return

    print(f"  Condition {condition} complete! Total processed this run: {processed}", flush=True)


def summarize():
    print(f"\n{'=' * 50}", flush=True)
    print("SUMMARY", flush=True)
    for condition in CONDITIONS:
        output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"
        if not os.path.exists(output_path):
            print(f"  {condition}: not started", flush=True)
            continue
        results = load_jsonl(output_path)
        if not results:
            print(f"  {condition}: empty", flush=True)
            continue
        em = sum(1 for r in results if r["is_correct"]) / len(results) * 100
        wr = sum(1 for r in results if r["matches_wrong_answer"]) / len(results) * 100
        print(f"  {condition:8s}: N={len(results)} | EM={em:.1f}% | Wrong={wr:.1f}%", flush=True)


if __name__ == "__main__":
    for condition in CONDITIONS:
        run_condition(condition)
    summarize()
