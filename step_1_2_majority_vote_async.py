"""Step 1.2 - Majority Vote with 3x concurrency. Supports resume."""
import asyncio
import json
import os
import time
from collections import Counter
import httpx

API_KEY = "sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD"
BASE_URL = "https://api.chatanywhere.tech/v1/chat/completions"
MODEL = "gpt-4o-mini"
CONCURRENCY = 3
CONDITIONS = ["clean", "amp_8"]
OUTPUT_PREFIX = "results_majority_vote"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


async def extract_answer_async(client, semaphore, question, doc_text):
    prompt = f"""Based ONLY on the following document, answer the question with a short, direct answer (a few words or a short phrase). If the document does not contain relevant information, say "NOT FOUND".

Question: {question}

Document:
{doc_text}

Answer:"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.post(BASE_URL, json=payload, headers=headers, timeout=30)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(3 * (attempt + 1))
    return ""


def majority_vote(answers):
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


async def process_sample(client, semaphore, sample):
    tasks = [
        extract_answer_async(client, semaphore, sample["question"], doc["text"])
        for doc in sample["documents"]
    ]
    doc_answers = await asyncio.gather(*tasks)
    prediction = majority_vote(doc_answers)
    is_correct = normalized_exact_match(prediction, sample["gold_answers"])
    matches_wrong = normalized_exact_match(prediction, [sample["wrong_answer"]]) if sample["wrong_answer"] else False
    return {
        "pilot_id": sample["pilot_id"],
        "question": sample["question"],
        "gold_answers": sample["gold_answers"],
        "wrong_answer": sample["wrong_answer"],
        "prediction": prediction,
        "doc_answers": list(doc_answers),
        "is_correct": is_correct,
        "matches_wrong_answer": matches_wrong,
        "num_docs": sample["num_total_docs"],
    }


async def run_condition(condition):
    input_path = rf"d:\pythonProject\dataset_{condition}.jsonl"
    output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"

    samples = load_jsonl(input_path)
    existing = load_jsonl(output_path)
    completed_ids = {r["pilot_id"] for r in existing}
    remaining_samples = [s for s in samples if s["pilot_id"] not in completed_ids]

    print(f"\n{'=' * 50}", flush=True)
    print(f"{condition}: Total={len(samples)} Done={len(completed_ids)} Remaining={len(remaining_samples)}", flush=True)

    if not remaining_samples:
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        with open(output_path, "a", encoding="utf-8") as output_file:
            for index, sample in enumerate(remaining_samples):
                result = await process_sample(client, semaphore, sample)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()

                done_total = len(completed_ids) + index + 1
                if (index + 1) % 10 == 0 or index == len(remaining_samples) - 1:
                    elapsed = time.time() - start_time
                    rate = (index + 1) / elapsed
                    eta = (len(remaining_samples) - index - 1) / rate / 60
                    all_results = existing + load_jsonl(output_path)[len(existing):]
                    # Quick EM from what we have so far
                    print(f"  [{done_total}/{len(samples)}] rate={rate:.1f} samples/s ETA={eta:.1f}min", flush=True)

    final_results = load_jsonl(output_path)
    em = sum(1 for r in final_results if r["is_correct"]) / len(final_results) * 100
    wr = sum(1 for r in final_results if r["matches_wrong_answer"]) / len(final_results) * 100
    print(f"  FINAL {condition}: EM={em:.1f}% Wrong={wr:.1f}% N={len(final_results)}", flush=True)


async def main():
    for condition in CONDITIONS:
        await run_condition(condition)

    print(f"\n{'=' * 50}", flush=True)
    print("SUMMARY", flush=True)
    for condition in CONDITIONS:
        output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"
        results = load_jsonl(output_path)
        if results:
            em = sum(1 for r in results if r["is_correct"]) / len(results) * 100
            wr = sum(1 for r in results if r["matches_wrong_answer"]) / len(results) * 100
            print(f"  {condition:8s}: N={len(results)} | EM={em:.1f}% | Wrong={wr:.1f}%", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as fatal_error:
        with open(r"d:\pythonProject\majority_vote_error.log", "a", encoding="utf-8") as error_log:
            import traceback
            error_log.write(f"\n{'='*50}\n{time.strftime('%H:%M:%S')}\n")
            error_log.write(traceback.format_exc())
        raise
