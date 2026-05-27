"""
Step 1.4 - MADAM-RAG (Multi-Agent Debate) with 5x concurrency.
Adapted from official run_madam_rag.py to use GPT-4o-mini API.
Supports resume via append mode.
"""
import asyncio
import json
import os
import re
import string
import time
from typing import List
import httpx

API_KEY = "sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD"
BASE_URL = "https://api.chatanywhere.tech/v1/chat/completions"
MODEL = "gpt-4o-mini"
CONCURRENCY = 5
NUM_ROUNDS = 3
CONDITIONS = ["clean", "amp_8"]
OUTPUT_PREFIX = "results_madam_rag"
LOG_PATH = r"d:\pythonProject\madam_rag_progress.log"


def log(message: str):
    print(message, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def normalized_exact_match(prediction: str, gold_answers: list) -> bool:
    if not prediction:
        return False
    pred_norm = normalize_answer(prediction)
    return any(
        normalize_answer(gold) in pred_norm or pred_norm in normalize_answer(gold)
        for gold in gold_answers
    )


async def call_llm_async(client, semaphore, prompt, max_tokens=256):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.post(BASE_URL, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                elif resp.status_code == 429:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(5 * (attempt + 1))
    return ""


def build_agent_prompt(query, document, history=""):
    if history:
        return f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following reponses are from other agents as additional information.
{history}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.  
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""
    else:
        return f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""


def build_aggregation_prompt(query, responses):
    joined = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
    return f"""You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963. 
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{joined}
"""


def extract_answer_from_response(response):
    if "Answer:" in response and "Explanation" in response:
        return response[response.find("Answer:") + len("Answer:"):response.find("Explanation")].strip().rstrip(".")
    return response[:100]


def extract_final_answer(aggregation):
    """Extract answer from aggregation output like 'All Correct Answers: ["X", "Y"].'"""
    if "All Correct Answers:" in aggregation:
        start = aggregation.find("All Correct Answers:") + len("All Correct Answers:")
        end = aggregation.find("]", start) + 1
        try:
            answers_str = aggregation[start:end].strip()
            answers = json.loads(answers_str)
            if isinstance(answers, list) and answers:
                return answers[0]
        except (json.JSONDecodeError, IndexError):
            pass
    # Fallback: try to find first quoted answer
    match = re.search(r'"([^"]+)"', aggregation)
    if match:
        return match.group(1)
    return aggregation[:100]


async def run_madam_debate(client, semaphore, query, documents):
    num_agents = len(documents)
    agent_outputs = []

    # Round 1: each agent answers independently
    round1_tasks = [
        call_llm_async(client, semaphore, build_agent_prompt(query, doc))
        for doc in documents
    ]
    agent_outputs = await asyncio.gather(*round1_tasks)

    round1_answers = [extract_answer_from_response(r) for r in agent_outputs]

    # Aggregation for round 1
    agg1 = await call_llm_async(client, semaphore, build_aggregation_prompt(query, agent_outputs), max_tokens=512)

    final_aggregation = agg1
    actual_rounds = 1

    # Additional rounds (up to NUM_ROUNDS)
    for round_num in range(2, NUM_ROUNDS + 1):
        # Each agent sees all other agents' previous responses
        round_tasks = []
        for i, doc in enumerate(documents):
            history = "\n".join(
                [f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i]
            )
            round_tasks.append(
                call_llm_async(client, semaphore, build_agent_prompt(query, doc, history))
            )
        new_outputs = await asyncio.gather(*round_tasks)

        new_answers = [extract_answer_from_response(r) for r in new_outputs]
        prev_answers = [extract_answer_from_response(r) for r in agent_outputs]

        # Check convergence
        converged = True
        for k in range(len(new_answers)):
            na = normalize_answer(new_answers[k])
            pa = normalize_answer(prev_answers[k])
            if na not in pa and pa not in na:
                converged = False
                break

        agent_outputs = new_outputs
        actual_rounds = round_num

        if converged:
            break

        # Aggregation
        agg = await call_llm_async(client, semaphore, build_aggregation_prompt(query, agent_outputs), max_tokens=512)
        final_aggregation = agg

    return {
        "final_aggregation": final_aggregation,
        "final_answer": extract_final_answer(final_aggregation),
        "rounds_used": actual_rounds,
        "round1_answers": round1_answers,
    }


async def run_condition(condition):
    input_path = rf"d:\pythonProject\dataset_{condition}.jsonl"
    output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"

    samples = load_jsonl(input_path)
    existing = load_jsonl(output_path)
    completed_ids = {r["pilot_id"] for r in existing}
    remaining = [s for s in samples if s["pilot_id"] not in completed_ids]

    log(f"\n{'='*50}")
    log(f"MADAM-RAG {condition}: Total={len(samples)} Done={len(completed_ids)} Remaining={len(remaining)}")

    if not remaining:
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        with open(output_path, "a", encoding="utf-8") as out_file:
            for idx, sample in enumerate(remaining):
                documents = [doc["text"] for doc in sample["documents"]]
                debate_result = await run_madam_debate(client, semaphore, sample["question"], documents)

                prediction = debate_result["final_answer"]
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
                    "rounds_used": debate_result["rounds_used"],
                    "round1_answers": debate_result["round1_answers"],
                    "num_docs": sample["num_total_docs"],
                }
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()

                done_total = len(completed_ids) + idx + 1
                if (idx + 1) % 5 == 0 or idx == len(remaining) - 1:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (len(remaining) - idx - 1) / rate / 60
                    log(f"  [{done_total}/{len(samples)}] rate={rate:.2f} s/s ETA={eta:.1f}min rounds={debate_result['rounds_used']}")

    final_results = load_jsonl(output_path)
    em = sum(1 for r in final_results if r["is_correct"]) / len(final_results) * 100
    wr = sum(1 for r in final_results if r["matches_wrong_answer"]) / len(final_results) * 100
    log(f"  FINAL {condition}: EM={em:.1f}% Wrong={wr:.1f}% N={len(final_results)}")


async def main():
    open(LOG_PATH, "a", encoding="utf-8").write(f"\n=== MADAM-RAG started at {time.strftime('%H:%M:%S')} ===\n")
    for condition in CONDITIONS:
        await run_condition(condition)

    log(f"\n{'='*50}")
    log("MADAM-RAG SUMMARY")
    for condition in CONDITIONS:
        output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"
        results = load_jsonl(output_path)
        if results:
            em = sum(1 for r in results if r["is_correct"]) / len(results) * 100
            wr = sum(1 for r in results if r["matches_wrong_answer"]) / len(results) * 100
            log(f"  {condition:8s}: N={len(results)} EM={em:.1f}% Wrong={wr:.1f}%")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        import traceback
        with open(r"d:\pythonProject\madam_rag_error.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n{time.strftime('%H:%M:%S')}\n{traceback.format_exc()}")
        raise
