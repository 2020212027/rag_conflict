"""
MADAM-RAG baseline on AmbigDocs, adapted to GPT-4o-mini API.
Prompts are identical to RAMDocs/run_madam_rag.py (official MADAM-RAG).
Eval uses same strict accuracy as run_ambigdocs.py.
Supports checkpoint/resume.
"""
import json, time, os, re, string
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MODEL = "gpt-4o-mini"
MAX_ROUNDS = 3
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)


def log(msg, log_path=None):
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def call_llm(prompt, max_tokens=256):
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=30,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 * (attempt + 1)
            if attempt < 4:
                time.sleep(wait)
    return ""


# ─── MADAM-RAG prompts (identical to official) ───────────────────────────────
def agent_response(query: str, document: str, history: str = ""):
    if history:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following reponses are from other agents as additional information.
{history}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.  
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""
    else:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

    return call_llm(prompt)


def aggregate_responses(query: str, responses: List[str]):
    joined = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
    prompt = f"""You are an aggregator reading answers from multiple agents.

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
    return call_llm(prompt, max_tokens=512)


MAX_WORKERS = 5


def parse_answer(resp):
    ans_start = resp.find("Answer: ")
    exp_start = resp.find("Explanation")
    if ans_start >= 0 and exp_start >= 0:
        return resp[ans_start + len("Answer: "):exp_start].strip()
    return resp[:100]


def multi_agent_debate(query: str, documents: List[str], num_rounds: int = 3):
    """Run MADAM-RAG debate with concurrent agent calls."""
    num_agents = len(documents)
    api_calls = 0

    # Round 1: concurrent
    agent_outputs = [None] * num_agents
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(agent_response, query, doc): i for i, doc in enumerate(documents)}
        for future in as_completed(futures):
            idx = futures[future]
            agent_outputs[idx] = future.result()
            api_calls += 1

    aggregation = aggregate_responses(query, agent_outputs)
    api_calls += 1

    prev_answers = [parse_answer(resp) for resp in agent_outputs]

    # Additional rounds
    for t in range(1, num_rounds):
        histories = []
        for i in range(num_agents):
            histories.append("\n".join([f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i]))

        new_outputs = [None] * num_agents
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(agent_response, query, documents[i], histories[i]): i for i in range(num_agents)}
            for future in as_completed(futures):
                idx = futures[future]
                new_outputs[idx] = future.result()
                api_calls += 1

        agent_outputs = new_outputs
        current_answers = [parse_answer(resp) for resp in agent_outputs]

        converged = True
        for k in range(len(current_answers)):
            ca = normalize_answer(current_answers[k])
            pa = normalize_answer(prev_answers[k])
            if ca in pa or pa in ca:
                continue
            else:
                converged = False

        if converged:
            break

        aggregation = aggregate_responses(query, agent_outputs)
        api_calls += 1
        prev_answers = current_answers

    return aggregation, api_calls


# ─── Evaluation (same as run_ambigdocs.py) ────────────────────────────────────
def answers_match(pred_norm, gold_norm):
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if len(gold_norm) >= 3 and gold_norm in pred_norm:
        return True
    return False


def compute_answer_recall(prediction, gold_answers):
    if not gold_answers:
        return 1.0
    pred_norm = normalize_answer(prediction)
    hits = 0
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if answers_match(pred_norm, gold_norm):
            hits += 1
    return hits / len(gold_answers)


def compute_strict_accuracy(prediction, gold_answers):
    return compute_answer_recall(prediction, gold_answers) == 1.0


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="MADAM-RAG on AmbigDocs")
    parser.add_argument("--dataset", type=str, default="AmbigDocs_data/test.json")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    output_path = os.path.join(RESULTS_DIR, "results_madam_ambigdocs.jsonl")
    ckpt_path = os.path.join(RESULTS_DIR, "ckpt_madam_ambigdocs.jsonl")
    log_path = os.path.join(LOGS_DIR, "madam_ambigdocs.log")

    with open(args.dataset, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if args.limit > 0:
        all_data = all_data[:args.limit]

    # Resume
    completed = set()
    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                completed.add(r["idx"])
                results.append(r)
        log(f"Resumed: {len(completed)} completed", log_path)

    log(f"=== MADAM-RAG on AmbigDocs (N={len(all_data)}) ===", log_path)
    log(f"Model: {MODEL}, Rounds: {MAX_ROUNDS}", log_path)
    log(f"Output: {output_path}\n", log_path)

    for idx, sample in enumerate(all_data):
        if idx in completed:
            continue

        question = sample["question"]
        documents = [d["text"] for d in sample["documents"]]
        gold_answers = [d["answer"] for d in sample["documents"]]

        # Run MADAM-RAG
        final_aggregation, api_calls = multi_agent_debate(question, documents, MAX_ROUNDS)

        recall = compute_answer_recall(final_aggregation, gold_answers)
        strict = compute_strict_accuracy(final_aggregation, gold_answers)

        result = {
            "idx": idx,
            "question": question,
            "gold_answers": gold_answers,
            "num_docs": len(documents),
            "madam_answer": final_aggregation,
            "recall": recall,
            "strict": strict,
            "api_calls": api_calls,
        }

        results.append(result)
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        log(f"  [{idx}] recall={recall:.2f} strict={'Y' if strict else 'N'} "
            f"| docs={len(documents)} | calls={api_calls}", log_path)

        if (len(results)) % 20 == 0 or idx == len(all_data) - 1:
            avg_recall = sum(r["recall"] for r in results) / len(results) * 100
            avg_strict = sum(r["strict"] for r in results) / len(results) * 100
            avg_calls = sum(r["api_calls"] for r in results) / len(results)
            log(f"\n--- Progress: {len(results)}/{len(all_data)} ---", log_path)
            log(f"  MADAM Answer Recall: {avg_recall:.1f}%", log_path)
            log(f"  MADAM Strict ACC:    {avg_strict:.1f}%", log_path)
            log(f"  Avg API calls:       {avg_calls:.1f}\n", log_path)

    # Final
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_recall = sum(r["recall"] for r in results) / len(results) * 100
    avg_strict = sum(r["strict"] for r in results) / len(results) * 100
    avg_calls = sum(r["api_calls"] for r in results) / len(results)

    log(f"\n{'='*60}", log_path)
    log(f"FINAL — MADAM-RAG on AmbigDocs (N={len(results)})", log_path)
    log(f"{'='*60}", log_path)
    log(f"  Answer Recall:  {avg_recall:.1f}%", log_path)
    log(f"  Strict ACC:     {avg_strict:.1f}%", log_path)
    log(f"  Avg API calls:  {avg_calls:.1f}", log_path)
    log(f"{'='*60}", log_path)


if __name__ == "__main__":
    main()
