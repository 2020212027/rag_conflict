"""
Step 2 - Plan B Pilot: LLM-as-Judge Dependence Detection
1. Select 5 queries from amp_8 dataset
2. Extract structured evidence units from each document
3. Run pairwise dependence judgment on same-answer pairs
4. Compare: amplified pairs (expected D3-D4) vs independent pairs (expected D0-D1)
"""
import json
import time
import asyncio
from itertools import combinations
from openai import AsyncOpenAI

MODEL = "gpt-4o-mini"
INPUT_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
OUTPUT_EVIDENCE = r"d:\pythonProject\pilot_evidence_units.jsonl"
OUTPUT_PAIRS = r"d:\pythonProject\pilot_dependence_pairs.jsonl"
LOG_PATH = r"d:\pythonProject\pilot_dependence.log"
NUM_QUERIES = 5
CONCURRENCY = 5

client = AsyncOpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

semaphore = asyncio.Semaphore(CONCURRENCY)


def log(message: str) -> None:
    print(message, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ─── Evidence Extraction ───────────────────────────────────────────────────────

EVIDENCE_PROMPT = """Given a question and ONE document, extract a structured evidence unit.

Question: {question}
Document: {doc_text}

Extract the following fields:
1. answer: What answer does this document suggest? (short, factual)
2. supporting_facts: Why does it support this answer? List specific reasons. NOT the answer itself.
3. specific_details: Unique numbers, dates, names, quotes, unusual phrases found in this document.
4. evidence_span: The single most relevant sentence from the document.

Output ONLY valid JSON:
{{"answer": "...", "supporting_facts": ["...", "..."], "specific_details": ["...", "..."], "evidence_span": "..."}}"""


async def extract_evidence(question: str, doc_text: str, doc_idx: int, query_idx: int) -> dict:
    async with semaphore:
        prompt = EVIDENCE_PROMPT.format(question=question, doc_text=doc_text[:2000])
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(content)
            return result
        except Exception as e:
            log(f"  [WARN] extract_evidence q{query_idx} doc{doc_idx}: {e}")
            return {"answer": "", "supporting_facts": [], "specific_details": [], "evidence_span": ""}


# ─── Dependence Judgment ───────────────────────────────────────────────────────

DEPENDENCE_PROMPT = """You are judging whether two evidence units are INDEPENDENT or DEPENDENT.

Both evidence units support the same answer to the same question.
Your job is NOT to judge whether the answer is correct.
Your job is to judge whether these two units provide independent information,
or whether they are likely copied, paraphrased, or derived from the same source.

IMPORTANT RULES:
- Two units supporting the same answer does NOT make them dependent.
- Focus on: Do they share the same supporting reasoning, the same rare details,
  the same wording, or the same errors?
- Generic facts shared by both are weak evidence of dependence.
- Rare details (specific numbers, dates, quotes, unusual names) shared by both
  are strong evidence of dependence.

Question: {question}
Candidate answer: {answer}

Evidence Unit A:
  supporting_facts: {facts_a}
  specific_details: {details_a}
  evidence_span: {span_a}

Evidence Unit B:
  supporting_facts: {facts_b}
  specific_details: {details_b}
  evidence_span: {span_b}

Rate the dependence level:
- D0: Independent. Different information paths, no shared rare details.
- D1: Minimal overlap. Same answer, maybe one shared common fact.
- D2: Moderate overlap. Several shared supporting facts, but different details.
- D3: High overlap. Shared rare details or highly similar reasoning chain. Likely same source.
- D4: Near-duplicate. Same wording, same details, same errors. Almost certainly same source.

Output ONLY valid JSON:
{{"level": "D0/D1/D2/D3/D4", "score": 0.0, "key_evidence": "one sentence explanation"}}"""


async def judge_dependence(question: str, answer: str, unit_a: dict, unit_b: dict,
                           pair_id: str) -> dict:
    async with semaphore:
        prompt = DEPENDENCE_PROMPT.format(
            question=question,
            answer=answer,
            facts_a=json.dumps(unit_a.get("supporting_facts", []), ensure_ascii=False),
            details_a=json.dumps(unit_a.get("specific_details", []), ensure_ascii=False),
            span_a=unit_a.get("evidence_span", ""),
            facts_b=json.dumps(unit_b.get("supporting_facts", []), ensure_ascii=False),
            details_b=json.dumps(unit_b.get("specific_details", []), ensure_ascii=False),
            span_b=unit_b.get("evidence_span", "")
        )
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(content)
            level_to_score = {"D0": 0.0, "D1": 0.25, "D2": 0.5, "D3": 0.75, "D4": 1.0}
            level = result.get("level", "D2")
            if "score" not in result or result["score"] == 0.0:
                result["score"] = level_to_score.get(level, 0.5)
            return result
        except Exception as e:
            log(f"  [WARN] judge_dependence {pair_id}: {e}")
            return {"level": "ERROR", "score": -1, "key_evidence": str(e)}


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

async def run_pilot():
    open(LOG_PATH, "w", encoding="utf-8").close()
    log("=" * 60)
    log("Plan B Pilot: LLM-as-Judge Dependence Detection")
    log(f"Model: {MODEL}, Concurrency: {CONCURRENCY}")
    log("=" * 60)

    # Load data - pick first NUM_QUERIES queries
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    samples = all_data[:NUM_QUERIES]
    log(f"\nSelected {len(samples)} queries for pilot test")

    # Phase 1: Extract evidence units
    log("\n--- Phase 1: Evidence Extraction ---")
    all_evidence = []  # list of {query_idx, doc_idx, type, evidence_unit}

    for qi, sample in enumerate(samples):
        question = sample["question"]
        docs = sample["documents"]
        log(f"\nQuery {qi}: {question[:60]}...")
        log(f"  Docs: {len(docs)} total, types: {[d['type'] for d in docs]}")

        tasks = []
        for di, doc in enumerate(docs):
            if doc["type"] == "noise":
                continue  # skip noise docs (answer=unknown)
            tasks.append((di, doc, extract_evidence(question, doc["text"], di, qi)))

        results = await asyncio.gather(*[t[2] for t in tasks])

        for (di, doc, _), evidence in zip(tasks, results):
            record = {
                "query_idx": qi,
                "doc_idx": di,
                "doc_type": doc["type"],
                "doc_answer": doc["answer"],
                "question": question,
                "evidence_unit": evidence
            }
            all_evidence.append(record)

        log(f"  Extracted {len(results)} evidence units")

    # Save evidence
    with open(OUTPUT_EVIDENCE, "w", encoding="utf-8") as f:
        for rec in all_evidence:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"\nSaved {len(all_evidence)} evidence units to {OUTPUT_EVIDENCE}")

    # Phase 2: Pairwise dependence judgment
    log("\n--- Phase 2: Pairwise Dependence Judgment ---")
    all_pairs = []

    for qi, sample in enumerate(samples):
        question = sample["question"]
        query_evidence = [e for e in all_evidence if e["query_idx"] == qi]

        # Group by answer
        answer_groups = {}
        for ev in query_evidence:
            ans = ev["evidence_unit"].get("answer", "").lower().strip()
            if ans not in answer_groups:
                answer_groups[ans] = []
            answer_groups[ans].append(ev)

        log(f"\nQuery {qi}: {len(answer_groups)} answer groups")

        for answer, group in answer_groups.items():
            if len(group) < 2:
                continue
            log(f"  Answer '{answer[:30]}': {len(group)} docs -> {len(list(combinations(range(len(group)), 2)))} pairs")

            pair_tasks = []
            for (idx_a, idx_b) in combinations(range(len(group)), 2):
                ev_a = group[idx_a]
                ev_b = group[idx_b]
                # Determine pair category
                type_a = ev_a["doc_type"]
                type_b = ev_b["doc_type"]
                if type_a in ("misinfo", "misinfo_amplified") and type_b in ("misinfo", "misinfo_amplified"):
                    pair_category = "amplified_pair"
                elif type_a == "correct" and type_b == "correct":
                    pair_category = "independent_pair"
                else:
                    pair_category = "mixed_pair"

                pair_id = f"q{qi}_d{ev_a['doc_idx']}_d{ev_b['doc_idx']}"
                task = judge_dependence(
                    question, answer,
                    ev_a["evidence_unit"], ev_b["evidence_unit"],
                    pair_id
                )
                pair_tasks.append({
                    "pair_id": pair_id,
                    "query_idx": qi,
                    "category": pair_category,
                    "type_a": type_a,
                    "type_b": type_b,
                    "task": task
                })

            results = await asyncio.gather(*[p["task"] for p in pair_tasks])

            for pair_info, result in zip(pair_tasks, results):
                record = {
                    "pair_id": pair_info["pair_id"],
                    "query_idx": pair_info["query_idx"],
                    "category": pair_info["category"],
                    "type_a": pair_info["type_a"],
                    "type_b": pair_info["type_b"],
                    "judgment": result
                }
                all_pairs.append(record)

    # Save pairs
    with open(OUTPUT_PAIRS, "w", encoding="utf-8") as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"\nSaved {len(all_pairs)} pair judgments to {OUTPUT_PAIRS}")

    # Phase 3: Analysis
    log("\n--- Phase 3: Results Analysis ---")
    analyze_results(all_pairs)


def analyze_results(all_pairs: list) -> None:
    categories = {}
    for pair in all_pairs:
        cat = pair["category"]
        score = pair["judgment"].get("score", -1)
        level = pair["judgment"].get("level", "ERROR")
        if score < 0:
            continue
        if cat not in categories:
            categories[cat] = {"scores": [], "levels": []}
        categories[cat]["scores"].append(score)
        categories[cat]["levels"].append(level)

    log("\n" + "=" * 60)
    log("PILOT RESULTS SUMMARY")
    log("=" * 60)

    for cat in ["amplified_pair", "independent_pair", "mixed_pair"]:
        if cat not in categories:
            continue
        scores = categories[cat]["scores"]
        levels = categories[cat]["levels"]
        avg_score = sum(scores) / len(scores) if scores else 0
        level_dist = {}
        for lv in levels:
            level_dist[lv] = level_dist.get(lv, 0) + 1

        log(f"\n[{cat}] ({len(scores)} pairs)")
        log(f"  Avg dependence score: {avg_score:.3f}")
        log(f"  Level distribution: {level_dist}")
        log(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")

    # Separability check
    if "amplified_pair" in categories and "independent_pair" in categories:
        amp_scores = categories["amplified_pair"]["scores"]
        ind_scores = categories["independent_pair"]["scores"]
        amp_mean = sum(amp_scores) / len(amp_scores)
        ind_mean = sum(ind_scores) / len(ind_scores)
        gap = amp_mean - ind_mean

        log(f"\n--- SEPARABILITY ---")
        log(f"  Amplified mean:   {amp_mean:.3f}")
        log(f"  Independent mean: {ind_mean:.3f}")
        log(f"  Gap:              {gap:.3f}")

        # Simple AUROC approximation
        correct_predictions = 0
        total_comparisons = 0
        for a_score in amp_scores:
            for i_score in ind_scores:
                total_comparisons += 1
                if a_score > i_score:
                    correct_predictions += 1
                elif a_score == i_score:
                    correct_predictions += 0.5
        auroc = correct_predictions / total_comparisons if total_comparisons > 0 else 0
        log(f"  AUROC:            {auroc:.3f}")

        if auroc > 0.8:
            log(f"\n  ✓ GOOD: Distributions are well-separated. Method is viable.")
        elif auroc > 0.6:
            log(f"\n  ~ MODERATE: Some separation. May need stronger model or better prompts.")
        else:
            log(f"\n  ✗ POOR: Distributions overlap heavily. Method foundation is weak.")
    else:
        log("\n  [NOTE] Not enough independent pairs for separability analysis.")
        log("  This is expected if correct docs don't share the same extracted answer.")

    log("\n" + "=" * 60)
    log("Pilot complete.")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(run_pilot())
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
