"""
Step 2 Pilot v2 mini: Minimal docs, safe batches.
Only extract what's needed for 80 ind + 80 amp pairs.
"""
import json, time, asyncio
from itertools import combinations
from collections import defaultdict
from openai import AsyncOpenAI

MODEL = "gpt-4o-mini"
CLEAN_PATH = r"d:\pythonProject\dataset_clean.jsonl"
AMP8_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
OUTPUT_PAIRS = r"d:\pythonProject\pilot_v2b_pairs.jsonl"
LOG_PATH = r"d:\pythonProject\pilot_v2b.log"

client = AsyncOpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1",
    timeout=30.0
)

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

EVIDENCE_PROMPT = """Given a question and ONE document, extract:
1. answer: what answer does this document suggest? (short)
2. supporting_facts: why? (list reasons, NOT the answer)
3. specific_details: unique numbers, dates, names, quotes
4. evidence_span: most relevant sentence

Question: {question}
Document: {doc_text}

Output ONLY valid JSON:
{{"answer":"...","supporting_facts":["..."],"specific_details":["..."],"evidence_span":"..."}}"""

DEPENDENCE_PROMPT = """You judge whether two evidence units are INDEPENDENT or DEPENDENT.

Both support the same answer to the same question.
Judge whether they provide independent information or are copied/paraphrased from same source.

RULES:
- Same answer does NOT mean dependent.
- Focus on: shared rare details, same wording, same errors.
- Generic shared facts = weak evidence of dependence.
- Rare shared details = strong evidence.

CALIBRATION:
- Same well-known fact with DIFFERENT wording/examples/details = D0 or D1, NOT D3/D4.
- D3/D4 requires: shared RARE details not in the question/answer, near-identical structure, or shared errors.
- Ask: "Could two people independently write these?" If yes -> D0/D1.

Question: {question}
Answer: {answer}

Unit A:
  facts: {facts_a}
  details: {details_a}
  span: {span_a}

Unit B:
  facts: {facts_b}
  details: {details_b}
  span: {span_b}

Levels:
- D0: Independent. Different paths, no shared rare details.
- D1: Minimal overlap. Maybe one shared common fact.
- D2: Moderate overlap. Several shared facts, different details.
- D3: High overlap. Shared rare details, similar reasoning. Likely same source.
- D4: Near-duplicate. Same wording, details, errors.

Output ONLY valid JSON:
{{"level":"D0/D1/D2/D3/D4","score":0.0,"key_evidence":"one sentence"}}"""


async def call_api(prompt, max_tokens=400):
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=max_tokens
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content)
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(3)
            else:
                return None
    return None


async def batch_call(prompts, max_tokens=400, batch_size=3, delay=2.0):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_res = await asyncio.gather(*[call_api(p, max_tokens) for p in batch])
        results.extend(batch_res)
        if i + batch_size < len(prompts):
            await asyncio.sleep(delay)
        # Progress
        if (i // batch_size) % 10 == 0:
            log(f"    batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
    return results


async def main():
    open(LOG_PATH, "w", encoding="utf-8").close()
    log("=" * 60)
    log("Pilot v2 mini - fixed pairing + calibrated prompt")
    log(f"Model: {MODEL}")
    log("=" * 60)

    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        clean_data = [json.loads(l) for l in f]
    with open(AMP8_PATH, "r", encoding="utf-8") as f:
        amp8_data = [json.loads(l) for l in f]

    # ─── Select minimal independent docs ───
    # Find query groups where correct docs share same answer
    ind_groups = []  # (query_idx, answer, [doc_texts])
    for idx, item in enumerate(clean_data):
        correct = [d for d in item["documents"] if d["type"] == "correct"]
        by_ans = defaultdict(list)
        for d in correct:
            by_ans[d["answer"].lower().strip()].append(d["text"])
        for ans, texts in by_ans.items():
            if len(texts) >= 2:
                ind_groups.append((idx, item["question"], ans, texts))
    ind_groups.sort(key=lambda x: -len(x[3]))

    # Take enough groups to get ~80 pairs
    ind_docs_to_extract = []  # (question, text, group_key)
    target_pairs = 80
    pair_count = 0
    selected_groups = []
    for qi, question, ans, texts in ind_groups:
        n = len(texts)
        p = n * (n - 1) // 2
        selected_groups.append((qi, question, ans, texts))
        for t in texts:
            ind_docs_to_extract.append((question, t, f"{qi}_{ans}"))
        pair_count += p
        if pair_count >= target_pairs:
            break

    # Select amplified docs (just take first 3 queries with most misinfo docs)
    amp_docs_to_extract = []
    amp_queries = []
    for idx, item in enumerate(amp8_data):
        misinfo = [d for d in item["documents"] if d["type"] in ("misinfo", "misinfo_amplified")]
        if len(misinfo) >= 5:
            amp_queries.append((idx, item["question"], misinfo))
    amp_queries.sort(key=lambda x: -len(x[2]))

    amp_pair_count = 0
    selected_amp = []
    for qi, question, docs in amp_queries:
        selected_amp.append((qi, question, docs))
        for d in docs:
            amp_docs_to_extract.append((question, d["text"], f"{qi}"))
        amp_pair_count += len(docs) * (len(docs) - 1) // 2
        if amp_pair_count >= target_pairs:
            break

    log(f"\nIndependent: {len(ind_docs_to_extract)} docs -> ~{pair_count} pairs")
    log(f"Amplified:   {len(amp_docs_to_extract)} docs -> ~{amp_pair_count} pairs")
    log(f"Total API calls needed: ~{len(ind_docs_to_extract) + len(amp_docs_to_extract) + 160}")

    # ─── Phase 1: Extract evidence ───
    log("\n--- Phase 1: Evidence Extraction ---")
    log("  Independent docs...")
    ind_prompts = [EVIDENCE_PROMPT.format(question=q, doc_text=t[:1500]) for q, t, _ in ind_docs_to_extract]
    ind_ev = await batch_call(ind_prompts, max_tokens=400, batch_size=3, delay=2.0)
    ok = sum(1 for e in ind_ev if e and e.get("evidence_span"))
    log(f"  Done: {ok}/{len(ind_prompts)} valid")

    log("  Amplified docs...")
    amp_prompts = [EVIDENCE_PROMPT.format(question=q, doc_text=t[:1500]) for q, t, _ in amp_docs_to_extract]
    amp_ev = await batch_call(amp_prompts, max_tokens=400, batch_size=3, delay=2.0)
    ok = sum(1 for e in amp_ev if e and e.get("evidence_span"))
    log(f"  Done: {ok}/{len(amp_prompts)} valid")

    # ─── Phase 2: Build pairs ───
    log("\n--- Phase 2: Pair Construction ---")

    # Independent pairs: group evidence by group_key, pair within group
    ind_by_group = defaultdict(list)
    for (q, t, gk), ev in zip(ind_docs_to_extract, ind_ev):
        if ev and ev.get("evidence_span") and ev.get("answer"):
            ind_by_group[gk].append({"question": q, "ev": ev})

    ind_pair_list = []
    for gk, units in ind_by_group.items():
        if len(units) < 2:
            continue
        for i, j in combinations(range(len(units)), 2):
            ind_pair_list.append(("independent_pair", units[i], units[j]))
            if len(ind_pair_list) >= 80:
                break
        if len(ind_pair_list) >= 80:
            break

    # Amplified pairs: group by query, then by extracted answer
    amp_by_query = defaultdict(list)
    for (q, t, gk), ev in zip(amp_docs_to_extract, amp_ev):
        if ev and ev.get("evidence_span") and ev.get("answer"):
            amp_by_query[gk].append({"question": q, "ev": ev})

    amp_pair_list = []
    for gk, units in amp_by_query.items():
        # Group by answer
        by_ans = defaultdict(list)
        for u in units:
            by_ans[u["ev"]["answer"].lower().strip()].append(u)
        for ans, group in by_ans.items():
            if len(group) < 2:
                continue
            for i, j in combinations(range(len(group)), 2):
                amp_pair_list.append(("amplified_pair", group[i], group[j]))
                if len(amp_pair_list) >= 80:
                    break
            if len(amp_pair_list) >= 80:
                break
        if len(amp_pair_list) >= 80:
            break

    log(f"  Independent pairs: {len(ind_pair_list)}")
    log(f"  Amplified pairs:   {len(amp_pair_list)}")

    # ─── Phase 3: Judge dependence ───
    log("\n--- Phase 3: Dependence Judgment ---")
    all_pairs_input = ind_pair_list + amp_pair_list

    judge_prompts = []
    for cat, ua, ub in all_pairs_input:
        ea, eb = ua["ev"], ub["ev"]
        judge_prompts.append(DEPENDENCE_PROMPT.format(
            question=ua["question"],
            answer=ea.get("answer", ""),
            facts_a=json.dumps(ea.get("supporting_facts", []), ensure_ascii=False),
            details_a=json.dumps(ea.get("specific_details", []), ensure_ascii=False),
            span_a=ea.get("evidence_span", ""),
            facts_b=json.dumps(eb.get("supporting_facts", []), ensure_ascii=False),
            details_b=json.dumps(eb.get("specific_details", []), ensure_ascii=False),
            span_b=eb.get("evidence_span", "")
        ))

    log(f"  Judging {len(judge_prompts)} pairs...")
    judgments = await batch_call(judge_prompts, max_tokens=200, batch_size=3, delay=2.0)
    ok = sum(1 for j in judgments if j)
    log(f"  Done: {ok}/{len(judgments)} valid")

    # ─── Save & Analyze ───
    level_map = {"D0": 0.0, "D1": 0.25, "D2": 0.5, "D3": 0.75, "D4": 1.0}
    results = []
    for (cat, ua, ub), j in zip(all_pairs_input, judgments):
        if j is None:
            j = {"level": "ERROR", "score": -1, "key_evidence": "failed"}
        else:
            lv = j.get("level", "D2")
            if j.get("score", 0.0) == 0.0:
                j["score"] = level_map.get(lv, 0.5)
        results.append({"category": cat, "judgment": j})

    with open(OUTPUT_PAIRS, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Analysis
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    cats = defaultdict(list)
    for r in results:
        s = r["judgment"].get("score", -1)
        if s >= 0:
            cats[r["category"]].append((s, r["judgment"].get("level", "?")))

    from collections import Counter
    for cat in ["amplified_pair", "independent_pair"]:
        if cat not in cats:
            continue
        scores = [x[0] for x in cats[cat]]
        levels = [x[1] for x in cats[cat]]
        avg = sum(scores) / len(scores)
        dist = dict(Counter(levels).most_common())
        log(f"\n[{cat}] n={len(scores)}")
        log(f"  Avg: {avg:.3f}, Range: [{min(scores):.2f}, {max(scores):.2f}]")
        log(f"  Levels: {dist}")

    if "amplified_pair" in cats and "independent_pair" in cats:
        amp_s = [x[0] for x in cats["amplified_pair"]]
        ind_s = [x[0] for x in cats["independent_pair"]]
        correct = sum(1 for a in amp_s for i in ind_s if a > i) + \
                  0.5 * sum(1 for a in amp_s for i in ind_s if a == i)
        total = len(amp_s) * len(ind_s)
        auroc = correct / total if total else 0
        log(f"\n  Amp mean:  {sum(amp_s)/len(amp_s):.3f} (n={len(amp_s)})")
        log(f"  Ind mean:  {sum(ind_s)/len(ind_s):.3f} (n={len(ind_s)})")
        log(f"  Gap:       {sum(amp_s)/len(amp_s) - sum(ind_s)/len(ind_s):.3f}")
        log(f"  AUROC:     {auroc:.3f}")
        if auroc > 0.85:
            log("\n  >>> RESULT: Method viable with gpt-4o-mini!")
        elif auroc > 0.7:
            log("\n  >>> RESULT: Moderate. Consider stronger model.")
        else:
            log("\n  >>> RESULT: Poor separation. Reconsider.")

    log("\n" + "=" * 60)
    log("Done.")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"\nTotal: {time.time()-t0:.1f}s")
