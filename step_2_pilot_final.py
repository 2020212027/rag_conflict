"""
Step 2 Pilot v2 Final - Sync with checkpoint, robust timeout.
"""
import json, time, os
from itertools import combinations
from collections import defaultdict
import urllib.request
import urllib.error
import ssl

MODEL = "gpt-4o-mini"
CLEAN_PATH = r"d:\pythonProject\dataset_clean.jsonl"
AMP8_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
CHECKPOINT = r"d:\pythonProject\pilot_checkpoint.json"
OUTPUT_PAIRS = r"d:\pythonProject\pilot_v2b_pairs.jsonl"
LOG_PATH = r"d:\pythonProject\pilot_v2b.log"
API_KEY = "sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD"
BASE_URL = "https://api.chatanywhere.tech/v1/chat/completions"


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def call_llm(prompt, max_tokens=400):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens
    }).encode("utf-8")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(BASE_URL, data=payload, headers=headers, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5)
            else:
                time.sleep(2)
        except Exception:
            time.sleep(3)
    return None


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

Both support the same answer. Judge if they are from same source or independent.

RULES:
- Same answer does NOT mean dependent.
- Focus on: shared rare details, same wording, same errors.

CALIBRATION:
- Same fact with DIFFERENT wording/details = D0/D1, NOT D3/D4.
- D3/D4 requires: shared RARE details not in question/answer, near-identical structure, or shared errors.
- Ask: "Could two people independently write these?" If yes -> D0/D1.

Question: {question}
Answer: {answer}

Unit A: facts={facts_a} details={details_a} span="{span_a}"
Unit B: facts={facts_b} details={details_b} span="{span_b}"

D0=Independent D1=Minimal D2=Moderate D3=High(same source) D4=Near-duplicate

Output ONLY valid JSON:
{{"level":"D0/D1/D2/D3/D4","score":0.0,"key_evidence":"one sentence"}}"""


def save_checkpoint(data):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    # Check for checkpoint
    ckpt = load_checkpoint()
    if ckpt and ckpt.get("phase") == "judge":
        log("Resuming from checkpoint (judgment phase)...")
        ind_pairs = ckpt["ind_pairs"]
        amp_pairs = ckpt["amp_pairs"]
        judgments = ckpt.get("judgments", [])
        start_idx = len(judgments)
    else:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            pass
        log("=" * 60)
        log("Pilot v2 Final - robust sync with checkpoint")
        log(f"Model: {MODEL}")
        log("=" * 60)

        with open(CLEAN_PATH, "r", encoding="utf-8") as f:
            clean_data = [json.loads(l) for l in f]
        with open(AMP8_PATH, "r", encoding="utf-8") as f:
            amp8_data = [json.loads(l) for l in f]

        # Select data
        ind_groups = []
        for idx, item in enumerate(clean_data):
            correct = [d for d in item["documents"] if d["type"] == "correct"]
            by_ans = defaultdict(list)
            for d in correct:
                by_ans[d["answer"].lower().strip()].append(d["text"])
            for ans, texts in by_ans.items():
                if len(texts) >= 2:
                    ind_groups.append((idx, item["question"], ans, texts))
        ind_groups.sort(key=lambda x: -len(x[3]))

        ind_docs = []
        pair_count = 0
        for qi, question, ans, texts in ind_groups:
            for t in texts:
                ind_docs.append((question, t, f"{qi}_{ans}"))
            pair_count += len(texts) * (len(texts) - 1) // 2
            if pair_count >= 80:
                break

        amp_queries = []
        for idx, item in enumerate(amp8_data):
            misinfo = [d for d in item["documents"] if d["type"] in ("misinfo", "misinfo_amplified")]
            if len(misinfo) >= 5:
                amp_queries.append((idx, item["question"], misinfo))
        amp_queries.sort(key=lambda x: -len(x[2]))

        amp_docs = []
        for qi, question, docs in amp_queries[:3]:
            for d in docs:
                amp_docs.append((question, d["text"], f"{qi}"))
            if len(amp_docs) >= 25:
                break

        log(f"\nInd docs: {len(ind_docs)}, Amp docs: {len(amp_docs)}")

        # Phase 1: Extract evidence
        log("\n--- Phase 1: Evidence Extraction ---")
        ind_ev = []
        for i, (q, t, gk) in enumerate(ind_docs):
            ev = call_llm(EVIDENCE_PROMPT.format(question=q, doc_text=t[:1500]))
            ind_ev.append(ev)
            if (i + 1) % 5 == 0:
                log(f"  ind {i+1}/{len(ind_docs)}")
        log(f"  ind done: {sum(1 for e in ind_ev if e and e.get('evidence_span'))}/{len(ind_docs)} valid")

        amp_ev = []
        for i, (q, t, gk) in enumerate(amp_docs):
            ev = call_llm(EVIDENCE_PROMPT.format(question=q, doc_text=t[:1500]))
            amp_ev.append(ev)
            if (i + 1) % 5 == 0:
                log(f"  amp {i+1}/{len(amp_docs)}")
        log(f"  amp done: {sum(1 for e in amp_ev if e and e.get('evidence_span'))}/{len(amp_docs)} valid")

        # Phase 2: Build pairs
        log("\n--- Phase 2: Pair Construction ---")
        ind_by_group = defaultdict(list)
        for (q, t, gk), ev in zip(ind_docs, ind_ev):
            if ev and ev.get("evidence_span") and ev.get("answer"):
                ind_by_group[gk].append({"question": q, "ev": ev})

        ind_pairs = []
        for gk, units in ind_by_group.items():
            if len(units) < 2:
                continue
            for i, j in combinations(range(len(units)), 2):
                ind_pairs.append({"cat": "independent_pair", "q": units[i]["question"],
                                  "a": units[i]["ev"], "b": units[j]["ev"]})
                if len(ind_pairs) >= 80:
                    break
            if len(ind_pairs) >= 80:
                break

        amp_by_query = defaultdict(list)
        for (q, t, gk), ev in zip(amp_docs, amp_ev):
            if ev and ev.get("evidence_span") and ev.get("answer"):
                amp_by_query[gk].append({"question": q, "ev": ev})

        amp_pairs = []
        for gk, units in amp_by_query.items():
            by_ans = defaultdict(list)
            for u in units:
                by_ans[u["ev"]["answer"].lower().strip()].append(u)
            for ans, group in by_ans.items():
                if len(group) < 2:
                    continue
                for i, j in combinations(range(len(group)), 2):
                    amp_pairs.append({"cat": "amplified_pair", "q": group[i]["question"],
                                      "a": group[i]["ev"], "b": group[j]["ev"]})
                    if len(amp_pairs) >= 80:
                        break
                if len(amp_pairs) >= 80:
                    break
            if len(amp_pairs) >= 80:
                break

        log(f"  Ind pairs: {len(ind_pairs)}, Amp pairs: {len(amp_pairs)}")
        judgments = []
        start_idx = 0
        save_checkpoint({"phase": "judge", "ind_pairs": ind_pairs, "amp_pairs": amp_pairs, "judgments": []})

    # Phase 3: Judge with checkpoint
    all_input = ind_pairs + amp_pairs
    log(f"\n--- Phase 3: Judgment ({start_idx}/{len(all_input)} done) ---")

    level_map = {"D0": 0.0, "D1": 0.25, "D2": 0.5, "D3": 0.75, "D4": 1.0}

    for i in range(start_idx, len(all_input)):
        p = all_input[i]
        ea, eb = p["a"], p["b"]
        j = call_llm(DEPENDENCE_PROMPT.format(
            question=p["q"], answer=ea.get("answer", ""),
            facts_a=json.dumps(ea.get("supporting_facts", []), ensure_ascii=False),
            details_a=json.dumps(ea.get("specific_details", []), ensure_ascii=False),
            span_a=ea.get("evidence_span", ""),
            facts_b=json.dumps(eb.get("supporting_facts", []), ensure_ascii=False),
            details_b=json.dumps(eb.get("specific_details", []), ensure_ascii=False),
            span_b=eb.get("evidence_span", "")
        ), max_tokens=200)

        if j is None:
            j = {"level": "ERROR", "score": -1, "key_evidence": "failed"}
        else:
            lv = j.get("level", "D2")
            if j.get("score", 0.0) == 0.0:
                j["score"] = level_map.get(lv, 0.5)
        judgments.append({"category": p["cat"], "judgment": j})

        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(all_input)}")
            save_checkpoint({"phase": "judge", "ind_pairs": ind_pairs, "amp_pairs": amp_pairs, "judgments": judgments})

    # Save final
    save_checkpoint({"phase": "judge", "ind_pairs": ind_pairs, "amp_pairs": amp_pairs, "judgments": judgments})
    with open(OUTPUT_PAIRS, "w", encoding="utf-8") as f:
        for r in judgments:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Analysis
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    from collections import Counter
    cats = defaultdict(list)
    for r in judgments:
        s = r["judgment"].get("score", -1)
        if s >= 0:
            cats[r["category"]].append((s, r["judgment"].get("level", "?")))

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
            log("\n  >>> Method viable with gpt-4o-mini!")
        elif auroc > 0.7:
            log("\n  >>> Moderate. Consider stronger model.")
        else:
            log("\n  >>> Poor separation.")

    log("\nDone.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
