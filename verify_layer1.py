"""
Verify Layer 1 (TF-IDF cosine) on pilot pairs.
Pure Python - no numpy/sklearn needed.
"""
import json
import math
import re
from itertools import combinations
from collections import defaultdict, Counter

STOPWORDS = set("a an the is was were be been being have has had do does did will would shall should may might can could and or but if then else when at by for with about against between through during before after above below to from up down in out on off over under again further once here there all each every both few more most other some such no nor not only own same so than too very".split())


def tokenize(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOPWORDS and len(w) > 1]


def tfidf_cosine(text_a, text_b):
    """Compute TF-IDF cosine between two texts (self-contained)."""
    tokens_a = tokenize(text_a[:800])
    tokens_b = tokenize(text_b[:800])
    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    vocab = set(tf_a.keys()) | set(tf_b.keys())

    # IDF based on just these 2 docs
    doc_freq = {}
    for w in vocab:
        doc_freq[w] = (1 if w in tf_a else 0) + (1 if w in tf_b else 0)

    # TF-IDF vectors
    vec_a = {}
    vec_b = {}
    for w in vocab:
        idf = math.log(2.0 / doc_freq[w] + 1)
        vec_a[w] = tf_a.get(w, 0) * idf
        vec_b[w] = tf_b.get(w, 0) * idf

    # Cosine
    dot = sum(vec_a[w] * vec_b[w] for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(text_a, text_b):
    """Word-level Jaccard as secondary metric."""
    tokens_a = set(tokenize(text_a[:800]))
    tokens_b = set(tokenize(text_b[:800]))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def build_pairs():
    with open(r"d:\pythonProject\dataset_clean.jsonl", "r", encoding="utf-8") as f:
        clean_data = [json.loads(l) for l in f if l.strip()]
    with open(r"d:\pythonProject\dataset_amp_8.jsonl", "r", encoding="utf-8") as f:
        amp8_data = [json.loads(l) for l in f if l.strip()]

    # Independent pairs: correct docs grouped by answer
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

    ind_pairs = []
    pair_count = 0
    for qi, question, ans, texts in ind_groups:
        for t1, t2 in combinations(texts, 2):
            ind_pairs.append({"text_a": t1, "text_b": t2, "question": question})
            pair_count += 1
        if pair_count >= 80:
            break

    # Amplified pairs: misinfo docs
    amp_queries = []
    for idx, item in enumerate(amp8_data):
        misinfo = [d for d in item["documents"] if d["type"] in ("misinfo", "misinfo_amplified")]
        if len(misinfo) >= 5:
            amp_queries.append((idx, item["question"], misinfo))
    amp_queries.sort(key=lambda x: -len(x[2]))

    amp_pairs = []
    for qi, question, docs in amp_queries[:3]:
        texts = [d["text"] for d in docs]
        for t1, t2 in combinations(texts, 2):
            amp_pairs.append({"text_a": t1, "text_b": t2, "question": question})
            if len(amp_pairs) >= 80:
                break
        if len(amp_pairs) >= 80:
            break

    return ind_pairs[:80], amp_pairs[:80]


def main():
    ind_pairs, amp_pairs = build_pairs()
    print(f"Built {len(ind_pairs)} independent pairs, {len(amp_pairs)} amplified pairs")

    ind_cosines = [tfidf_cosine(p["text_a"], p["text_b"]) for p in ind_pairs]
    amp_cosines = [tfidf_cosine(p["text_a"], p["text_b"]) for p in amp_pairs]

    ind_jaccards = [jaccard_similarity(p["text_a"], p["text_b"]) for p in ind_pairs]
    amp_jaccards = [jaccard_similarity(p["text_a"], p["text_b"]) for p in amp_pairs]

    print("\n" + "=" * 60)
    print("LAYER 1 VERIFICATION: TF-IDF Cosine Similarity")
    print("=" * 60)

    def stats(arr):
        arr_s = sorted(arr)
        n = len(arr_s)
        return (sum(arr_s)/n, arr_s[n//2], min(arr_s), max(arr_s))

    m, med, mn, mx = stats(ind_cosines)
    print(f"\nIndependent cosine: mean={m:.3f} median={med:.3f} min={mn:.3f} max={mx:.3f}")
    m, med, mn, mx = stats(amp_cosines)
    print(f"Amplified cosine:   mean={m:.3f} median={med:.3f} min={mn:.3f} max={mx:.3f}")

    m, med, mn, mx = stats(ind_jaccards)
    print(f"\nIndependent jaccard: mean={m:.3f} median={med:.3f} min={mn:.3f} max={mx:.3f}")
    m, med, mn, mx = stats(amp_jaccards)
    print(f"Amplified jaccard:   mean={m:.3f} median={med:.3f} min={mn:.3f} max={mx:.3f}")

    print(f"\n--- Threshold Analysis (TF-IDF Cosine) ---")
    print(f"{'Thresh':<8} {'Ind pass':<10} {'Amp pass':<10} {'Ind%':<8} {'Amp%':<8} {'Precision':<10}")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        ind_pass = sum(1 for s in ind_cosines if s >= thresh)
        amp_pass = sum(1 for s in amp_cosines if s >= thresh)
        total_pass = ind_pass + amp_pass
        prec = amp_pass / total_pass * 100 if total_pass > 0 else 0
        print(f"  {thresh:<6.2f} {ind_pass:<10} {amp_pass:<10} "
              f"{ind_pass/len(ind_cosines)*100:<8.1f} {amp_pass/len(amp_cosines)*100:<8.1f} {prec:<10.1f}")

    print(f"\n--- Threshold Analysis (Jaccard) ---")
    print(f"{'Thresh':<8} {'Ind pass':<10} {'Amp pass':<10} {'Ind%':<8} {'Amp%':<8} {'Precision':<10}")
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        ind_pass = sum(1 for s in ind_jaccards if s >= thresh)
        amp_pass = sum(1 for s in amp_jaccards if s >= thresh)
        total_pass = ind_pass + amp_pass
        prec = amp_pass / total_pass * 100 if total_pass > 0 else 0
        print(f"  {thresh:<6.2f} {ind_pass:<10} {amp_pass:<10} "
              f"{ind_pass/len(ind_jaccards)*100:<8.1f} {amp_pass/len(amp_jaccards)*100:<8.1f} {prec:<10.1f}")

    # Histogram
    print(f"\n--- Cosine Histogram ---")
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"{'Range':<12} {'Independent':<14} {'Amplified':<14}")
    for lo, hi in buckets:
        ind_n = sum(1 for s in ind_cosines if lo <= s < hi)
        amp_n = sum(1 for s in amp_cosines if lo <= s < hi)
        print(f"  [{lo:.1f},{hi:.1f})  {ind_n:<14} {amp_n:<14}")


if __name__ == "__main__":
    main()
