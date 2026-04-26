from tf_idf import TFIDFRetriever
from minhash_lsh import MinHashLSHRetriever
from simhash_retrieval import SimHashRetriever

# -----------------------------
# Fixed Evaluation Queries
# -----------------------------
QUERIES = [
    "What is the minimum GPA requirement?",
    "What is the attendance policy?",
    "What happens if a student fails a course?",
    "What is the grading criteria?",
    "Are there counseling services?",
    "What is hostel policy?",
    "How many credits are required to graduate?",
    "What is academic probation?",
    "Can a course be repeated?",
    "What are disciplinary actions?",
    "Does NUST provide student support?",
    "What is relative grading system?",
    "What is withdrawal policy?",
    "What are examination rules?",
    "What is campus life like?"
]


# -----------------------------
# Helper: Simple relevance check
# (manual keyword-based proxy)
# -----------------------------
def is_relevant(query, text):
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())

    overlap = len(q_words.intersection(t_words))

    # stricter threshold
    return overlap >= max(2, len(q_words) * 0.5)


# -----------------------------
# Evaluate a retriever
# -----------------------------
def evaluate_retriever(name, retriever, method="query"):
    print(f"\n================ {name} ================\n")

    total_precision = 0

    for q in QUERIES:
        results = retriever.query(q)

        if not results:
            print(f"[{q}] -> No results")
            continue

        # Top-5 evaluation
        top_k = results[:5]

        relevant = 0
        for r in top_k:
            if is_relevant(q, r["text"]):
                relevant += 1

        precision = relevant / len(top_k)
        total_precision += precision

        print(f"Query: {q}")
        print(f"Precision@5: {precision:.2f}")
        print("-" * 50)

    avg_precision = total_precision / len(QUERIES)

    print(f"\n📊 FINAL AVG Precision@5 for {name}: {avg_precision:.3f}\n")
    return avg_precision


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\n📊 Academic Policy QA System — Evaluation Module\n")

    tfidf = TFIDFRetriever()
    lsh = MinHashLSHRetriever()
    simhash = SimHashRetriever()

    tfidf_score = evaluate_retriever("TF-IDF (Exact)", tfidf)
    lsh_score = evaluate_retriever("MinHash + LSH (Approx)", lsh)
    simhash_score = evaluate_retriever("SimHash (Fingerprint)", simhash)

    print("\n================ FINAL COMPARISON ================\n")
    print(f"TF-IDF     Precision@5: {tfidf_score:.3f}")
    print(f"LSH        Precision@5: {lsh_score:.3f}")
    print(f"SimHash    Precision@5: {simhash_score:.3f}")

    print("\n✔ Evaluation Complete\n")


if __name__ == "__main__":
    main()