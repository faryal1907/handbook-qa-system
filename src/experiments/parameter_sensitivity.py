from src.minhash_lsh import MinHashLSHRetriever

QUERIES = [
    "minimum GPA requirement",
    "attendance policy",
    "grading system"
]

THRESHOLDS = [0.1, 0.2, 0.4, 0.6]


def run_lsh_sensitivity():
    print("\n---LSH PARAMETER SENSITIVITY EXPERIMENT---\n")

    for t in THRESHOLDS:
        print(f"\n================ Threshold = {t} ================\n")

        retriever = MinHashLSHRetriever()

        # dynamically adjust threshold
        retriever.lsh.threshold = t

        for q in QUERIES:
            results = retriever.query(q)

            print(f"\nQuery: {q}")
            print(f"Results: {len(results)}")

            for r in results[:2]:
                print("-", r["text"][:120])


if __name__ == "__main__":
    run_lsh_sensitivity()