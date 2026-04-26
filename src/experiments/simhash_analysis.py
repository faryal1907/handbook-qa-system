from src.simhash_retrieval import SimHashRetriever


def run_simhash_analysis():
    print("\n---SIMHASH DISTANCE ANALYSIS---\n")

    retriever = SimHashRetriever()

    queries = [
        "what is GPA requirement",
        "attendance rules",
        "grading criteria"
    ]

    for q in queries:
        print(f"\n================ Query: {q} ================\n")

        results = retriever.query(q)

        for r in results:
            dist = r["distance"]

            if dist < 10:
                label = "VERY SIMILAR"
            elif dist < 20:
                label = "MODERATE"
            else:
                label = "WEAK MATCH"

            print(f"[{label}] Distance={dist}")
            print(r["text"][:120])
            print("-")


if __name__ == "__main__":
    run_simhash_analysis()