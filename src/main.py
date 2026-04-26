from tf_idf import TFIDFRetriever
from minhash_lsh import MinHashLSHRetriever
from simhash_retrieval import SimHashRetriever


def main():
    print("\n---Academic Policy QA System (Day 2 - Comparison Mode)---\n")

    tfidf = TFIDFRetriever()
    lsh = MinHashLSHRetriever()
    simhash = SimHashRetriever()

    while True:
        query = input("\nEnter your question (or 'exit'): ")

        if query.lower() == "exit":
            break

        print("\n====== TF-IDF Results ======")
        tfidf_results = tfidf.query(query)
        for r in tfidf_results:
            print(f"\nScore: {r['score']:.4f}")
            print(r["text"][:200])

        print("\n====== LSH Results ======")
        lsh_results = lsh.query(query)
        for r in lsh_results:
            print("\nChunk:")
            print(r["text"][:200])

        print("\n====== SimHash Results ======")
        sim_results = simhash.query(query)
        for r in sim_results:
            print(f"\nDistance: {r['distance']}")
            print(r["text"][:200])


if __name__ == "__main__":
    main()