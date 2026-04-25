from tf_idf import TFIDFRetriever


def main():
    print("\n---Academic Policy QA System (Day 1 - TF-IDF Baseline)---\n")

    retriever = TFIDFRetriever()

    while True:
        query = input("\nEnter your question (or 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting system.")
            break

        results = retriever.query(query)

        print("\n---Top Results---\n")

        for i, r in enumerate(results):
            print(f"\n[{i+1}] Score: {r['score']:.4f}")
            print(f"Chunk ID: {r['chunk_id']}")
            print(f"Text: {r['text'][:300]}...")
            print("-" * 80)


if __name__ == "__main__":
    main()