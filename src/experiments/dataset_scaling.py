import json
import time
from src.tf_idf import TFIDFRetriever


def expand_dataset(input_file="chunks.json", multiplier=5):
    print("\n---EXPANDING DATASET (Big Data Simulation)---\n")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    large_data = data * multiplier

    output_file = "chunks_large.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(large_data, f)

    print(f"New dataset size: {len(large_data)} chunks")
    print(f"Saved to: {output_file}")


def run_large_scale_test():
    print("\n---LARGE SCALE RETRIEVAL TEST---\n")

    tfidf = TFIDFRetriever()

    queries = [
        "attendance policy",
        "GPA requirement",
        "grading system"
    ]

    for q in queries:
        start = time.time()
        tfidf.query(q)
        end = time.time()

        print(f"Query: {q} | Time: {end - start:.5f} sec")


if __name__ == "__main__":
    expand_dataset()
    run_large_scale_test()