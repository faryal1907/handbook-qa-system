import time
from src.tf_idf import TFIDFRetriever
from src.minhash_lsh import MinHashLSHRetriever
from src.simhash_retrieval import SimHashRetriever


def measure_time(name, func, query):
    start = time.time()
    func(query)
    end = time.time()

    print(f"{name} Time: {end - start:.5f} sec")


def run_scalability_test():
    print("\n---SCALABILITY / LATENCY EXPERIMENT---\n")

    query = "what is attendance policy"

    tfidf = TFIDFRetriever()
    lsh = MinHashLSHRetriever()
    simhash = SimHashRetriever()

    measure_time("TF-IDF", tfidf.query, query)
    measure_time("LSH", lsh.query, query)
    measure_time("SimHash", simhash.query, query)


if __name__ == "__main__":
    run_scalability_test()