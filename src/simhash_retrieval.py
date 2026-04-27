import json
from simhash import Simhash
from config import CHUNKS_FILE


def get_features(text):
    return text.split()


class SimHashRetriever:
    def __init__(self, chunk_file=None):
        chunk_file = chunk_file or CHUNKS_FILE
        print("---Loading chunks for SimHash---")

        with open(chunk_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print("---Building SimHash fingerprints---")

        self.fingerprints = []
        for item in self.data:
            simhash = Simhash(get_features(item["text"]))
            self.fingerprints.append(simhash)

    def query(self, text, top_k=5):
        query_hash = Simhash(get_features(text))

        distances = []
        for i, fp in enumerate(self.fingerprints):
            dist = query_hash.distance(fp)
            distances.append((i, dist))

        distances.sort(key=lambda x: x[1])

        results = []
        for idx, dist in distances[:top_k]:
            results.append({
                "chunk_id": idx,
                "text": self.data[idx]["text"],
                "distance": dist
            })

        return results