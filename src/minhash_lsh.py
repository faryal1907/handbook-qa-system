import json
from datasketch import MinHash, MinHashLSH


def get_shingles(text, k=2):
    words = text.split()
    return set([" ".join(words[i:i+k]) for i in range(len(words)-k+1)])


class MinHashLSHRetriever:
    def __init__(self, chunk_file="chunks.json", num_perm=128):
        print("---Loading chunks for LSH---")

        with open(chunk_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=0.2, num_perm=num_perm)

        self.minhashes = {}

        print("---Building MinHash + LSH index---")

        for item in self.data:
            shingles = get_shingles(item["text"])

            m = MinHash(num_perm=num_perm)
            for shingle in shingles:
                m.update(shingle.encode("utf-8"))

            self.lsh.insert(str(item["id"]), m)
            self.minhashes[str(item["id"])] = m

    def query(self, text, top_k=5):
        shingles = get_shingles(text)

        m = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle.encode("utf-8"))

        result_ids = self.lsh.query(m)

        results = []

        # ✅ Normal LSH results
        if result_ids:
            for rid in result_ids:
                idx = int(rid)
                results.append({
                    "chunk_id": idx,
                    "text": self.data[idx]["text"]
                })

        # ✅ Fallback (VERY IMPORTANT)
        else:
            print("⚠️ No LSH matches found — using fallback similarity")

            similarities = []
            for key, mh in self.minhashes.items():
                sim = m.jaccard(mh)
                similarities.append((int(key), sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            for idx, sim in similarities[:top_k]:
                results.append({
                    "chunk_id": idx,
                    "text": self.data[idx]["text"],
                    "similarity": sim
                })

        return results[:top_k]