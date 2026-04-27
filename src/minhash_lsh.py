import json
import re
import math
from collections import Counter
from datasketch import MinHash, MinHashLSH
from config import CHUNKS_FILE


# -----------------------------
# TEXT PROCESSING
# -----------------------------
def get_shingles(text, k=2):  # k=2 gives more overlap than k=3
    text = text.lower()
    words = re.findall(r"\w+", text)
    
    # Remove stopwords to reduce noise
    stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "for", 
                 "that", "this", "with", "are", "be", "as", "at", "by"}
    words = [w for w in words if w not in stopwords]
    
    if len(words) < k:
        return set(words)
    
    return {
        " ".join(words[i:i + k])
        for i in range(len(words) - k + 1)
    }


def compute_tf_idf_weights(corpus_texts):
    """
    Compute TF-IDF weight for each term across corpus.
    Returns: dict of {term: idf_score}
    """
    N = len(corpus_texts)
    df = Counter()
    
    for text in corpus_texts:
        words = set(re.findall(r"\w+", text.lower()))
        for w in words:
            df[w] += 1
    
    idf = {term: math.log(N / (1 + freq)) for term, freq in df.items()}
    return idf


# -----------------------------
# WEIGHTED MINHASH HELPER
# -----------------------------
def build_weighted_minhash(text, idf_weights, num_perm=128):
    """
    Repeats each shingle proportional to its TF-IDF weight,
    simulating weighted Jaccard similarity within standard MinHash.
    """
    mh = MinHash(num_perm=num_perm)
    words = re.findall(r"\w+", text.lower())
    tf = Counter(words)
    total = sum(tf.values())
    
    for word, count in tf.items():
        tf_score = count / total
        idf_score = idf_weights.get(word, 0.0)
        weight = tf_score * idf_score
        
        # Repeat the shingle proportional to weight (integer rounding)
        repeats = max(1, int(weight * 100))
        for i in range(repeats):
            mh.update(f"{word}_{i}".encode("utf-8"))
    
    return mh


# -----------------------------
# FIXED LSH RETRIEVER
# -----------------------------
class MinHashLSHRetriever:

    def __init__(self, chunk_file=None, num_perm=128, threshold=0.05):
        chunk_file = chunk_file or CHUNKS_FILE
        print("\n---Loading chunks for LSH---")

        with open(chunk_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.num_perm = num_perm
        
        # Compute IDF weights across full corpus FIRST
        print("---Computing TF-IDF weights across corpus---")
        corpus_texts = [item["text"] for item in self.data]
        self.idf_weights = compute_tf_idf_weights(corpus_texts)

        # Low threshold is correct for sparse academic text
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}

        print("---Building weighted MinHash + LSH index---")

        for item in self.data:
            mh = build_weighted_minhash(
                item["text"], self.idf_weights, num_perm
            )
            key = str(item["id"])
            self.lsh.insert(key, mh)
            self.minhashes[key] = mh

        print(f"Indexed {len(self.data)} chunks into LSH\n")

    def query(self, text, top_k=5):
        # Build weighted MinHash for query using same IDF weights
        query_mh = build_weighted_minhash(
            text, self.idf_weights, self.num_perm
        )

        result_ids = self.lsh.query(query_mh)

        if not result_ids:
            # Fallback: rank ALL chunks by weighted Jaccard similarity
            print("⚠️ Sparse LSH match — using weighted MinHash similarity fallback")
            scores = [
                (idx, query_mh.jaccard(mh))
                for idx, mh in self.minhashes.items()
            ]
            scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {
                    "chunk_id": int(rid),
                    "text": self.data[int(rid)]["text"],
                    "score": round(score, 4),
                    "method": "fallback"
                }
                for rid, score in scores[:top_k]
            ]

        # Normal LSH bucket hit — rank results by similarity score
        scored = [
            (rid, query_mh.jaccard(self.minhashes[rid]))
            for rid in result_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "chunk_id": int(rid),
                "text": self.data[int(rid)]["text"],
                "score": round(score, 4),
                "method": "lsh"
            }
            for rid, score in scored[:top_k]
        ]