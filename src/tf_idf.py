import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from config import CHUNKS_FILE


class TFIDFRetriever:
    def __init__(self, chunk_file=None):
        chunk_file = chunk_file or CHUNKS_FILE
        print("---Loading chunks---")
        with open(chunk_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

                # -----------------------------
        # MAP STEP (Big Data simulation)
        # -----------------------------
        # Define common English stopwords to filter out
        stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "for", 
                     "that", "this", "with", "are", "be", "as", "at", "by"}

        def map_remove_stopwords(doc):
            # Genuinely useful distributed task: Tokenize and filter out noise
            words = doc["text"].split()
            filtered = [w for w in words if w not in stopwords]
            return " ".join(filtered)

        print("---Map step: removing stopwords in parallel---")
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.chunks = list(executor.map(map_remove_stopwords, self.data))


        # -----------------------------
        # REDUCE STEP (aggregation)
        # -----------------------------
        print("---Reduce step: assembling dataset---")
        self.chunks = [c for c in self.chunks]

        # -----------------------------
        # TF-IDF INDEXING
        # -----------------------------
        print("---Building TF-IDF index---")
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def query(self, question, top_k=5):
        print("---Processing query---")

        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx, score in ranked[:top_k]:
            results.append({
                "chunk_id": self.data[idx]["id"],
                "text": self.data[idx]["text"],
                "score": float(score)
            })

        return results