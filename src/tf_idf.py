import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    def __init__(self, chunk_file="chunks.json"):
        print("---Loading chunks---")
        with open(chunk_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.chunks = [d["text"] for d in self.data]

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