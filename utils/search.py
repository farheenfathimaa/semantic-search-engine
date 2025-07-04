import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle

class SemanticSearchEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.papers = []
        self.embeddings = None

    def create_embeddings(self, papers):
        self.papers = papers
        texts = [f"{p['title']} {p['abstract']}" for p in papers]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        with open("data/embeddings.pkl", "wb") as f:
            pickle.dump((self.papers, self.embeddings), f)

    def load_embeddings(self):
        with open("data/embeddings.pkl", "rb") as f:
            self.papers, self.embeddings = pickle.load(f)

    def search(self, query, top_k=10, year_range=None, venue=None):
        query_embedding = self.embedding_model.encode(query)
        cosine_scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(-cosine_scores)[:50]

        filtered = []
        for idx in top_indices:
            p = self.papers[idx]
            if year_range:
                if not (year_range[0] <= p["year"] <= year_range[1]):
                    continue
            if venue and venue.lower() not in p["venue"].lower():
                continue
            filtered.append((p, cosine_scores[idx]))

        if not filtered:
            return []

        rerank_inputs = [(query, f"{p['title']} {p['abstract']}") for p, _ in filtered]
        rerank_scores = self.cross_encoder.predict(rerank_inputs)
        reranked = sorted(zip(filtered, rerank_scores), key=lambda x: -x[1])

        return [
            {
                "title": p["title"],
                "abstract": p["abstract"],
                "url": p["url"],
                "year": p["year"],
                "venue": p["venue"],
                "score": f"{score:.2f}"
            }
            for ((p, _), score) in reranked[:top_k]
        ]
