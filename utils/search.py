# utils/search.py
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

class SemanticSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.papers = []
        self.embeddings = None

    def create_embeddings(self, papers: List[Dict[str, Any]]):
        """Create embeddings for research papers"""
        self.papers = papers
        texts = [f"{p['title']} {p['abstract']}" for p in papers]
        print(f"Creating embeddings for {len(texts)} papers...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype(np.float32))

        self.save_embeddings()

    def save_embeddings(self):
        """Save embeddings and papers to disk"""
        os.makedirs("data", exist_ok=True)
        data_to_save = {
            'embeddings': self.embeddings,
            'papers': self.papers,
            'index': faiss.serialize_index(self.index)
        }
        with open("data/embeddings.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

    def load_embeddings(self):
        """Load embeddings and papers from disk"""
        with open("data/embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.papers = data['papers']
        self.index = faiss.deserialize_index(data['index'])

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar papers"""
        if self.index is None:
            raise ValueError("Search index not initialized")

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.papers):
                paper = self.papers[idx].copy()
                paper['similarity_score'] = float(score)
                paper['rank'] = i + 1
                results.append(paper)
        return results

    def search_with_filters(
        self,
        query: str,
        top_k: int = 10,
        year_range: tuple = None,
        venue: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar papers with optional filtering by year range and venue.
        """
        if self.index is None:
            raise ValueError("Search index not initialized")

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k * 2)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.papers):
                paper = self.papers[idx].copy()
                paper['similarity_score'] = float(score)
                paper['rank'] = i + 1

                if year_range:
                    if not (year_range[0] <= paper.get("year", 0) <= year_range[1]):
                        continue
                if venue:
                    if venue.lower() not in paper.get("venue", "").lower():
                        continue

                results.append(paper)
                if len(results) >= top_k:
                    break

        return results
