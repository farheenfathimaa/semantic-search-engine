import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import json
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
        
        # Prepare text for embedding (title + abstract)
        texts = []
        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            texts.append(text)
        
        print(f"Creating embeddings for {len(texts)} papers...")
        
        # Create embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype(np.float32))
        
        # Save embeddings and papers
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
            
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.papers):
                paper = self.papers[idx].copy()
                paper['similarity_score'] = float(score)
                paper['rank'] = i + 1
                results.append(paper)
                
        return results