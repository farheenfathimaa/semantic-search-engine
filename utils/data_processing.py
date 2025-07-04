import json
import os
import requests
import arxiv
from typing import List, Dict, Any
import random
import torch
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
    def load_sample_data(self) -> List[Dict[str, Any]]:
        """Load sample research papers data"""
        sample_file = os.path.join(self.data_dir, "sample_papers.json")
        
        if os.path.exists(sample_file):
            # Check if empty or invalid
            if os.path.getsize(sample_file) == 0:
                print("Sample data file is empty. Recreating...")
            else:
                with open(sample_file, "r", encoding="utf-8") as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        print("Sample data file is corrupt. Recreating...")
        # If file missing, empty, or corrupt, regenerate
        papers = self.create_sample_papers()
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        return papers
    
    def create_sample_papers(self) -> List[Dict[str, Any]]:
        """Create sample research papers dataset"""
        sample_papers = [
            {
                "id": "1",
                "title": "Deep Learning for Natural Language Processing: A Survey",
                "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
                "abstract": "This survey provides a comprehensive overview of deep learning techniques applied to natural language processing tasks...",
                "year": 2023,
                "venue": "Journal of Machine Learning Research",
                "keywords": ["deep learning", "natural language processing", "transformers", "LSTM"],
                "url": "https://example.com/paper1"
            },
            # ... (remaining sample papers unchanged)
        ]

        additional_papers = [
            {
                "id": "6",
                "title": "Quantum Computing: Algorithms and Applications",
                "authors": ["Wilson, P.", "Taylor, K.", "Anderson, M."],
                "abstract": "This comprehensive review examines quantum computing algorithms and their potential applications...",
                "year": 2023,
                "venue": "Nature Quantum Information",
                "keywords": ["quantum computing", "quantum algorithms", "quantum supremacy", "Shor's algorithm"],
                "url": "https://example.com/paper6"
            },
            # ... (remaining additional papers unchanged)
        ]

        return sample_papers + additional_papers
    
    def fetch_arxiv_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv (optional enhancement)"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                "id": result.entry_id,
                "title": result.title,
                "authors": [str(author) for author in result.authors],
                "abstract": result.summary,
                "year": result.published.year,
                "venue": "arXiv",
                "keywords": [category for category in result.categories],
                "url": result.entry_id
            }
            papers.append(paper)
        return papers
