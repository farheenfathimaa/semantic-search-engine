import json
import os
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import arxiv

class DataProcessor:
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def load_sample_data(self) -> List[Dict[str, Any]]:
        """Load or fetch research papers data"""
        sample_file = os.path.join(self.data_dir, "sample_papers.json")

        if os.path.exists(sample_file):
            if os.path.getsize(sample_file) == 0:
                print("Sample data file is empty. Recreating...")
            else:
                with open(sample_file, "r", encoding="utf-8") as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        print("Sample data file is corrupt. Recreating...")

        # Fetch real papers if no valid cache
        papers = self.fetch_arxiv_papers(
            query="machine learning OR deep learning OR computer vision OR natural language processing",
            max_results=100   # SAFE LIMIT
        )
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        return papers

    def fetch_arxiv_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv API using simple Search API (no Client)"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in search.results():
            paper = {
                "id": result.entry_id,
                "title": result.title.strip().replace("\n", " "),
                "authors": [str(author) for author in result.authors],
                "abstract": result.summary.strip().replace("\n", " "),
                "year": result.published.year,
                "venue": "arXiv",
                "keywords": result.categories,
                "url": result.entry_id
            }
            papers.append(paper)

        print(f"Fetched {len(papers)} papers from arXiv.")
        return papers
