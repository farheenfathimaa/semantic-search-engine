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
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def load_sample_data(self) -> List[Dict[str, Any]]:
        sample_file = os.path.join(self.data_dir, "sample_papers.json")
        if os.path.exists(sample_file):
            with open(sample_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Multi-query search
        queries = [
            "deep learning healthcare",
            "machine learning healthcare",
            "medical NLP",
            "AI in medical diagnosis",
            "clinical decision support"
        ]

        all_papers = []
        seen_ids = set()
        for q in queries:
            papers = self.fetch_arxiv_papers(q, max_results=100)
            for p in papers:
                if p["id"] not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p["id"])

        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)

        return all_papers

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
