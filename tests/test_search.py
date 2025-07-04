import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.search import SemanticSearchEngine
from utils.data_processing import DataProcessor

def test_data_processor():
    """Test data processor functionality"""
    processor = DataProcessor()
    papers = processor.load_sample_data()
    
    assert len(papers) > 0
    assert 'title' in papers[0]
    assert 'abstract' in papers[0]
    assert 'authors' in papers[0]

def test_search_engine():
    """Test search engine functionality"""
    # Create a simple test
    search_engine = SemanticSearchEngine()
    
    # Test with sample data
    processor = DataProcessor()
    papers = processor.create_sample_papers()[:5]  # Use first 5 papers for testing
    
    search_engine.create_embeddings(papers)
    
    # Test search
    results = search_engine.search("machine learning", top_k=3)
    
    assert len(results) > 0
    assert 'similarity_score' in results[0]
    assert 'rank' in results[0]
    assert results[0]['rank'] == 1

if __name__ == "__main__":
    test_data_processor()
    test_search_engine()
    print("All tests passed!")