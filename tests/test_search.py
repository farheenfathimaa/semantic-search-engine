import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.search import SemanticSearchEngine
from utils.data_processing import DataProcessor

def test_data_processor():
    """Test data processor functionality"""
    try:
        processor = DataProcessor()
        papers = processor.load_sample_data()
        
        assert len(papers) > 0, "No papers loaded"
        
        # Check required fields exist in first paper
        first_paper = papers[0]
        required_fields = ['title', 'abstract', 'authors']
        
        for field in required_fields:
            assert field in first_paper, f"Missing required field: {field}"
        
        print(f"✓ Loaded {len(papers)} papers successfully")
        return True
        
    except Exception as e:
        print(f"✗ test_data_processor failed: {e}")
        return False

def test_search_engine_basic():
    """Test basic search engine functionality"""
    try:
        search_engine = SemanticSearchEngine()
        
        # Create simple mock data that matches expected format
        mock_papers = [
            {
                'title': 'Deep Learning for Computer Vision',
                'abstract': 'This paper presents a comprehensive study of deep learning techniques for computer vision applications.',
                'authors': ['John Doe', 'Jane Smith'],
                'year': 2023,
                'venue': 'CVPR',
                'url': 'https://example.com/paper1'
            },
            {
                'title': 'Natural Language Processing with Transformers',
                'abstract': 'We explore the use of transformer models for various natural language processing tasks.',
                'authors': ['Alice Johnson', 'Bob Wilson'],
                'year': 2023,
                'venue': 'ACL',
                'url': 'https://example.com/paper2'
            },
            {
                'title': 'Machine Learning in Healthcare',
                'abstract': 'This work investigates machine learning applications in healthcare and medical diagnosis.',
                'authors': ['Carol Brown', 'David Lee'],
                'year': 2022,
                'venue': 'Nature Medicine',
                'url': 'https://example.com/paper3'
            }
        ]
        
        # Create embeddings
        search_engine.create_embeddings(mock_papers)
        print("✓ Embeddings created successfully")
        
        # Test search
        results = search_engine.search("deep learning computer vision", top_k=2)
        
        assert isinstance(results, list), "Results should be a list"
        assert len(results) > 0, "Should return at least one result"
        
        # Check the structure of results (adapt to your actual format)
        first_result = results[0]
        print(f"✓ First result structure: {list(first_result.keys())}")
        
        # Basic checks that should work regardless of exact format
        assert isinstance(first_result, dict), "Each result should be a dictionary"
        
        return True
        
    except Exception as e:
        print(f"✗ test_search_engine_basic failed: {e}")
        return False

def test_search_engine_with_real_data():
    """Test search engine with real data from processor"""
    try:
        # Load real data
        processor = DataProcessor()
        papers = processor.load_sample_data()
        
        if len(papers) == 0:
            print("⚠ No papers loaded, skipping real data test")
            return True
        
        # Use first 3 papers for testing
        test_papers = papers[:3]
        
        # Ensure all papers have required fields
        for i, paper in enumerate(test_papers):
            if 'url' not in paper:
                paper['url'] = f"https://example.com/paper{i+1}"
        
        search_engine = SemanticSearchEngine()
        search_engine.create_embeddings(test_papers)
        
        # Test search with a generic query
        results = search_engine.search("machine learning", top_k=2)
        
        assert isinstance(results, list), "Results should be a list"
        print(f"✓ Search returned {len(results)} results")
        
        if len(results) > 0:
            print(f"✓ First result keys: {list(results[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ test_search_engine_with_real_data failed: {e}")
        return False

def test_search_engine_edge_cases():
    """Test search engine edge cases"""
    try:
        search_engine = SemanticSearchEngine()
        
        # Create minimal mock data
        mock_papers = [
            {
                'title': 'Test Paper',
                'abstract': 'This is a test paper abstract.',
                'authors': ['Test Author'],
                'year': 2023,
                'venue': 'Test Venue',
                'url': 'https://example.com/test'
            }
        ]
        
        search_engine.create_embeddings(mock_papers)
        
        # Test empty query
        try:
            results = search_engine.search("", top_k=1)
            assert isinstance(results, list), "Should return list even for empty query"
            print("✓ Empty query handled gracefully")
        except Exception as e:
            print(f"⚠ Empty query handling: {e}")
        
        # Test normal query
        results = search_engine.search("test", top_k=1)
        assert isinstance(results, list), "Should return list for normal query"
        print("✓ Normal query works")
        
        return True
        
    except Exception as e:
        print(f"✗ test_search_engine_edge_cases failed: {e}")
        return False

def inspect_search_results():
    """Inspect the actual structure of search results"""
    try:
        print("\n=== Inspecting Search Results Structure ===")
        
        search_engine = SemanticSearchEngine()
        
        # Create test data
        mock_papers = [
            {
                'title': 'Deep Learning Research',
                'abstract': 'A comprehensive study of deep learning methods.',
                'authors': ['Researcher A'],
                'year': 2023,
                'venue': 'AI Conference',
                'url': 'https://example.com/dl'
            }
        ]
        
        search_engine.create_embeddings(mock_papers)
        results = search_engine.search("deep learning", top_k=1)
        
        if results:
            print(f"Result type: {type(results[0])}")
            print(f"Result keys: {list(results[0].keys())}")
            print(f"Sample result: {results[0]}")
        else:
            print("No results returned")
        
        return True
        
    except Exception as e:
        print(f"✗ inspect_search_results failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive tests...\n")
    
    # Run tests
    test_results = []
    
    test_results.append(("Data Processor", test_data_processor()))
    test_results.append(("Search Engine Basic", test_search_engine_basic()))
    test_results.append(("Search Engine Real Data", test_search_engine_with_real_data()))
    test_results.append(("Search Engine Edge Cases", test_search_engine_edge_cases()))
    test_results.append(("Inspect Results", inspect_search_results()))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_results)} tests")
    
    if passed == len(test_results):
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed - check the output above for details")