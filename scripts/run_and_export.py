# test_search.py
from utils.search import SemanticSearchEngine
from utils.export import export_results

# Initialize
search_engine = SemanticSearchEngine()
search_engine.load_embeddings()

# Perform filtered search
results = search_engine.search_with_filters(
    query="transformer",
    top_k=5,
    year_range=(2018, 2023),
    venue="NeurIPS"
)

# Export
export_results(results, "search_results.csv", format="csv")
export_results(results, "search_results.json", format="json")

print("Export completed successfully!")
