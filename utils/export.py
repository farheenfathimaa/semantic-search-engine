# utils/export.py
import csv
import json
from typing import List, Dict, Any

def export_results(results: List[Dict[str, Any]], filepath: str, format: str = "csv"):
    """
    Export search results to CSV or JSON.
    Args:
        results: List of result dictionaries.
        filepath: Output file path.
        format: 'csv' or 'json'.
    """
    if format == "csv":
        keys = list(results[0].keys()) if results else []
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    elif format == "json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'json'.")
