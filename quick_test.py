"""
Quick test script - Process only first 5 queries for testing
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from instance_search import InstanceSearchSystem, HybridFeatureExtractor

def quick_test():
    """Quick test with first 5 queries only"""
    
    print("="*60)
    print("QUICK TEST MODE - Processing first 5 queries only")
    print("="*60)
    
    gallery_path = "gallery/gallery"
    query_path = "gallery/query"
    query_txt_path = "query_txt"
    
    # Method 1: ResNet50
    print("\nTesting Method 1: ResNet50...")
    system_resnet = InstanceSearchSystem(gallery_path, query_path, query_txt_path)
    
    for query_id in range(5):
        print(f"\nProcessing query {query_id}...")
        ranked_list = system_resnet.process_query(query_id, method='resnet')
        
        print(f"Top 10 results for query {query_id}:")
        print(ranked_list[:10])
        
        # Visualize
        system_resnet.visualize_results(query_id, ranked_list, 'ResNet50_test', top_k=10)
    
    print("\n✓ Method 1 test complete!")
    
    # Method 2: SIFT + Histogram
    print("\n" + "="*60)
    print("Testing Method 2: SIFT + Histogram...")
    system_hybrid = HybridFeatureExtractor(gallery_path, query_path, query_txt_path)
    
    for query_id in range(5):
        print(f"\nProcessing query {query_id}...")
        ranked_list = system_hybrid.process_query(query_id)
        
        print(f"Top 10 results for query {query_id}:")
        print(ranked_list[:10])
    
    print("\n✓ Method 2 test complete!")
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE!")
    print("="*60)
    print("\nGenerated visualization files:")
    for i in range(5):
        print(f"  - results_query_{i}_ResNet50_test.png")
    
    print("\nIf everything looks good, run the full script:")
    print("  python instance_search.py")

if __name__ == "__main__":
    quick_test()
