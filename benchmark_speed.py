"""
Performance Benchmark: Old vs Fast Implementation
Run this to see the speed improvement!
"""

import time
import sys

def benchmark_old_method():
    """Test original serial implementation"""
    print("\n" + "="*60)
    print("BENCHMARKING OLD METHOD (Serial I/O)")
    print("="*60)
    
    from instance_search import InstanceSearchSystem
    
    gallery_path = "gallery/gallery"
    query_path = "gallery/query"
    query_txt_path = "query_txt"
    
    system = InstanceSearchSystem(gallery_path, query_path, query_txt_path)
    
    # Test with first 3 queries
    start_time = time.time()
    for query_id in range(3):
        print(f"\nProcessing query {query_id}...")
        ranked_list = system.process_query(query_id, method='resnet')
    
    elapsed = time.time() - start_time
    avg_time = elapsed / 3
    
    print(f"\n{'='*60}")
    print(f"Old Method Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average per query: {avg_time:.2f}s")
    print(f"  Estimated time for 50 queries: {avg_time * 50:.1f}s ({avg_time * 50 / 60:.1f} min)")
    print(f"{'='*60}\n")
    
    return avg_time


def benchmark_fast_method():
    """Test new parallel GPU implementation"""
    print("\n" + "="*60)
    print("BENCHMARKING FAST METHOD (GPU + Parallel + Index)")
    print("="*60)
    
    from instance_search_fast import FastInstanceSearchSystem
    
    gallery_path = "gallery/gallery"
    query_path = "gallery/query"
    query_txt_path = "query_txt"
    
    system = FastInstanceSearchSystem(
        gallery_path, query_path, query_txt_path,
        batch_size=128,
        num_workers=8
    )
    
    # Build index (one-time cost)
    print("\nBuilding index (one-time operation)...")
    index_start = time.time()
    system.build_gallery_index('gallery_index_resnet.npz', force_rebuild=False)
    index_time = time.time() - index_start
    print(f"Index build/load time: {index_time:.2f}s")
    
    # Test with first 3 queries
    print("\nProcessing queries...")
    start_time = time.time()
    for query_id in range(3):
        ranked_list = system.process_query_fast(query_id)
        print(f"  Query {query_id}: Done")
    
    elapsed = time.time() - start_time
    avg_time = elapsed / 3
    
    print(f"\n{'='*60}")
    print(f"Fast Method Results:")
    print(f"  Index time: {index_time:.2f}s (one-time cost)")
    print(f"  Query time (3 queries): {elapsed:.2f}s")
    print(f"  Average per query: {avg_time:.3f}s")
    print(f"  Estimated time for 50 queries: {avg_time * 50:.1f}s ({avg_time * 50 / 60:.1f} min)")
    print(f"  Total time including index: {index_time + avg_time * 50:.1f}s")
    print(f"{'='*60}\n")
    
    return avg_time, index_time


def main():
    print("\n" + "="*70)
    print(" "*15 + "PERFORMANCE BENCHMARK")
    print("="*70)
    
    choice = input("\nChoose benchmark:\n1. Old method only\n2. Fast method only\n3. Both (comparison)\n\nChoice (1-3): ")
    
    old_time = None
    fast_time = None
    index_time = 0
    
    if choice == '1' or choice == '3':
        old_time = benchmark_old_method()
    
    if choice == '2' or choice == '3':
        fast_time, index_time = benchmark_fast_method()
    
    if choice == '3' and old_time and fast_time:
        print("\n" + "="*70)
        print(" "*20 + "COMPARISON RESULTS")
        print("="*70)
        print(f"\nOld method per query:  {old_time:.2f}s")
        print(f"Fast method per query: {fast_time:.3f}s")
        print(f"\nSpeedup: {old_time/fast_time:.1f}x faster! 🚀")
        print(f"\nFor 50 queries:")
        print(f"  Old method:  ~{old_time * 50:.1f}s ({old_time * 50 / 60:.1f} min)")
        print(f"  Fast method: ~{fast_time * 50 + index_time:.1f}s ({(fast_time * 50 + index_time) / 60:.1f} min)")
        print(f"  Time saved:  ~{old_time * 50 - fast_time * 50 - index_time:.1f}s ({(old_time * 50 - fast_time * 50 - index_time) / 60:.1f} min)")
        print("="*70)
        print("\n✨ Your RTX 3070 is being fully utilized!")
        print("✨ Parallel data loading saves tons of I/O time!")
        print("✨ Vectorized similarity computation is blazing fast!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
