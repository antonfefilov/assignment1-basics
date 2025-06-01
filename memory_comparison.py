#!/usr/bin/env python3
"""
Compare memory usage between standard and memory-mapped BPE training.
"""

import psutil
import os
import time
from pathlib import Path


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def monitor_training(training_func, *args, **kwargs):
    """Monitor memory usage during training."""
    print("Starting memory monitoring...")
    
    initial_memory = get_memory_usage()
    max_memory = initial_memory
    
    start_time = time.time()
    
    # Start training in background and monitor
    try:
        result = training_func(*args, **kwargs)
        current_memory = get_memory_usage()
        max_memory = max(max_memory, current_memory)
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None, 0, 0
    
    end_time = time.time()
    final_memory = get_memory_usage()
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Peak memory: {max_memory:.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Memory increase: {max_memory - initial_memory:.1f} MB")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    return result, max_memory - initial_memory, end_time - start_time


def compare_approaches(input_file: str, vocab_size: int = 1000, num_processes: int = 2):
    """Compare standard vs memory-mapped approaches."""
    
    print("=" * 60)
    print("MEMORY USAGE COMPARISON")
    print("=" * 60)
    
    # Test 1: Standard BPE
    print("\n1. Standard BPE Training:")
    print("-" * 30)
    
    from tokenizer.bpe import BPE
    
    def train_standard():
        tokenizer = BPE()
        return tokenizer.train(
            input_path=input_file,
            vocab_size=vocab_size,
            num_processes=num_processes
        )
    
    result1, memory1, time1 = monitor_training(train_standard)
    
    # Test 2: Memory-mapped BPE
    print("\n2. Memory-Mapped BPE Training:")
    print("-" * 30)
    
    from tokenizer.bpe_mmap import MemoryEfficientBPE
    
    def train_mmap():
        tokenizer = MemoryEfficientBPE()
        return tokenizer.train(
            input_path=input_file,
            vocab_size=vocab_size,
            num_processes=num_processes
        )
    
    result2, memory2, time2 = monitor_training(train_mmap)
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if memory1 > 0 and memory2 > 0:
        memory_savings = ((memory1 - memory2) / memory1) * 100
        print(f"Standard BPE memory usage: {memory1:.1f} MB")
        print(f"Memory-mapped BPE usage:   {memory2:.1f} MB")
        print(f"Memory savings: {memory_savings:.1f}% ({memory1 - memory2:.1f} MB)")
        
        if time1 > 0 and time2 > 0:
            time_diff = ((time2 - time1) / time1) * 100
            print(f"Time difference: {time_diff:+.1f}% ({time2 - time1:+.1f}s)")
    
    return {
        'standard_memory': memory1,
        'mmap_memory': memory2,
        'standard_time': time1,
        'mmap_time': time2
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare BPE memory usage")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size for testing")
    parser.add_argument("--processes", type=int, default=2, help="Number of processes")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    file_size = Path(args.input).stat().st_size / (1024 * 1024)
    print(f"Input file size: {file_size:.1f} MB")
    
    results = compare_approaches(args.input, args.vocab_size, args.processes)
    
    print("\nRecommendations:")
    if results['standard_memory'] > 0 and results['mmap_memory'] > 0:
        if results['mmap_memory'] < results['standard_memory'] * 0.8:
            print("✓ Use memory-mapped version for better memory efficiency")
        else:
            print("• Both versions have similar memory usage")
    
    return 0


if __name__ == "__main__":
    exit(main())