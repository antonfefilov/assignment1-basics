#!/usr/bin/env python3
"""Profile script for tokenizer training"""

import cProfile
import pstats
import sys
from tokenizer.training import train

def profile_train():
    """Profile the training function with a small dataset"""
    input_path = "tests/fixtures/tinystories_sample.txt"
    vocab_size = 500  # Small vocab for quick profiling
    special_tokens = ["<|pad|>"]
    
    # Run the training
    vocab, merges = train(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=1
    )
    
    print(f"Completed training: {len(vocab)} vocab tokens, {len(merges)} merges")

if __name__ == "__main__":
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    profile_train()
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions