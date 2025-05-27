#!/usr/bin/env python3
"""Profile script for tokenizer training"""

import argparse
import cProfile
import pstats
import sys
from tokenizer.training import train

def profile_train(vocab_size: int = 500, input_path: str = "tests/fixtures/tinystories_sample.txt", 
                 num_processes: int = 1, num_stats: int = 30):
    """Profile the training function with configurable parameters
    
    Args:
        vocab_size: Target vocabulary size for training
        input_path: Path to the input training file
        num_processes: Number of processes for parallel processing
        num_stats: Number of top functions to display in profiling stats
    """
    special_tokens = ["<|pad|>"]
    
    print(f"Profiling BPE training with vocab_size={vocab_size}, input_path={input_path}")
    
    # Run the training
    vocab, merges = train(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    
    print(f"Completed training: {len(vocab)} vocab tokens, {len(merges)} merges")
    return vocab, merges

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Profile BPE tokenizer training")
    parser.add_argument("--vocab-size", type=int, default=500, 
                       help="Target vocabulary size (default: 500)")
    parser.add_argument("--input-path", type=str, default="tests/fixtures/tinystories_sample.txt",
                       help="Path to input training file")
    parser.add_argument("--num-processes", type=int, default=1,
                       help="Number of processes for parallel processing (default: 1)")
    parser.add_argument("--num-stats", type=int, default=30,
                       help="Number of top functions to show in stats (default: 30)")
    parser.add_argument("--no-profile", action="store_true",
                       help="Run without profiling (just time the execution)")
    
    args = parser.parse_args()
    
    if args.no_profile:
        # Just run without profiling
        profile_train(args.vocab_size, args.input_path, args.num_processes, args.num_stats)
    else:
        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        profile_train(args.vocab_size, args.input_path, args.num_processes, args.num_stats)
        
        profiler.disable()
        
        # Print stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(args.num_stats)

if __name__ == "__main__":
    main()