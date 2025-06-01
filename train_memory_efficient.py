#!/usr/bin/env python3
"""
Memory-efficient BPE training script using memory-mapped files.

This script demonstrates how to train BPE with reduced memory consumption
for large datasets by using memory-mapped files and streaming processing.
"""

import argparse
import time
from pathlib import Path
from tokenizer.bpe_mmap import MemoryEfficientBPE


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient BPE training")
    parser.add_argument("--input", required=True, help="Input text file path")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes (recommend 4-8 for large files)")
    parser.add_argument("--output-prefix", default="memory_efficient", help="Output file prefix")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"], help="Special tokens to include")
    
    args = parser.parse_args()
    
    print(f"Starting memory-efficient BPE training...")
    print(f"Input file: {args.input}")
    print(f"Target vocabulary size: {args.vocab_size}")
    print(f"Number of processes: {args.processes}")
    print(f"Special tokens: {args.special_tokens}")
    
    # Check input file size
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    file_size_gb = input_path.stat().st_size / (1024 ** 3)
    print(f"Input file size: {file_size_gb:.2f} GB")
    
    # Recommend process count based on file size and available RAM
    if file_size_gb > 5 and args.processes > 8:
        print(f"Warning: Large file ({file_size_gb:.1f}GB) with {args.processes} processes may consume too much RAM")
        print("Consider using --processes 4-8 for better memory efficiency")
    
    # Initialize tokenizer
    tokenizer = MemoryEfficientBPE()
    
    # Train with timing
    start_time = time.time()
    
    try:
        success = tokenizer.train(
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            num_processes=args.processes
        )
        
        if not success:
            print("Training failed")
            return 1
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 1
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Save outputs
    output_dir = Path("trainings")
    output_dir.mkdir(exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = output_dir / f"{args.output_prefix}_tokenizer.pkl"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to: {tokenizer_path}")
    
    # Save vocabulary
    vocab_path = output_dir / f"{args.output_prefix}_vocabulary.json"
    tokenizer.save_vocabulary(str(vocab_path))
    print(f"Saved vocabulary to: {vocab_path}")
    
    # Save merges
    merges_path = output_dir / f"{args.output_prefix}_merges.txt"
    tokenizer.save_merges(str(merges_path))
    print(f"Saved merges to: {merges_path}")
    
    print("Memory-efficient BPE training completed successfully!")
    print(f"Final vocabulary size: {len(tokenizer.vocabulary)}")
    
    return 0


if __name__ == "__main__":
    exit(main())