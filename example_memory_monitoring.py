#!/usr/bin/env python3
"""
Example script demonstrating different ways to monitor memory usage during BPE training.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from tokenizer.bpe import BPE
from tokenizer.bpe_with_monitoring import BPEWithMonitoring, train_bpe_with_monitoring


def example_1_enhanced_profiling():
    """Example 1: Using the enhanced profiling script"""
    print("=" * 60)
    print("EXAMPLE 1: Enhanced Profiling Script")
    print("=" * 60)
    print("Run this command to profile with memory monitoring:")
    print("python profile_training.py --vocab-size 500 --memory-only --save-memory-log")
    print()
    print("Or for both CPU and memory profiling:")
    print("python profile_training.py --vocab-size 500 --save-memory-log")
    print()


def example_2_standalone_monitor():
    """Example 2: Using the standalone memory monitor"""
    print("=" * 60)
    print("EXAMPLE 2: Standalone Memory Monitor")
    print("=" * 60)
    print("1. First, start your BPE training in one terminal:")
    print(
        "   python -c \"from tokenizer.bpe import BPE; bpe = BPE(); bpe.train(input_path='tests/fixtures/tinystories_sample.txt', vocab_size=1000)\""
    )
    print()
    print("2. In another terminal, find the Python process:")
    print("   python memory_monitor.py --list-python")
    print()
    print("3. Monitor the process by PID:")
    print("   python memory_monitor.py --pid <PID> --save-log training_memory.csv")
    print()
    print("Or monitor by process name:")
    print("   python memory_monitor.py --process-name python --save-log training_memory.csv")
    print()


def example_3_integrated_monitoring():
    """Example 3: Using the BPE class with integrated monitoring"""
    print("=" * 60)
    print("EXAMPLE 3: Integrated Memory Monitoring")
    print("=" * 60)

    # Check if test file exists
    test_file = "tests/fixtures/tinystories_sample.txt"
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Creating a small sample...")
        os.makedirs("tests/fixtures", exist_ok=True)
        with open(test_file, "w") as f:
            f.write("Hello world. This is a test file for BPE training. " * 100)

    print("Training BPE with integrated memory monitoring...")

    # Use the enhanced BPE class
    bpe = BPEWithMonitoring(monitor_interval=0.1)  # Monitor every 100ms

    success = bpe.train(
        input_path=test_file, vocab_size=300, special_tokens=["<|pad|>"], enable_monitoring=True, save_memory_log=True
    )

    if success:
        print(f"\nTraining completed successfully!")
        print(f"Vocabulary size: {len(bpe.vocabulary)}")
        print(f"Number of merges: {len(bpe.merges)}")

        # Get memory statistics
        memory_stats = bpe.get_memory_stats()
        if memory_stats:
            print(f"\nMemory Statistics Summary:")
            for key, value in memory_stats.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


def example_4_convenience_function():
    """Example 4: Using the convenience function"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Convenience Function")
    print("=" * 60)

    test_file = "tests/fixtures/tinystories_sample.txt"
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Skipping this example.")
        return

    print("Using the convenience function for quick training with monitoring...")

    # Use the convenience function
    bpe, memory_stats = train_bpe_with_monitoring(
        input_path=test_file,
        vocab_size=200,
        special_tokens=["<|pad|>", "<|unk|>"],
        monitor_interval=0.2,
        save_memory_log=True,
    )

    print(f"\nTraining completed!")
    print(f"Vocabulary size: {len(bpe.vocabulary)}")
    print(f"Memory stats: {memory_stats}")


def main():
    """Run all examples"""
    print("BPE Training Memory Monitoring Examples")
    print("=" * 60)

    # Show command-line examples
    example_1_enhanced_profiling()
    example_2_standalone_monitor()

    # Run actual training examples
    try:
        example_3_integrated_monitoring()
        example_4_convenience_function()
    except Exception as e:
        print(f"Error running training examples: {e}")
        print("Make sure you have the required test files and dependencies.")

    print("\n" + "=" * 60)
    print("SUMMARY OF MONITORING OPTIONS")
    print("=" * 60)
    print("1. Enhanced profiling script: profile_training.py")
    print("   - Best for: Detailed CPU + memory profiling")
    print("   - Usage: python profile_training.py --memory-only --save-memory-log")
    print()
    print("2. Standalone monitor: memory_monitor.py")
    print("   - Best for: Monitoring external processes")
    print("   - Usage: python memory_monitor.py --process-name python")
    print()
    print("3. Integrated monitoring: BPEWithMonitoring class")
    print("   - Best for: Built-in monitoring in your code")
    print("   - Usage: from tokenizer.bpe_with_monitoring import BPEWithMonitoring")
    print()
    print("4. Convenience function: train_bpe_with_monitoring()")
    print("   - Best for: Quick one-liner training with monitoring")
    print("   - Usage: bpe, stats = train_bpe_with_monitoring(input_path, vocab_size)")


if __name__ == "__main__":
    main()
