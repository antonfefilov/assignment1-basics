#!/usr/bin/env python3
"""Profile script for tokenizer training with memory monitoring"""

import argparse
import cProfile
import pstats
import sys
import time
import threading
import psutil
import os
from typing import List, Tuple
from tokenizer.bpe import BPE

class MemoryMonitor:
    """Real-time memory monitoring class"""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.memory_usage: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.peak_memory = 0.0
        self.process = psutil.Process(os.getpid())
        self._monitor_thread = None

    def start_monitoring(self):
        """Start memory monitoring in a separate thread"""
        self.monitoring = True
        self.memory_usage = []
        self.peak_memory = 0.0
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print(f"Started memory monitoring (interval: {self.interval}s)")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        print(f"Stopped memory monitoring. Peak memory: {self.peak_memory:.2f} MB")

    def _monitor_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        while self.monitoring:
            try:
                # Get memory info
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

                timestamp = time.time() - start_time
                self.memory_usage.append((timestamp, memory_mb))

                # Update peak memory
                if memory_mb > self.peak_memory:
                    self.peak_memory = memory_mb

                time.sleep(self.interval)
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                break

    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024
        except:
            return 0.0

    def print_memory_stats(self):
        """Print detailed memory statistics"""
        if not self.memory_usage:
            print("No memory data collected")
            return

        memories = [mem for _, mem in self.memory_usage]
        avg_memory = sum(memories) / len(memories)
        min_memory = min(memories)
        max_memory = max(memories)

        print("\n" + "=" * 50)
        print("MEMORY USAGE STATISTICS")
        print("=" * 50)
        print(f"Peak Memory Usage:    {max_memory:.2f} MB")
        print(f"Average Memory Usage: {avg_memory:.2f} MB")
        print(f"Minimum Memory Usage: {min_memory:.2f} MB")
        print(f"Total Samples:        {len(self.memory_usage)}")
        print(f"Monitoring Duration:  {self.memory_usage[-1][0]:.2f} seconds")

        # Memory growth analysis
        if len(memories) > 10:
            initial_avg = sum(memories[:10]) / 10
            final_avg = sum(memories[-10:]) / 10
            growth = final_avg - initial_avg
            print(f"Memory Growth:        {growth:+.2f} MB")

        print("=" * 50)

    def save_memory_log(self, filename: str):
        """Save memory usage data to a file"""
        with open(filename, "w") as f:
            f.write("timestamp_seconds,memory_mb\n")
            for timestamp, memory in self.memory_usage:
                f.write(f"{timestamp:.3f},{memory:.2f}\n")
        print(f"Memory log saved to: {filename}")


def profile_train_with_memory(
    vocab_size: int = 500,
    input_path: str = "tests/fixtures/tinystories_sample.txt",
    num_processes: int = 1,
    num_stats: int = 30,
    monitor_interval: float = 0.1,
    save_memory_log: bool = False,
):
    """Profile the training function with memory monitoring

    Args:
        vocab_size: Target vocabulary size for training
        input_path: Path to the input training file
        num_processes: Number of processes for parallel processing
        num_stats: Number of top functions to display in profiling stats
        monitor_interval: Memory monitoring interval in seconds
        save_memory_log: Whether to save memory usage log to file
    """
    special_tokens = ["<|pad|>"]
    
    print(f"Profiling BPE training with vocab_size={vocab_size}, input_path={input_path}")

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(interval=monitor_interval)

    # Print initial memory state
    initial_memory = memory_monitor.get_current_memory()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Start memory monitoring
    memory_monitor.start_monitoring()

    try:
        # Run the training
        start_time = time.time()
        bpe = BPE()
        bpe.train(
            input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens, num_processes=num_processes
        )
        end_time = time.time()

        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Completed training: {len(bpe.vocabulary)} vocab tokens, {len(bpe.merges)} merges")

    finally:
        # Stop monitoring and print stats
        memory_monitor.stop_monitoring()
        memory_monitor.print_memory_stats()

        if save_memory_log:
            log_filename = f"memory_log_vocab{vocab_size}_{int(time.time())}.csv"
            memory_monitor.save_memory_log(log_filename)

    return bpe.vocabulary, bpe.merges


def profile_train(
    vocab_size: int = 500,
    input_path: str = "tests/fixtures/tinystories_sample.txt",
    num_processes: int = 1,
    num_stats: int = 30,
):
    """Original profile function for backward compatibility"""
    special_tokens = ["<|pad|>"]

    print(f"Profiling BPE training with vocab_size={vocab_size}, input_path={input_path}")

    # Run the training
    bpe = BPE()
    bpe.train(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens, num_processes=num_processes)

    print(f"Completed training: {len(bpe.vocabulary)} vocab tokens, {len(bpe.merges)} merges")
    return bpe.vocabulary, bpe.merges

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Profile BPE tokenizer training with memory monitoring")
    parser.add_argument("--vocab-size", type=int, default=500, 
                       help="Target vocabulary size (default: 500)")
    parser.add_argument("--input-path", type=str, default="tests/fixtures/tinystories_sample.txt",
                       help="Path to input training file")
    parser.add_argument("--num-processes", type=int, default=1,
                       help="Number of processes for parallel processing (default: 1)")
    parser.add_argument("--num-stats", type=int, default=30,
                       help="Number of top functions to show in stats (default: 30)")
    parser.add_argument("--no-profile", action="store_true", help="Run without CPU profiling (just time the execution)")
    parser.add_argument("--memory-only", action="store_true", help="Only monitor memory, skip CPU profiling")
    parser.add_argument(
        "--monitor-interval", type=float, default=0.1, help="Memory monitoring interval in seconds (default: 0.1)"
    )
    parser.add_argument("--save-memory-log", action="store_true", help="Save memory usage log to CSV file")
    
    args = parser.parse_args()

    if args.memory_only or args.no_profile:
        # Run with memory monitoring only
        profile_train_with_memory(
            args.vocab_size,
            args.input_path,
            args.num_processes,
            args.num_stats,
            args.monitor_interval,
            args.save_memory_log,
        )
    else:
        # Profile with both CPU and memory monitoring
        profiler = cProfile.Profile()
        profiler.enable()

        profile_train_with_memory(
            args.vocab_size,
            args.input_path,
            args.num_processes,
            args.num_stats,
            args.monitor_interval,
            args.save_memory_log,
        )
        
        profiler.disable()

        # Print CPU profiling stats
        print("\n" + "=" * 50)
        print("CPU PROFILING STATISTICS")
        print("=" * 50)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(args.num_stats)

if __name__ == "__main__":
    main()