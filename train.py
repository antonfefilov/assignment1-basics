from multiprocessing import freeze_support
from tokenizer.bpe import BPE
import time
import threading
import psutil
import os
from typing import List, Tuple
import json
import pickle
from pathlib import Path
import argparse


class MemoryMonitor:
    """Real-time memory monitoring class"""

    def __init__(self, interval: float = 0.5):
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
            initial_avg = sum(memories[:5]) / 5
            final_avg = sum(memories[-5:]) / 5
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


if __name__ == "__main__":
    freeze_support()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer with memory monitoring")
    parser.add_argument(
        "-i",
        "--input",
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Input training file (default: data/TinyStoriesV2-GPT4-train.txt)",
    )
    parser.add_argument("-v", "--vocab-size", type=int, default=10000, help="Target vocabulary size (default: 10000)")
    parser.add_argument(
        "-s",
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens (default: ['<|endoftext|>'])",
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=4, help="Number of processes for parallel processing (default: 4)"
    )
    parser.add_argument(
        "-m", "--memory-interval", type=float, default=0.5, help="Memory monitoring interval in seconds (default: 0.5)"
    )

    args = parser.parse_args()

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(interval=args.memory_interval)

    # Print initial memory state
    initial_memory = memory_monitor.get_current_memory()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Start memory monitoring
    memory_monitor.start_monitoring()

    try:
        # Run the training
        start_time = time.time()
        print("Starting BPE training...")

        bpe = BPE()
        bpe.train(
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            num_processes=args.processes,
        )

        # Save the trained tokenizer state
        print("Saving tokenizer state...")

        # Extract base name from input file for naming output files
        input_file = Path(args.input)
        base_name = "trainings/" + input_file.stem

        # Save vocabulary using BPE's built-in method
        vocab_path = f"{base_name}_vocabulary.json"
        bpe.save_vocabulary(vocab_path)
        print(f"Vocabulary saved to: {vocab_path}")

        # Save merges using BPE's built-in method
        merges_path = f"{base_name}_merges.txt"
        bpe.save_merges(merges_path)
        print(f"Merges saved to: {merges_path}")

        # Save complete tokenizer using BPE's built-in method
        tokenizer_path = f"{base_name}_tokenizer.pkl"
        bpe.save(tokenizer_path)
        print(f"Complete tokenizer saved to: {tokenizer_path}")

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Vocabulary size: {len(bpe.vocabulary)}")
        print(f"Number of merges: {len(bpe.merges)}")

    finally:
        # Stop monitoring and print stats
        memory_monitor.stop_monitoring()
        memory_monitor.print_memory_stats()

        # Save memory log with timestamp
        input_base = Path(args.input).stem
        log_filename = f"memory_log_{input_base}_{int(time.time())}.csv"
        memory_monitor.save_memory_log(log_filename)
