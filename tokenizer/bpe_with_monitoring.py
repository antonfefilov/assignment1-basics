"""
BPE tokenizer with built-in memory monitoring capabilities.
This is an enhanced version of the BPE class that includes memory tracking.
"""

import time
import threading
import psutil
import os
from typing import List, Tuple, Optional
from .bpe import BPE


class BPEWithMonitoring(BPE):
    """
    BPE tokenizer with built-in memory monitoring.
    Extends the base BPE class to add memory tracking capabilities.
    """

    def __init__(self, monitor_interval: float = 0.5):
        super().__init__()
        self.monitor_interval = monitor_interval
        self.memory_monitor: Optional["MemoryTracker"] = None

    def train(
        self,
        *,
        input_path: str,
        vocab_size: int = 10000,
        special_tokens: list[str] = [],
        num_processes: int = 1,
        enable_monitoring: bool = True,
        save_memory_log: bool = False,
    ) -> bool:
        """
        Train a BPE tokenizer with optional memory monitoring.

        Args:
            input_path: Path to the input training file
            vocab_size: Target vocabulary size (default: 10000)
            special_tokens: List of special tokens to include in vocabulary
            num_processes: Number of processes for parallel pretokenization
            enable_monitoring: Whether to enable memory monitoring during training
            save_memory_log: Whether to save memory usage log to file

        Returns:
            True if training completed successfully
        """
        if enable_monitoring:
            self.memory_monitor = MemoryTracker(self.monitor_interval)
            self.memory_monitor.start_monitoring()
            print(f"Memory monitoring enabled (interval: {self.monitor_interval}s)")

        try:
            # Call the parent train method
            result = super().train(
                input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens, num_processes=num_processes
            )

            return result

        finally:
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
                self.memory_monitor.print_stats()

                if save_memory_log:
                    log_filename = f"bpe_memory_log_vocab{vocab_size}_{int(time.time())}.csv"
                    self.memory_monitor.save_log(log_filename)

    def get_memory_stats(self) -> dict:
        """Get memory statistics from the last training run"""
        if not self.memory_monitor or not self.memory_monitor.memory_usage:
            return {}

        memories = [mem for _, mem in self.memory_monitor.memory_usage]
        return {
            "peak_memory_mb": max(memories),
            "average_memory_mb": sum(memories) / len(memories),
            "minimum_memory_mb": min(memories),
            "memory_growth_mb": max(memories) - min(memories),
            "samples_collected": len(memories),
            "monitoring_duration_seconds": self.memory_monitor.memory_usage[-1][0],
        }


class MemoryTracker:
    """Lightweight memory tracking for BPE training"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitoring = False
        self.memory_usage: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.peak_memory = 0.0
        self.process = psutil.Process(os.getpid())
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self.memory_usage = []
        self.peak_memory = 0.0
        self._start_time = time.time()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                timestamp = time.time() - self._start_time

                self.memory_usage.append((timestamp, memory_mb))

                if memory_mb > self.peak_memory:
                    self.peak_memory = memory_mb

                time.sleep(self.interval)
            except Exception:
                break

    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def print_stats(self):
        """Print memory usage statistics"""
        if not self.memory_usage:
            print("No memory monitoring data available")
            return

        memories = [mem for _, mem in self.memory_usage]
        avg_memory = sum(memories) / len(memories)
        min_memory = min(memories)
        max_memory = max(memories)
        duration = self.memory_usage[-1][0]

        print("\n" + "=" * 50)
        print("BPE TRAINING MEMORY STATISTICS")
        print("=" * 50)
        print(f"Peak Memory Usage:    {max_memory:.2f} MB")
        print(f"Average Memory Usage: {avg_memory:.2f} MB")
        print(f"Minimum Memory Usage: {min_memory:.2f} MB")
        print(f"Memory Growth:        {max_memory - min_memory:+.2f} MB")
        print(f"Monitoring Duration:  {duration:.2f} seconds")
        print(f"Samples Collected:    {len(self.memory_usage)}")
        print("=" * 50)

    def save_log(self, filename: str):
        """Save memory usage to CSV file"""
        if not self.memory_usage:
            return

        with open(filename, "w") as f:
            f.write("timestamp_seconds,memory_mb\n")
            for timestamp, memory in self.memory_usage:
                f.write(f"{timestamp:.3f},{memory:.2f}\n")
        print(f"Memory log saved to: {filename}")


# Convenience function for quick memory-monitored training
def train_bpe_with_monitoring(
    input_path: str,
    vocab_size: int = 10000,
    special_tokens: list[str] | None = None,
    num_processes: int = 1,
    monitor_interval: float = 0.5,
    save_memory_log: bool = False,
) -> Tuple[BPEWithMonitoring, dict]:
    """
    Convenience function to train BPE with memory monitoring.

    Returns:
        Tuple of (trained_bpe_instance, memory_stats_dict)
    """
    if special_tokens is None:
        special_tokens = []

    bpe = BPEWithMonitoring(monitor_interval=monitor_interval)
    bpe.train(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
        enable_monitoring=True,
        save_memory_log=save_memory_log,
    )

    memory_stats = bpe.get_memory_stats()
    return bpe, memory_stats
